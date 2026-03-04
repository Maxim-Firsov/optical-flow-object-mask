from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


Roi = Tuple[int, int, int, int]
FrameSize = Tuple[int, int]

OVERLAY_ALPHA = 0.35
ECC_ITERATIONS = 50
ECC_EPSILON = 1e-4
MOG2_HISTORY = 240
MOG2_VAR_THRESHOLD = 32.0
SMALL_KERNEL = 5
LARGE_KERNEL = 9
SEED_KERNEL = 15
GRABCUT_PADDING = 24
GRABCUT_ITERATIONS = 2
EDGE_BLUR_SIZE = 5
MAX_COMPONENT_ASPECT = 6.0
MIN_FILL_RATIO = 0.08
MIN_SOLIDITY = 0.22


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the foreground masking pipeline."""

    threshold: float
    downscale: float
    fps_override: float | None
    stabilize: bool
    keep_blobs: int
    min_area: int
    ema: float
    roi: Roi | None


@dataclass(frozen=True)
class VideoOutputs:
    """Output paths written by the processor."""

    mask_path: Path
    overlay_path: Path
    metadata_path: Path


@dataclass
class ProcessorState:
    """Per-video state carried between frames."""

    prev_gray: np.ndarray
    ema_mask: np.ndarray | None = None


@dataclass(frozen=True)
class ComponentCandidate:
    """Scored connected component candidate."""

    mask: np.ndarray
    area: int
    score: float


def parse_roi(roi_text: str | None) -> Roi | None:
    """Parse an ROI string in x,y,w,h form."""
    if roi_text is None:
        return None

    parts = [part.strip() for part in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be in x,y,w,h format.")

    try:
        x, y, w, h = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--roi must contain integer values.") from exc

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError("--roi requires x,y >= 0 and w,h > 0.")

    return x, y, w, h


def validate_roi(roi: Roi | None, frame_size: FrameSize) -> Roi | None:
    """Ensure the ROI remains inside the frame."""
    if roi is None:
        return None

    frame_width, frame_height = frame_size
    x, y, w, h = roi
    if x + w > frame_width or y + h > frame_height:
        raise ValueError(
            f"ROI {roi} exceeds frame bounds {frame_width}x{frame_height}."
        )
    return roi


def resize_frame(frame: np.ndarray, downscale: float) -> np.ndarray:
    """Resize a frame for faster processing when requested."""
    if downscale >= 1.0:
        return frame

    height, width = frame.shape[:2]
    target_width = max(1, int(width * downscale))
    target_height = max(1, int(height * downscale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def create_writer(output_path: Path, fps: float, frame_size: FrameSize) -> cv2.VideoWriter:
    """Create an MP4 video writer and fail fast if initialization fails."""
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
        True,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")
    return writer


def prepare_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to grayscale used for stabilization and differencing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def estimate_ecc_warp(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray | None:
    """Estimate an affine warp that aligns the current frame to the previous frame."""
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_ITERATIONS,
        ECC_EPSILON,
    )

    try:
        cv2.findTransformECC(
            templateImage=prev_gray,
            inputImage=curr_gray,
            warpMatrix=warp_matrix,
            motionType=cv2.MOTION_AFFINE,
            criteria=criteria,
        )
    except cv2.error:
        return None

    return warp_matrix


def warp_to_previous(frame: np.ndarray, warp_matrix: np.ndarray | None) -> np.ndarray:
    """Warp a current-frame image into previous-frame coordinates."""
    if warp_matrix is None:
        return frame

    height, width = frame.shape[:2]
    return cv2.warpAffine(
        frame,
        warp_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_mask_to_current(mask: np.ndarray, warp_matrix: np.ndarray | None) -> np.ndarray:
    """Move a mask from stabilized coordinates back to the original current frame."""
    if warp_matrix is None:
        return mask

    height, width = mask.shape[:2]
    return cv2.warpAffine(
        mask,
        warp_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def create_foreground_model() -> cv2.BackgroundSubtractor:
    """Create the primary foreground model."""
    # MOG2 models each pixel as a mixture of Gaussians so persistent background
    # modes are retained while transient moving regions are emitted as foreground.
    return cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False,
    )


def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Denoise and connect a binary mask."""
    small_kernel = np.ones((SMALL_KERNEL, SMALL_KERNEL), dtype=np.uint8)
    large_kernel = np.ones((LARGE_KERNEL, LARGE_KERNEL), dtype=np.uint8)
    # Median blur removes speckle noise; opening removes isolated artifacts;
    # closing reconnects fragmented foreground regions.
    cleaned = cv2.medianBlur(mask, 5)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, small_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, large_kernel)
    return cleaned


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes inside connected regions."""
    if mask.size == 0:
        return mask

    flood = mask.copy()
    h, w = mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def mask_bounding_box(mask: np.ndarray) -> Roi | None:
    """Return the tight bounding box around a binary mask."""
    points = cv2.findNonZero(mask)
    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    return int(x), int(y), int(w), int(h)


def expand_box(box: Roi, frame_size: FrameSize, padding: int) -> Roi:
    """Expand a box by padding while keeping it inside the frame."""
    x, y, w, h = box
    frame_width, frame_height = frame_size
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame_width, x + w + padding)
    y2 = min(frame_height, y + h + padding)
    return x1, y1, x2 - x1, y2 - y1


def build_motion_seed(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Build a motion seed from stabilized frame differencing."""
    # Frame differencing provides a second opinion beyond background subtraction,
    # which helps suppress static false positives from the MOG2 model alone.
    delta = cv2.absdiff(prev_gray, curr_gray)
    delta = cv2.GaussianBlur(delta, (5, 5), 0)
    delta_threshold = max(6, int(round(threshold * 6)))
    seed = np.where(delta >= delta_threshold, 255, 0).astype(np.uint8)
    seed = clean_binary_mask(seed)
    seed_kernel = np.ones((SEED_KERNEL, SEED_KERNEL), dtype=np.uint8)
    return cv2.dilate(seed, seed_kernel, iterations=1)


def blend_with_history(current_mask: np.ndarray, previous_ema: np.ndarray | None, ema: float) -> np.ndarray:
    """Use prior mask support to fill brief interior dropouts without introducing large trails."""
    current = (current_mask > 0).astype(np.float32)
    if ema <= 0.0 or previous_ema is None or previous_ema.shape != current.shape:
        return current

    history = previous_ema * (1.0 - ema)
    # Historical support is clipped to a dilation of the current mask so temporal
    # smoothing fills small holes without dragging large foreground ghosts forward.
    constrained_history = np.minimum(
        history,
        cv2.dilate(current, np.ones((LARGE_KERNEL, LARGE_KERNEL), dtype=np.uint8)).astype(np.float32),
    )
    return np.maximum(current, constrained_history)


def build_grabcut_seed_mask(coarse_mask: np.ndarray) -> np.ndarray:
    """Convert a coarse binary mask into a GrabCut trimap."""
    trimap = np.full(coarse_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    if not np.any(coarse_mask):
        trimap[:, :] = cv2.GC_BGD
        return trimap

    # GrabCut works best with a trimap that separates definite foreground from
    # probable foreground and background, so the coarse matte is expanded and eroded.
    outer = cv2.dilate(
        coarse_mask,
        np.ones((LARGE_KERNEL * 2 + 1, LARGE_KERNEL * 2 + 1), dtype=np.uint8),
        iterations=1,
    )
    inner = cv2.erode(
        coarse_mask,
        np.ones((SMALL_KERNEL, SMALL_KERNEL), dtype=np.uint8),
        iterations=1,
    )

    trimap[outer == 0] = cv2.GC_BGD
    trimap[coarse_mask > 0] = cv2.GC_PR_FGD
    trimap[inner > 0] = cv2.GC_FGD
    return trimap


def edge_aware_cleanup(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Snap the matte back toward visible image edges without introducing holes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    constrained = cv2.bitwise_or(mask, cv2.bitwise_and(edges, cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8))))
    constrained = cv2.GaussianBlur(constrained, (EDGE_BLUR_SIZE, EDGE_BLUR_SIZE), 0)
    cleaned = np.where(constrained >= 32, 255, 0).astype(np.uint8)
    return fill_mask_holes(clean_binary_mask(cleaned))


def refine_mask_with_grabcut(frame: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
    """Refine a coarse subject mask with GrabCut initialized from the current matte."""
    bbox = mask_bounding_box(coarse_mask)
    if bbox is None:
        return coarse_mask

    frame_height, frame_width = coarse_mask.shape[:2]
    x, y, w, h = expand_box(bbox, (frame_width, frame_height), GRABCUT_PADDING)
    frame_crop = frame[y : y + h, x : x + w]
    mask_crop = coarse_mask[y : y + h, x : x + w]
    trimap = build_grabcut_seed_mask(mask_crop)

    if not np.any(trimap == cv2.GC_FGD):
        return coarse_mask

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            frame_crop,
            trimap,
            None,
            bg_model,
            fg_model,
            GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return coarse_mask

    refined_crop = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    refined_crop = fill_mask_holes(clean_binary_mask(refined_crop))

    refined = np.zeros_like(coarse_mask)
    refined[y : y + h, x : x + w] = refined_crop
    refined = cv2.bitwise_or(refined, cv2.erode(coarse_mask, np.ones((3, 3), dtype=np.uint8), iterations=1))
    return edge_aware_cleanup(frame, refined)


def score_component(
    mask: np.ndarray,
    motion_overlap: int,
    bbox: Roi,
    area: int,
) -> ComponentCandidate | None:
    """Score a component by generic region quality and motion support."""
    _, _, w, h = bbox
    if min(w, h) < 8:
        return None

    bbox_area = max(w * h, 1)
    fill_ratio = float(area / bbox_area)
    aspect_ratio = float(max(w / max(h, 1), h / max(w, 1)))
    if fill_ratio < MIN_FILL_RATIO or aspect_ratio > MAX_COMPONENT_ASPECT:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    hull_area = float(cv2.contourArea(cv2.convexHull(contours[0])))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0
    if solidity < MIN_SOLIDITY:
        return None

    overlap_ratio = float(motion_overlap / max(area, 1))
    # The score is heuristic rather than learned: larger, denser, more convex
    # regions with stronger motion support are preferred as candidate subjects.
    score = (area * 0.01) + (fill_ratio * 2.0) + (solidity * 1.5) + (overlap_ratio * 3.0)
    return ComponentCandidate(mask=mask, area=area, score=score)


def select_components(
    base_mask: np.ndarray,
    motion_seed: np.ndarray,
    keep_blobs: int,
    min_area: int,
) -> np.ndarray:
    """Keep the best foreground components supported by motion."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(base_mask, connectivity=8)
    candidates: list[ComponentCandidate] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        component_mask = np.zeros_like(base_mask)
        component_mask[labels == label] = 255
        motion_overlap = int(np.count_nonzero((component_mask > 0) & (motion_seed > 0)))
        if motion_overlap == 0:
            continue

        filled_component = fill_mask_holes(component_mask)
        candidate = score_component(filled_component, motion_overlap, (x, y, w, h), area)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return np.zeros_like(base_mask)

    candidates.sort(key=lambda candidate: (candidate.score, candidate.area), reverse=True)
    selected = np.zeros_like(base_mask)
    for candidate in candidates[:keep_blobs]:
        selected = cv2.bitwise_or(selected, candidate.mask)

    return fill_mask_holes(clean_binary_mask(selected))


def paste_roi_mask(mask_roi: np.ndarray, frame_size: FrameSize, roi: Roi | None) -> np.ndarray:
    """Paste an ROI-sized mask back into full-frame coordinates."""
    if roi is None:
        return mask_roi

    frame_width, frame_height = frame_size
    x, y, w, h = roi
    full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    full_mask[y : y + h, x : x + w] = mask_roi[:h, :w]
    return full_mask


def create_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a semi-transparent green mask onto the current frame."""
    overlay_color = np.zeros_like(frame)
    overlay_color[..., 1] = mask
    return cv2.addWeighted(frame, 1.0, overlay_color, OVERLAY_ALPHA, 0.0)


class MotionMaskProcessor:
    """Foreground masking pipeline optimized for output alignment and full-region coverage."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.foreground_model = create_foreground_model()

    def _output_fps(self, capture: cv2.VideoCapture) -> float:
        input_fps = capture.get(cv2.CAP_PROP_FPS)
        output_fps = self.config.fps_override if self.config.fps_override is not None else input_fps
        return output_fps if output_fps and output_fps > 0 else 30.0

    def process(self, input_path: Path, out_dir: Path) -> VideoOutputs:
        """Process a video and write editor-friendly mask and overlay MP4s."""
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / "mask.mp4"
        overlay_path = out_dir / "overlay.mp4"
        metadata_path = out_dir / "run_metadata.json"

        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            capture.release()
            raise RuntimeError(f"No frames found in video: {input_path}")

        input_height, input_width = first_frame.shape[:2]
        first_frame = resize_frame(first_frame, self.config.downscale)
        frame_height, frame_width = first_frame.shape[:2]
        frame_size = (frame_width, frame_height)
        roi = validate_roi(self.config.roi, frame_size)

        output_fps = self._output_fps(capture)
        mask_writer = create_writer(mask_path, output_fps, frame_size)
        overlay_writer = create_writer(overlay_path, output_fps, frame_size)

        first_gray = prepare_gray(first_frame)
        state = ProcessorState(prev_gray=first_gray)
        warmup_frame = first_frame if roi is None else first_frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
        # A full-learning-rate warmup seeds the background model from the first frame
        # before incremental updates begin on subsequent frames.
        self.foreground_model.apply(warmup_frame, learningRate=1.0)
        processed_frames = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                frame = resize_frame(frame, self.config.downscale)
                if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                    frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

                current_gray = prepare_gray(frame)
                warp_matrix = None
                if self.config.stabilize:
                    warp_matrix = estimate_ecc_warp(state.prev_gray, current_gray)

                # The pipeline combines motion differencing and background subtraction:
                # differencing emphasizes change, while MOG2 supplies a region prior.
                aligned_gray = warp_to_previous(current_gray, warp_matrix)
                motion_seed_aligned = build_motion_seed(state.prev_gray, aligned_gray, self.config.threshold)
                motion_seed = warp_mask_to_current(motion_seed_aligned, warp_matrix)

                if roi is None:
                    model_frame = frame
                    motion_seed_roi = motion_seed
                else:
                    x, y, w, h = roi
                    model_frame = frame[y : y + h, x : x + w]
                    motion_seed_roi = motion_seed[y : y + h, x : x + w]

                raw_foreground = self.foreground_model.apply(model_frame, learningRate=-1)
                foreground_mask = np.where(raw_foreground > 0, 255, 0).astype(np.uint8)
                foreground_mask = clean_binary_mask(foreground_mask)

                selected_mask = select_components(
                    foreground_mask,
                    motion_seed_roi,
                    self.config.keep_blobs,
                    self.config.min_area,
                )
                # GrabCut and edge cleanup refine the coarse selected region into a matte
                # that is better suited to export as a mask or overlay.
                selected_mask = refine_mask_with_grabcut(model_frame, selected_mask)

                history_mask = blend_with_history(selected_mask, state.ema_mask, self.config.ema)
                state.ema_mask = history_mask
                final_mask_roi = np.where(history_mask >= 0.5, 255, 0).astype(np.uint8)
                final_mask_roi = fill_mask_holes(clean_binary_mask(final_mask_roi))

                final_mask = paste_roi_mask(final_mask_roi, frame_size, roi)
                mask_writer.write(cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))
                overlay_writer.write(create_overlay(frame, final_mask))
                state.prev_gray = current_gray
                processed_frames += 1
        finally:
            capture.release()
            mask_writer.release()
            overlay_writer.release()

        metadata_path.write_text(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "input_frame_size": {"width": input_width, "height": input_height},
                    "output_frame_size": {"width": frame_width, "height": frame_height},
                    "processed_frames": processed_frames,
                    "output_fps": output_fps,
                    "config": {
                        "threshold": self.config.threshold,
                        "downscale": self.config.downscale,
                        "fps_override": self.config.fps_override,
                        "stabilize": self.config.stabilize,
                        "keep_blobs": self.config.keep_blobs,
                        "min_area": self.config.min_area,
                        "ema": self.config.ema,
                        "roi": self.config.roi,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return VideoOutputs(mask_path=mask_path, overlay_path=overlay_path, metadata_path=metadata_path)
