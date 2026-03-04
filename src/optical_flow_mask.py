from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


DEFAULT_THRESHOLD = 1.5
DEFAULT_DOWNSCALE = 1.0
DEFAULT_OUT_DIR = "outputs"
MORPH_KERNEL_SIZE = 5
OVERLAY_ALPHA = 0.35
DEFAULT_KEEP_BLOBS = 1
DEFAULT_MIN_AREA = 500
DEFAULT_EMA = 0.0
ECC_ITERATIONS = 50
ECC_EPSILON = 1e-4

Roi = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the optical flow mask pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate motion mask and overlay videos from dense optical flow."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Directory for output videos. Defaults to '{DEFAULT_OUT_DIR}'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Optical-flow magnitude threshold used to create the motion mask.",
    )
    parser.add_argument(
        "--downscale",
        type=float,
        default=DEFAULT_DOWNSCALE,
        help="Optional frame downscale factor for faster processing. Must be > 0.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional output FPS override. If omitted, the input video FPS is preserved.",
    )
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Disable ECC-based camera motion compensation.",
    )
    parser.add_argument(
        "--keep-blobs",
        type=int,
        default=DEFAULT_KEEP_BLOBS,
        help="Keep the largest N connected regions in the mask after filtering.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help="Discard connected regions smaller than this many pixels.",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=DEFAULT_EMA,
        help="EMA factor for smoothing optical-flow magnitude before thresholding. Use 0 to disable.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional ROI as x,y,w,h. Flow is computed only inside the ROI and pasted back into the full frame.",
    )
    return parser.parse_args()


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


def validate_args(args: argparse.Namespace) -> Tuple[Path, Path, Roi | None]:
    """Validate CLI arguments and return normalized paths."""
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    roi = parse_roi(args.roi)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if args.threshold < 0:
        raise ValueError("--threshold must be >= 0.")
    if args.downscale <= 0:
        raise ValueError("--downscale must be > 0.")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be > 0 when provided.")
    if args.keep_blobs < 1:
        raise ValueError("--keep-blobs must be >= 1.")
    if args.min_area < 0:
        raise ValueError("--min-area must be >= 0.")
    if not 0.0 <= args.ema <= 1.0:
        raise ValueError("--ema must be between 0.0 and 1.0.")

    return input_path, out_dir, roi


def create_writer(output_path: Path, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create an MP4 writer and fail fast if initialization does not succeed."""
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


def resize_frame(frame: np.ndarray, downscale: float) -> np.ndarray:
    """Resize the frame when downscale < 1.0; otherwise return it unchanged."""
    if downscale >= 1.0:
        return frame

    height, width = frame.shape[:2]
    target_width = max(1, int(width * downscale))
    target_height = max(1, int(height * downscale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def validate_roi(roi: Roi | None, frame_size: Tuple[int, int]) -> Roi | None:
    """Ensure the ROI fits within the current frame size."""
    if roi is None:
        return None

    frame_width, frame_height = frame_size
    x, y, w, h = roi
    if x + w > frame_width or y + h > frame_height:
        raise ValueError(
            f"ROI {roi} exceeds frame bounds {frame_width}x{frame_height}."
        )
    return roi


def estimate_ecc_warp(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray | None:
    """Estimate an affine warp that maps the current frame onto the previous frame."""
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


def warp_frame_to_previous(frame: np.ndarray, warp_matrix: np.ndarray | None) -> np.ndarray:
    """Warp the current frame into the previous frame's coordinate system when a warp exists."""
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


def filter_mask_components(mask: np.ndarray, keep_blobs: int, min_area: int) -> np.ndarray:
    """Keep only the largest connected components above the minimum area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    candidates: list[Tuple[int, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            candidates.append((label, area))

    if not candidates:
        return np.zeros_like(mask)

    candidates.sort(key=lambda item: item[1], reverse=True)
    kept_labels = {label for label, _ in candidates[:keep_blobs]}

    filtered = np.zeros_like(mask)
    for label in kept_labels:
        filtered[labels == label] = 255

    return filtered


def build_motion_mask(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> np.ndarray:
    """Compute dense optical flow magnitude between two aligned grayscale frames."""
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude


def finalize_mask(
    magnitude: np.ndarray,
    threshold: float,
    kernel: np.ndarray,
    keep_blobs: int,
    min_area: int,
) -> np.ndarray:
    """Threshold, denoise, and keep the strongest motion regions."""
    mask = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return filter_mask_components(mask, keep_blobs, min_area)


def create_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a colored motion mask onto the original frame."""
    overlay_color = np.zeros_like(frame)
    overlay_color[..., 1] = mask
    return cv2.addWeighted(frame, 1.0, overlay_color, OVERLAY_ALPHA, 0.0)


def process_video(
    input_path: Path,
    out_dir: Path,
    threshold: float,
    downscale: float,
    fps_override: float | None,
    stabilize: bool,
    keep_blobs: int,
    min_area: int,
    ema: float,
    roi: Roi | None,
) -> Tuple[Path, Path]:
    """Read the input video, compute frame-to-frame motion masks, and write outputs."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "mask.mp4"
    overlay_path = out_dir / "overlay.mp4"

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"No frames found in video: {input_path}")

    first_frame = resize_frame(first_frame, downscale)
    frame_height, frame_width = first_frame.shape[:2]
    roi = validate_roi(roi, (frame_width, frame_height))

    input_fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = fps_override if fps_override is not None else input_fps
    if output_fps is None or output_fps <= 0:
        output_fps = 30.0

    frame_size = (frame_width, frame_height)
    mask_writer = create_writer(mask_path, output_fps, frame_size)
    overlay_writer = create_writer(overlay_path, output_fps, frame_size)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    ema_magnitude: np.ndarray | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            frame = resize_frame(frame, downscale)
            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aligned_gray = curr_gray
            if stabilize:
                warp_matrix = estimate_ecc_warp(prev_gray, curr_gray)
                aligned_gray = warp_frame_to_previous(curr_gray, warp_matrix)

            if roi is None:
                prev_flow_gray = prev_gray
                curr_flow_gray = aligned_gray
            else:
                x, y, w, h = roi
                prev_flow_gray = prev_gray[y : y + h, x : x + w]
                curr_flow_gray = aligned_gray[y : y + h, x : x + w]

            magnitude = build_motion_mask(prev_flow_gray, curr_flow_gray)
            if ema > 0.0:
                if ema_magnitude is None or ema_magnitude.shape != magnitude.shape:
                    ema_magnitude = magnitude
                else:
                    ema_magnitude = (ema * magnitude) + ((1.0 - ema) * ema_magnitude)
                magnitude = ema_magnitude

            motion_mask = finalize_mask(magnitude, threshold, kernel, keep_blobs, min_area)

            if roi is None:
                mask = motion_mask
            else:
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                mask[y : y + h, x : x + w] = motion_mask

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay = create_overlay(frame, mask)

            mask_writer.write(mask_bgr)
            overlay_writer.write(overlay)
            prev_gray = curr_gray
    finally:
        capture.release()
        mask_writer.release()
        overlay_writer.release()

    return mask_path, overlay_path


def main() -> int:
    """Entry point for the CLI."""
    try:
        args = parse_args()
        input_path, out_dir, roi = validate_args(args)
        mask_path, overlay_path = process_video(
            input_path=input_path,
            out_dir=out_dir,
            threshold=args.threshold,
            downscale=args.downscale,
            fps_override=args.fps,
            stabilize=not args.no_stabilize,
            keep_blobs=args.keep_blobs,
            min_area=args.min_area,
            ema=args.ema,
            roi=roi,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Mask video written to: {mask_path}")
    print(f"Overlay video written to: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
