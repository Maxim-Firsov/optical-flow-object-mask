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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Validate CLI arguments and return normalized paths."""
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if args.threshold < 0:
        raise ValueError("--threshold must be >= 0.")
    if args.downscale <= 0:
        raise ValueError("--downscale must be > 0.")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be > 0 when provided.")

    return input_path, out_dir


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


def build_motion_mask(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    threshold: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """Compute dense optical flow and convert its magnitude into a cleaned binary mask."""
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
    mask = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


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

    input_fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = fps_override if fps_override is not None else input_fps
    if output_fps is None or output_fps <= 0:
        output_fps = 30.0

    frame_size = (frame_width, frame_height)
    mask_writer = create_writer(mask_path, output_fps, frame_size)
    overlay_writer = create_writer(overlay_path, output_fps, frame_size)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            frame = resize_frame(frame, downscale)
            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = build_motion_mask(prev_gray, curr_gray, threshold, kernel)

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
        input_path, out_dir = validate_args(args)
        mask_path, overlay_path = process_video(
            input_path=input_path,
            out_dir=out_dir,
            threshold=args.threshold,
            downscale=args.downscale,
            fps_override=args.fps,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Mask video written to: {mask_path}")
    print(f"Overlay video written to: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
