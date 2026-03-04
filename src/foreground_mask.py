from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .motion_mask_pipeline import MotionMaskProcessor, PipelineConfig, parse_roi


DEFAULT_THRESHOLD = 1.5
DEFAULT_DOWNSCALE = 1.0
DEFAULT_OUT_DIR = "outputs"
DEFAULT_KEEP_BLOBS = 1
DEFAULT_MIN_AREA = 500
DEFAULT_EMA = 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the motion-mask processor."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate mask.mp4 and overlay.mp4 from generic foreground motion using "
            "stabilization and background subtraction."
        )
    )
    parser.add_argument("--input", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Directory for output videos. Defaults to '{DEFAULT_OUT_DIR}'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum temporal foreground support. Higher values suppress short-lived artifacts.",
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
        help="Keep the largest N coherent moving regions per frame.",
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
        help="EMA factor for smoothing the foreground mask over time. Use 0 to disable.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional ROI as x,y,w,h. Motion is computed only inside the ROI.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, PipelineConfig]:
    """Validate CLI arguments and package them into a pipeline config."""
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

    config = PipelineConfig(
        threshold=args.threshold,
        downscale=args.downscale,
        fps_override=args.fps,
        stabilize=not args.no_stabilize,
        keep_blobs=args.keep_blobs,
        min_area=args.min_area,
        ema=args.ema,
        roi=roi,
    )
    return input_path, out_dir, config


def main() -> int:
    """CLI entry point."""
    try:
        args = parse_args()
        input_path, out_dir, config = validate_args(args)
        outputs = MotionMaskProcessor(config).process(input_path, out_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Mask video written to: {outputs.mask_path}")
    print(f"Overlay video written to: {outputs.overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
