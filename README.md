# Foreground Matting Video Mask

Video-processing tool that generates a foreground mask and an overlay preview from an input clip. The implementation uses a classical computer-vision pipeline rather than a deep-learning matte model.

## Problem Statement

For quick post-production workflows, it is useful to isolate the dominant moving subject from handheld footage and export a mask for compositing, rough cutouts, or downstream review. The challenge is keeping the subject region stable while suppressing flicker and camera-motion artifacts.

## Technical Approach

- Runtime profiles select practical defaults for large clips.
- Optional ECC-based frame stabilization reduces camera-motion noise when the clip size is manageable.
- MOG2 background subtraction estimates moving foreground regions.
- Morphological cleanup and connected-component ranking keep coherent subject blobs.
- GrabCut refinement and edge-aware cleanup tighten the mask around the subject.
- Temporal smoothing through an exponential moving average reduces per-frame flicker.

## Architecture

```text
Input video
    |
    v
Profile resolution + frame resize
    |
    v
Optional ECC stabilization
    |
    v
Motion seed + MOG2 foreground model
    |
    v
Morphology + component selection
    |
    v
GrabCut refinement + temporal smoothing
    |
    +--> mask.mp4
    |
    +--> overlay.mp4
    |
    +--> run_metadata.json
```

## Repository Layout

- `src/foreground_mask.py`: CLI entry point, profile selection, and argument validation
- `src/motion_mask_pipeline.py`: reusable processing pipeline
- `tests/test_pipeline_config.py`: tests for ROI validation and runtime profile behavior
- `data/demo.mp4`: local 4K demo clip
- `.github/workflows/ci.yml`: automated test workflow

## Example Usage

Verified auto-profile run on the included 4K demo:

```powershell
python -m src.foreground_mask --input data\demo.mp4 --out-dir outputs\demo_auto --ema 0.2 --keep-blobs 1
```

Observed output metadata:

```json
{
  "input_frame_size": {"width": 4096, "height": 2160},
  "output_frame_size": {"width": 1024, "height": 540},
  "processed_frames": 166,
  "output_fps": 25.0,
  "config": {
    "downscale": 0.25,
    "stabilize": false
  }
}
```

## Design Decisions

- `auto` is the default profile because a 4K-safe path that completes reliably is more valuable than an expensive default that times out.
- The pipeline writes `run_metadata.json` so each output can be tied to a specific processing configuration.
- The implementation favors deterministic classical CV steps over opaque learned models, which makes the behavior easier to inspect and discuss.

## Limitations

- This is not a semantic segmentation model and will struggle with low-motion subjects or highly dynamic backgrounds.
- The quality profile may be expensive on large 4K footage depending on hardware.
- Output quality depends on scene composition and foreground/background contrast.

## Running The Project

```powershell
python -m pip install -r requirements.txt
python -m src.foreground_mask --input data\demo.mp4 --out-dir outputs\demo_auto --ema 0.2 --keep-blobs 1
python -m unittest discover -s tests
```
