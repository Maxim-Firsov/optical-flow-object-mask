# Optical Flow Object Mask

OpenCV and NumPy foreground masking tool for generating:

- `mask.mp4`: white foreground motion on black
- `overlay.mp4`: original video with a semi-transparent green overlay

The pipeline is designed for generic MP4 foreground extraction rather than one specific clip. It combines:

- optional ECC-based camera stabilization
- background subtraction with MOG2
- morphological cleanup
- connected-component quality filtering
- temporal support maps to suppress flicker and short-lived artifacts

## Usage

```powershell
python -m src.foreground_mask --input data\demo.mp4
```

Useful options:

- `--threshold`: temporal support threshold for foreground persistence
- `--ema`: smooth the foreground mask over time
- `--keep-blobs`: keep the strongest moving regions
- `--min-area`: reject tiny regions
- `--roi x,y,w,h`: limit processing to a region of interest
- `--no-stabilize`: disable ECC stabilization

## Repo layout

- `src/foreground_mask.py`: CLI entry point
- `src/motion_mask_pipeline.py`: reusable processing pipeline
- `data/`: local demo video inputs
- `tests/`: unit tests
