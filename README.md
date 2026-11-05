# Document Scanner

An OpenCV-based document scanner that takes a photo of a document (receipt, page, whiteboard, etc.), detects the edges, applies a perspective transform to produce a flat top-down view, and optionally converts it to black-and-white for a clean scanned look.

## How it works

1. **Edge detection** — the image is resized, blurred and passed through a Canny edge detector.
2. **Contour detection** — the largest four-sided contour is picked as the document boundary.
3. **Perspective warp** — a four-point perspective transform straightens the document.
4. **Adaptive threshold** — (optional) converts the result to black-and-white using adaptive Gaussian thresholding.

## Usage

```bash
python scanner.py photo.jpg                   # saves photo_scanned.jpg
python scanner.py photo.jpg -o output.png     # custom output name
python scanner.py photo.jpg --no-bw           # keep colour, skip B&W step
```

If the document edges cannot be detected, the original image is returned unchanged.

## Requirements

```bash
pip install opencv-python numpy
```

## Example

| Input | Output |
|-------|--------|
| A photo of a receipt taken at an angle | A flat, black-and-white scan of the same receipt |
