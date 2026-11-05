"""
Document scanner using OpenCV.

Takes an image of a document (photo, receipt, whiteboard, etc.),
detects the document edges, applies a perspective transform to get
a flat top-down view, and optionally converts to black-and-white
for a clean scanned look.

Usage
-----
    python scanner.py photo.jpg                 # saves photo_scanned.jpg
    python scanner.py photo.jpg -o output.png   # custom output path
    python scanner.py photo.jpg --no-bw         # keep colour, skip B&W step
"""

import argparse
import os
import sys

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort four points into [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp *image* so that the quadrilateral defined by *pts* fills the frame."""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def find_document_contour(image: np.ndarray) -> np.ndarray | None:
    """Return the four-corner contour of the largest document in *image*."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    # close small gaps in the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def scan(image: np.ndarray, bw: bool = True) -> np.ndarray:
    """Detect a document in *image*, warp it flat, and optionally apply B&W."""
    ratio = image.shape[0] / 500.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    contour = find_document_contour(resized)
    if contour is None:
        print("warning: could not detect document edges, returning original image")
        return image

    # scale contour back to original image size
    contour = contour * ratio
    warped = four_point_transform(image, contour)

    if bw:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9
        )

    return warped


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a document from a photo.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("-o", "--output", help="Path for the output image")
    parser.add_argument(
        "--no-bw",
        action="store_true",
        help="Keep the colour output instead of converting to black-and-white",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"error: cannot read '{args.image}'")
        sys.exit(1)

    result = scan(image, bw=not args.no_bw)

    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.image)
        out_path = f"{base}_scanned{ext}"

    cv2.imwrite(out_path, result)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
