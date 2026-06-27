#!/usr/bin/env python3
"""Test any image file without Streamlit. Usage:
   conda activate ml
   python test_image.py path/to/photo.jpg
"""
import sys
from pathlib import Path

from streamlit_app import decode_image, run_model, WEIGHTS

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image.jpg|png|...>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    if not WEIGHTS.exists():
        print(f"Missing model weights: {WEIGHTS}")
        print("Run from cv_final_project folder.")
        sys.exit(1)

    data = path.read_bytes()
    print(f"File: {path.name} ({len(data)} bytes)")

    arr = decode_image(data)
    if arr is None:
        print("FAILED to decode.")
        sys.exit(1)

    print(f"Decoded: {arr.shape}")
    label, conf, probs = run_model(arr)
    print(f"Prediction: {label} ({conf:.1%})")
    for n, p in zip(("with_mask", "mask_weared_incorrect", "without_mask"), probs):
        print(f"  {n}: {p:.1%}")

if __name__ == "__main__":
    main()
