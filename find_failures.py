"""Save misclassified validation face crops for the project report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.data import CLASS_NAMES, crop_face, parse_annotations, stratified_split
from src.model import load_model_for_inference
from src.train import predict_face_crop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("archive-4"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("face_mask_artifacts/face_mask_mobilenet.keras"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("face_mask_artifacts/eval/failure_cases"),
    )
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = parse_annotations(args.data_dir)
    _, val_samples = stratified_split(samples, val_ratio=0.2, seed=42)
    model = load_model_for_inference(args.model_path)

    failures: list[dict] = []
    for sample in val_samples:
        crop = crop_face(sample)
        pred_label, confidence = predict_face_crop(model, crop)
        if pred_label != sample.label:
            failures.append(
                {
                    "sample": sample,
                    "crop": crop,
                    "pred_label": pred_label,
                    "confidence": confidence,
                }
            )

    failures.sort(key=lambda x: x["confidence"], reverse=True)
    selected = failures[: args.max_cases]

    summary = []
    for i, case in enumerate(selected, start=1):
        sample = case["sample"]
        filename = f"failure_{i}_true-{sample.label}_pred-{case['pred_label']}.png"
        out_path = args.output_dir / filename

        img = Image.fromarray(case["crop"])
        draw = ImageDraw.Draw(img)
        caption = f"True: {sample.label}\nPred: {case['pred_label']} ({case['confidence']:.0%})"
        draw.rectangle((0, 0, 224, 40), fill=(0, 0, 0))
        draw.text((4, 4), caption.replace("\n", " | "), fill=(255, 255, 255))
        img.save(out_path)

        summary.append(
            {
                "file": filename,
                "true_label": sample.label,
                "predicted_label": case["pred_label"],
                "confidence": case["confidence"],
                "source_image": str(sample.image_path),
                "bbox": list(sample.bbox),
                "likely_cause": _likely_cause(sample.label, case["pred_label"]),
            }
        )

    (args.output_dir / "failure_cases.json").write_text(json.dumps(summary, indent=2))
    print(f"Found {len(failures)} misclassifications on validation set")
    print(f"Saved {len(selected)} examples to {args.output_dir}")
    for item in summary:
        print(f"  {item['file']}: {item['true_label']} -> {item['predicted_label']}")


def _likely_cause(true_label: str, pred_label: str) -> str:
    if true_label == "mask_weared_incorrect":
        return "Small class size and ambiguous partial-mask appearance; model defaults to with/without mask."
    if true_label == "without_mask" and pred_label == "with_mask":
        return "Hand near face, shadow, or angle may resemble mask edge."
    if true_label == "with_mask" and pred_label == "without_mask":
        return "Light-colored or thin mask with low contrast against skin."
    if pred_label == "mask_weared_incorrect":
        return "Mask boundary unclear; model uncertain between correct and incorrect wear."
    return "Occlusion, lighting, or crop alignment reduced discriminative features."


if __name__ == "__main__":
    main()
