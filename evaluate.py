"""Evaluate a trained face-mask classifier locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data import CLASS_NAMES, parse_annotations, stratified_split
from src.model import load_model_for_inference
from src.train import make_tf_dataset


def plot_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face-mask MobileNet classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("archive-4"),
        help="Path to MAKS dataset root (images/ + annotations/)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("face_mask_artifacts/face_mask_mobilenet.keras"),
        help="Path to saved .keras model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("face_mask_artifacts/eval"),
        help="Directory for evaluation outputs",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = parse_annotations(args.data_dir)
    _, val_samples = stratified_split(samples, val_ratio=0.2, seed=42)
    val_ds = make_tf_dataset(val_samples, batch_size=args.batch_size, shuffle=False, augment=False)

    model = load_model_for_inference(args.model_path)

    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(CLASS_NAMES),
        output_dict=True,
    )
    (args.output_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    print(classification_report(y_true, y_pred, target_names=list(CLASS_NAMES)))

    plot_confusion_matrix(y_true, y_pred, args.output_dir / "confusion_matrix.png")
    print(f"Saved evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
