"""Gradio app for face mask classification (Capstone deployment)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.data import CLASS_NAMES
from src.model import load_model_for_inference
from src.train import predict_face_crop

ARTIFACT_DIR = Path("face_mask_artifacts")
MODEL_PATH = ARTIFACT_DIR / "face_mask_mobilenet.weights.h5"

DISPLAY_NAMES = {
    "with_mask": "With Mask",
    "mask_weared_incorrect": "Mask Worn Incorrectly",
    "without_mask": "Without Mask",
}

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise gr.Error(f"Model weights not found at {MODEL_PATH}")
        _model = load_model_for_inference(MODEL_PATH)
    return _model


def classify(image):
    """Classify a face image."""
    if image is None:
        raise gr.Error("Please upload a face image first, then click Classify.")

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    image = image.astype(np.uint8)
    label, confidence = predict_face_crop(get_model(), image)
    probs = _all_probabilities(image)

    top = f"{DISPLAY_NAMES[label]} ({confidence:.1%} confidence)"
    breakdown = "\n".join(
        f"{DISPLAY_NAMES[name]}: {probs[i]:.1%}" for i, name in enumerate(CLASS_NAMES)
    )
    return top, breakdown


def _all_probabilities(image: np.ndarray) -> np.ndarray:
    from src.model import IMG_SIZE
    import tensorflow as tf

    if image.shape[:2] != IMG_SIZE:
        image = np.asarray(
            Image.fromarray(image.astype(np.uint8)).resize(IMG_SIZE, Image.BILINEAR),
            dtype=np.uint8,
        )
    batch = np.expand_dims(image.astype(np.float32), axis=0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    return get_model().predict(batch, verbose=0)[0]


def _subtitle() -> str:
    subtitle = "MobileNetV2 transfer learning on MAKS face-mask dataset"
    history_path = ARTIFACT_DIR / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        val_acc = max(history.get("val_accuracy", [0]))
        subtitle += f" · val accuracy {val_acc:.1%}"
    return subtitle


def _example_paths() -> list[str]:
    examples_dir = ARTIFACT_DIR / "eval" / "failure_cases"
    if not examples_dir.exists():
        return []
    return [str(p) for p in sorted(examples_dir.glob("*.png"))[:6]]


with gr.Blocks(title="Face Mask Classifier") as demo:
    gr.Markdown(
        f"""
        # Face Mask Classifier
        {_subtitle()}

        Upload a **face image**, then click **Classify**.

        Classes: With Mask · Mask Worn Incorrectly · Without Mask
        """
    )
    with gr.Row():
        image_in = gr.Image(type="numpy", label="Face image")
        with gr.Column():
            text_out = gr.Textbox(label="Top prediction", interactive=False)
            prob_out = gr.Textbox(label="All class probabilities", interactive=False, lines=4)

    gr.Button("Classify", variant="primary").click(
        fn=classify,
        inputs=image_in,
        outputs=[text_out, prob_out],
    )

    examples = _example_paths()
    if examples:
        gr.Examples(examples=examples, inputs=image_in, label="Example images")


# HF Spaces launches `demo` automatically — do not call demo.launch() there.
if __name__ == "__main__" and not os.getenv("SPACE_ID"):
    demo.launch()
