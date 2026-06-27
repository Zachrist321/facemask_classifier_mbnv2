"""Gradio app for face mask classification (Capstone deployment)."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.data import CLASS_NAMES
from src.model import load_model_for_inference
from src.train import predict_face_crop

ARTIFACT_DIR = Path("face_mask_artifacts")
MODEL_PATH = ARTIFACT_DIR / "face_mask_mobilenet.keras"

DISPLAY_NAMES = {
    "with_mask": "With Mask",
    "mask_weared_incorrect": "Mask Worn Incorrectly",
    "without_mask": "Without Mask",
}

model = load_model_for_inference(MODEL_PATH)


def predict(image: np.ndarray) -> tuple[str, dict[str, float]]:
    if image is None:
        raise gr.Error("Please upload or capture a face image.")

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]

    label, confidence = predict_face_crop(model, image.astype(np.uint8))
    probs = _all_probabilities(image)
    return (
        f"{DISPLAY_NAMES[label]} ({confidence:.1%} confidence)",
        {DISPLAY_NAMES[name]: float(probs[i]) for i, name in enumerate(CLASS_NAMES)},
    )


def _all_probabilities(image: np.ndarray) -> np.ndarray:
    from src.model import IMG_SIZE

    if image.shape[:2] != IMG_SIZE:
        image = np.asarray(
            Image.fromarray(image.astype(np.uint8)).resize(IMG_SIZE, Image.BILINEAR),
            dtype=np.uint8,
        )
    batch = np.expand_dims(image.astype(np.float32), axis=0)
    import tensorflow as tf

    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    return model.predict(batch, verbose=0)[0]


def build_app() -> gr.Blocks:
    config = json.loads((ARTIFACT_DIR / "config.json").read_text())
    val_acc = None
    history_path = ARTIFACT_DIR / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        val_acc = max(history.get("val_accuracy", [0]))

    subtitle = "MobileNetV2 transfer learning on MAKS face-mask dataset"
    if val_acc is not None:
        subtitle += f" · val accuracy {val_acc:.1%}"

    with gr.Blocks(title="Face Mask Classifier") as demo:
        gr.Markdown(
            f"""
            # Face Mask Classifier
            {subtitle}

            Upload or capture a **face image**. The model classifies:
            - **With Mask**
            - **Mask Worn Incorrectly**
            - **Without Mask**

            Works best on a clear, centered face crop (224×224).
            """
        )
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Face image", sources=["upload", "webcam"])
            with gr.Column():
                label_out = gr.Label(label="Prediction probabilities")
                text_out = gr.Textbox(label="Top prediction", interactive=False)

        gr.Examples(
            examples=_example_paths(),
            inputs=image_in,
            label="Example face crops (from validation set)",
        )

        image_in.change(predict, inputs=image_in, outputs=[text_out, label_out])

    return demo


def _example_paths() -> list[str]:
    examples_dir = ARTIFACT_DIR / "eval" / "failure_cases"
    if not examples_dir.exists():
        return []
    return [str(p) for p in sorted(examples_dir.glob("*.png"))[:6]]


demo = build_app()

if __name__ == "__main__":
    demo.launch()
