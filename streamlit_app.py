"""Streamlit app for face mask classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

CLASS_NAMES = ("with_mask", "mask_weared_incorrect", "without_mask")
DISPLAY = {
    "with_mask": "With Mask",
    "mask_weared_incorrect": "Mask Worn Incorrectly",
    "without_mask": "Without Mask",
}
IMG_SIZE = (224, 224)
WEIGHTS_PATH = Path("face_mask_artifacts/face_mask_mobilenet.weights.h5")
HISTORY_PATH = Path("face_mask_artifacts/history.json")


@st.cache_resource(show_spinner="Loading model…")
def load_model() -> tf.keras.Model:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights=None, input_tensor=inputs
    )
    base.trainable = False
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="predictions")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="face_mask_mobilenet")
    model.load_weights(WEIGHTS_PATH)
    return model


def predict(image: Image.Image) -> tuple[str, float, dict[str, float]]:
    model = load_model()
    image = image.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8)
    batch = np.expand_dims(arr.astype(np.float32), axis=0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)

    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    prob_map = {DISPLAY[name]: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
    return label, float(probs[idx]), prob_map


def _val_accuracy() -> float | None:
    if not HISTORY_PATH.exists():
        return None
    history = json.loads(HISTORY_PATH.read_text())
    vals = history.get("val_accuracy", [])
    return max(vals) if vals else None


def main() -> None:
    st.set_page_config(page_title="Face Mask Classifier", page_icon="😷", layout="centered")

    st.title("😷 Face Mask Classifier")
    subtitle = "MobileNetV2 transfer learning on the MAKS face-mask dataset"
    val_acc = _val_accuracy()
    if val_acc is not None:
        subtitle += f" · validation accuracy **{val_acc:.1%}**"
    st.caption(subtitle)

    st.markdown(
        "Upload or capture a **face image**. The model predicts:\n"
        "- **With Mask**\n"
        "- **Mask Worn Incorrectly**\n"
        "- **Without Mask**"
    )

    tab_upload, tab_camera = st.tabs(["Upload", "Webcam"])

    image: Image.Image | None = None
    with tab_upload:
        uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded is not None:
            image = Image.open(uploaded)

    with tab_camera:
        camera = st.camera_input("Take a photo")
        if camera is not None:
            image = Image.open(camera)

    if image is None:
        st.info("Upload or capture a face image to get a prediction.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input image", use_container_width=True)

    with col2:
        with st.spinner("Classifying…"):
            label, confidence, prob_map = predict(image)

        st.success(f"**{DISPLAY[label]}**")
        st.metric("Confidence", f"{confidence:.1%}")
        st.bar_chart(prob_map)

        with st.expander("All probabilities"):
            for name, prob in prob_map.items():
                st.write(f"{name}: {prob:.1%}")


if __name__ == "__main__":
    main()
