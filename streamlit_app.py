"""Streamlit face mask classifier — simple, works with JPG/JPEG/PNG."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFile, ImageOps
from tensorflow.keras import layers, models

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES = ("with_mask", "mask_weared_incorrect", "without_mask")
LABELS = {
    "with_mask": "With Mask",
    "mask_weared_incorrect": "Mask Worn Incorrectly",
    "without_mask": "Without Mask",
}
IMG_SIZE = (224, 224)
WEIGHTS = Path("face_mask_artifacts/face_mask_mobilenet.weights.h5")


@st.cache_resource
def get_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    base = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(3, activation="softmax")(x)
    model = models.Model(inputs, out)
    model.load_weights(WEIGHTS)
    return model


def decode_bytes(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    if data[:2] == b"\xff\xd8":
        try:
            return tf.io.decode_jpeg(data, channels=3).numpy().astype(np.uint8)
        except Exception:
            pass
    try:
        return tf.io.decode_image(data, channels=3, expand_animations=False).numpy().astype(np.uint8)
    except Exception:
        pass
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        return np.array(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def classify(arr: np.ndarray) -> tuple[str, float, list[float]]:
    img = Image.fromarray(arr).resize(IMG_SIZE, Image.BILINEAR)
    batch = np.expand_dims(np.array(img, dtype=np.float32), 0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    probs = get_model().predict(batch, verbose=0)[0]
    i = int(np.argmax(probs))
    return CLASS_NAMES[i], float(probs[i]), probs.tolist()


def main():
    st.set_page_config(page_title="Face Mask Classifier", page_icon="😷", layout="centered")
    st.title("😷 Face Mask Classifier")
    st.caption("MobileNetV2 · Upload a face photo (JPG, JPEG, PNG)")

    uploaded = st.file_uploader("Upload face image", type=None)

    if uploaded is None:
        st.info("Upload a face photo to classify.")
        return

    arr = decode_bytes(uploaded.getvalue())
    if arr is None:
        st.error(
            f"Could not read **{uploaded.name}**. "
            "If from iPhone: save as PNG first, then upload."
        )
        return

    st.image(arr, caption=uploaded.name, use_container_width=True)

    with st.spinner("Classifying…"):
        label, conf, probs = classify(arr)

    st.success(f"**{LABELS[label]}** — {conf:.1%} confidence")
    for name, p in zip(CLASS_NAMES, probs):
        st.progress(float(p), text=f"{LABELS[name]}: {p:.1%}")


if __name__ == "__main__":
    main()
