"""Face mask classifier — upload or webcam."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFile, ImageOps
from tensorflow.keras import layers, models

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

CLASS_NAMES = ("with_mask", "mask_weared_incorrect", "without_mask")
LABELS = {
    "with_mask": "With Mask",
    "mask_weared_incorrect": "Mask Worn Incorrectly",
    "without_mask": "Without Mask",
}
IMG_SIZE = (224, 224)
WEIGHTS = Path(__file__).parent / "face_mask_artifacts" / "face_mask_mobilenet.weights.h5"


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    base = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(3, activation="softmax")(x)
    model = models.Model(inputs, out)
    model.load_weights(WEIGHTS)
    return model


def read_uploaded_file(uploaded) -> tuple[bytes, str]:
    """Read bytes from Streamlit UploadedFile reliably."""
    name = getattr(uploaded, "name", None) or "image"
    uploaded.seek(0)
    data = uploaded.read()
    uploaded.seek(0)
    return data, name


def decode_image(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception:
        pass
    if data[:2] == b"\xff\xd8":
        try:
            return tf.io.decode_jpeg(data, channels=3).numpy().astype(np.uint8)
        except Exception:
            pass
    try:
        t = tf.io.decode_image(data, channels=3, expand_animations=False)
        return t.numpy().astype(np.uint8)
    except Exception:
        return None


def run_model(arr: np.ndarray) -> tuple[str, float, list[float]]:
    img = Image.fromarray(arr).resize(IMG_SIZE, Image.BILINEAR)
    batch = np.expand_dims(np.array(img, dtype=np.float32), 0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    probs = get_model().predict(batch, verbose=0)[0]
    i = int(np.argmax(probs))
    return CLASS_NAMES[i], float(probs[i]), probs.tolist()


def main():
    st.set_page_config(page_title="Face Mask Classifier", page_icon="😷")
    st.title("😷 Face Mask Classifier")

    if not WEIGHTS.exists():
        st.error("Model weights missing. Run from `cv_final_project` folder.")
        st.stop()

    mode = st.radio("Input", ["Upload file", "Webcam"], horizontal=True)

    uploaded = None
    if mode == "Upload file":
        uploaded = st.file_uploader(
            "Select a face photo",
            type=["jpg", "jpeg", "png", "webp", "bmp", "heic", "heif"],
        )
    else:
        uploaded = st.camera_input("Take a photo")

    if uploaded is None:
        st.info("Choose a file or take a photo above.")
        return

    data, filename = read_uploaded_file(uploaded)
    if not data:
        st.error("File is empty.")
        return

    arr = decode_image(data)
    if arr is None:
        st.error(f"Could not read **{filename}**.")
        st.caption("Try exporting as JPG or PNG from Preview/Photos.")
        return

    st.image(arr, caption=filename, use_container_width=True)

    with st.spinner("Classifying…"):
        label, conf, probs = run_model(arr)

    st.success(f"**{LABELS[label]}** — {conf:.1%} confidence")
    for name, p in zip(CLASS_NAMES, probs):
        st.progress(float(p), text=f"{LABELS[name]}: {p:.1%}")


if __name__ == "__main__":
    main()
