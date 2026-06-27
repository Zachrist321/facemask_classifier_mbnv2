"""Face mask classifier — upload or webcam, JPEG/PNG supported."""

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
COLORS = {"with_mask": "#16a34a", "mask_weared_incorrect": "#ca8a04", "without_mask": "#dc2626"}
IMG_SIZE = (224, 224)
WEIGHTS = Path("face_mask_artifacts/face_mask_mobilenet.weights.h5")


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS}")
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    base = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(3, activation="softmax")(x)
    model = models.Model(inputs, out)
    model.load_weights(WEIGHTS)
    return model


def decode_bytes(data: bytes) -> np.ndarray | None:
    """Decode image bytes to RGB uint8 array. Tries several methods for JPEG."""
    if not data or len(data) < 10:
        return None

    errors = []

    # Method 1: TensorFlow JPEG (best for camera + phone JPG)
    if data[:2] == b"\xff\xd8":
        for attempt in ("strict", "relaxed"):
            try:
                if attempt == "strict":
                    t = tf.io.decode_jpeg(data, channels=3)
                else:
                    t = tf.io.decode_jpeg(data, channels=3, fancy_upscaling=False, dct_method="INTEGER_FAST")
                return t.numpy().astype(np.uint8)
            except Exception as e:
                errors.append(f"jpeg-{attempt}: {e}")

    # Method 2: PIL + EXIF (handles rotated phone photos)
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        if img.mode == "CMYK":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except Exception as e:
        errors.append(f"pil: {e}")

    # Method 3: TensorFlow generic
    try:
        t = tf.io.decode_image(data, channels=3, expand_animations=False)
        return t.numpy().astype(np.uint8)
    except Exception as e:
        errors.append(f"tf-image: {e}")

    st.session_state["decode_errors"] = errors
    return None


def classify(arr: np.ndarray) -> tuple[str, float, list[float]]:
    img = Image.fromarray(arr).resize(IMG_SIZE, Image.BILINEAR)
    batch = np.expand_dims(np.array(img, dtype=np.float32), 0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    probs = get_model().predict(batch, verbose=0)[0]
    i = int(np.argmax(probs))
    return CLASS_NAMES[i], float(probs[i]), probs.tolist()


def main():
    st.set_page_config(page_title="Face Mask Classifier", page_icon="😷", layout="wide")
    st.title("😷 Face Mask Classifier")
    st.caption("MobileNetV2 · Upload or use your webcam")

    col_left, col_right = st.columns(2)

    with col_left:
        tab_up, tab_cam = st.tabs(["📁 Upload", "📷 Webcam"])

        raw_bytes: bytes | None = None
        source_name = ""

        with tab_up:
            file = st.file_uploader(
                "Choose image (JPG, JPEG, PNG, WEBP)",
                type=None,
                key="upload",
            )
            if file is not None:
                raw_bytes = file.getvalue()
                source_name = file.name or "upload"

        with tab_cam:
            st.caption("Allow camera access, then click **Take Photo**")
            photo = st.camera_input("Webcam", key="camera")
            if photo is not None:
                raw_bytes = photo.getvalue()
                source_name = "webcam"

        if raw_bytes is None:
            st.info("Upload an image or take a webcam photo.")
            return

        arr = decode_bytes(raw_bytes)
        if arr is None:
            st.error(f"Could not read image ({source_name}).")
            if "decode_errors" in st.session_state:
                with st.expander("Debug info"):
                    for err in st.session_state["decode_errors"]:
                        st.code(err)
            st.warning(
                "**iPhone users:** Photos are often HEIC, not JPEG. "
                "Open the photo → **Share → Save to Files** as PNG/JPG, then upload."
            )
            return

        st.image(arr, caption=source_name, use_container_width=True)
        st.session_state["face_array"] = arr

    with col_right:
        st.subheader("Prediction")
        arr = st.session_state.get("face_array")
        if arr is None:
            st.info("Add a face image on the left.")
            return

        if st.button("Classify", type="primary", use_container_width=True):
            with st.spinner("Running model…"):
                label, conf, probs = classify(arr)

            color = COLORS[label]
            st.markdown(
                f"<h2 style='color:{color};'>{LABELS[label]}</h2>"
                f"<p style='font-size:1.2rem;'>Confidence: <b>{conf:.1%}</b></p>",
                unsafe_allow_html=True,
            )
            for name, p in zip(CLASS_NAMES, probs):
                st.progress(float(p), text=f"{LABELS[name]}: {p:.1%}")


if __name__ == "__main__":
    main()
