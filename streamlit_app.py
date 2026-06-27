"""Streamlit app for face mask classification."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from tensorflow.keras import layers, models

# Allow slightly truncated JPEGs from phone cameras
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES = ("with_mask", "mask_weared_incorrect", "without_mask")
DISPLAY = {
    "with_mask": "With Mask",
    "mask_weared_incorrect": "Mask Worn Incorrectly",
    "without_mask": "Without Mask",
}
EMOJI = {
    "with_mask": "✅",
    "mask_weared_incorrect": "⚠️",
    "without_mask": "❌",
}
COLORS = {
    "with_mask": "#16a34a",
    "mask_weared_incorrect": "#ca8a04",
    "without_mask": "#dc2626",
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


def _read_upload_bytes(uploaded) -> bytes:
    if hasattr(uploaded, "seek"):
        uploaded.seek(0)
    if hasattr(uploaded, "getvalue"):
        data = uploaded.getvalue()
    elif hasattr(uploaded, "read"):
        data = uploaded.read()
    else:
        data = bytes(uploaded)
    if hasattr(uploaded, "seek"):
        uploaded.seek(0)
    return data or b""


def open_image(uploaded) -> Image.Image | None:
    """Safely open an uploaded/camera file as RGB PIL image."""
    if uploaded is None:
        return None

    data = _read_upload_bytes(uploaded)
    if not data:
        return None

    # 1) PIL (PNG, most JPEG)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, ValueError):
        pass

    # 2) TensorFlow decoder — robust for JPEG/JPG from phones
    try:
        tensor = tf.io.decode_image(data, channels=3, expand_animations=False)
        tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8)
        return Image.fromarray(tensor.numpy())
    except Exception:
        return None


def predict(image: Image.Image) -> tuple[str, float, dict[str, float]]:
    model = load_model()
    image = image.resize(IMG_SIZE, Image.BILINEAR)
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


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0 0.5rem;
        }
        .main-header h1 { margin-bottom: 0.25rem; font-size: 2.2rem; }
        .main-header p { color: #64748b; font-size: 1rem; margin: 0; }
        .result-card {
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            margin: 1rem 0;
            border: 2px solid #e2e8f0;
        }
        .result-label { font-size: 1.6rem; font-weight: 700; margin: 0.5rem 0; }
        .result-conf { font-size: 1.1rem; color: #475569; }
        .prob-row { margin: 0.6rem 0; }
        .prob-name { font-weight: 600; font-size: 0.95rem; }
        div[data-testid="stFileUploader"] section {
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_probs(prob_map: dict[str, float], highlight: str) -> None:
    for name, prob in prob_map.items():
        st.markdown(f'<p class="prob-name">{name}</p>', unsafe_allow_html=True)
        st.progress(min(max(prob, 0.0), 1.0), text=f"{prob:.1%}")
        if name == highlight:
            st.caption("↑ predicted class")


def main() -> None:
    st.set_page_config(
        page_title="Face Mask Classifier",
        page_icon="😷",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_css()

    val_acc = _val_accuracy()
    acc_text = f" · {val_acc:.1%} validation accuracy" if val_acc else ""

    st.markdown(
        f"""
        <div class="main-header">
            <h1>😷 Face Mask Classifier</h1>
            <p>MobileNetV2 · MAKS dataset{acc_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Input")
        source = st.radio(
            "Image source",
            ["Upload image", "Use webcam"],
            horizontal=True,
            label_visibility="collapsed",
        )

        uploaded_file = None
        if source == "Upload image":
            uploaded_file = st.file_uploader(
                "Choose a face photo",
                type=None,
                accept_multiple_files=False,
                label_visibility="collapsed",
            )
        else:
            uploaded_file = st.camera_input("Take a photo", label_visibility="collapsed")

        image = open_image(uploaded_file)

        if uploaded_file is not None and image is None:
            st.error("Could not read that image. Try JPG or PNG.")
            return

        if image is None:
            st.info("👆 Upload or capture a clear **face photo**, then click **Classify**.")
            return

        st.image(image, caption="Your image", use_container_width=True)

    with col_out:
        st.subheader("Result")

        if st.button("🔍 Classify face", type="primary", use_container_width=True):
            with st.spinner("Analysing face…"):
                try:
                    label, confidence, prob_map = predict(image)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    return

            color = COLORS[label]
            st.markdown(
                f"""
                <div class="result-card" style="border-color: {color}; background: {color}11;">
                    <div style="font-size: 3rem;">{EMOJI[label]}</div>
                    <div class="result-label" style="color: {color};">{DISPLAY[label]}</div>
                    <div class="result-conf">{confidence:.1%} confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Class probabilities")
            _render_probs(prob_map, DISPLAY[label])
        else:
            st.markdown(
                """
                <div style="text-align:center; padding: 3rem 1rem; color: #94a3b8;">
                    <p style="font-size: 3rem; margin: 0;">🎯</p>
                    <p>Click <strong>Classify face</strong> to see the prediction</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.caption(
        "Classes: ✅ With Mask · ⚠️ Mask Worn Incorrectly · ❌ Without Mask · "
        "Best results on a clear, centered face crop."
    )


if __name__ == "__main__":
    main()
