"""MobileNetV2 transfer-learning model with a mostly frozen backbone."""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

from .data import CLASS_NAMES

IMG_SIZE = (224, 224)


def build_mobilenet_classifier(
    num_classes: int = len(CLASS_NAMES),
    dropout: float = 0.3,
    unfreeze_last_n_layers: int = 0,
    backbone_weights: str | None = "imagenet",
) -> tf.keras.Model:
    """
    Build MobileNetV2 with ImageNet weights.

    By default the entire backbone is frozen; only the classification head trains.
    Set unfreeze_last_n_layers > 0 to fine-tune the last N backbone layers.
    """
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=backbone_weights,
        input_tensor=inputs,
        pooling=None,
    )
    base_model.trainable = False

    if unfreeze_last_n_layers > 0:
        for layer in base_model.layers[-unfreeze_last_n_layers:]:
            layer.trainable = True

    x = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="face_mask_mobilenet")
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_trainable_layers(model: tf.keras.Model) -> tuple[int, int]:
    trainable = sum(int(layer.trainable) for layer in model.layers)
    return trainable, len(model.layers)


def save_artifacts(
    model: tf.keras.Model,
    output_dir: str | Path,
    history: dict | None = None,
) -> None:
    """Save model, weights, class labels, and optional training history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / "face_mask_mobilenet.keras")
    model.save_weights(output_dir / "face_mask_mobilenet.weights.h5")

    (output_dir / "class_names.json").write_text(json.dumps(list(CLASS_NAMES), indent=2))
    if history is not None:
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))


def load_model_for_inference(model_path: str | Path) -> tf.keras.Model:
    path = Path(model_path)
    if path.suffix == ".h5" or "weights" in path.name:
        model = build_mobilenet_classifier(backbone_weights=None)
        model.load_weights(path)
        return model
    return tf.keras.models.load_model(path)
