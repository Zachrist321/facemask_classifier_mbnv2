"""TensorFlow data pipeline and training helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np
import tensorflow as tf

from .data import CLASS_TO_INDEX, FaceSample, crop_face
from .model import IMG_SIZE


def _sample_to_dict(sample: FaceSample) -> dict:
    return {
        "image_path": str(sample.image_path),
        "label": CLASS_TO_INDEX[sample.label],
        "bbox": sample.bbox,
    }


def make_tf_dataset(
    samples: list[FaceSample],
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    records = [_sample_to_dict(s) for s in samples]

    def generator():
        for record in records:
            yield record

    output_signature = {
        "image_path": tf.TensorSpec(shape=(), dtype=tf.string),
        "label": tf.TensorSpec(shape=(), dtype=tf.int32),
        "bbox": tf.TensorSpec(shape=(4,), dtype=tf.int32),
    }
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(records), 1024), reshuffle_each_iteration=True)

    ds = ds.map(_load_and_crop, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _load_and_crop(record: dict) -> tuple[tf.Tensor, tf.Tensor]:
    image_path = record["image_path"]
    label = record["label"]
    bbox = record["bbox"]

    def _py_load(path, box) -> np.ndarray:
        from .data import FaceSample

        if hasattr(path, "numpy"):
            path = path.numpy()
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        else:
            path = str(path)

        if hasattr(box, "numpy"):
            box = box.numpy()

        sample = FaceSample(
            image_path=path,
            label="with_mask",
            bbox=tuple(int(v) for v in box.tolist()),
        )
        return crop_face(sample, img_size=IMG_SIZE)

    image = tf.py_function(
        func=_py_load,
        inp=[image_path, bbox],
        Tout=tf.uint8,
    )
    image.set_shape((*IMG_SIZE, 3))
    return tf.cast(image, tf.float32), label


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def _preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def evaluate_model(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
) -> dict[str, float]:
    results = model.evaluate(val_ds, verbose=0)
    metric_names = model.metrics_names
    return {name: float(value) for name, value in zip(metric_names, results)}


def predict_face_crop(
    model: tf.keras.Model,
    image: np.ndarray,
) -> tuple[str, float]:
    """Run inference on a single RGB uint8 crop of shape (H, W, 3)."""
    from .data import CLASS_NAMES

    if image.shape[:2] != IMG_SIZE:
        from PIL import Image

        image = np.asarray(
            Image.fromarray(image).resize(IMG_SIZE, Image.BILINEAR),
            dtype=np.uint8,
        )

    batch = np.expand_dims(image.astype(np.float32), axis=0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])
