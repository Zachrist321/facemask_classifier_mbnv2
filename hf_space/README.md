---
title: Face Mask Classifier
emoji: 😷
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Face Mask Classifier

MobileNetV2 transfer-learning model that classifies face images into:

- **With Mask**
- **Mask Worn Incorrectly**
- **Without Mask**

Trained on the [MAKS face mask dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) (Kaggle). Validation accuracy: **~92.5%**.

Upload a face image or use your webcam. Works best on a clear, centered face crop.

## Model

| Setting | Value |
|---------|-------|
| Architecture | MobileNetV2 (frozen ImageNet backbone) |
| Input | 224×224 RGB |
| Classes | 3 |

## Limitations

- Trained on cropped faces; full-scene photos without a clear face may be less accurate.
- `mask_weared_incorrect` is the hardest class (smallest training set).
