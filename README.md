# Face Mask Classification — MSc AI: Computer Vision Project

**Live app:** [https://facemask-classifier-mbnv2.streamlit.app](https://facemask-classifier-mbnv2.streamlit.app)

**GitHub:** [https://github.com/Zachrist321/facemask_classifier_mbnv2](https://github.com/Zachrist321/facemask_classifier_mbnv2)

**Report (PDF):** [report.pdf](report.pdf)

Image classification using **MobileNetV2** transfer learning on the [MAKS face mask dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset).

**Problem statement:** Given a face image, the system classifies mask compliance (`with_mask`, `mask_weared_incorrect`, `without_mask`) so that public-health teams can monitor mask usage.

## Results (validation set)

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.5% |
| **Weighted F1** | 0.91 |
| with_mask F1 | 0.96 |
| without_mask F1 | 0.85 |
| mask_weared_incorrect F1 | 0.14 (small class — 24 val samples) |

Artifacts: `face_mask_artifacts/`

## Project layout

```
cv_final_project/
├── archive-4/                    # Kaggle dataset
├── face_mask_artifacts/          # Trained model + eval outputs
│   ├── face_mask_mobilenet.keras
│   ├── eval/
│   │   ├── classification_report.json
│   │   ├── confusion_matrix.png
│   │   └── failure_cases/        # Misclassified examples for report
├── notebooks/train_mobilenet_colab.ipynb
├── src/                          # Data, model, training utilities
├── streamlit_app.py              # Streamlit app (deploy this)
├── DEPLOY.md                     # Streamlit Cloud instructions
├── app.py                        # Legacy Gradio app (optional)
├── evaluate.py                   # Local evaluation
└── find_failures.py              # Export failure cases
```

## Setup (local — use your `ml` conda env)

Libraries live in your **conda `ml` environment**, not system Python.

```bash
conda activate ml
pip install -r requirements-local.txt   # optional; you likely already have these
streamlit run streamlit_app.py
```

Always use `python` / `streamlit` from `ml` after `conda activate ml` — **not** bare `python3`.

| Where | Python | TensorFlow |
|-------|--------|------------|
| **Your Mac (`ml` env)** | 3.13 | 2.21 |
| **Streamlit Cloud** | 3.11 | 2.15.1 (`requirements.txt`) |

Cloud installs from `requirements.txt` on their servers — separate from your local venv.

## Evaluate locally

```bash
python evaluate.py
```

## Run the app (Streamlit)

```bash
conda activate ml
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud

See **[DEPLOY.md](DEPLOY.md)** for full steps:

1. Push repo to GitHub (include `face_mask_artifacts/face_mask_mobilenet.weights.h5`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → Create app
3. Main file: `streamlit_app.py`
4. Deploy → use your `*.streamlit.app` URL for submission

## Train on Colab

See `notebooks/train_mobilenet_colab.ipynb` — mount Google Drive, train on GPU, save weights to `face_mask_artifacts/`.

## Model

| Setting | Value |
|---------|-------|
| Backbone | MobileNetV2 (ImageNet, frozen) |
| Head | GAP → Dropout(0.3) → Dense(3) |
| Input | 224×224 face crops |
| Best val accuracy | ~92.4% |

## Failure cases

Run `python find_failures.py` to export 5 misclassified validation crops with explanations in `face_mask_artifacts/eval/failure_cases/failure_cases.json`.

Common failure modes:
- **mask_weared_incorrect** confused with with/without mask (class imbalance)
- Thin or light-colored masks mistaken for no mask
- Shadows / hands near face mistaken for mask edges
