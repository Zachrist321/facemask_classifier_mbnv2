# Face Mask Classification — CV Capstone

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
├── app.py                        # Gradio deployment app
├── evaluate.py                   # Local evaluation
└── find_failures.py              # Export failure cases
```

## Setup

```bash
conda activate ml
pip install -r requirements.txt
```

## Evaluate locally

```bash
python evaluate.py
```

## Run the app (Gradio)

```bash
python app.py
```

Opens a local web UI — upload or webcam a face image for prediction.

## Deploy to Hugging Face Spaces

Everything is ready in `hf_space/`. One-time setup, then one command:

**1. Create a token** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (type: **Write**).

**2. Log in** (in terminal):

```bash
conda activate ml
hf auth login
```

**3. Deploy:**

```bash
cd /Users/zach/python/cv_final_project
export HF_USERNAME=your_hf_username   # e.g. zach123
./hf_space/deploy.sh
```

Optional: custom space name (default `face-mask-classifier`):

```bash
export HF_SPACE_NAME=my-face-mask-app
./hf_space/deploy.sh
```

Your app will be live at:

`https://huggingface.co/spaces/YOUR_USERNAME/face-mask-classifier`

First build takes **5–10 minutes** (TensorFlow install on HF). Use that URL for your capstone submission.

**Manual alternative:** create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space) (SDK: Gradio), then upload the contents of `hf_space/staging/` after running the deploy script once locally (it creates the staging folder), or copy `app.py`, `requirements.txt`, `hf_space/README.md` → `README.md`, `src/`, and `face_mask_artifacts/`.

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
