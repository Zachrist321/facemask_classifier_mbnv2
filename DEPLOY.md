# Deploy to Streamlit Community Cloud

Push this repo to GitHub, then deploy at [share.streamlit.io](https://share.streamlit.io).

## Files Streamlit Cloud needs

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main app (set as entry point) |
| `requirements.txt` | Python dependencies |
| `face_mask_artifacts/face_mask_mobilenet.weights.h5` | Trained weights (~9.5 MB) |
| `face_mask_artifacts/history.json` | Optional — shows val accuracy in UI |

## Steps

1. **Push to GitHub** (include the weights file):
   ```bash
   git add streamlit_app.py requirements.txt .streamlit/ face_mask_artifacts/
   git commit -m "Add Streamlit deployment"
   git push
   ```

2. **Open** [share.streamlit.io](https://share.streamlit.io) → **Create app**

3. **Connect** your GitHub repo

4. **Settings:**
   - **Main file path:** `streamlit_app.py`
   - **Branch:** `main` (or your branch)

5. Click **Deploy** — first build takes ~5–10 min (TensorFlow install)

Your public URL will look like:

`https://your-app-name.streamlit.app`

Use that link for your capstone submission.

## Run locally

```bash
conda activate ml
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Troubleshooting

- **Model not found:** ensure `face_mask_mobilenet.weights.h5` is committed and pushed (not gitignored).
- **Build fails on TensorFlow:** `requirements.txt` uses `tensorflow-cpu` for Cloud compatibility.
- **App sleeps:** free tier apps sleep after inactivity; first visit may take ~30s to wake.
