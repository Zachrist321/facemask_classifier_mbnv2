#!/usr/bin/env bash
# Bundle project files and push to a Hugging Face Space.
# Usage:
#   export HF_USERNAME=your_hf_username
#   ./hf_space/deploy.sh
#
# First time: hf auth login  (paste a Write token from huggingface.co/settings/tokens)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="$ROOT/hf_space/staging"
SPACE_NAME="${HF_SPACE_NAME:-face-mask-classifier}"
HF_USERNAME="${HF_USERNAME:-}"

if [[ -z "$HF_USERNAME" ]]; then
  echo "Set your Hugging Face username:"
  echo "  export HF_USERNAME=your_username"
  exit 1
fi

REPO_ID="${HF_USERNAME}/${SPACE_NAME}"

echo "Staging files for Space: $REPO_ID"
rm -rf "$STAGING"
mkdir -p "$STAGING/face_mask_artifacts/eval/failure_cases" "$STAGING/src"

cp "$ROOT/app.py" "$STAGING/"
cp "$ROOT/hf_space/requirements.txt" "$STAGING/"
cp "$ROOT/hf_space/README.md" "$STAGING/"
cp "$ROOT/src/"*.py "$STAGING/src/"
cp "$ROOT/face_mask_artifacts/face_mask_mobilenet.keras" "$STAGING/face_mask_artifacts/"
cp "$ROOT/face_mask_artifacts/config.json" "$STAGING/face_mask_artifacts/"
cp "$ROOT/face_mask_artifacts/class_names.json" "$STAGING/face_mask_artifacts/"
cp "$ROOT/face_mask_artifacts/history.json" "$STAGING/face_mask_artifacts/"
cp "$ROOT/face_mask_artifacts/eval/failure_cases/"*.png "$STAGING/face_mask_artifacts/eval/failure_cases/" 2>/dev/null || true

echo "Creating Space (if needed) and uploading..."
hf repo create "$SPACE_NAME" --type space --space-sdk gradio --exist-ok 2>/dev/null \
  || huggingface-cli repo create "$SPACE_NAME" --type space --space-sdk gradio --exist-ok

hf upload "$REPO_ID" "$STAGING/." . --repo-type space --commit-message "Deploy face mask classifier" \
  || huggingface-cli upload "$REPO_ID" "$STAGING/." . --repo-type space --commit-message "Deploy face mask classifier"

echo ""
echo "Done! Space URL:"
echo "  https://huggingface.co/spaces/${REPO_ID}"
echo ""
echo "First build may take 5–10 minutes (TensorFlow install)."
