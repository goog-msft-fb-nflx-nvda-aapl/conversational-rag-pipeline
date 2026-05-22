#!/bin/bash
# Download fine-tuned retriever and reranker models from HuggingFace Hub
set -e

mkdir -p models/retriever models/reranker

echo "Downloading retriever..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='jameswatanabegoogle2024/adl-hw3-retriever',
    local_dir='models/retriever',
    local_dir_use_symlinks=False
)
"

echo "Downloading reranker..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='jameswatanabegoogle2024/adl-hw3-reranker',
    local_dir='models/reranker',
    local_dir_use_symlinks=False
)
"

echo "Download complete."
ls -lh models/retriever/model.safetensors models/reranker/model.safetensors
