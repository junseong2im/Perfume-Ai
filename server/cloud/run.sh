#!/bin/bash
# Fragrance AI v6 - SOTA Training Pipeline
# Run this on Elice Cloud GPU instance
set -e

echo "========================================"
echo "  Fragrance AI v6 SOTA - Setup & Training"
echo "========================================"

# 1. Install dependencies
echo "[1/5] Installing dependencies..."
pip install -q torch-geometric rdkit-pypi transformers pandas requests

# 2. Check GPU
echo "[2/5] Checking GPU..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 3. Self-supervised GNN pretraining (optional but recommended)
echo "[3/5] Self-supervised GNN pretraining..."
if [ ! -f "weights/v6/gnn_pretrained.pt" ]; then
    python pretrain_ssl.py \
        --n-molecules 50000 \
        --epochs 50 \
        --batch-size 128 \
        --device cuda \
        --save-path weights/v6/gnn_pretrained.pt \
        --source generate
    echo "  GNN pretraining complete!"
else
    echo "  Using existing pretrained GNN weights"
fi

# 4. Generate ChemBERTa cache
echo "[4/5] ChemBERTa cache will be generated automatically..."

# 5. Run main training with pretrained GNN
echo "[5/5] Starting SOTA training..."
python train_v6.py \
    --model odor \
    --device cuda \
    --epochs 300 \
    --batch-size 64 \
    --accumulation 4 \
    --pretrained-gnn weights/v6/gnn_pretrained.pt

echo ""
echo "========================================"
echo "  Training Complete!"
echo "  Weights saved to: weights/v6/"
echo "========================================"
