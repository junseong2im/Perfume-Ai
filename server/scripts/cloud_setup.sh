#!/bin/bash
set -e
echo "=== Setup Cloud Training ==="

# Unzip
cd ~
mkdir -p cloud
cd cloud
unzip -o ~/cloud_training_fixed.zip
echo "=== Unzipped ==="

# Check environment
echo "=== Python ==="
python3 --version

echo "=== PyTorch ==="
python3 -c "import torch; print('torch='+torch.__version__); print('cuda='+str(torch.cuda.is_available())); print('gpu='+str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')); print('vram='+str(round(torch.cuda.get_device_properties(0).total_mem/1e9,1)) if torch.cuda.is_available() else '')" 2>/dev/null || echo "PyTorch missing"

echo "=== Check Packages ==="
python3 -c "import torch_geometric; print('pyg='+torch_geometric.__version__)" 2>/dev/null || echo "torch_geometric MISSING"
python3 -c "import rdkit; print('rdkit OK')" 2>/dev/null || echo "rdkit MISSING"
python3 -c "import transformers; print('transformers='+transformers.__version__)" 2>/dev/null || echo "transformers MISSING"

echo "=== Disk ==="
df -h ~ | tail -1

echo "=== Done ==="
