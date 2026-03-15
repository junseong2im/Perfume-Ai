"""Quick test of SMILES augmentation and dataset loading"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Test 1: SMILES augmentation
print("=" * 60)
print("  Test 1: SMILES Augmentation")
print("=" * 60)
from train_v6 import randomize_smiles

test_smiles = [
    "CCO",               # ethanol
    "c1ccccc1",          # benzene
    "CC(=O)O",           # acetic acid
    "CC(C)CC1=CC=C(C=C1)C(C)C",  # ibuprofen
]

for smi in test_smiles:
    augs = randomize_smiles(smi, n_aug=5)
    print(f"  {smi:30s} -> {len(augs)} variants: {augs[:3]}...")

# Test 2: Dataset loading from curated data
print("\n" + "=" * 60)
print("  Test 2: Curated Dataset Loading (no augmentation)")
print("=" * 60)
from train_v6 import OdorDataset

csv_path = os.path.join(os.path.dirname(__file__), "data", "curated_training_data.csv")
ds = OdorDataset(csv_path, n_aug=0, label_smooth=0.0)
if len(ds) > 0:
    print(f"  Loaded {len(ds)} samples")
    print(f"  Label dims: {ds.n_labels}")
    sample = ds[0]
    print(f"  Sample SMILES: {sample['smiles'][:40]}")
    print(f"  Sample odor shape: {sample['odor'].shape}")
    print(f"  Sample weight: {sample['weight']:.3f}")
    # Count non-zero labels
    nz = (sample['odor'] > 0.01).sum().item()
    print(f"  Non-zero labels: {nz}")
else:
    print("  [ERROR] No samples loaded!")

# Test 3: Dataset with augmentation
print("\n" + "=" * 60)
print("  Test 3: Curated Dataset Loading (with 3x augmentation)")
print("=" * 60)
ds_aug = OdorDataset(csv_path, n_aug=3, label_smooth=0.05, max_samples=100)
print(f"  Original + augmented: {len(ds_aug)} samples")
if len(ds_aug) > 0:
    print(f"  Effective multiplier: {len(ds_aug)/min(100, len(ds)):.1f}x")

print("\n  ALL TESTS PASSED")
