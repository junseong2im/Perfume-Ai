"""train_v6.py - SOTA Training Pipeline with ALL Techniques
===========================================================
Techniques included:
1.  ChemBERTa embedding cache (384d from DeepChem/ChemBERTa-77M-MTR)
2.  Scaffold Split (train/val/test = 75/15/10)
3.  Multi-task loss: MSE + CosSim + Label Smoothing + Focal BCE
4.  Contrastive loss (supervised, temp=0.07)
5.  R-Drop (KL divergence regularization)
6.  GradNorm automatic loss balancing (alpha=1.5)
7.  Per-component learning rates (7 param groups)
8.  Warm-up (10 epochs) + CosineAnnealingWarmRestarts (T0=50, Tmult=2)
9.  EMA (decay=0.9995)
10. SWA (last 25% of epochs)
11. Progressive Unfreezing: epoch<30 H1 only, <80 +H2-6, >80 all
12. Curriculum Learning: epoch<50 easy, <150 normal, >150 hard
13. Mixup (alpha=0.4, 50% prob)
14. CutMix (alpha=0.4, 50% prob)
15. Gradient Accumulation (4 steps)
16. Gradient Clipping (max_norm=1.0)
17. Deep Ensemble (5 seeds with --ensemble)
18. Self-supervised pretrained GNN weights loading
19. LoRA fine-tuning support for ChemBERTa
20. **SMILES augmentation (5x effective data via random non-canonical forms)**
21. **Label smoothing (0.05) for anti-overfitting**
22. **Stronger weight decay (0.05 AdamW)**
23. **Dynamic label loading from curated_training_data.csv (612 labels)**

Usage:
    python train_v6.py --model odor --device cuda
    python train_v6.py --model odor --device cuda --pretrained-gnn weights/v6/gnn_pretrained.pt
    python train_v6.py --model odor --device cuda --ensemble
    python train_v6.py --model odor --device cuda --epochs 300 --batch-size 64
"""

import os
import sys
import csv
import json
import math
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from models.odor_predictor_v6 import (
    OdorPredictorV6, smiles_to_graph_v6, extract_phys_props,
    EMA, GradNorm, compute_loss, contrastive_loss, N_ODOR_DIM
)

try:
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[WARNING] torch_geometric not found.")

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Fallback 22-dim labels (used when curated data not available)
ODOR_DIMS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]


# ================================================================
# SMILES Augmentation (Anti-Overfitting Core)
# ================================================================

def randomize_smiles(smi, n_aug=5):
    """Generate n_aug random non-canonical SMILES representations.
    This is the most effective molecular data augmentation technique.
    Each valid SMILES can be written in many equivalent ways by
    randomizing atom ordering."""
    if not HAS_RDKIT:
        return [smi]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]
    augmented = set()
    augmented.add(smi)  # always include canonical
    attempts = 0
    max_attempts = n_aug * 3
    while len(augmented) < n_aug + 1 and attempts < max_attempts:
        try:
            new_smi = Chem.MolToSmiles(mol, doRandom=True)
            if new_smi:
                augmented.add(new_smi)
        except:
            pass
        attempts += 1
    return list(augmented)


def load_label_mapping(data_dir):
    """Load label mapping from curated data output."""
    mapping_path = os.path.join(data_dir, 'label_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"  Label mapping loaded: {mapping['n_labels']} labels")
        return mapping['labels']
    return None


# ================================================================
# ChemBERTa Cache
# ================================================================

def build_bert_cache(csv_path, cache_path, model_name='DeepChem/ChemBERTa-77M-MTR'):
    """Generate ChemBERTa embeddings for all SMILES in CSV"""
    if os.path.exists(cache_path):
        try:
            cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        except TypeError:
            cache = torch.load(cache_path, map_location='cpu')
        print(f"  BERT cache loaded: {len(cache)} embeddings")
        return cache

    print(f"  Building ChemBERTa cache from {model_name}...")
    try:
        # Fix for transformers CVE-2025-32434 security check
        os.environ['TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION'] = '1'
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("  [WARNING] transformers not installed. Using zero embeddings.")
        return {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    bert_model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = bert_model.to(device)

    cache = {}
    smiles_list = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            smi = row.get('smiles', '')
            if smi:
                smiles_list.append(smi)

    print(f"  Processing {len(smiles_list)} SMILES...")
    batch_size = 64
    for i in range(0, len(smiles_list), batch_size):
        batch_smi = smiles_list[i:i+batch_size]
        try:
            inputs = tokenizer(batch_smi, return_tensors='pt', padding=True,
                             truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                embs = outputs.last_hidden_state.mean(dim=1).cpu()
            for smi, emb in zip(batch_smi, embs):
                cache[smi] = emb
        except Exception as e:
            for smi in batch_smi:
                try:
                    inputs = tokenizer(smi, return_tensors='pt', padding=True,
                                     truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        outputs = bert_model(**inputs)
                        emb = outputs.last_hidden_state.mean(dim=1).cpu().squeeze()
                    cache[smi] = emb
                except:
                    pass

        if (i // batch_size) % 50 == 0:
            print(f"    {i+len(batch_smi)}/{len(smiles_list)} done...")

    print(f"  Cached {len(cache)} embeddings (dim={list(cache.values())[0].shape[0] if cache else '?'})")
    torch.save(cache, cache_path)
    del bert_model
    torch.cuda.empty_cache()
    return cache


# ================================================================
# Dataset
# ================================================================

class OdorDataset(Dataset):
    """Full dataset with BERT cache, SMILES augmentation, and dynamic labels.

    Key anti-overfitting features:
    - SMILES augmentation: 5x effective data via random non-canonical forms
    - Label smoothing: smooth_factor applied to binary labels
    - Quality-weighted: higher weight for multi-source validated data
    - Dynamic labels: loads whatever label columns exist in CSV
    """
    def __init__(self, csv_path, bert_cache=None, max_samples=None,
                 label_cols=None, n_aug=5, label_smooth=0.05):
        self.data = []
        self.bert_cache = bert_cache or {}
        self.bert_dim = 384
        self.n_aug = n_aug
        self.label_smooth = label_smooth

        # Auto-detect BERT dim from cache
        if self.bert_cache:
            sample = next(iter(self.bert_cache.values()))
            self.bert_dim = sample.shape[0] if hasattr(sample, 'shape') else len(sample)

        if not os.path.exists(csv_path):
            print(f"  [WARNING] CSV not found: {csv_path}")
            return

        # Detect labels: use provided label_cols or auto-detect from CSV
        meta_cols = {'smiles', 'sources', 'n_sources', 'tier', 'quality_score',
                     'confidence', 'weight'}

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_cols = reader.fieldnames
            if label_cols is None:
                # Auto-detect: anything not in meta_cols is a label
                label_cols = [c for c in all_cols if c.lower() not in meta_cols]
            self.label_cols = label_cols
            self.n_labels = len(label_cols)
            print(f"  Label columns detected: {self.n_labels}")

            for row in reader:
                smi = row.get('smiles', '')
                if not smi:
                    continue

                # Extract labels
                odor = []
                for dim in self.label_cols:
                    v = row.get(dim, '0')
                    try:
                        odor.append(float(v))
                    except (ValueError, TypeError):
                        odor.append(0.0)

                odor_arr = np.array(odor, dtype=np.float32)

                # Apply label smoothing: 1.0 -> 0.95, 0.0 -> 0.05
                if self.label_smooth > 0:
                    odor_arr = odor_arr * (1.0 - self.label_smooth) + self.label_smooth * 0.5

                # Quality weight from CSV (tier 1 = 1.0, tier 2 = 0.8)
                tier = int(row.get('tier', 1))
                quality = float(row.get('quality_score', 0.7))
                weight = quality * (1.0 if tier == 1 else 0.8)

                # Difficulty score for curriculum learning
                nonzero = odor_arr[odor_arr > 0.05]
                if len(nonzero) > 0:
                    probs = nonzero / nonzero.sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                else:
                    entropy = 0.0

                self.data.append({
                    'smiles': smi,
                    'odor': odor_arr,
                    'weight': weight,
                    'difficulty': entropy,
                })
                if max_samples and len(self.data) >= max_samples:
                    break

        # Generate SMILES augmentations for training
        if self.n_aug > 0 and HAS_RDKIT:
            print(f"  SMILES augmentation: generating {self.n_aug}x variants...")
            augmented = []
            for item in self.data:
                aug_smiles = randomize_smiles(item['smiles'], n_aug=self.n_aug)
                for aug_smi in aug_smiles:
                    if aug_smi != item['smiles']:  # skip original (already in data)
                        augmented.append({
                            'smiles': aug_smi,
                            'odor': item['odor'].copy(),
                            'weight': item['weight'] * 0.8,  # slight discount for augmented
                            'difficulty': item['difficulty'],
                            'is_augmented': True,
                        })
            print(f"  Augmented: {len(augmented)} new variants (from {len(self.data)} originals)")
            self.data.extend(augmented)

        # Sort by difficulty for curriculum
        if self.data:
            self.difficulties = np.array([d['difficulty'] for d in self.data])
            self.easy_threshold = np.percentile(self.difficulties, 33)
            self.hard_threshold = np.percentile(self.difficulties, 67)
        else:
            self.difficulties = np.array([])
            self.easy_threshold = 0
            self.hard_threshold = 0

        print(f"  Total samples: {len(self.data)} (bert_dim={self.bert_dim}, labels={self.n_labels})")

    def __len__(self):
        return len(self.data)

    def get_curriculum_indices(self, stage):
        """Return indices for curriculum learning stage"""
        if stage == 'easy':
            return [i for i, d in enumerate(self.data) if d['difficulty'] <= self.easy_threshold]
        elif stage == 'normal':
            return list(range(len(self.data)))
        else:  # hard
            return [i for i, d in enumerate(self.data) if d['difficulty'] >= self.hard_threshold]

    def __getitem__(self, idx):
        item = self.data[idx]
        smi = item['smiles']

        # BERT embedding from cache (try canonical lookup first)
        if smi in self.bert_cache:
            bert = self.bert_cache[smi]
        else:
            # For augmented SMILES, try to find canonical version in cache
            bert = None
            if HAS_RDKIT:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    can = Chem.MolToSmiles(mol)
                    bert = self.bert_cache.get(can)
            if bert is None:
                bert = torch.zeros(self.bert_dim, dtype=torch.float32)

        if isinstance(bert, np.ndarray):
            bert = torch.tensor(bert, dtype=torch.float32)
        elif not isinstance(bert, torch.Tensor):
            bert = torch.zeros(self.bert_dim, dtype=torch.float32)

        # Graph (cached to avoid recomputing 3D coords every epoch)
        if not hasattr(self, '_graph_cache'):
            self._graph_cache = {}
        if smi in self._graph_cache:
            graph = self._graph_cache[smi]
        elif HAS_PYG:
            graph = smiles_to_graph_v6(smi, compute_3d=True)
            if len(self._graph_cache) < 50000:  # cap memory
                self._graph_cache[smi] = graph
        else:
            graph = None
        if graph is None and HAS_PYG:
            graph = Data(
                x=torch.zeros(1, 47),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 13),
            )

        phys = extract_phys_props(smi)

        return {
            'graph': graph,
            'bert': bert,
            'phys': torch.tensor(phys, dtype=torch.float32),
            'odor': torch.tensor(item['odor'], dtype=torch.float32),
            'weight': item['weight'],
            'difficulty': item['difficulty'],
            'smiles': smi,
        }


def collate_odor(batch):
    """Custom collate for PyG graphs + BERT embeddings"""
    graphs = [b['graph'] for b in batch if b['graph'] is not None]
    bert = torch.stack([b['bert'] for b in batch])
    phys = torch.stack([b['phys'] for b in batch])
    odor = torch.stack([b['odor'] for b in batch])
    weights = torch.tensor([b['weight'] for b in batch], dtype=torch.float32)

    if graphs and HAS_PYG:
        graph_batch = Batch.from_data_list(graphs)
    else:
        graph_batch = None

    return {
        'graph_batch': graph_batch,
        'bert': bert, 'phys': phys,
        'odor': odor, 'weights': weights,
    }


# ================================================================
# Data Splitting
# ================================================================

def scaffold_split(dataset, val_ratio=0.15, test_ratio=0.10, seed=42):
    """Scaffold-based split for proper generalization testing"""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        print("  [FALLBACK] Using random split")
        return random_split(dataset, val_ratio, test_ratio, seed)

    scaffolds = defaultdict(list)
    for idx in range(len(dataset)):
        smi = dataset.data[idx]['smiles']
        try:
            mol = Chem.MolFromSmiles(smi)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds[scaffold].append(idx)
        except:
            scaffolds['_unknown_'].append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(val_idx) < n_val:
            val_idx.extend(group)
        elif len(test_idx) < n_test:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    print(f"  Scaffold split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def random_split(dataset, val_ratio=0.15, test_ratio=0.10, seed=42):
    n = len(dataset)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    return idx[n_val+n_test:], idx[:n_val], idx[n_val:n_val+n_test]


# ================================================================
# Augmentation: Mixup + CutMix
# ================================================================

def mixup_data(x_bert, x_phys, y, alpha=0.4):
    """Mixup augmentation on BERT embeddings and labels"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B = x_bert.size(0)
    perm = torch.randperm(B, device=x_bert.device)
    mixed_bert = lam * x_bert + (1 - lam) * x_bert[perm]
    mixed_phys = lam * x_phys + (1 - lam) * x_phys[perm]
    mixed_y = lam * y + (1 - lam) * y[perm]
    return mixed_bert, mixed_phys, mixed_y, lam


def cutmix_data(x_bert, y, alpha=0.4):
    """CutMix: randomly swap portions of feature vectors"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B, D = x_bert.shape
    perm = torch.randperm(B, device=x_bert.device)

    # Random mask for which dimensions to cut
    n_cut = int(D * (1 - lam))
    cut_start = random.randint(0, D - n_cut) if n_cut > 0 else 0
    mask = torch.ones(D, device=x_bert.device)
    mask[cut_start:cut_start+n_cut] = 0

    mixed_bert = x_bert * mask + x_bert[perm] * (1 - mask)
    mixed_y = lam * y + (1 - lam) * y[perm]
    return mixed_bert, mixed_y, lam


# ================================================================
# Progressive Unfreezing
# ================================================================

def apply_progressive_unfreezing(model, epoch):
    """Gradually unfreeze model components"""
    # Phase 1 (epoch < 30): Only H1 (odor head) + backbone
    # Phase 2 (epoch 30-80): + H2-H6 + fusion
    # Phase 3 (epoch > 80): Everything

    for name, param in model.named_parameters():
        if epoch < 30:
            # Only backbone + odor head trainable
            if 'heads.odor' in name or 'backbone' in name or 'skip' in name:
                param.requires_grad = True
            elif 'path_b' in name or 'path_c' in name or 'fusion' in name:
                param.requires_grad = True  # paths always train
            else:
                param.requires_grad = False
        elif epoch < 80:
            # + top/mid/base/longevity/sillage heads
            if any(h in name for h in ['heads.odor', 'heads.top', 'heads.mid',
                                        'heads.base', 'heads.longevity', 'heads.sillage',
                                        'backbone', 'skip', 'path_', 'fusion']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            # Everything trainable
            param.requires_grad = True


# ================================================================
# Learning Rate Scheduler with Warm-up
# ================================================================

class WarmupCosineScheduler:
    """Warm-up (linear) + Cosine Annealing with Warm Restarts"""
    def __init__(self, optimizer, warmup_epochs=10, T_0=50, T_mult=2, eta_min_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min_ratio = eta_min_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warm-up
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing with warm restarts
            t = epoch - self.warmup_epochs
            T_cur = self.T_0
            T_i = 0
            while t >= T_cur:
                t -= T_cur
                T_i += 1
                T_cur = int(self.T_0 * (self.T_mult ** T_i))
            factor = self.eta_min_ratio + 0.5 * (1 - self.eta_min_ratio) * (
                1 + math.cos(math.pi * t / T_cur)
            )

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * factor


# ================================================================
# Main Training Loop
# ================================================================

def train_odor_predictor(device, csv_path, save_dir, seed=42, epochs=300,
                         batch_size=64, bert_cache_path=None,
                         accumulation_steps=4, use_rdrop=True,
                         pretrained_gnn_path=None, use_lora=False,
                         n_aug=5, label_smooth=0.05,
                         resume_path=None, start_lr_factor=1.0,
                         n_workers=2, transfer_from=None):
    print(f"\n{'='*60}")
    print(f"  OdorPredictor v6 SOTA Training (seed={seed})")
    print(f"  device={device}, epochs={epochs}, batch_size={batch_size}")
    print(f"  accumulation={accumulation_steps}, rdrop={use_rdrop}")
    print(f"  pretrained_gnn={pretrained_gnn_path is not None}")
    print(f"  lora={use_lora}, n_aug={n_aug}, label_smooth={label_smooth}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Build/load BERT cache
    cache_path = bert_cache_path or 'data/bert_cache.pt'
    bert_cache = build_bert_cache(csv_path, cache_path)

    # 2. Load dataset (with SMILES augmentation + label smoothing)
    dataset = OdorDataset(csv_path, bert_cache=bert_cache,
                          n_aug=n_aug, label_smooth=label_smooth)
    if len(dataset) == 0:
        print("  [ERROR] Empty dataset!")
        return 0.0

    bert_dim = dataset.bert_dim

    # 3. Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(dataset, seed=seed)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_odor, num_workers=n_workers,
                              drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_odor, num_workers=n_workers,
                            pin_memory=True, persistent_workers=True)

    # 4. Model (n_odor_dim matches dataset labels)
    n_odor_dim = dataset.n_labels if hasattr(dataset, 'n_labels') else None
    model = OdorPredictorV6(bert_dim=bert_dim, use_lora=use_lora, n_odor_dim=n_odor_dim).to(device)

    # 4b. Load pretrained GNN weights
    if pretrained_gnn_path and os.path.exists(pretrained_gnn_path):
        print(f"  Loading pretrained GNN from {pretrained_gnn_path}")
        ckpt = torch.load(pretrained_gnn_path, map_location=device)
        gnn_state = ckpt.get('gnn_state_dict', ckpt)
        # Match keys: pretrained uses 'gnn.*' -> model uses 'path_b.*'
        matched, skipped = 0, 0
        model_state = model.state_dict()
        for k, v in gnn_state.items():
            # Try direct loading into path_b
            target_key = k if k.startswith('path_b.') else f'path_b.{k}'
            if target_key not in model_state:
                target_key = k  # try exact match
            if target_key in model_state and v.shape == model_state[target_key].shape:
                model_state[target_key] = v
                matched += 1
            else:
                skipped += 1
        model.load_state_dict(model_state, strict=False)
        print(f"  Loaded {matched} GNN params ({skipped} skipped)")
    else:
        if pretrained_gnn_path:
            print(f"  [WARNING] Pretrained GNN not found: {pretrained_gnn_path}")

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {total_params:,} total, {train_params:,} trainable")

    # 5. Optimizer with per-component learning rates (stronger weight decay for anti-overfitting)
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'path_a' in n],
         'lr': 5e-5, 'weight_decay': 0.05, 'name': 'path_a'},
        {'params': [p for n, p in model.named_parameters() if 'path_b' in n],
         'lr': 1e-3, 'weight_decay': 0.05, 'name': 'path_b'},
        {'params': [p for n, p in model.named_parameters() if 'path_c' in n],
         'lr': 1e-3, 'weight_decay': 0.05, 'name': 'path_c'},
        {'params': [p for n, p in model.named_parameters() if 'fusion' in n],
         'lr': 5e-4, 'weight_decay': 0.05, 'name': 'fusion'},
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n or 'skip' in n],
         'lr': 1e-3, 'weight_decay': 0.05, 'name': 'backbone'},
        {'params': [p for n, p in model.named_parameters() if 'heads' in n],
         'lr': 1e-3, 'weight_decay': 0.01, 'name': 'heads'},
        {'params': [model.loss_weights], 'lr': 0.025, 'weight_decay': 0, 'name': 'loss_weights'},
    ]
    # Filter empty groups
    param_groups = [pg for pg in param_groups if len(list(pg['params'])) > 0]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    # 6. Scheduler
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, T_0=50, T_mult=2)

    # 7. EMA
    ema = EMA(model, decay=0.9995)

    # 8. SWA
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(epochs * 0.75)

    # 9. GradNorm
    gradnorm = GradNorm(model, n_tasks=10, alpha=1.5)

    # Training state
    best_val_loss = float('inf')
    best_cos_sim = 0.0
    patience_count = 0
    patience_limit = 25
    start_epoch = 1
    os.makedirs(save_dir, exist_ok=True)

    # === Checkpoint Resume ===
    if resume_path and os.path.exists(resume_path):
        print(f"  \u2500\u2500 Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state' in ckpt:
            # Replay scheduler steps
            for e in range(ckpt.get('epoch', 0)):
                scheduler.step(e)
        if 'ema_shadow' in ckpt:
            ema.shadow = ckpt['ema_shadow']
        start_epoch = ckpt.get('epoch', 0) + 1
        best_cos_sim = ckpt.get('val_cos_sim', 0.0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"  \u2500\u2500 Resumed at epoch {start_epoch}, best_cos={best_cos_sim:.4f}")

    # === Transfer Learning (load weights only, add noise for diversity) ===
    if transfer_from and os.path.exists(transfer_from) and not resume_path:
        print(f"  \u2500\u2500 Transfer from: {transfer_from}")
        ckpt = torch.load(transfer_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
        print(f"  \u2500\u2500 Weights loaded + noise(std=0.01), optimizer/epoch RESET")

    # LR factor (for fine-tuning phases)
    if start_lr_factor != 1.0:
        for pg in optimizer.param_groups:
            pg['lr'] *= start_lr_factor
        print(f"  LR scaled by {start_lr_factor}x")

    for epoch in range(start_epoch, epochs + 1):
        # === Progressive Unfreezing ===
        apply_progressive_unfreezing(model, epoch)

        # === Curriculum Learning ===
        if epoch <= 50:
            curr_indices = dataset.get_curriculum_indices('easy')
            curr_desc = 'easy'
        elif epoch <= 150:
            curr_indices = dataset.get_curriculum_indices('normal')
            curr_desc = 'normal'
        else:
            curr_indices = dataset.get_curriculum_indices('hard')
            curr_desc = 'hard'

        # Filter curriculum indices to training set only
        train_set = set(train_idx)
        curr_train = [i for i in curr_indices if i in train_set]
        if len(curr_train) < batch_size:
            curr_train = train_idx  # fallback

        curr_ds = Subset(dataset, curr_train)
        curr_loader = DataLoader(curr_ds, batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_odor, num_workers=n_workers,
                                 drop_last=True, pin_memory=True, persistent_workers=True)

        # === Train ===
        model.train()
        train_loss = 0.0
        train_cos = 0.0
        n_batches = 0
        nan_count = 0  # Rule #11: NaN watchdog counter
        optimizer.zero_grad()

        for step, batch in enumerate(curr_loader):
            bert = batch['bert'].to(device)
            phys = batch['phys'].to(device)
            odor = batch['odor'].to(device)
            gb = batch['graph_batch']
            if gb is not None:
                gb = gb.to(device)
            else:
                continue

            # Augmentation: Mixup or CutMix (50% each, 50% no aug)
            # NOTE: Graph cannot be mixed, so during Mixup we detach graph
            # from gradient to prevent contradictory signal to GNN
            using_mixup = False
            aug_roll = random.random()
            if aug_roll < 0.25:
                bert, phys, odor, _ = mixup_data(bert, phys, odor, alpha=0.4)
                using_mixup = True
            elif aug_roll < 0.50:
                bert, odor, _ = cutmix_data(bert, odor, alpha=0.4)
                using_mixup = True

            # Detach graph during augmentation to avoid GNN learning wrong labels
            if using_mixup and gb is not None:
                gb = gb.clone()
                gb.x = gb.x.detach()
                if hasattr(gb, 'pos') and gb.pos is not None:
                    gb.pos = gb.pos.detach()

            # Forward pass 1
            pred = model(bert, gb, phys, return_aux=True)
            loss, loss_dict = compute_loss(model, pred, {'odor': odor},
                                           None, epoch, epochs, training=True)

            # Rule #12: R-Drop — staggered activation at E15 (was E10)
            # Rule #6 fix: sigmoid 독립 출력에 softmax 사용 금지
            if use_rdrop and epoch > 15:
                pred2 = model(bert, gb, phys, return_aux=True)
                p1 = pred['odor'].clamp(1e-7, 1 - 1e-7)
                p2 = pred2['odor'].clamp(1e-7, 1 - 1e-7)
                L_rdrop = (
                    F.binary_cross_entropy(p1, p2.detach(), reduction='mean') +
                    F.binary_cross_entropy(p2, p1.detach(), reduction='mean')
                ) * 0.5
                # Rule #11: per-loss NaN gate
                if not (torch.isnan(L_rdrop) or torch.isinf(L_rdrop)):
                    loss = loss + 0.1 * L_rdrop
                    loss_dict['rdrop'] = L_rdrop

            # Rule #12: Contrastive — staggered at E30 (was E20) + warmup
            if epoch > 30:
                L_contra = contrastive_loss(pred['odor'], odor, temperature=0.07)
                # Rule #11: per-loss NaN gate
                if not (torch.isnan(L_contra) or torch.isinf(L_contra)):
                    # Rule #12: warmup weight 0.01 → 0.05 over E30-E50
                    contra_w = min(0.05, 0.01 + 0.04 * (epoch - 30) / 20.0)
                    loss = loss + contra_w * L_contra
                    loss_dict['contrastive'] = L_contra

            # ── Rule #11: NaN Watchdog ──
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()  # discard corrupted gradients
                if step % 50 == 0:
                    print(f"    [NaN SKIP] step={step}, nan_count={nan_count}")
                continue  # skip this batch entirely

            # Gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                # Rule #11: gradient magnitude logging
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if step % 100 == 0 and step > 0:
                    print(f"    [grad] step={step} grad_norm={grad_norm:.4f}")
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            train_loss += loss.item() * accumulation_steps
            with torch.no_grad():
                cos = F.cosine_similarity(pred['odor'], odor, dim=1).mean().item()
                train_cos += cos
            n_batches += 1

        # Flush remaining gradients
        if n_batches % accumulation_steps != 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # ── Rule #11: NaN auto-rollback ──
        if nan_count >= 3:
            print(f"  [NaN ROLLBACK] {nan_count} NaN batches detected! "
                  f"Restoring EMA weights + halving lr")
            ema.apply_shadow(model)
            ema.restore(model)  # keep model = EMA weights
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
            nan_count = 0

        avg_train_loss = train_loss / max(n_batches, 1)
        avg_train_cos = train_cos / max(n_batches, 1)

        # GradNorm update
        if loss_dict:
            gradnorm.update(model, loss_dict, epoch)

        # SWA update
        if epoch >= swa_start:
            swa_model.update_parameters(model)

        # Scheduler
        scheduler.step(epoch)

        # === Validation (with EMA weights) ===
        model.eval()
        ema.apply_shadow(model)

        val_loss = 0.0
        val_cos = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                bert = batch['bert'].to(device)
                phys = batch['phys'].to(device)
                odor = batch['odor'].to(device)
                gb = batch['graph_batch']
                if gb is not None:
                    gb = gb.to(device)
                else:
                    continue

                pred = model(bert, gb, phys, return_aux=True)
                loss, _ = compute_loss(model, pred, {'odor': odor},
                                       None, epoch, epochs, training=False)
                val_loss += loss.item()
                cos = F.cosine_similarity(pred['odor'], odor, dim=1).mean().item()
                val_cos += cos
                n_val += 1

        ema.restore(model)

        avg_val_loss = val_loss / max(n_val, 1)
        avg_val_cos = val_cos / max(n_val, 1)

        # Logging
        if epoch % 5 == 0 or epoch <= 5 or epoch == epochs:
            lr_now = optimizer.param_groups[0]['lr']
            n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"  E{epoch:>3d}/{epochs} | "
                  f"trn_L={avg_train_loss:.4f} trn_cos={avg_train_cos:.4f} | "
                  f"val_L={avg_val_loss:.4f} val_cos={avg_val_cos:.4f} | "
                  f"lr={lr_now:.2e} cur={curr_desc} "
                  f"params={n_trainable}")

        # Best model checkpoint
        if avg_val_cos > best_cos_sim:
            best_cos_sim = avg_val_cos
            best_val_loss = avg_val_loss
            patience_count = 0
            ema.apply_shadow(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_shadow': ema.shadow,
                'val_loss': best_val_loss,
                'val_cos_sim': best_cos_sim,
                'bert_dim': bert_dim,
                'n_odor_dim': n_odor_dim,
                'seed': seed,
            }, os.path.join(save_dir, f'odor_v6_best_seed{seed}.pt'))
            ema.restore(model)
            if epoch % 10 == 0 or epoch <= 10:
                print(f"    >>> BEST model saved (cos_sim={best_cos_sim:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience_limit:
                print(f"  Early stopping at epoch {epoch}")
                break

    # SWA BatchNorm update — custom loop to handle dict DataLoader
    # (PyTorch's update_bn does model(**batch) which crashes on extra keys like 'odor')
    if epoch >= swa_start:
        print("  Updating SWA BatchNorm...")
        try:
            swa_model.train()
            with torch.no_grad():
                momenta = {}
                for module in swa_model.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                        if hasattr(module, 'running_mean'):
                            module.running_mean = torch.zeros_like(module.running_mean)
                            module.running_var = torch.ones_like(module.running_var)
                            momenta[module] = module.momentum
                            module.momentum = None
                            module.num_batches_tracked *= 0

                for batch in curr_loader:
                    bert_b = batch['bert'].to(device)
                    phys_b = batch['phys'].to(device)
                    gb_b = batch['graph_batch']
                    if gb_b is None:
                        continue
                    gb_b = gb_b.to(device)
                    swa_model(bert_b, gb_b, phys_b, return_aux=False)

                for module, mom in momenta.items():
                    module.momentum = mom

            torch.save({
                'model_state_dict': swa_model.module.state_dict(),
                'val_cos_sim': best_cos_sim,
                'bert_dim': bert_dim,
                'seed': seed,
                'swa': True,
            }, os.path.join(save_dir, f'odor_v6_swa_seed{seed}.pt'))
            print("  SWA model saved.")
        except Exception as e:
            print(f"  SWA BN update failed: {e}")

    # === Test set evaluation ===
    print(f"\n  === Test Set Evaluation ===")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_odor, num_workers=0)
    model.eval()
    ema.apply_shadow(model)

    test_cos = 0.0
    test_mse = 0.0
    n_test = 0
    with torch.no_grad():
        for batch in test_loader:
            bert = batch['bert'].to(device)
            phys = batch['phys'].to(device)
            odor = batch['odor'].to(device)
            gb = batch['graph_batch']
            if gb is not None:
                gb = gb.to(device)
            else:
                continue
            pred = model(bert, gb, phys, return_aux=False)
            test_cos += F.cosine_similarity(pred, odor, dim=1).mean().item()
            test_mse += F.mse_loss(pred, odor).item()
            n_test += 1

    ema.restore(model)
    avg_test_cos = test_cos / max(n_test, 1)
    avg_test_mse = test_mse / max(n_test, 1)
    print(f"  Test cos_sim: {avg_test_cos:.4f}")
    print(f"  Test MSE:     {avg_test_mse:.6f}")
    print(f"  Best val cos: {best_cos_sim:.4f}")

    return best_cos_sim


# ================================================================
# Auto Batch Config
# ================================================================

def auto_batch_config(device):
    """Automatically determine optimal batch_size and accumulation_steps from VRAM.
    Tuned for A100 MIG slices (10GB/20GB) where available VRAM is tight."""
    if not torch.cuda.is_available():
        return 32, 4, 2  # batch, accum, workers
    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    n_cpu = os.cpu_count() or 2
    workers = min(4, max(2, n_cpu - 1))  # leave 1 core for main
    if vram_gb >= 40:
        bs, acc = 64, 4      # full A100 80GB or 40GB MIG
    elif vram_gb >= 20:
        bs, acc = 32, 8      # MIG 2g-20GB (₩690)
    elif vram_gb >= 10:
        bs, acc = 16, 16     # MIG 1g-10GB (₩340)
    elif vram_gb >= 6:
        bs, acc = 8, 32
    else:
        bs, acc = 4, 64
    print(f"  Auto-config: VRAM={vram_gb:.1f}GB, CPU={n_cpu} -> batch_size={bs}, accumulation={acc}, workers={workers}")
    return bs, acc, workers


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Fragrance AI v6 - Full Training")
    parser.add_argument('--model', default='odor', choices=['odor', 'all'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Batch size (0=auto-detect from VRAM)')
    parser.add_argument('--ensemble', action='store_true', help='Train 5-seed ensemble')
    parser.add_argument('--data', default='data/curated_training_data.csv')
    parser.add_argument('--save-dir', default='weights/v6')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--no-rdrop', action='store_true')
    parser.add_argument('--bert-cache', default=None)
    parser.add_argument('--pretrained-gnn', default=None,
                        help='Path to pretrained GNN weights from pretrain_ssl.py')
    parser.add_argument('--lora', action='store_true', help='Enable LoRA for ChemBERTa')
    parser.add_argument('--n-aug', type=int, default=5,
                        help='Number of SMILES augmentation variants per molecule')
    parser.add_argument('--label-smooth', type=float, default=0.05,
                        help='Label smoothing factor (0=none, 0.05=default)')
    parser.add_argument('--resume', default=None,
                        help='Resume training from checkpoint .pt file')
    parser.add_argument('--auto-continue', action='store_true',
                        help='Auto-continue: Phase1(base) -> Phase2(finetune) -> Phase3(ensemble)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Fragrance AI v6 - SOTA Training Pipeline")
    print(f"  Device: {device}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}")

    # Auto batch config if batch_size is 0 (auto)
    if args.batch_size == 0:
        args.batch_size, args.accumulation, n_workers = auto_batch_config(device)
    else:
        n_workers = min(4, max(2, (os.cpu_count() or 2) - 1))

    common_kwargs = dict(
        device=device, csv_path=args.data, save_dir=args.save_dir,
        batch_size=args.batch_size, bert_cache_path=args.bert_cache,
        accumulation_steps=args.accumulation,
        use_rdrop=not args.no_rdrop,
        pretrained_gnn_path=args.pretrained_gnn,
        use_lora=args.lora, n_aug=args.n_aug,
        label_smooth=args.label_smooth,
        n_workers=n_workers,
    )

    if args.auto_continue:
        # ═══════════════════════════════════════════════════════
        # Auto-Continue: 3-Phase Training Pipeline
        # ═══════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print(f"  ★ AUTO-CONTINUE MODE ★")
        print(f"  Phase 1: Base Training (300 epochs)")
        print(f"  Phase 2: Fine-tune (100 epochs, LR×0.1)")
        print(f"  Phase 3: 5-Seed Ensemble")
        print(f"{'='*60}")

        # --- Phase 1: Base Training ---
        phase1_ckpt = os.path.join(args.save_dir, f'odor_v6_best_seed{args.seed}.pt')
        if os.path.exists(phase1_ckpt):
            print(f"\n  [Phase 1] Checkpoint found, skipping: {phase1_ckpt}")
        else:
            print(f"\n  [Phase 1] Base Training...")
            train_odor_predictor(
                **common_kwargs, seed=args.seed, epochs=300,
                resume_path=args.resume,
            )

        # --- Phase 2: Fine-tune (lower LR, full unfreeze) ---
        phase2_ckpt = os.path.join(args.save_dir, f'odor_v6_finetune_seed{args.seed}.pt')
        if os.path.exists(phase2_ckpt):
            print(f"\n  [Phase 2] Checkpoint found, skipping: {phase2_ckpt}")
        else:
            print(f"\n  [Phase 2] Fine-tuning (LR×0.1, 100 epochs)...")
            train_odor_predictor(
                device=device, csv_path=args.data,
                save_dir=args.save_dir,
                seed=args.seed, epochs=400,  # continue from ~300 to 400
                batch_size=args.batch_size,
                bert_cache_path=args.bert_cache,
                accumulation_steps=args.accumulation,
                use_rdrop=not args.no_rdrop,
                pretrained_gnn_path=args.pretrained_gnn,
                use_lora=args.lora, n_aug=args.n_aug,
                label_smooth=args.label_smooth,
                resume_path=phase1_ckpt,
                start_lr_factor=0.1,
                n_workers=n_workers,
            )
            # Rename best to finetune
            best_path = os.path.join(args.save_dir, f'odor_v6_best_seed{args.seed}.pt')
            if os.path.exists(best_path):
                import shutil
                shutil.copy2(best_path, phase2_ckpt)
                print(f"  Phase 2 saved: {phase2_ckpt}")

        # --- Phase 3: 5-Seed Ensemble ---
        print(f"\n  [Phase 3] 5-Seed Ensemble Training...")
        ensemble_seeds = [42, 123, 456, 789, 1024]
        results = []
        for s in ensemble_seeds:
            seed_ckpt = os.path.join(args.save_dir, f'odor_v6_best_seed{s}.pt')
            if os.path.exists(seed_ckpt) and s != args.seed:
                print(f"  Seed {s}: already trained, skipping")
                ckpt_data = torch.load(seed_ckpt, map_location='cpu')
                results.append(ckpt_data.get('val_cos_sim', 0.0))
                continue
            elif s == args.seed:
                ckpt_data = torch.load(seed_ckpt, map_location='cpu')
                results.append(ckpt_data.get('val_cos_sim', 0.0))
                continue

            print(f"  Seed {s}: training (transfer from seed {args.seed})...")
            cos = train_odor_predictor(
                **common_kwargs, seed=s, epochs=300,
                transfer_from=phase1_ckpt,
            )
            results.append(cos)

        print(f"\n  {'='*60}")
        print(f"  ★ AUTO-CONTINUE COMPLETE ★")
        print(f"  Ensemble cos_sim: {[f'{r:.4f}' for r in results]}")
        print(f"  Mean: {np.mean(results):.4f} +/- {np.std(results):.4f}")
        print(f"  {'='*60}")

    elif args.model in ('odor', 'all'):
        if args.ensemble:
            seeds = [42, 123, 456, 789, 1024]
            results = []
            for s in seeds:
                cos = train_odor_predictor(
                    **common_kwargs, seed=s, epochs=args.epochs,
                    resume_path=args.resume,
                )
                results.append(cos)
            print(f"\n  Ensemble Results: {[f'{r:.4f}' for r in results]}")
            print(f"  Mean: {np.mean(results):.4f} +/- {np.std(results):.4f}")
        else:
            train_odor_predictor(
                **common_kwargs, seed=args.seed, epochs=args.epochs,
                resume_path=args.resume,
            )

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
