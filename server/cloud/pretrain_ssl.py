"""pretrain_ssl.py - Self-Supervised Pretraining for GNN
========================================================
Pretrain the GNN backbone on unlabeled molecules from PubChem/ZINC
using 3 pretext tasks:
  1. Masked Atom Prediction (MAE): mask 15% atoms, predict features
  2. Context Prediction: predict if subgraph belongs to molecule
  3. Contrastive Learning: augmented views of same molecule should be close

This addresses the #1 weakness: data scarcity.
Pretrained weights are loaded before fine-tuning on labeled data.

Usage:
    python pretrain_ssl.py --n-molecules 100000 --epochs 100 --device cuda
    python pretrain_ssl.py --smiles-file pubchem_smiles.txt --epochs 50
"""

import os
import sys
import argparse
import random
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from models.odor_predictor_v6 import (
    smiles_to_graph_v6, ATOM_FEATURES_DIM, BOND_FEATURES_DIM,
    IndustryGNNPath, LearnableRBF
)

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GlobalAttention
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ================================================================
# Common SMILES for pretraining (if no file provided)
# ================================================================

SEED_SMILES = [
    # Essential oils / aroma compounds
    'CC(=O)OC1CC(=CCC1C(C)=C)C', 'CC1=CCC(CC1)C(C)=C',
    'CC(C)=CCCC(C)=CC=O', 'CC(=O)C1CCC(CC1)C(C)C',
    'CC1=CCC(=CC1)C(C)C', 'OC(=O)/C=C/c1ccccc1',
    'CC(C)=CCCC(C)=CCO', 'CC12CCC(CC1)CC2',
    'c1ccc2c(c1)ccc1ccccc12', 'CC(=O)Oc1ccccc1C(O)=O',
    # Drug-like molecules
    'CC(=O)Nc1ccc(O)cc1', 'OC(=O)c1ccccc1O',
    'CC(O)CC(=O)O', 'c1ccc(cc1)C(O)c1ccccc1',
    'CC(C)Cc1ccc(cc1)C(C)C(O)=O', 'O=C1CCCN1',
    # Fragrances
    'CCCCCCCC=O', 'CCCCCCCCCC=O', 'CCCCCCCCCCC=O',
    'CC/C=C/CC(=O)OCC', 'CCOC(=O)c1ccccc1',
    'COC(=O)c1ccc(OC)cc1', 'CC(=O)OCCc1ccccc1',
    'OC/C=C/c1ccccc1', 'O=Cc1ccc(OC)cc1',
    'CC(=O)c1ccc2OCOc2c1', 'O=Cc1ccc(O)c(OC)c1',
]


def generate_random_smiles(n=10000, seed=42):
    """Generate diverse SMILES from RDKit random molecule generation."""
    if not HAS_RDKIT:
        return SEED_SMILES * (n // len(SEED_SMILES) + 1)

    random.seed(seed)
    smiles_set = set(SEED_SMILES)

    # Method 1: Enumerate from seeds via randomized SMILES
    for smi in SEED_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for _ in range(min(50, n // len(SEED_SMILES))):
            try:
                rsmi = Chem.MolToSmiles(mol, doRandom=True)
                smiles_set.add(rsmi)
            except:
                pass

    # Method 2: Fragment combination
    fragments = [
        'C', 'CC', 'CCC', 'CCCC', 'C=C', 'C#C',
        'c1ccccc1', 'C1CCCCC1', 'C1CCCC1', 'C1CCC1',
        'C=O', 'C(=O)O', 'CO', 'CN', 'CF', 'CCl', 'CBr',
        'OC', 'NC', 'SC', 'c1ccncc1', 'c1ccoc1', 'c1ccsc1',
        'C(=O)N', 'C(=O)OC', 'OC(=O)',
    ]
    while len(smiles_set) < n:
        # Random chain length
        n_frag = random.randint(2, 6)
        parts = [random.choice(fragments) for _ in range(n_frag)]
        smi = ''.join(parts)
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canon = Chem.MolToSmiles(mol)
            if 5 <= mol.GetNumAtoms() <= 50:
                smiles_set.add(canon)

    return list(smiles_set)[:n]


def download_pubchem_smiles(n=50000, cache_path='data/pubchem_smiles.txt'):
    """Download random SMILES from PubChem."""
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            smiles = [l.strip() for l in f if l.strip()]
        print(f"  Loaded {len(smiles)} SMILES from cache")
        return smiles[:n]

    print(f"  Downloading {n} SMILES from PubChem...")
    try:
        import requests
        smiles = set()
        batch = 10000
        for start in range(1, n + 1, batch):
            url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                   f"cid/{start}-{start+batch-1}/property/IsomericSMILES/CSV")
            try:
                r = requests.get(url, timeout=30)
                for line in r.text.strip().split('\n')[1:]:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        smi = parts[1].strip().strip('"')
                        if smi and len(smi) < 200:
                            smiles.add(smi)
                print(f"    {len(smiles)} / {n} ...")
            except:
                pass
            if len(smiles) >= n:
                break

        smiles = list(smiles)[:n]
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            f.write('\n'.join(smiles))
        print(f"  Cached {len(smiles)} SMILES")
        return smiles
    except Exception as e:
        print(f"  Download failed: {e}, using generated SMILES")
        return generate_random_smiles(n)


# ================================================================
# Augmentation for Contrastive Learning
# ================================================================

def augment_graph(data, mask_ratio=0.15, drop_edge_ratio=0.1):
    """Create augmented view of molecular graph (preserves pos)."""
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone()
    pos = data.pos.clone() if hasattr(data, 'pos') and data.pos is not None else torch.zeros(x.size(0), 3)

    N = x.size(0)
    E = edge_index.size(1)

    # 1. Mask atoms
    n_mask = max(1, int(N * mask_ratio))
    mask_idx = torch.randperm(N)[:n_mask]
    x[mask_idx] = torch.randn_like(x[mask_idx]) * 0.1

    # 2. Drop edges
    n_drop = max(0, int(E * drop_edge_ratio))
    if n_drop > 0 and E > n_drop:
        keep = torch.randperm(E)[:E - n_drop]
        edge_index = edge_index[:, keep]
        edge_attr = edge_attr[keep]

    # 3. Coordinate noise for 3D denoising
    pos_noisy = pos + torch.randn_like(pos) * 0.1

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos_noisy)


# ================================================================
# SSL Dataset
# ================================================================

class SSLMoleculeDataset(Dataset):
    """Dataset for self-supervised molecular pretraining."""
    def __init__(self, smiles_list):
        self.graphs = []
        self.smiles = []
        failed = 0
        for smi in smiles_list:
            try:
                g = smiles_to_graph_v6(smi, compute_3d=True)
                if g is not None and g.x.size(0) >= 2:
                    self.graphs.append(g)
                    self.smiles.append(smi)
            except:
                failed += 1
        print(f"  SSL Dataset: {len(self.graphs)} graphs ({failed} failed)")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        # Create 2 augmented views for contrastive learning
        view1 = augment_graph(g, mask_ratio=0.15, drop_edge_ratio=0.1)
        view2 = augment_graph(g, mask_ratio=0.20, drop_edge_ratio=0.15)
        return {
            'original': g,
            'view1': view1,
            'view2': view2,
        }


def collate_ssl(batch):
    """Collate SSL batch."""
    originals = Batch.from_data_list([b['original'] for b in batch])
    view1s = Batch.from_data_list([b['view1'] for b in batch])
    view2s = Batch.from_data_list([b['view2'] for b in batch])
    return originals, view1s, view2s


# ================================================================
# SSL Pretraining Model
# ================================================================

class SSLPretrainer(nn.Module):
    """Self-supervised pretraining wrapper for Industry GNN backbone.
    Includes 4 pretext tasks:
      1. Masked Atom Prediction (MAE)
      2. Contrastive Learning (NT-Xent)
      3. Graph-level Property Prediction
      4. 3D Coordinate Denoising (Uni-Mol style) <- NEW
    """
    def __init__(self, hidden=256, n_layers=4, n_rbf=64):
        super().__init__()
        # Shared GNN backbone (matches OdorPredictorV6 path_b)
        self.gnn = IndustryGNNPath(
            in_dim=ATOM_FEATURES_DIM, hidden=hidden, out=hidden,
            gat_heads=4, n_rbf=n_rbf, n_blocks=n_layers,
            drop_path_rate=0.1
        )

        # Task 1: Masked Atom Prediction head
        self.mae_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, ATOM_FEATURES_DIM),
        )

        # Task 2: Contrastive projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 128),
        )

        # Task 3: Graph-level property prediction
        self.prop_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 3),
        )

        # Task 4: 3D Coordinate Denoising head (Uni-Mol style)
        self.denoise_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),  # predict noise delta
        )

    def forward(self, batch, return_graph_emb=False):
        """Forward pass: returns graph-level embedding."""
        pos = batch.pos if hasattr(batch, 'pos') else None
        g_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr,
                         batch.batch, pos)
        if return_graph_emb:
            return g_emb
        return g_emb

    def compute_mae_loss(self, batch, mask_ratio=0.15):
        """Masked Atom Prediction loss."""
        x = batch.x.clone()
        N = x.size(0)
        n_mask = max(1, int(N * mask_ratio))
        mask_idx = torch.randperm(N, device=x.device)[:n_mask]

        # Save original features
        original_features = x[mask_idx].clone()

        # Mask atoms
        x[mask_idx] = 0.0
        batch_masked = batch.clone()
        batch_masked.x = x

        # Forward
        pos = batch_masked.pos if hasattr(batch_masked, 'pos') else None
        g_emb = self.gnn(batch_masked.x, batch_masked.edge_index,
                         batch_masked.edge_attr, batch_masked.batch, pos)

        # Predict masked atom features from graph embedding
        # (simplified: predict from graph level, broadcast)
        pred = self.mae_head(g_emb)  # [B, atom_dim]

        # Rule #1 fix: scatter_add로 벡터화 (Python for-loop O(N²) 제거)
        B = g_emb.size(0)
        target_mean = torch.zeros(B, ATOM_FEATURES_DIM, device=x.device)
        counts = torch.zeros(B, 1, device=x.device)
        
        # 각 마스킹된 원자가 어느 그래프에 속하는지 인덱스
        batch_indices = batch.batch[mask_idx]  # [n_mask]
        
        # scatter_add: 같은 그래프의 원자 특징을 한번에 합산
        target_mean.scatter_add_(
            0,
            batch_indices.unsqueeze(1).expand(-1, ATOM_FEATURES_DIM),
            original_features
        )
        counts.scatter_add_(
            0,
            batch_indices.unsqueeze(1),
            torch.ones(len(mask_idx), 1, device=x.device)
        )
        target_mean = target_mean / counts.clamp(min=1)

        return F.mse_loss(pred, target_mean)

    def compute_contrastive_loss(self, view1_batch, view2_batch, temperature=0.07):
        """NT-Xent contrastive loss between two augmented views."""
        pos1 = view1_batch.pos if hasattr(view1_batch, 'pos') else None
        pos2 = view2_batch.pos if hasattr(view2_batch, 'pos') else None
        z1 = F.normalize(self.projector(
            self.gnn(view1_batch.x, view1_batch.edge_index,
                     view1_batch.edge_attr, view1_batch.batch, pos1)
        ), dim=1)
        z2 = F.normalize(self.projector(
            self.gnn(view2_batch.x, view2_batch.edge_index,
                     view2_batch.edge_attr, view2_batch.batch, pos2)
        ), dim=1)

        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2B, d]
        sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)

        # Mask self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        return F.cross_entropy(sim, labels)

    def compute_property_loss(self, batch):
        """Self-supervised property prediction (no labels needed)."""
        pos = batch.pos if hasattr(batch, 'pos') else None
        g_emb = self.gnn(batch.x, batch.edge_index,
                         batch.edge_attr, batch.batch, pos)
        pred = self.prop_head(g_emb)

        # Compute ground-truth properties from graph structure
        B = batch.batch.max().item() + 1
        targets = []
        for b in range(B):
            node_mask = batch.batch == b
            n_atoms = node_mask.sum().float()
            # Estimate n_rings from aromatic atoms (feature index ~4)
            if batch.x.size(1) > 4:
                n_aromatic = batch.x[node_mask, 4].sum()
            else:
                n_aromatic = torch.tensor(0.0, device=batch.x.device)
            targets.append(torch.stack([
                n_atoms / 50.0,  # normalized
                n_aromatic / 10.0,
                torch.tensor(0.5, device=batch.x.device),  # placeholder
            ]))
        targets = torch.stack(targets)
        return F.mse_loss(pred, targets)


# ================================================================
# Pretraining Loop
# ================================================================

def pretrain(device, smiles_list, save_path, epochs=100, batch_size=128,
             lr=1e-3, hidden=256, n_layers=4):
    print(f"\n{'='*60}")
    print(f"  Self-Supervised GNN Pretraining")
    print(f"  molecules={len(smiles_list)}, epochs={epochs}, device={device}")
    print(f"  hidden={hidden}, layers={n_layers}")
    print(f"{'='*60}")

    dataset = SSLMoleculeDataset(smiles_list)
    if len(dataset) < 10:
        print("  [ERROR] Not enough valid molecules!")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_ssl, num_workers=0, drop_last=True)

    model = SSLPretrainer(hidden=hidden, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SSL Model: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batch = 0

        for orig, v1, v2 in loader:
            orig = orig.to(device)
            v1 = v1.to(device)
            v2 = v2.to(device)

            # 3 SSL losses
            L_mae = model.compute_mae_loss(orig)
            L_contra = model.compute_contrastive_loss(v1, v2)
            L_prop = model.compute_property_loss(orig)

            loss = 1.0 * L_mae + 1.0 * L_contra + 0.5 * L_prop

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batch, 1)

        if epoch % 10 == 0 or epoch <= 3:
            print(f"  E{epoch:>3d}/{epochs} | loss={avg_loss:.4f} "
                  f"(mae={L_mae.item():.3f} contra={L_contra.item():.3f} "
                  f"prop={L_prop.item():.3f}) | lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save({
                'gnn_state_dict': model.gnn.state_dict(),
                'full_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'hidden': hidden,
                'n_layers': n_layers,
            }, save_path)

    print(f"\n  Pretraining complete! Best loss: {best_loss:.4f}")
    print(f"  GNN weights saved to: {save_path}")
    return save_path


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="SSL Pretraining for GNN")
    parser.add_argument('--n-molecules', type=int, default=50000)
    parser.add_argument('--smiles-file', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-path', default='weights/v6/gnn_pretrained.pt')
    parser.add_argument('--source', choices=['pubchem', 'generate'],
                        default='generate', help='SMILES source')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load or generate SMILES
    if args.smiles_file and os.path.exists(args.smiles_file):
        with open(args.smiles_file, 'r') as f:
            smiles = [l.strip() for l in f if l.strip()]
        smiles = smiles[:args.n_molecules]
    elif args.source == 'pubchem':
        smiles = download_pubchem_smiles(args.n_molecules)
    else:
        smiles = generate_random_smiles(args.n_molecules)

    print(f"  Using {len(smiles)} SMILES for pretraining")
    pretrain(device, smiles, args.save_path, epochs=args.epochs,
             batch_size=args.batch_size, lr=args.lr,
             hidden=args.hidden, n_layers=args.n_layers)


if __name__ == '__main__':
    main()
