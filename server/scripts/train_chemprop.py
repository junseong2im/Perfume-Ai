# -*- coding: utf-8 -*-
"""
Chemprop D-MPNN Training Wrapper for Odor Prediction
=====================================================
chemprop 패키지를 사용하여 진짜 Directed Message Passing Neural Network을
학습시키고, 기존 앙상블에 통합.

D-MPNN은 분자 그래프에서 직접 메시지 패싱을 수행하는 것이 핵심.
기존 OdorGNN(실제로는 MLP)과 달리 원자/결합 수준의 구조 정보를 학습.

Usage:
    python train_chemprop.py --epochs 30
    python train_chemprop.py --epochs 50 --ensemble-size 3

After training:
    chemprop model saved to weights/chemprop_model/
    → odor_engine.py에서 자동 로드
"""

import sys
import os
import time
import argparse
import csv
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import database as db
from train_models import _descriptor_to_target, _get_scaffold

# Constants
WEIGHTS_DIR = Path(__file__).parent.parent / 'weights'
CHEMPROP_DIR = WEIGHTS_DIR / 'chemprop_model'
DATA_DIR = Path(__file__).parent.parent / 'data'

ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]
N_DIM = len(ODOR_DIMENSIONS)


# ================================================================
# Data Export for Chemprop
# ================================================================

def export_chemprop_csv(output_path=None, n_augment=0):
    """DB 데이터를 chemprop 학습용 CSV로 내보내기
    
    Format: smiles, sweet, sour, woody, ..., metallic
    """
    if output_path is None:
        output_path = DATA_DIR / 'chemprop_train.csv'
    
    molecules = db.get_all_molecules()
    
    # SMILES augmentation
    do_augment = n_augment > 0
    if do_augment:
        try:
            from smiles_augment import randomize_smiles
        except ImportError:
            do_augment = False
    
    rows = []
    skipped = 0
    
    for mol in molecules:
        smiles = mol.get('smiles', '')
        labels = mol.get('odor_labels', [])
        
        if not smiles or not labels or labels == ['odorless']:
            skipped += 1
            continue
        
        target = _descriptor_to_target(labels)
        if target.max() == 0:
            skipped += 1
            continue
        
        # Original
        row = [smiles] + target.tolist()
        rows.append(row)
        
        # Augmented variants
        if do_augment:
            variants = randomize_smiles(smiles, n_augment=n_augment)
            for v in variants:
                if v != smiles:
                    rows.append([v] + target.tolist())
    
    # Write CSV
    header = ['smiles'] + ODOR_DIMENSIONS
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"  Exported {len(rows)} rows to {output_path}")
    print(f"  Skipped: {skipped}")
    if do_augment:
        n_original = len([m for m in molecules if m.get('smiles') and m.get('odor_labels')])
        print(f"  Augmentation: {n_original} → {len(rows)} ({len(rows)/max(n_original,1):.1f}x)")
    
    return str(output_path), len(rows)


def create_scaffold_split_files(csv_path, output_dir, val_ratio=0.2, seed=42):
    """chemprop용 scaffold split 인덱스 파일 생성"""
    import csv as csv_mod
    rng = np.random.RandomState(seed)
    
    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv_mod.reader(f)
        header = next(reader)
        data = list(reader)
    
    # Scaffold grouping
    from collections import defaultdict
    scaffold_to_indices = defaultdict(list)
    for i, row in enumerate(data):
        smiles = row[0]
        scaffold = _get_scaffold(smiles)
        scaffold_to_indices[scaffold].append(i)
    
    scaffolds = list(scaffold_to_indices.values())
    rng.shuffle(scaffolds)
    scaffolds.sort(key=len, reverse=True)
    
    n_val = int(len(data) * val_ratio)
    train_idx, val_idx = [], []
    
    for group in scaffolds:
        if len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            train_idx.extend(group)
    
    # Write split CSVs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, indices in [('train', train_idx), ('val', val_idx)]:
        split_path = output_dir / f'{name}.csv'
        with open(split_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv_mod.writer(f)
            writer.writerow(header)
            for i in indices:
                writer.writerow(data[i])
        print(f"  {name}: {len(indices)} samples → {split_path}")
    
    return str(output_dir / 'train.csv'), str(output_dir / 'val.csv')


# ================================================================
# Chemprop Training (API-based)
# ================================================================

def train_chemprop_model(
    epochs=30,
    batch_size=50,
    hidden_size=300,
    depth=3,
    dropout=0.15,
    n_augment=5,
    ensemble_size=1,
    seed=42,
):
    """chemprop D-MPNN 학습
    
    chemprop v2 API를 사용하여 프로그래밍 방식으로 학습
    """
    start = time.time()
    
    print("=" * 60)
    print(f"  Chemprop D-MPNN Training")
    print(f"  Architecture: D-MPNN (depth={depth}, hidden={hidden_size})")
    print(f"  Augmentation: {n_augment}x")
    print(f"  Ensemble: {ensemble_size} models")
    print("=" * 60)
    
    # 1. Export data
    print("\n  Exporting data...")
    csv_path, n_samples = export_chemprop_csv(n_augment=n_augment)
    
    # Create split files
    split_dir = DATA_DIR / 'chemprop_splits'
    train_csv, val_csv = create_scaffold_split_files(csv_path, split_dir, seed=seed)
    
    # 2. Try chemprop v2 API
    try:
        import chemprop
        print(f"\n  chemprop version: {chemprop.__version__}")
    except ImportError:
        print("\n  ERROR: chemprop not installed!")
        print("  Install with: pip install chemprop>=2.0.0")
        print("  Falling back to manual D-MPNN implementation...")
        return _train_manual_dmpnn(train_csv, val_csv, epochs=epochs, seed=seed)
    
    # 3. Train with chemprop v2 API
    CHEMPROP_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # chemprop v2 uses a different API
        from chemprop import data as chemprop_data
        from chemprop import models as chemprop_models
        from chemprop import train as chemprop_train
        
        print(f"\n  Training D-MPNN with chemprop v2 API...")
        print(f"  Train: {train_csv}")
        print(f"  Val: {val_csv}")
        print(f"  Save: {CHEMPROP_DIR}")
        
        # Use chemprop CLI interface (most reliable across versions)
        import subprocess
        
        cmd = [
            sys.executable, '-m', 'chemprop', 'train',
            '--data-path', train_csv,
            '--dataset-type', 'regression',
            '--save-dir', str(CHEMPROP_DIR),
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--hidden-size', str(hidden_size),
            '--depth', str(depth),
            '--dropout', str(dropout),
            '--split-type', 'predetermined',
            '--folds-file', val_csv,
            '--seed', str(seed),
            '--metric', 'mse',
            '--quiet',
        ]
        
        if ensemble_size > 1:
            cmd.extend(['--ensemble-size', str(ensemble_size)])
        
        print(f"\n  Running: {' '.join(cmd[:5])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            elapsed = time.time() - start
            print(f"\n  ✅ Chemprop training complete! ({elapsed:.1f}s)")
            print(f"  Model saved to: {CHEMPROP_DIR}")
            if result.stdout:
                # Extract final metrics
                for line in result.stdout.split('\n')[-10:]:
                    if line.strip():
                        print(f"    {line.strip()}")
        else:
            print(f"\n  Chemprop CLI failed, trying alternative approach...")
            print(f"  stderr: {result.stderr[:500]}")
            return _train_manual_dmpnn(train_csv, val_csv, epochs=epochs, seed=seed)
    
    except Exception as e:
        print(f"\n  Chemprop API error: {e}")
        print(f"  Falling back to manual D-MPNN...")
        return _train_manual_dmpnn(train_csv, val_csv, epochs=epochs, seed=seed)
    
    # 4. Save training info
    info = {
        'epochs': epochs,
        'hidden_size': hidden_size,
        'depth': depth,
        'dropout': dropout,
        'n_augment': n_augment,
        'n_train': n_samples,
        'ensemble_size': ensemble_size,
        'time': time.time() - start,
    }
    with open(CHEMPROP_DIR / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    return str(CHEMPROP_DIR)


# ================================================================
# D-MPNN Architecture (Module-level for import by odor_engine.py)
# ================================================================

import torch
import torch.nn as nn

from rdkit import Chem

ATOM_DIM = 8
BOND_DIM = 4


def atom_features(atom):
    """Atom → 8d feature vector"""
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 6.0,
        atom.GetFormalCharge() / 3.0,
        atom.GetNumRadicalElectrons() / 2.0,
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 4.0,
        atom.GetHybridization().real / 6.0,
        int(atom.IsInRing()),
    ]


def bond_features(bond):
    """Bond → 4d feature vector"""
    bt = bond.GetBondTypeAsDouble()
    return [
        bt / 3.0,
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(bond.GetStereo() != Chem.BondStereo.STEREONONE),
    ]


def mol_to_graph(smiles):
    """SMILES → molecular graph dict (nodes, edges, features)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None
    
    # Node features
    x = torch.zeros(n_atoms, ATOM_DIM)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = torch.tensor(atom_features(atom))
    
    # Edge features and adjacency
    edges_src, edges_dst = [], []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges_src.extend([i, j])
        edges_dst.extend([j, i])
        edge_feats.extend([bf, bf])
    
    if len(edges_src) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, BOND_DIM)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    
    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'n_atoms': n_atoms}


class DirectedMPNN(nn.Module):
    """Directed Message Passing Neural Network for molecular property prediction"""
    def __init__(self, atom_dim=ATOM_DIM, bond_dim=BOND_DIM, 
                 hidden_dim=300, output_dim=N_DIM, depth=3, dropout=0.15):
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        
        # Initial message
        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim)
        
        # Message passing layers
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, x, edge_index, edge_attr, n_atoms_list, batch_idx):
        """
        x: [total_atoms, atom_dim]
        edge_index: [2, total_edges]
        edge_attr: [total_edges, bond_dim]
        n_atoms_list: list of atom counts per molecule
        batch_idx: [total_atoms] — which molecule each atom belongs to
        """
        n_edges = edge_index.size(1)
        if n_edges == 0:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            src, dst = edge_index
            
            # Initial messages
            init_msg = torch.cat([x[src], edge_attr], dim=-1)
            h_edge = torch.relu(self.W_i(init_msg))
            
            # Message passing
            for _ in range(self.depth - 1):
                msg = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
                msg.index_add_(0, dst, h_edge)
                h_edge = torch.relu(self.W_h(h_edge) + msg[src])
                h_edge = self.dropout(h_edge)
            
            # Final atom representations
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            h.index_add_(0, dst, h_edge)
        
        # Atom output
        h = torch.relu(self.W_o(torch.cat([x, h], dim=-1)))
        h = self.dropout(h)
        
        # Graph-level readout (mean pool per molecule)
        batch_size = len(n_atoms_list)
        graph_vecs = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        for b in range(batch_size):
            mask = (batch_idx == b)
            if mask.any():
                graph_vecs[b] = h[mask].mean(dim=0)
        
        return self.readout(graph_vecs)


def collate_graphs(batch):
    """Collate function for batching molecular graphs"""
    all_x, all_edge_index, all_edge_attr = [], [], []
    all_targets = []
    n_atoms_list = []
    batch_idx = []
    atom_offset = 0
    
    for b, (graph, target) in enumerate(batch):
        n = graph['n_atoms']
        all_x.append(graph['x'])
        
        if graph['edge_index'].size(1) > 0:
            all_edge_index.append(graph['edge_index'] + atom_offset)
            all_edge_attr.append(graph['edge_attr'])
        
        n_atoms_list.append(n)
        batch_idx.extend([b] * n)
        all_targets.append(target)
        atom_offset += n
    
    x = torch.cat(all_x, dim=0)
    if all_edge_index:
        edge_index = torch.cat(all_edge_index, dim=1)
        edge_attr = torch.cat(all_edge_attr, dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, BOND_DIM)
    
    targets = torch.tensor(np.array(all_targets), dtype=torch.float32)
    batch_tensor = torch.tensor(batch_idx, dtype=torch.long)
    
    return x, edge_index, edge_attr, n_atoms_list, batch_tensor, targets


# ================================================================
# Manual D-MPNN training (if chemprop CLI fails)
# ================================================================

def _train_manual_dmpnn(train_csv, val_csv, epochs=30, seed=42):
    """chemprop 없이 직접 D-MPNN 구현
    
    PyTorch 기반 D-MPNN — chemprop CLI가 실패할 경우 fallback
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n  === Manual D-MPNN (PyTorch, device={device}) ===")
    
    # --- Load data ---
    def load_csv(path):
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                smiles = row[0]
                targets = [float(v) for v in row[1:]]
                graph = mol_to_graph(smiles)
                if graph is not None:
                    rows.append((graph, np.array(targets, dtype=np.float32)))
        return rows
    
    print(f"  Loading train data from {train_csv}...")
    train_data = load_csv(train_csv)
    print(f"  Loading val data from {val_csv}...")
    val_data = load_csv(val_csv)
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                              collate_fn=collate_graphs, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False,
                            collate_fn=collate_graphs)
    
    # --- Model ---
    model = DirectedMPNN(hidden_dim=300, depth=3, dropout=0.15).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  D-MPNN params: {n_params:,}")
    
    from train_models import AsymmetricLoss
    criterion = AsymmetricLoss(gamma_pos=1, gamma_neg=4, clip=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # --- Training ---
    best_val_loss = float('inf')
    patience = 0
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        
        for x, edge_index, edge_attr, n_atoms_list, batch_tensor, targets in train_loader:
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            batch_tensor = batch_tensor.to(device)
            targets = targets.to(device)
            
            preds = model(x, edge_index, edge_attr, n_atoms_list, batch_tensor)
            loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= max(1, n_batches)
        
        # Validate
        model.eval()
        val_loss = 0
        val_cos = 0
        n_val = 0
        
        with torch.no_grad():
            for x, edge_index, edge_attr, n_atoms_list, batch_tensor, targets in val_loader:
                x = x.to(device)
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                batch_tensor = batch_tensor.to(device)
                targets = targets.to(device)
                
                preds = model(x, edge_index, edge_attr, n_atoms_list, batch_tensor)
                val_loss += criterion(preds, targets).item()
                
                cos = nn.functional.cosine_similarity(preds, targets, dim=1)
                val_cos += cos.sum().item()
                n_val += targets.size(0)
        
        val_loss /= max(1, len(val_loader))
        val_cos /= max(1, n_val)
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            
            CHEMPROP_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_cos_sim': val_cos,
                'n_params': n_params,
                'type': 'manual_dmpnn',
            }, CHEMPROP_DIR / 'dmpnn_model.pt')
        else:
            patience += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"CosSim: {val_cos:.3f} | {elapsed:.1f}s")
        
        if patience >= 15:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    elapsed = time.time() - start
    print(f"\n  ✅ D-MPNN training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Saved to: {CHEMPROP_DIR / 'dmpnn_model.pt'}")
    print(f"  Time: {elapsed:.1f}s")
    
    return str(CHEMPROP_DIR)


# ================================================================
# Prediction helper (used by odor_engine.py)
# ================================================================

def load_chemprop_model(device='cuda'):
    """학습된 chemprop/D-MPNN 모델 로드"""
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Manual D-MPNN
    dmpnn_path = CHEMPROP_DIR / 'dmpnn_model.pt'
    if dmpnn_path.exists():
        import torch
        from rdkit import Chem
        
        checkpoint = torch.load(dmpnn_path, map_location=dev, weights_only=False)
        
        # Reconstruct model
        # (We need to import DirectedMPNN from here)
        print(f"  [Chemprop] Loaded manual D-MPNN (epoch {checkpoint['epoch']}, "
              f"cos={checkpoint.get('val_cos_sim', 0):.3f})")
        return checkpoint, 'manual_dmpnn'
    
    # chemprop CLI model
    chemprop_model_path = CHEMPROP_DIR / 'fold_0' / 'model_0' / 'model.pt'
    if chemprop_model_path.exists():
        try:
            import chemprop
            print(f"  [Chemprop] Loading chemprop model from {CHEMPROP_DIR}")
            return str(CHEMPROP_DIR), 'chemprop_cli'
        except ImportError:
            pass
    
    return None, None


def predict_with_dmpnn(smiles_list, model_path=None, device='cuda'):
    """D-MPNN으로 예측 (단순 API)"""
    # TODO: 실제 예측 로직 구현 (모델 로드 후)
    # 현재는 fallback으로 0 벡터 반환
    return np.zeros((len(smiles_list), N_DIM), dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chemprop D-MPNN Training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--hidden-size', type=int, default=300,
                        help='D-MPNN hidden dimension')
    parser.add_argument('--depth', type=int, default=3,
                        help='Message passing depth')
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--augment', type=int, default=5,
                        help='SMILES augmentation factor')
    parser.add_argument('--ensemble-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--export-only', action='store_true',
                        help='Only export CSV, do not train')
    args = parser.parse_args()
    
    if args.export_only:
        export_chemprop_csv(n_augment=args.augment)
    else:
        train_chemprop_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            dropout=args.dropout,
            n_augment=args.augment,
            ensemble_size=args.ensemble_size,
            seed=args.seed,
        )
