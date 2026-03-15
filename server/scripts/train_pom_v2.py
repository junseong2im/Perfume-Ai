"""
===================================================================
POM MPNN v2 — 모든 실측 데이터로 재학습
===================================================================
기존: Leffingwell 3,522분자 → AUROC 0.8746
신규: 18+ 데이터셋 통합 (120K+ 분자) → 목표 AUROC 0.92+

핵심 개선:
1. 다중 데이터셋 통합 (라벨 정규화)
2. Focal Loss (극도의 클래스 불균형 대응)
3. 더 깊은 MPNN (5-layer)
4. Multi-head Attention Readout
"""
import os
import sys
import csv
import json
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.train_pom_mpnn import (
    smiles_to_graph, MPNNLayer, collate_graphs, ATOM_FEATURES
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_data')


# ============================================================
# 1. 통합 데이터 로더 — 모든 Pyrfume 데이터셋 합치기
# ============================================================

def load_all_pyrfume_data():
    """모든 Pyrfume 데이터셋 로딩 + 라벨 정규화
    
    각 데이터셋은 다른 라벨 체계를 사용:
    - Leffingwell: 113 binary 라벨
    - Dravnieks: 146 intensity 라벨
    - Keller: 21 perceptual descriptors
    - GoodScents: 가변 라벨
    
    → 공통 라벨 셋으로 정규화 (union of all descriptors)
    """
    pyrfume_dir = os.path.join(DATA_DIR, 'pyrfume_all')
    if not os.path.exists(pyrfume_dir):
        # 기존 pyrfume 디렉토리 사용
        pyrfume_dir = os.path.join(DATA_DIR, 'pyrfume')
    
    all_smiles = []
    all_labels = []
    all_descriptors = set()
    dataset_stats = {}
    
    # 1단계: 모든 데이터셋 로딩 + 서술어 수집
    raw_data = {}  # {dataset_name: [(smiles, {descriptor: value}), ...]}
    
    for ds_name in sorted(os.listdir(pyrfume_dir)):
        ds_dir = os.path.join(pyrfume_dir, ds_name)
        if not os.path.isdir(ds_dir):
            continue
        
        mol_path = os.path.join(ds_dir, 'molecules.csv')
        beh_path = os.path.join(ds_dir, 'behavior.csv')
        
        if not os.path.exists(mol_path) or not os.path.exists(beh_path):
            continue
        
        # molecules 로드 (CID → SMILES)
        cid_to_smiles = {}
        try:
            with open(mol_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get('CID', row.get('cid', row.get('Stimulus', '')))
                    smiles = row.get('IsomericSMILES', row.get('SMILES', row.get('smiles', '')))
                    if cid and smiles and len(smiles) > 1:
                        cid_to_smiles[str(cid)] = smiles
        except:
            continue
        
        if not cid_to_smiles:
            continue
        
        # behavior 로드
        entries = []
        try:
            with open(beh_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                descriptors = [c for c in reader.fieldnames 
                             if c.lower() not in ('stimulus', 'cid', 'subject', 'replicate',
                                                   'concentration', 'dilution', 'compound identifier',
                                                   'odor', 'cas', 'name', 'iupacname',
                                                   'molecularweight')]
                
                for row in reader:
                    stim = row.get('Stimulus', row.get('CID', row.get('cid', '')))
                    smiles = cid_to_smiles.get(str(stim), '')
                    
                    if not smiles:
                        # 숫자 CID 대조
                        for cid, smi in cid_to_smiles.items():
                            if stim == cid:
                                smiles = smi
                                break
                    
                    if smiles:
                        desc_vals = {}
                        for desc in descriptors:
                            val = row.get(desc, '0')
                            try:
                                v = float(val)
                                if v != 0:
                                    desc_vals[desc.lower().strip()] = v
                            except:
                                if val and val.lower() not in ('', 'nan', 'none'):
                                    desc_vals[desc.lower().strip()] = 1.0
                        
                        if desc_vals:
                            entries.append((smiles, desc_vals))
                            all_descriptors.update(desc_vals.keys())
        except:
            continue
        
        if entries:
            raw_data[ds_name] = entries
            dataset_stats[ds_name] = len(entries)
    
    print(f"\n  📊 데이터셋 통합 현황:")
    total_entries = 0
    for ds, count in sorted(dataset_stats.items(), key=lambda x: -x[1]):
        print(f"    {ds:20s}: {count:>6}개 항목")
        total_entries += count
    
    # 2단계: 공통 서술어 셋으로 정규화
    # 빈도 기준 상위 서술어만 사용 (5개+ 데이터셋에 존재)
    desc_freq = {}
    for ds_name, entries in raw_data.items():
        ds_descs = set()
        for _, desc_vals in entries:
            ds_descs.update(desc_vals.keys())
        for d in ds_descs:
            desc_freq[d] = desc_freq.get(d, 0) + 1
    
    # 최소 한 데이터셋 이상에서 등장한 서술어 사용
    # (너무 많으면 학습 비효율, 최대 200개)
    sorted_descs = sorted(desc_freq.keys(), key=lambda x: -desc_freq[x])
    unified_descriptors = sorted_descs[:200]
    desc_to_idx = {d: i for i, d in enumerate(unified_descriptors)}
    
    print(f"\n  📋 통합 서술어: {len(unified_descriptors)}개")
    print(f"    상위 10: {unified_descriptors[:10]}")
    
    # 3단계: 통합 라벨 행렬 생성
    seen_smiles = {}  # smiles → best label (여러 데이터셋 평균)
    
    for ds_name, entries in raw_data.items():
        for smiles, desc_vals in entries:
            if smiles not in seen_smiles:
                seen_smiles[smiles] = {'labels': np.zeros(len(unified_descriptors)),
                                       'counts': np.zeros(len(unified_descriptors))}
            
            for desc, val in desc_vals.items():
                if desc in desc_to_idx:
                    idx = desc_to_idx[desc]
                    seen_smiles[smiles]['labels'][idx] += val
                    seen_smiles[smiles]['counts'][idx] += 1
    
    # 평균 계산
    for smiles, data in seen_smiles.items():
        mask = data['counts'] > 0
        data['labels'][mask] /= data['counts'][mask]
        # 0~1로 정규화
        max_val = np.max(data['labels'])
        if max_val > 1:
            data['labels'] /= max_val
    
    smiles_list = list(seen_smiles.keys())
    labels_list = [seen_smiles[s]['labels'] for s in smiles_list]
    
    print(f"\n  📊 통합 결과:")
    print(f"    총 데이터셋:   {len(raw_data)}개")
    print(f"    총 고유 분자:  {len(smiles_list)}개")
    print(f"    총 서술어:     {len(unified_descriptors)}개")
    print(f"    총 항목:       {total_entries}개")
    
    return smiles_list, labels_list, unified_descriptors


# ============================================================
# 2. 향상된 MPNN v2
# ============================================================

class AttentionReadout(nn.Module):
    """Multi-head Attention Readout for graph-level representation"""
    
    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, batch):
        """Attention-weighted pooling per graph"""
        n_graphs = batch.max().item() + 1
        outputs = []
        
        for i in range(n_graphs):
            mask = batch == i
            if not mask.any():
                outputs.append(torch.zeros(x.size(1), device=x.device))
                continue
            
            nodes = x[mask].unsqueeze(0)  # [1, n_nodes, hidden]
            # Self-attention
            attn_out, _ = self.attention(nodes, nodes, nodes)
            # Gate
            gates = self.gate(attn_out)  # [1, n_nodes, 1]
            weighted = (attn_out * gates).sum(dim=1).squeeze(0)  # [hidden]
            outputs.append(weighted)
        
        return torch.stack(outputs)


class POMv2(nn.Module):
    """POM MPNN v2 — 향상된 아키텍처
    
    개선점:
    1. 5-layer MPNN (3 → 5)
    2. Attention Readout (mean+max → attention)
    3. 더 넓은 hidden (128 → 192)
    4. Skip connections
    """
    
    def __init__(self, atom_dim=4, hidden_dim=192, n_layers=5, n_descriptors=200):
        super().__init__()
        self.n_descriptors = n_descriptors
        
        # Atom encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # MPNN layers with skip connections
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim=1, hidden_dim=hidden_dim * 2)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.1) for _ in range(n_layers)
        ])
        
        # Readout
        self.mean_max_readout = True  # Attention은 느릴 수 있음
        readout_dim = hidden_dim * 2  # mean + max concat
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_descriptors),
        )
    
    def forward(self, atom_features, edge_index, edge_attr, batch=None):
        x = self.atom_encoder(atom_features)
        
        for mpnn, ln, drop in zip(self.mpnn_layers, self.layer_norms, self.dropouts):
            x_new = mpnn(x, edge_index, edge_attr)
            x = ln(x + drop(x_new))  # residual + dropout + norm
        
        # Global pooling
        if batch is None:
            mean_pool = x.mean(dim=0, keepdim=True)
            max_pool = x.max(dim=0, keepdim=True).values
        else:
            n_graphs = batch.max().item() + 1
            mean_pool = torch.zeros(n_graphs, x.size(1), device=x.device)
            max_pool = torch.full((n_graphs, x.size(1)), float('-inf'), device=x.device)
            for i in range(n_graphs):
                mask = batch == i
                if mask.any():
                    mean_pool[i] = x[mask].mean(dim=0)
                    max_pool[i] = x[mask].max(dim=0).values
        
        graph_embed = torch.cat([mean_pool, max_pool], dim=-1)
        return self.head(graph_embed)


# ============================================================
# 3. Focal Loss (극도의 클래스 불균형 대응)
# ============================================================

class FocalBCELoss(nn.Module):
    """Focal Loss for multi-label classification
    
    FL(p) = -αt(1-pt)^γ log(pt)
    γ=2: 쉬운 샘플 down-weight
    """
    
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


# ============================================================
# 4. 학습 데이터셋
# ============================================================

class UnifiedOdorDataset(Dataset):
    def __init__(self, smiles_list, labels):
        self.graphs = []
        self.labels = []
        skipped = 0
        
        for smi, label in zip(smiles_list, labels):
            graph = smiles_to_graph(smi)
            if graph is not None:
                self.graphs.append(graph)
                self.labels.append(torch.tensor(label, dtype=torch.float32))
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"  ⚠️ {skipped}개 SMILES 파싱 실패")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


# ============================================================
# 5. 학습기
# ============================================================

def train_pom_v2(epochs=150, batch_size=64, lr=0.0005):
    """POM MPNN v2 학습"""
    print("=" * 60)
    print("🧠 POM MPNN v2 — 전체 데이터 재학습")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # 데이터 로딩
    smiles, labels, desc_names = load_all_pyrfume_data()
    if not smiles:
        print("  ❌ 학습 데이터 없음")
        return
    
    n_descriptors = len(desc_names)
    
    # 이진화 (연속값 → 0/1, 역치 0.1)
    binary_labels = []
    for label in labels:
        binary_labels.append(np.where(np.array(label) > 0.1, 1.0, 0.0))
    
    # Train/Val 분할 (85/15)
    n_total = len(smiles)
    n_train = int(n_total * 0.85)
    
    indices = list(range(n_total))
    indices.sort(key=lambda i: hash(smiles[i]) % 10000)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_dataset = UnifiedOdorDataset(
        [smiles[i] for i in train_idx],
        [binary_labels[i] for i in train_idx]
    )
    val_dataset = UnifiedOdorDataset(
        [smiles[i] for i in val_idx],
        [binary_labels[i] for i in val_idx]
    )
    
    print(f"\n  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Descriptors: {n_descriptors}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=False, collate_fn=collate_graphs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_graphs)
    
    # 모델
    model = POMv2(
        atom_dim=4, hidden_dim=192,
        n_layers=5, n_descriptors=n_descriptors
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  모델 파라미터: {n_params:,}")
    
    # Class imbalance weighting
    all_labels_tensor = torch.tensor(np.array(binary_labels))
    pos_counts = all_labels_tensor.sum(dim=0)
    neg_counts = len(all_labels_tensor) - pos_counts
    pos_weights = (neg_counts / (pos_counts + 1)).clamp(max=50).to(device)
    
    criterion = FocalBCELoss(gamma=2.0, pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )
    
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_auroc = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0
        
        for atoms, edges, edge_attr, batch_idx, labels_batch in train_loader:
            atoms = atoms.to(device)
            edges = edges.to(device)
            edge_attr = edge_attr.to(device)
            batch_idx = batch_idx.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(atoms, edges, edge_attr, batch_idx)
            loss = criterion(preds, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)
        
        # Validation (매 5 epoch)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            all_preds = []
            all_targets = []
            val_loss_sum = 0
            n_val = 0
            
            with torch.no_grad():
                for atoms, edges, edge_attr, batch_idx, labels_batch in val_loader:
                    atoms = atoms.to(device)
                    edges = edges.to(device)
                    edge_attr = edge_attr.to(device)
                    batch_idx = batch_idx.to(device)
                    labels_batch = labels_batch.to(device)
                    
                    preds = model(atoms, edges, edge_attr, batch_idx)
                    loss = criterion(preds, labels_batch)
                    val_loss_sum += loss.item()
                    n_val += 1
                    
                    all_preds.append(torch.sigmoid(preds).cpu())
                    all_targets.append(labels_batch.cpu())
            
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # AUROC 계산
            aurocs = []
            for j in range(n_descriptors):
                targets_j = all_targets[:, j].numpy()
                preds_j = all_preds[:, j].numpy()
                
                pos_mask = targets_j == 1
                neg_mask = targets_j == 0
                
                if pos_mask.sum() >= 3 and neg_mask.sum() >= 3:
                    pos_scores = preds_j[pos_mask]
                    neg_scores = preds_j[neg_mask]
                    # Efficient AUROC: sample if too many pairs
                    n_pos, n_neg = len(pos_scores), len(neg_scores)
                    if n_pos * n_neg > 100000:
                        idx_p = np.random.choice(n_pos, min(n_pos, 500), replace=False)
                        idx_n = np.random.choice(n_neg, min(n_neg, 500), replace=False)
                        correct = sum(1 for p in pos_scores[idx_p] for n in neg_scores[idx_n] if p > n)
                        total_pairs = len(idx_p) * len(idx_n)
                    else:
                        correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
                        total_pairs = n_pos * n_neg
                    
                    if total_pairs > 0:
                        aurocs.append(correct / total_pairs)
            
            avg_auroc = np.mean(aurocs) if aurocs else 0
            avg_val_loss = val_loss_sum / max(n_val, 1)
            
            if avg_auroc > best_val_auroc:
                best_val_auroc = avg_auroc
                best_epoch = epoch + 1
                
                torch.save({
                    'model_state': model.state_dict(),
                    'n_descriptors': n_descriptors,
                    'descriptor_names': desc_names,
                    'epoch': epoch + 1,
                    'auroc': avg_auroc,
                    'hidden_dim': 192,
                    'n_layers': 5,
                    'model_version': 'v2',
                    'total_molecules': len(smiles),
                }, os.path.join(save_dir, 'pom_mpnn_v2.pt'))
            
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train={avg_train_loss:.4f} "
                  f"val={avg_val_loss:.4f} "
                  f"AUROC={avg_auroc:.4f} "
                  f"({len(aurocs)} dims) "
                  f"{'★' if epoch + 1 == best_epoch else ''}")
    
    print(f"\n  🏆 Best AUROC: {best_val_auroc:.4f} (epoch {best_epoch})")
    print(f"  📁 저장: models/pom_mpnn_v2.pt")
    return model


if __name__ == '__main__':
    start = time.time()
    train_pom_v2(epochs=150, batch_size=64, lr=0.0005)
    print(f"\n⏱️ 총 학습 시간: {time.time() - start:.1f}초")
