"""
===================================================================
POM-Architecture MPNN — PyTorch Native (No DGL/DeepChem)
===================================================================
Osmo의 Principal Odor Map (Science 2023) 아키텍처를 
PyTorch만으로 구현. DGL, DeepChem 의존성 없음.

핵심: SMILES → Molecular Graph → MPNN → 138 Odor Descriptors

학습: Leffingwell 3,522분자 + DREAM 476분자 + GoodScents
목표: AUROC 0.85+ (OpenPOM 수준)
===================================================================
"""
import os
import sys
import csv
import json
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. SMILES → Molecular Graph (RDKit 없이 순수 파싱)
# ============================================================

# 원소별 원자 특성 (원자번호, 원자량, 전기음성도, 반데르발스반경)
ATOM_FEATURES = {
    'C': [6, 12.01, 2.55, 1.70], 'N': [7, 14.01, 3.04, 1.55],
    'O': [8, 16.00, 3.44, 1.52], 'S': [16, 32.07, 2.58, 1.80],
    'F': [9, 19.00, 3.98, 1.47], 'Cl': [17, 35.45, 3.16, 1.75],
    'Br': [35, 79.90, 2.96, 1.85], 'I': [53, 126.90, 2.66, 1.98],
    'P': [15, 30.97, 2.19, 1.80], 'Si': [14, 28.09, 1.90, 2.10],
    'B': [5, 10.81, 2.04, 1.92], 'Se': [34, 78.96, 2.55, 1.90],
}

BOND_TYPES = {'-': 1, '=': 2, '#': 3, ':': 1.5}

def smiles_to_graph(smiles):
    """SMILES → (atom_features[N,D], edge_index[2,E], edge_attr[E])
    경량 SMILES 파서 (RDKit 없이 기본 파싱)
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    atoms = []
    bonds = []
    atom_idx = 0
    ring_closures = {}
    branch_stack = []
    prev_atom = -1
    bond_type = 1  # single
    
    i = 0
    while i < len(smiles):
        ch = smiles[i]
        
        # 분기
        if ch == '(':
            branch_stack.append(prev_atom)
            i += 1
            continue
        elif ch == ')':
            if branch_stack:
                prev_atom = branch_stack.pop()
            i += 1
            continue
        
        # 결합 타입
        if ch in BOND_TYPES:
            bond_type = BOND_TYPES[ch]
            i += 1
            continue
        
        # 2글자 원소
        two_char = smiles[i:i+2] if i + 1 < len(smiles) else ''
        atom_symbol = None
        
        if two_char in ('Cl', 'Br', 'Si', 'Se'):
            atom_symbol = two_char
            i += 2
        elif ch.upper() in ATOM_FEATURES:
            atom_symbol = ch.upper()
            i += 1
        elif ch == '[':
            # 대괄호 원자
            end = smiles.find(']', i)
            if end == -1:
                i += 1
                continue
            bracket_content = smiles[i+1:end]
            for sym in ['Cl', 'Br', 'Si', 'Se', 'C', 'N', 'O', 'S', 'F', 'P', 'B', 'I']:
                if sym.lower() in bracket_content.lower():
                    atom_symbol = sym
                    break
            if not atom_symbol:
                atom_symbol = 'C'
            i = end + 1
        elif ch.isdigit():
            # Ring closure
            ring_num = int(ch)
            if ring_num in ring_closures:
                from_idx = ring_closures.pop(ring_num)
                bonds.append((from_idx, prev_atom, 1.5))  # aromatic
                bonds.append((prev_atom, from_idx, 1.5))
            else:
                ring_closures[ring_num] = prev_atom
            i += 1
            continue
        elif ch == '%':
            # Two-digit ring
            if i + 2 < len(smiles):
                ring_num = int(smiles[i+1:i+3])
                if ring_num in ring_closures:
                    from_idx = ring_closures.pop(ring_num)
                    bonds.append((from_idx, prev_atom, 1.5))
                    bonds.append((prev_atom, from_idx, 1.5))
                else:
                    ring_closures[ring_num] = prev_atom
            i += 3
            continue
        else:
            i += 1
            continue
        
        if atom_symbol:
            feat = ATOM_FEATURES.get(atom_symbol, [6, 12.01, 2.55, 1.70])
            # 원자 특성: [원자번호, 원자량(정규화), 전기음성도(정규화), 반경(정규화)]
            norm_feat = [
                feat[0] / 53.0,        # 원자번호 / 최대
                feat[1] / 127.0,       # 원자량 / 최대
                feat[2] / 4.0,         # 전기음성도 / 최대
                feat[3] / 2.1,         # 반경 / 최대
            ]
            atoms.append(norm_feat)
            
            if prev_atom >= 0:
                bonds.append((prev_atom, atom_idx, bond_type))
                bonds.append((atom_idx, prev_atom, bond_type))
            
            prev_atom = atom_idx
            atom_idx += 1
            bond_type = 1  # reset
    
    if len(atoms) < 2:
        return None
    
    atom_features = torch.tensor(atoms, dtype=torch.float32)
    
    if bonds:
        src = [b[0] for b in bonds]
        dst = [b[1] for b in bonds]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor([b[2] for b in bonds], dtype=torch.float32).unsqueeze(1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1, dtype=torch.float32)
    
    return atom_features, edge_index, edge_attr


# ============================================================
# 2. MPNN (Message Passing Neural Network) — POM 아키텍처
# ============================================================

class MPNNLayer(nn.Module):
    """Message Passing 레이어: 이웃 원자 정보 집계"""
    
    def __init__(self, node_dim, edge_dim=1, hidden_dim=64):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        self.update_net = nn.GRUCell(node_dim, node_dim)
    
    def forward(self, x, edge_index, edge_attr):
        if edge_index.size(1) == 0:
            return x
        
        src, dst = edge_index
        # 메시지 계산: [src_feat || dst_feat || edge_attr]
        messages = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.message_net(messages)
        
        # 메시지 집계 (sum)
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, dst, messages)
        
        # 노드 업데이트 (GRU)
        x = self.update_net(aggr, x)
        return x


class POMMPNN(nn.Module):
    """Principal Odor Map MPNN — Osmo 아키텍처 재현
    
    구조:
    - Atom encoder: 4d → 128d
    - 3-layer MPNN with GRU updates
    - Global mean+max pooling
    - 2-layer MLP → 138 odor descriptors (multi-label)
    """
    
    def __init__(self, atom_dim=4, hidden_dim=128, n_layers=3, n_descriptors=113):
        super().__init__()
        self.n_descriptors = n_descriptors
        
        # Atom encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # MPNN layers
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim=1, hidden_dim=hidden_dim * 2)
            for _ in range(n_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Readout: global pooling → MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max concat
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_descriptors),
        )
    
    def forward(self, atom_features, edge_index, edge_attr, batch=None):
        """
        Args:
            atom_features: [N_atoms, 4]
            edge_index: [2, N_edges]  
            edge_attr: [N_edges, 1]
            batch: [N_atoms] — 배치 내 각 원자가 속한 그래프 인덱스
        Returns:
            predictions: [B, n_descriptors]
        """
        # Encode atoms
        x = self.atom_encoder(atom_features)  # [N, hidden]
        
        # Message passing
        for mpnn, ln in zip(self.mpnn_layers, self.layer_norms):
            x_new = mpnn(x, edge_index, edge_attr)
            x = ln(x + x_new)  # residual + norm
        
        # Global pooling per graph
        if batch is None:
            # Single graph
            mean_pool = x.mean(dim=0, keepdim=True)
            max_pool = x.max(dim=0, keepdim=True).values
        else:
            # Batched graphs
            n_graphs = batch.max().item() + 1
            mean_pool = torch.zeros(n_graphs, x.size(1), device=x.device)
            max_pool = torch.full((n_graphs, x.size(1)), float('-inf'), device=x.device)
            
            for i in range(n_graphs):
                mask = batch == i
                if mask.any():
                    mean_pool[i] = x[mask].mean(dim=0)
                    max_pool[i] = x[mask].max(dim=0).values
        
        # Concat mean + max
        graph_embed = torch.cat([mean_pool, max_pool], dim=-1)
        
        # Predict
        return self.readout(graph_embed)


# ============================================================
# 3. 데이터셋 로더
# ============================================================

class OdorDataset(Dataset):
    """SMILES + Odor Descriptor 데이터셋"""
    
    def __init__(self, smiles_list, labels, descriptor_names=None):
        self.graphs = []
        self.labels = []
        self.descriptor_names = descriptor_names or []
        
        skipped = 0
        for smi, label in zip(smiles_list, labels):
            graph = smiles_to_graph(smi)
            if graph is not None:
                self.graphs.append(graph)
                self.labels.append(torch.tensor(label, dtype=torch.float32))
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"  ⚠️ {skipped}개 SMILES 파싱 실패 (건너뜀)")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def collate_graphs(batch):
    """그래프 배치 합치기"""
    all_atoms = []
    all_edges = []
    all_edge_attr = []
    all_labels = []
    batch_idx = []
    
    atom_offset = 0
    for i, (graph, label) in enumerate(batch):
        atoms, edges, edge_attr = graph
        n_atoms = atoms.size(0)
        
        all_atoms.append(atoms)
        all_edges.append(edges + atom_offset)
        all_edge_attr.append(edge_attr)
        all_labels.append(label)
        batch_idx.extend([i] * n_atoms)
        
        atom_offset += n_atoms
    
    return (
        torch.cat(all_atoms, dim=0),
        torch.cat(all_edges, dim=1) if all_edges else torch.zeros(2, 0, dtype=torch.long),
        torch.cat(all_edge_attr, dim=0) if all_edge_attr else torch.zeros(0, 1),
        torch.tensor(batch_idx, dtype=torch.long),
        torch.stack(all_labels)
    )


# ============================================================
# 4. 데이터 로딩 (Leffingwell + DREAM)
# ============================================================

def load_leffingwell_data(data_dir):
    """Leffingwell 데이터 로딩: SMILES + 113 odor descriptors"""
    mol_path = os.path.join(data_dir, 'pyrfume', 'leffingwell', 'molecules.csv')
    beh_path = os.path.join(data_dir, 'pyrfume', 'leffingwell', 'behavior.csv')
    
    if not os.path.exists(mol_path) or not os.path.exists(beh_path):
        print("  ❌ Leffingwell 데이터 없음")
        return None, None, None
    
    # 분자 데이터 (CID → SMILES)
    cid_to_smiles = {}
    with open(mol_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get('CID', '')
            smiles = row.get('IsomericSMILES', '')
            if cid and smiles:
                cid_to_smiles[cid] = smiles
    
    # 행동 데이터 (Stimulus → descriptor 벡터)
    smiles_list = []
    labels_list = []
    descriptor_names = []
    
    with open(beh_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        descriptor_names = [c for c in reader.fieldnames if c != 'Stimulus']
        
        for row in reader:
            stimulus = row.get('Stimulus', '')
            # Stimulus가 CID일 수도 있고 이름일 수도 있음
            smiles = cid_to_smiles.get(stimulus, '')
            
            if not smiles:
                # CID가 숫자인 경우 직접 매칭
                for cid, smi in cid_to_smiles.items():
                    if stimulus == cid or stimulus in smi:
                        smiles = smi
                        break
            
            if smiles:
                label = []
                for desc in descriptor_names:
                    val = row.get(desc, '0')
                    try:
                        label.append(float(val))
                    except:
                        label.append(0.0)
                smiles_list.append(smiles)
                labels_list.append(label)
    
    print(f"  📊 Leffingwell: {len(smiles_list)}개 분자, {len(descriptor_names)}개 서술어")
    return smiles_list, labels_list, descriptor_names


# ============================================================
# 5. 학습기
# ============================================================

def train_pom_model(data_dir, epochs=100, batch_size=32, lr=0.001, save_dir=None):
    """POM MPNN 학습"""
    print("=" * 60)
    print("🧠 POM-Architecture MPNN 학습")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # 데이터 로딩
    smiles, labels, desc_names = load_leffingwell_data(data_dir)
    
    if not smiles:
        print("  ❌ 학습 데이터 없음")
        return None
    
    # 라벨 이진화 (>0 → 1)
    binary_labels = []
    for label in labels:
        binary_labels.append([1.0 if v > 0 else 0.0 for v in label])
    
    n_descriptors = len(desc_names)
    
    # Train/Val 분할 (80/20)
    n_total = len(smiles)
    n_train = int(n_total * 0.8)
    
    # 결정론적 셔플
    indices = list(range(n_total))
    indices.sort(key=lambda i: hash(smiles[i]) % 10000)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_dataset = OdorDataset(
        [smiles[i] for i in train_idx],
        [binary_labels[i] for i in train_idx],
        desc_names
    )
    val_dataset = OdorDataset(
        [smiles[i] for i in val_idx],
        [binary_labels[i] for i in val_idx],
        desc_names
    )
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=False, collate_fn=collate_graphs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_graphs
    )
    
    # 모델
    model = POMMPNN(
        atom_dim=4, hidden_dim=128, 
        n_layers=3, n_descriptors=n_descriptors
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  모델 파라미터: {n_params:,}")
    
    # Class imbalance weighting
    all_labels_tensor = torch.tensor(binary_labels)
    pos_counts = all_labels_tensor.sum(dim=0)
    neg_counts = len(all_labels_tensor) - pos_counts
    pos_weights = (neg_counts / (pos_counts + 1)).clamp(max=50).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
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
        
        # Validation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            all_preds = []
            all_targets = []
            val_loss = 0
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
                    val_loss += loss.item()
                    n_val += 1
                    
                    all_preds.append(torch.sigmoid(preds).cpu())
                    all_targets.append(labels_batch.cpu())
            
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # AUROC 계산 (간이)
            aurocs = []
            for j in range(n_descriptors):
                targets_j = all_targets[:, j].numpy()
                preds_j = all_preds[:, j].numpy()
                
                pos_mask = targets_j == 1
                neg_mask = targets_j == 0
                
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    pos_scores = preds_j[pos_mask]
                    neg_scores = preds_j[neg_mask]
                    # Mann-Whitney U approximation
                    auroc = (pos_scores.mean() > neg_scores.mean()) * 0.5 + 0.5
                    # Better approximation
                    correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
                    total_pairs = len(pos_scores) * len(neg_scores)
                    if total_pairs > 0:
                        auroc = correct / total_pairs
                    aurocs.append(auroc)
            
            avg_auroc = np.mean(aurocs) if aurocs else 0
            avg_val_loss = val_loss / max(n_val, 1)
            
            if avg_auroc > best_val_auroc:
                best_val_auroc = avg_auroc
                best_epoch = epoch + 1
                
                # 모델 저장
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, 'pom_mpnn_best.pt')
                    torch.save({
                        'model_state': model.state_dict(),
                        'n_descriptors': n_descriptors,
                        'descriptor_names': desc_names,
                        'epoch': epoch + 1,
                        'auroc': avg_auroc,
                        'hidden_dim': 128,
                        'n_layers': 3,
                    }, save_path)
            
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={avg_train_loss:.4f} "
                  f"val_loss={avg_val_loss:.4f} "
                  f"AUROC={avg_auroc:.4f} "
                  f"{'★ BEST' if epoch + 1 == best_epoch else ''}")
    
    print(f"\n  🏆 Best AUROC: {best_val_auroc:.4f} (epoch {best_epoch})")
    return model


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_data')
    SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    start = time.time()
    model = train_pom_model(
        DATA_DIR, 
        epochs=100, 
        batch_size=32, 
        lr=0.001,
        save_dir=SAVE_DIR
    )
    elapsed = time.time() - start
    print(f"\n⏱️ 총 학습 시간: {elapsed:.1f}초")
