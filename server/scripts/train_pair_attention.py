"""
PairAttentionNet v2 -- Production Training
==========================================
- Expanded DB (5,330 molecules, pom_embeddings_v2.npz)
- Weighted BCE (pos_weight, clipped [1.0, 50.0])
- Molecule-level split (no molecule in both train/test)
- Per-label AUROC + macro/weighted averages
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

BASE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(BASE)
sys.path.insert(0, SERVER_DIR)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def canonicalize_smiles(smiles):
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        pass
    return None


def build_label_vocab(data):
    labels = set()
    for item in data:
        for note in item.get('blend_notes', []):
            n = note.strip().lower()
            if n:
                labels.add(n)
    labels = sorted(labels)
    return {l: i for i, l in enumerate(labels)}, labels


class OdorPairDataset(Dataset):
    def __init__(self, data, emb_mat, smi_idx, label2idx):
        self.samples = []
        n_labels = len(label2idx)
        skipped = 0
        
        for item in data:
            s1 = item.get('mol1', '')
            s2 = item.get('mol2', '')
            
            # Try canonical first, then raw
            c1 = canonicalize_smiles(s1) if s1 else None
            c2 = canonicalize_smiles(s2) if s2 else None
            
            idx1 = smi_idx.get(c1) if c1 else smi_idx.get(s1)
            idx2 = smi_idx.get(c2) if c2 else smi_idx.get(s2)
            
            if idx1 is None:
                idx1 = smi_idx.get(s1)
            if idx2 is None:
                idx2 = smi_idx.get(s2)
            
            if idx1 is None or idx2 is None:
                skipped += 1
                continue
            
            label = np.zeros(n_labels, dtype=np.float32)
            for note in item.get('blend_notes', []):
                n = note.strip().lower()
                if n in label2idx:
                    label[label2idx[n]] = 1.0
            
            x = np.stack([emb_mat[idx1], emb_mat[idx2]], axis=0).astype(np.float32)
            self.samples.append((x, label, s1, s2))
        
        print(f"  [DATA] {len(self.samples)} pairs loaded ({skipped} skipped)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y, _, _ = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


class PairAttentionNet(nn.Module):
    def __init__(self, pom_dim=256, num_labels=109, n_heads=4, hidden=256, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Linear(pom_dim, hidden)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, num_labels),
        )
        # NO sigmoid — BCEWithLogitsLoss handles it
    
    def forward(self, x):
        h = self.input_proj(x)
        attn_out, _ = self.attention(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = self.ffn_norm(h + self.ffn(h))
        return self.classifier(h.sum(dim=1))


def molecule_level_split(data, test_ratio=0.1, seed=42):
    """Split by molecule: molecules in test never appear in training"""
    rng = np.random.RandomState(seed)
    
    # Collect all unique molecules
    mol_set = set()
    for item in data:
        s1 = item.get('mol1', '')
        s2 = item.get('mol2', '')
        if s1: mol_set.add(s1)
        if s2: mol_set.add(s2)
    
    mol_list = sorted(mol_set)
    rng.shuffle(mol_list)
    
    n_test = max(int(len(mol_list) * test_ratio), 10)
    test_mols = set(mol_list[:n_test])
    
    train_data, test_data = [], []
    for item in data:
        s1 = item.get('mol1', '')
        s2 = item.get('mol2', '')
        # If EITHER molecule is in test set -> test
        if s1 in test_mols or s2 in test_mols:
            test_data.append(item)
        else:
            train_data.append(item)
    
    return train_data, test_data, test_mols


def compute_pos_weight(dataset, n_labels, clip_max=50.0):
    """Compute per-label pos_weight for Weighted BCE"""
    pos = np.zeros(n_labels)
    total = 0
    
    for _, y in dataset:
        pos += y.numpy()
        total += 1
    
    neg = total - pos
    pos_safe = np.clip(pos, 1, None)
    pw = neg / pos_safe
    pw = np.clip(pw, 1.0, clip_max)
    return torch.tensor(pw, dtype=torch.float32)


def train_model(train_loader, val_loader, model, label_names, pos_weight,
                epochs=50, lr=1e-3, save_dir=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    
    best_auroc = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss, n_b = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_b += 1
        scheduler.step()
        
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                preds.append(torch.sigmoid(model(x)).cpu().numpy())
                labels.append(y.numpy())
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        aurocs, aps, supports = [], [], []
        for i in range(labels.shape[1]):
            support = labels[:, i].sum()
            if support > 5:
                try:
                    aurocs.append(roc_auc_score(labels[:, i], preds[:, i]))
                    aps.append(average_precision_score(labels[:, i], preds[:, i]))
                    supports.append(support)
                except:
                    pass
        
        macro_auroc = np.mean(aurocs) if aurocs else 0
        macro_ap = np.mean(aps) if aps else 0
        w_auroc = np.average(aurocs, weights=supports) if aurocs else 0
        
        if (epoch+1) % 5 == 0 or macro_auroc > best_auroc:
            print(f"  Ep {epoch+1:3d}/{epochs}: loss={train_loss/max(n_b,1):.4f} "
                  f"AUROC={macro_auroc:.4f} AP={macro_ap:.4f} wAUROC={w_auroc:.4f} ({len(aurocs)} labels)")
        
        if macro_auroc > best_auroc:
            best_auroc = macro_auroc
            best_epoch = epoch + 1
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'model_state': model.state_dict(),
                    'label_names': label_names,
                    'num_labels': len(label_names),
                    'auroc': best_auroc,
                    'macro_ap': float(macro_ap),
                    'weighted_auroc': float(w_auroc),
                    'epoch': best_epoch,
                    'n_train_pairs': len(train_loader.dataset),
                    'n_val_pairs': len(val_loader.dataset),
                    'n_evaluated_labels': len(aurocs),
                    'split_method': 'molecule_level',
                }, os.path.join(save_dir, 'pair_attention_best.pt'))
    
    print(f"\n  [BEST] AUROC={best_auroc:.4f} at epoch {best_epoch}")
    return best_auroc


if __name__ == '__main__':
    t0 = time.time()
    print("=" * 60)
    print(f"  PairAttentionNet v2 -- Production Training (Device: {DEVICE})")
    print("=" * 60)
    
    # Load Odor-Pair
    data_path = os.path.join(SERVER_DIR, 'data', 'pom_upgrade', 'odor_pair', 'full.json')
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    print(f"\n  Odor-Pair: {len(raw_data)} pairs")
    
    label2idx, label_names = build_label_vocab(raw_data)
    print(f"  Labels: {len(label_names)}")
    
    # Load EXPANDED embedding DB (v2)
    emb_path = os.path.join(SERVER_DIR, 'models', 'openpom_ensemble', 'pom_embeddings_v2.npz')
    if not os.path.exists(emb_path):
        emb_path = os.path.join(SERVER_DIR, 'models', 'openpom_ensemble', 'pom_embeddings.npz')
        print("  [WARN] v2 DB not found, using v1")
    
    emb_data = np.load(emb_path, allow_pickle=True)
    emb_mat = emb_data['embeddings']
    emb_smiles = [str(s) for s in emb_data['smiles']]
    
    # Build index with BOTH canonical and raw SMILES
    smi_idx = {}
    for i, smi in enumerate(emb_smiles):
        smi_idx[smi] = i
        can = canonicalize_smiles(smi)
        if can:
            smi_idx[can] = i
    
    print(f"  Embedding DB: {len(emb_smiles)} molecules ({len(smi_idx)} index entries)")
    
    # MOLECULE-LEVEL split
    print("\n--- Molecule-level split ---")
    train_data, val_data, test_mols = molecule_level_split(raw_data, test_ratio=0.1)
    print(f"  Test molecules: {len(test_mols)}")
    print(f"  Train pairs: {len(train_data)}, Val pairs: {len(val_data)}")
    
    # Create datasets
    print("\n--- Building datasets ---")
    train_ds = OdorPairDataset(train_data, emb_mat, smi_idx, label2idx)
    val_ds = OdorPairDataset(val_data, emb_mat, smi_idx, label2idx)
    
    if len(train_ds) < 100:
        print("  [ERROR] Not enough data")
        sys.exit(1)
    
    # Compute Weighted BCE pos_weight
    print("\n--- Computing Weighted BCE pos_weight ---")
    pw = compute_pos_weight(train_ds, len(label_names), clip_max=50.0)
    active_labels = (pw < 50.0).sum().item()
    print(f"  Active labels (pw < 50): {active_labels}/{len(label_names)}")
    print(f"  Weight range: [{pw.min():.1f}, {pw.max():.1f}]")
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    
    model = PairAttentionNet(
        pom_dim=256, num_labels=len(label_names),
        n_heads=4, hidden=256, dropout=0.15,
    ).to(DEVICE)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {params:,} params")
    
    # Train
    print("\n--- Training (Weighted BCE + Molecule Split) ---")
    save_dir = os.path.join(SERVER_DIR, 'models', 'pair_attention')
    best = train_model(train_loader, val_loader, model, label_names, pw,
                       epochs=60, lr=1e-3, save_dir=save_dir)
    
    with open(os.path.join(save_dir, 'label_mapping.json'), 'w') as f:
        json.dump({'label2idx': label2idx, 'label_names': label_names}, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n[OK] Phase 2 complete ({elapsed:.1f}s)")
