"""OdorGATv5: Graph Attention Network + ChemBERTa Fusion for Odor Prediction

Architecture:
  AtomFeatures(9d) → GATConv(3-layer, 4-head) → global_mean+max_pool(128d)
                                                        ↓
                                              ChemBERTa(384d) → concat(512d)
                                                        ↓
                                              MLP(512→256→128→20) → Sigmoid
                                                        ↓
                                              20d Odor Vector

Atom features (9d): atomic_num, degree, n_hydrogens, formal_charge, 
                    aromatic, hybridization(3 one-hot), in_ring
Bond features (3d): bond_type, conjugated, in_ring
"""
import os, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem

# 22d odor space
ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]
N_DIM = 22


# ================================================================
# Molecular Graph Featurization
# ================================================================

ATOM_FEATURES_DIM = 9

def atom_features(atom):
    """Atom → 9d feature vector"""
    return [
        atom.GetAtomicNum() / 53.0,        # Normalized (max ~53 for I)
        atom.GetDegree() / 4.0,             # Normalized
        atom.GetTotalNumHs() / 4.0,         # Normalized
        atom.GetFormalCharge() / 2.0,        # Normalized
        float(atom.GetIsAromatic()),         # Boolean
        # Hybridization one-hot (sp, sp2, sp3)
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
        float(atom.IsInRing()),              # Boolean
    ]


BOND_FEATURES_DIM = 3

def bond_features(bond):
    """Bond → 3d feature vector"""
    bt = bond.GetBondType()
    return [
        float(bt == Chem.rdchem.BondType.SINGLE) + 
        float(bt == Chem.rdchem.BondType.DOUBLE) * 2 + 
        float(bt == Chem.rdchem.BondType.TRIPLE) * 3 +
        float(bt == Chem.rdchem.BondType.AROMATIC) * 1.5,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def smiles_to_graph(smiles, device='cpu'):
    """SMILES → PyG Data object (molecular graph)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Atom features
    atoms = []
    for atom in mol.GetAtoms():
        atoms.append(atom_features(atom))
    x = torch.tensor(atoms, dtype=torch.float32)
    
    # Edge index + features (bidirectional)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)
    
    if not edge_indices:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURES_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data.to(device)


# ================================================================
# OdorGATv5 Model
# ================================================================

class OdorGATv5(nn.Module):
    """Graph Attention Network + ChemBERTa fusion for 20d odor vectors.
    
    Dual-path architecture:
    Path A: Molecular graph → GATConv(3-layer) → global pool → 128d
    Path B: ChemBERTa embedding → 384d
    Fusion: concat(128+384=512d) → MLP → 20d
    
    ~300K parameters
    """
    
    def __init__(self, bert_dim=384, heads=4, gat_hidden=64, dropout=0.15):
        super().__init__()
        
        self.bert_dim = bert_dim
        
        # === Path A: GATConv layers ===
        self.gat1 = GATConv(ATOM_FEATURES_DIM, gat_hidden, heads=heads, 
                           dropout=dropout, edge_dim=BOND_FEATURES_DIM)
        self.gat2 = GATConv(gat_hidden * heads, gat_hidden, heads=heads,
                           dropout=dropout, edge_dim=BOND_FEATURES_DIM)
        self.gat3 = GATConv(gat_hidden * heads, gat_hidden, heads=1,
                           dropout=dropout, edge_dim=BOND_FEATURES_DIM, concat=False)
        
        self.gat_norm1 = nn.LayerNorm(gat_hidden * heads)
        self.gat_norm2 = nn.LayerNorm(gat_hidden * heads)
        self.gat_norm3 = nn.LayerNorm(gat_hidden)
        
        # Pool: mean+max concat → 128d
        pool_dim = gat_hidden * 2  # mean(64) + max(64) = 128
        
        # === Path B: ChemBERTa projection ===
        self.bert_proj = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
        )
        
        # === Fusion: concat(128 + 256) = 384 → 20d ===
        fusion_dim = pool_dim + 256
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, N_DIM),
            nn.Sigmoid(),
        )
    
    def forward_graph(self, data):
        """Molecular graph → 128d pooled vector"""
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # GAT layer 1
        h = self.gat1(x, edge_index, edge_attr=edge_attr)
        h = self.gat_norm1(h)
        h = F.elu(h)
        
        # GAT layer 2
        h = self.gat2(h, edge_index, edge_attr=edge_attr)
        h = self.gat_norm2(h)
        h = F.elu(h)
        
        # GAT layer 3
        h = self.gat3(h, edge_index, edge_attr=edge_attr)
        h = self.gat_norm3(h)
        h = F.elu(h)
        
        # Global pooling: mean + max → concat
        h_mean = global_mean_pool(h, batch)  # [B, 64]
        h_max = global_max_pool(h, batch)    # [B, 64]
        
        return torch.cat([h_mean, h_max], dim=-1)  # [B, 128]
    
    def forward(self, graph_data, bert_embedding):
        """
        Args:
            graph_data: PyG Data or Batch object
            bert_embedding: [B, 384] ChemBERTa embeddings
        Returns:
            [B, 20] odor vectors
        """
        # Path A: Graph → 128d
        graph_feat = self.forward_graph(graph_data)  # [B, 128]
        
        # Path B: ChemBERTa → 256d
        bert_feat = self.bert_proj(bert_embedding)    # [B, 256]
        
        # Fusion
        fused = torch.cat([graph_feat, bert_feat], dim=-1)  # [B, 384]
        return self.fusion(fused)  # [B, 20]
    
    def forward_graph_only(self, graph_data):
        """Fallback: graph only (no ChemBERTa available)"""
        graph_feat = self.forward_graph(graph_data)
        # Pad with zeros for bert part
        bert_zeros = torch.zeros(graph_feat.size(0), 256, device=graph_feat.device)
        fused = torch.cat([graph_feat, bert_zeros], dim=-1)
        return self.fusion(fused)
    
    @torch.no_grad()
    def extract_attention(self, data):
        """Extract per-atom attention weights from all 3 GAT layers.
        
        Uses entropy-based importance: atoms that receive concentrated
        attention (low entropy) are more informative than uniformly attended ones.
        
        Returns:
            atom_importance: [n_atoms] tensor — normalized 0-1 importance per atom
            layer_attentions: list of (edge_index, attention_weights) per layer
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        n_atoms = x.size(0)
        
        # Accumulate entropy-based importance per atom
        atom_entropy_sum = torch.zeros(n_atoms, device=x.device)
        layer_attentions = []
        
        # Layer 1: [n_atoms, 9] → [n_atoms, 256]
        h, (ei1, alpha1) = self.gat1(x, edge_index, edge_attr=edge_attr,
                                      return_attention_weights=True)
        h = self.gat_norm1(h)
        h = F.elu(h)
        self._accumulate_entropy_importance(ei1, alpha1, atom_entropy_sum, n_atoms)
        layer_attentions.append((ei1.cpu(), alpha1.cpu()))
        
        # Layer 2
        h, (ei2, alpha2) = self.gat2(h, edge_index, edge_attr=edge_attr,
                                      return_attention_weights=True)
        h = self.gat_norm2(h)
        h = F.elu(h)
        self._accumulate_entropy_importance(ei2, alpha2, atom_entropy_sum, n_atoms)
        layer_attentions.append((ei2.cpu(), alpha2.cpu()))
        
        # Layer 3
        h, (ei3, alpha3) = self.gat3(h, edge_index, edge_attr=edge_attr,
                                      return_attention_weights=True)
        h = self.gat_norm3(h)
        h = F.elu(h)
        self._accumulate_entropy_importance(ei3, alpha3, atom_entropy_sum, n_atoms)
        layer_attentions.append((ei3.cpu(), alpha3.cpu()))
        
        # ★ Fix: Low entropy = concentrated attention = MORE important atom
        # Invert so that important atoms get high scores
        max_entropy = atom_entropy_sum.max()
        if max_entropy > 0:
            atom_importance = 1.0 - (atom_entropy_sum / max_entropy)
        else:
            atom_importance = torch.zeros(n_atoms, device=x.device)
        
        return atom_importance.cpu(), layer_attentions
    
    @staticmethod
    def _accumulate_entropy_importance(edge_index, alpha, atom_entropy, n_atoms):
        """Compute attention entropy per target atom — low entropy = important."""
        alpha_mean = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha  # [n_edges]
        alpha_mean = alpha_mean.clamp(min=1e-8)
        
        for target_idx in range(n_atoms):
            # Get all attention weights pointing to this target
            mask = edge_index[1] == target_idx
            if not mask.any():
                continue
            weights = alpha_mean[mask]
            # Normalize to distribution
            weights = weights / weights.sum().clamp(min=1e-8)
            # Shannon entropy: uniform dist → high entropy (unimportant)
            entropy = -(weights * weights.log()).sum()
            atom_entropy[target_idx] += entropy.item()


# ================================================================
# Training function
# ================================================================

def train_v5(
    multi_source_csv,
    bert_cache_path,
    save_path,
    epochs=50,
    batch_size=32,
    lr=0.001,
    device='cuda',
):
    """Train OdorGATv5 with multi-source data"""
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  OdorGATv5 Training | Device: {dev}")
    print(f"{'='*60}")
    
    # 1. Load ChemBERTa cache
    print("  Loading ChemBERTa cache ...")
    cache_data = np.load(bert_cache_path)
    bert_smiles = cache_data['smiles']
    bert_embeds = cache_data['embeddings']
    bert_cache = {s: bert_embeds[i] for i, s in enumerate(bert_smiles)}
    print(f"  Cache: {len(bert_cache)} molecules, {bert_embeds.shape[1]}d")
    
    # 2. Load multi-source labels
    print("  Loading multi-source labels ...")
    with open(multi_source_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 3. Build training data (graph + bert + label)
    print("  Building training data ...")
    train_data = []
    skipped = 0
    
    for row in rows:
        smiles = row['smiles']
        
        # Must have ChemBERTa embedding
        bert_emb = bert_cache.get(smiles)
        if bert_emb is None:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                can = Chem.MolToSmiles(mol)
                bert_emb = bert_cache.get(can)
        
        if bert_emb is None:
            skipped += 1
            continue
        
        # Build graph
        graph = smiles_to_graph(smiles, device='cpu')
        if graph is None or graph.x.size(0) == 0:
            skipped += 1
            continue
        
        # Label vector
        label = np.array([float(row[dim]) for dim in ODOR_DIMENSIONS], dtype=np.float32)
        
        train_data.append({
            'graph': graph,
            'bert': torch.tensor(bert_emb, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
        })
    
    print(f"  Training samples: {len(train_data)} (skipped {skipped})")
    
    # 4. Train/val split
    np.random.seed(42)
    indices = np.random.permutation(len(train_data))
    split = int(len(train_data) * 0.9)
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # 5. Model
    model = OdorGATv5(bert_dim=384).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    mse_loss = nn.MSELoss()
    cos_loss = nn.CosineEmbeddingLoss()
    
    best_val_loss = float('inf')
    best_cos_sim = 0
    patience = 0
    max_patience = 20
    
    import time
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        # === Train ===
        model.train()
        np.random.shuffle(train_idx)
        train_loss_sum = 0
        n_batches = 0
        
        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[i:i+batch_size]
            
            graphs = [train_data[j]['graph'] for j in batch_idx]
            berts = torch.stack([train_data[j]['bert'] for j in batch_idx]).to(dev)
            labels = torch.stack([train_data[j]['label'] for j in batch_idx]).to(dev)
            
            batch_graph = Batch.from_data_list(graphs).to(dev)
            
            pred = model(batch_graph, berts)
            
            loss_mse = mse_loss(pred, labels)
            targets_cos = torch.ones(pred.size(0), device=dev)
            loss_cos = cos_loss(pred, labels, targets_cos)
            loss = loss_mse + 0.3 * loss_cos
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_train = train_loss_sum / max(n_batches, 1)
        
        # === Validate ===
        model.eval()
        val_loss_sum = 0
        val_cos_sum = 0
        n_val = 0
        
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_idx = val_idx[i:i+batch_size]
                
                graphs = [train_data[j]['graph'] for j in batch_idx]
                berts = torch.stack([train_data[j]['bert'] for j in batch_idx]).to(dev)
                labels = torch.stack([train_data[j]['label'] for j in batch_idx]).to(dev)
                
                batch_graph = Batch.from_data_list(graphs).to(dev)
                pred = model(batch_graph, berts)
                
                loss = mse_loss(pred, labels)
                val_loss_sum += loss.item() * len(batch_idx)
                
                # Cosine similarity
                for p, l in zip(pred, labels):
                    cos = F.cosine_similarity(p.unsqueeze(0), l.unsqueeze(0)).item()
                    val_cos_sum += cos
                
                n_val += len(batch_idx)
        
        val_loss = val_loss_sum / max(n_val, 1)
        val_cos = val_cos_sum / max(n_val, 1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_cos_sim = val_cos
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_cos_sim': val_cos,
                'n_params': n_params,
                'architecture': 'GATv5',
                'data_source': 'multi_source',
            }, save_path)
        else:
            patience += 1
        
        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Train: {avg_train:.4f} | "
                  f"Val: {val_loss:.4f} | CosSim: {val_cos:.3f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.0f}s"
                  + (" *" if patience == 0 else ""))
        
        if patience >= max_patience:
            print(f"  Early stop at epoch {epoch} (no improvement for {max_patience} epochs)")
            break
    
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE! Best epoch: {epoch - patience}, Val loss: {best_val_loss:.4f}")
    print(f"  Best CosSim: {best_cos_sim:.3f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Saved: {save_path}")
    print(f"{'='*60}")
    
    return model


if __name__ == "__main__":
    import sys
    BASE = os.path.dirname(os.path.dirname(__file__))
    
    model = train_v5(
        multi_source_csv=os.path.join(BASE, "data", "multi_source_unified.csv"),
        bert_cache_path=os.path.join(BASE, "chemberta_cache.npz"),
        save_path=os.path.join(BASE, "weights", "odor_gnn_v5.pt"),
        epochs=80,
        batch_size=32,
        lr=0.001,
    )
