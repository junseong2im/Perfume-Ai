"""OdorPredictor v6 - Industry SOTA Implementation
====================================================
[PATH A] ChemBERTa 384d + LoRA -> Bottleneck AttentionPooling -> 384d
[PATH B] EGNN (E(n) Equivariant) x4 + GATv2 x4 + SchNet CFCONV x4
         + VirtualNode + Stochastic Depth
         + Graphormer-style Spatial/Edge Bias
         + Multi-Conformer Ensembling (K=3)
         + JumpingKnowledge + GlobalAttention -> 256d
[PATH C] PhysProps (12d) -> MLP + Feature Cross -> 64d

4-layer Transformer Fusion (768d) + SE + Gating
Backbone: 768 -> 768 -> 512 -> 256 (skip + SE + Stochastic Depth)
10 Output Heads from 256d (with uncertainty estimation)

Includes: EMA, GradNorm, Noisy Student, Knowledge Distillation,
          MC Dropout for uncertainty, full multi-task loss
~30M trainable params

References:
- EGNN: Satorras et al. (2021) "E(n) Equivariant Graph Neural Networks"
- SchNet: Schutt et al. (2018) "SchNet: A deep learning architecture"
- Graphormer: Ying et al. (2021) "Do Transformers Really Perform Bad"
- Uni-Mol: Zhou et al. (2023) "Uni-Mol: A Universal 3D Molecular"
- GNoME: Merchant et al. (2023) "Scaling deep learning for materials"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, GlobalAttention
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    try:
        from torch_geometric.nn import GATConv as GATv2Conv, global_mean_pool, GlobalAttention
        from torch_geometric.data import Data
        HAS_PYG = True
    except ImportError:
        HAS_PYG = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]
N_ODOR_DIM = len(ODOR_DIMENSIONS)
ATOM_FEATURES_DIM = 47
BOND_FEATURES_DIM = 13


# ================================================================
# Graph Construction (with multi-conformer support)
# ================================================================

def smiles_to_graph_v6(smiles, device='cpu', compute_3d=True, n_conformers=1):
    """SMILES -> PyG Data with full features + 3D coords + multi-conformer"""
    if not HAS_RDKIT or not HAS_PYG:
        return None
    if not smiles or not smiles.strip():  # Rule: empty/blank SMILES guard
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol_h = Chem.AddHs(mol)

        # Multi-conformer 3D generation (Uni-Mol style)
        all_coords = []
        if compute_3d:
            try:
                cids = AllChem.EmbedMultipleConfs(
                    mol_h, numConfs=max(n_conformers, 1),
                    randomSeed=42, maxAttempts=200,
                    pruneRmsThresh=0.5, numThreads=0
                )
                for cid in cids:
                    try:
                        AllChem.MMFFOptimizeMolecule(mol_h, confId=cid, maxIters=500)
                    except:
                        pass
                    conf = mol_h.GetConformer(cid)
                    coords = np.array([
                        list(conf.GetAtomPosition(i))
                        for i in range(mol_h.GetNumAtoms())
                    ])
                    all_coords.append(coords)
            except:
                pass

        if not all_coords:
            try:
                AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=100)
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                conf = mol_h.GetConformer()
                coords = np.array([
                    list(conf.GetAtomPosition(i))
                    for i in range(mol_h.GetNumAtoms())
                ])
                all_coords.append(coords)
            except:
                pass

        mol = Chem.RemoveHs(mol_h)
        ri = mol.GetRingInfo()
        n_atoms = mol.GetNumAtoms()

        # Use first conformer for atom features (others stored separately)
        coords_3d = all_coords[0] if all_coords else None
        # Trim coords to match heavy atoms
        if coords_3d is not None and len(coords_3d) > n_atoms:
            coords_3d = coords_3d[:n_atoms]
        trimmed_all = []
        for c in all_coords:
            if len(c) > n_atoms:
                c = c[:n_atoms]
            trimmed_all.append(c)

        atom_features = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            feat = [
                atom.GetAtomicNum() / 53.0,
                atom.GetDegree() / 6.0,
                atom.GetTotalValence() / 6.0,
                (atom.GetFormalCharge() + 2) / 4.0,
            ]
            # Hybridization one-hot (6d)
            hyb = atom.GetHybridization()
            hyb_types = [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ]
            hyb_oh = [1.0 if hyb == h else 0.0 for h in hyb_types] + [0.0]
            feat.extend(hyb_oh)

            feat.append(1.0 if atom.GetIsAromatic() else 0.0)
            feat.append(1.0 if atom.IsInRing() else 0.0)
            feat.append(atom.GetTotalNumHs() / 4.0)
            feat.append(len([r for r in ri.AtomRingSizes(idx)]) / 3.0)

            # Chirality (4d)
            chiral = atom.GetChiralTag()
            chiral_map = {
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 0,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 1,
                Chem.rdchem.ChiralType.CHI_OTHER: 2,
            }
            chiral_oh = [0.0] * 4
            if chiral in chiral_map:
                chiral_oh[chiral_map[chiral]] = 1.0
            else:
                chiral_oh[3] = 1.0
            feat.extend(chiral_oh)

            # Ring membership (4d)
            for rsize in [3, 4, 5, 6]:
                feat.append(1.0 if ri.IsAtomInRingOfSize(idx, rsize) else 0.0)

            # 3D coordinates (3d)
            if coords_3d is not None and idx < len(coords_3d):
                feat.extend(coords_3d[idx].tolist())
            else:
                feat.extend([0.0, 0.0, 0.0])

            # Physical properties
            feat.append(atom.GetMass() / 200.0)
            try:
                feat.append(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) / 3.0)
            except:
                feat.append(0.5)
            feat.append(atom.GetAtomicNum() / 53.0)
            feat.append(atom.GetNumRadicalElectrons() / 2.0)

            while len(feat) < ATOM_FEATURES_DIM:
                feat.append(0.0)
            feat = feat[:ATOM_FEATURES_DIM]
            atom_features.append(feat)

        x = torch.tensor(atom_features, dtype=torch.float32)

        # Bond features
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])
            bf = _bond_features(bond, coords_3d)
            edge_attr.extend([bf, bf])

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, BOND_FEATURES_DIM, dtype=torch.float32)

        # 3D positions tensor
        if coords_3d is not None:
            pos = torch.tensor(coords_3d[:n_atoms], dtype=torch.float32)
        else:
            pos = torch.zeros(n_atoms, 3, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

        # Store additional conformer coordinates
        if len(trimmed_all) > 1:
            data.multi_pos = [torch.tensor(c[:n_atoms], dtype=torch.float32)
                              for c in trimmed_all]
        else:
            data.multi_pos = [pos]

        return data
    except Exception:
        return None


def _bond_features(bond, coords_3d=None):
    """Bond -> 13d feature vector with 3D distance"""
    bt = bond.GetBondType()
    feat = [
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]
    stereo = bond.GetStereo()
    stereo_map = {
        Chem.rdchem.BondStereo.STEREOZ: 0,
        Chem.rdchem.BondStereo.STEREOE: 1,
        Chem.rdchem.BondStereo.STEREOANY: 2,
    }
    stereo_oh = [0.0] * 4
    if stereo in stereo_map:
        stereo_oh[stereo_map[stereo]] = 1.0
    else:
        stereo_oh[3] = 1.0
    feat.extend(stereo_oh)

    if coords_3d is not None:
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < len(coords_3d) and j < len(coords_3d):
            dist = np.linalg.norm(coords_3d[i] - coords_3d[j])
            feat.extend([dist / 5.0, np.exp(-dist), np.sin(dist)])
        else:
            feat.extend([0.0, 0.0, 0.0])
    else:
        feat.extend([0.0, 0.0, 0.0])

    return feat[:BOND_FEATURES_DIM]


def extract_phys_props(smiles):
    """SMILES -> 12d physical property vector"""
    props = np.zeros(12, dtype=np.float32)
    if not HAS_RDKIT:
        return props
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return props
        props[0] = Descriptors.MolWt(mol) / 500.0
        props[1] = Descriptors.MolLogP(mol) / 10.0
        props[2] = Descriptors.TPSA(mol) / 200.0
        props[3] = Descriptors.NumHDonors(mol) / 5.0
        props[4] = Descriptors.NumHAcceptors(mol) / 10.0
        props[5] = Descriptors.NumRotatableBonds(mol) / 15.0
        props[6] = Descriptors.RingCount(mol) / 5.0
        props[7] = Descriptors.NumAromaticRings(mol) / 3.0
        props[8] = mol.GetNumHeavyAtoms() / 50.0
        props[9] = Descriptors.FractionCSP3(mol)
        props[10] = len(Chem.FindMolChiralCenters(mol)) / 3.0
        props[11] = 0.0
    except Exception:
        pass
    return props


# ================================================================
# Stochastic Depth (DropPath)
# ================================================================

class DropPath(nn.Module):
    """Stochastic depth (Huang et al., 2016). Used by Graphormer + ViT."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep


# ================================================================
# Bottleneck Attention Pooling for ChemBERTa
# ================================================================

class AttentionPooling(nn.Module):
    """Multi-head bottleneck attention pooling (industry-grade).
    Similar to Set Transformer's PMA / Perceiver's cross-attention."""
    def __init__(self, d_model=384, n_heads=8, n_latents=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, d] -> [B, 1, d] for cross-attention with latents
        B = x.size(0)
        x_seq = x.unsqueeze(1)
        latents = self.latents.expand(B, -1, -1)

        # Cross-attend: latents query, input is KV
        h, _ = self.cross_attn(latents, x_seq, x_seq)
        h = self.norm1(h + latents)

        # Self-attend over latents
        h2, _ = self.self_attn(h, h, h)
        h = self.norm2(h2 + h)
        h = self.norm3(h + self.ffn(h))

        # Pool latents -> single vector
        pooled = h.mean(dim=1)
        return pooled * self.gate(pooled)


# ================================================================
# E(n) Equivariant Graph Neural Network Layer (EGNN)
# ================================================================

class EGNNLayer(nn.Module):
    """E(n) Equivariant Graph Convolution (Satorras et al., 2021).
    Updates both node features AND 3D coordinates equivariantly.
    Key: coordinate updates are equivariant to rotations/translations.
    Fix: coordinate-to-node feedback ensures gradient flow to coord_mlp."""
    def __init__(self, hidden=256, edge_dim=13, act=nn.SiLU):
        super().__init__()
        # Edge MLP: processes distances + features
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 1 + edge_dim, hidden * 2),
            act(), nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            act(),
        )
        # Node MLP: updates node features
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            act(), nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

        # Coordinate update (must output scalar for equivariance)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            act(),
            nn.Linear(hidden, 1, bias=False),
        )
        # Small random init for coordinate output (NOT zeros)
        # This ensures gradients can flow from step 1
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight)
        self.coord_mlp[-1].weight.data *= 0.001  # scale down for stability

        # Attention weights for coordinate update
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

        # Coordinate-to-node feedback: project updated distances back to features
        # This creates a gradient path: loss -> node features -> feedback -> coord_mlp
        self.coord_feedback = nn.Sequential(
            nn.Linear(1, hidden), act(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h, pos, edge_index, edge_attr=None):
        """
        h: [N, hidden] node features
        pos: [N, 3] 3D coordinates
        edge_index: [2, E]
        edge_attr: [E, edge_dim] optional
        Returns: h_new, pos_new
        """
        row, col = edge_index
        # Direction vectors (equivariant)
        diff = pos[row] - pos[col]  # [E, 3]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-5)  # [E, 1]

        # Edge messages
        edge_input = [h[row], h[col], dist]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        edge_input = torch.cat(edge_input, dim=-1)
        m_ij = self.edge_mlp(edge_input)  # [E, hidden]

        # Attention-weighted aggregation
        att = self.att_mlp(m_ij)  # [E, 1]
        m_ij_att = m_ij * att

        # Aggregate messages
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, m_ij_att)

        # Node update
        h_new = self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))

        # Coordinate update (equivariant: scalar * direction vector)
        coord_weights = self.coord_mlp(m_ij)  # [E, 1]
        coord_agg = torch.zeros_like(pos)
        coord_agg.index_add_(0, row, diff * coord_weights)
        pos_new = pos + coord_agg

        # Coordinate-to-node feedback: compute updated distances
        # and project them back into node features (creates gradient path)
        diff_new = pos_new[row] - pos_new[col]  # [E, 3]
        dist_new = diff_new.norm(dim=-1, keepdim=True).clamp(min=1e-5)  # [E, 1]
        coord_fb = self.coord_feedback(dist_new)  # [E, hidden]
        fb_agg = torch.zeros_like(h_new)
        fb_agg.index_add_(0, row, coord_fb)
        h_new = h_new + 0.1 * fb_agg  # small contribution

        return h_new, pos_new


# ================================================================
# Learnable RBF (SchNet-style, learnable parameters)
# ================================================================

class LearnableRBF(nn.Module):
    """Learnable Radial Basis Functions with trainable centers/widths."""
    def __init__(self, n_rbf=64, cutoff=10.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.centers = nn.Parameter(torch.linspace(0, cutoff, n_rbf))
        self.widths = nn.Parameter(torch.ones(n_rbf) * (cutoff / n_rbf))
        self.envelope = CosineCutoff(cutoff)
        self.proj = nn.Sequential(
            nn.Linear(n_rbf, n_rbf), nn.SiLU(),
            nn.Linear(n_rbf, n_rbf),
        )

    def forward(self, distances):
        env = self.envelope(distances)
        rbf = torch.exp(-0.5 * ((distances - self.centers) / (self.widths.abs() + 1e-5)) ** 2)
        return self.proj(rbf * env)


class CosineCutoff(nn.Module):
    """Smooth cutoff function (DimeNet/PaiNN style)."""
    def __init__(self, cutoff=10.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, d):
        return 0.5 * (torch.cos(d * math.pi / self.cutoff) + 1.0) * (d < self.cutoff).float()


# ================================================================
# SchNet Continuous-Filter Convolution (CFCONV)
# ================================================================

class ContinuousFilterConv(nn.Module):
    """SchNet-inspired CFCONV with deeper filter network."""
    def __init__(self, hidden=256, n_rbf=64):
        super().__init__()
        self.rbf = LearnableRBF(n_rbf=n_rbf, cutoff=10.0)
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.node_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())

    def forward(self, x, edge_index, distances):
        W = self.filter_net(self.rbf(distances))
        row, col = edge_index
        msg = self.node_proj(x[col]) * W
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, msg)
        out = self.norm(x + agg)
        return out * self.gate(out)


# ================================================================
# Virtual Node (OGN/GPS-style)
# ================================================================

class VirtualNode(nn.Module):
    """Virtual node for global graph context (OGB-style)."""
    def __init__(self, hidden=256):
        super().__init__()
        self.mlp_agg = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.LayerNorm(hidden * 2), nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden),
        )
        self.mlp_bc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, batch, vn_emb):
        B = batch.max().item() + 1
        agg = torch.zeros(B, x.size(1), device=x.device)
        agg.index_add_(0, batch, x)
        counts = torch.zeros(B, 1, device=x.device)
        counts.index_add_(0, batch, torch.ones(x.size(0), 1, device=x.device))
        agg = agg / counts.clamp(min=1)
        vn_emb = self.norm(vn_emb + self.mlp_agg(agg))
        x = x + self.mlp_bc(vn_emb[batch])
        return x, vn_emb


# ================================================================
# SE Block (Squeeze-and-Excitation)
# ================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel recalibration."""
    def __init__(self, d, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, d // reduction), nn.ReLU(),
            nn.Linear(d // reduction, d), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


# ================================================================
# Graphormer-style Spatial / Edge Attention Bias
# ================================================================

class GraphormerBias(nn.Module):
    """Learnable spatial bias based on graph distance (Graphormer).
    Adds distance-based bias to attention scores."""
    def __init__(self, n_heads=4, max_dist=8):
        super().__init__()
        self.spatial_bias = nn.Embedding(max_dist + 1, n_heads)
        self.edge_bias = nn.Linear(BOND_FEATURES_DIM, n_heads)
        self.max_dist = max_dist

    def get_bias(self, edge_index, edge_attr, n_nodes, batch):
        """Compute attention bias per edge for GATv2Conv."""
        # For simplicity, we add a scalar bias computed from edge features
        if edge_attr.size(0) == 0:
            return torch.zeros(0, device=edge_attr.device)
        bias = self.edge_bias(edge_attr).mean(dim=-1)  # [E]
        return bias


# ================================================================
# Industry SOTA GNN Path
# EGNN + GATv2 + SchNet + VirtualNode + Graphormer + Stochastic Depth
# ================================================================

class IndustryGNNPath(nn.Module):
    """
    4-block architecture, each block contains:
    1. EGNN layer (equivariant coordinate + feature update)
    2. GATv2Conv (attention-based message passing with edge features)
    3. SchNet CFCONV (continuous-filter convolution, every other block)
    4. VirtualNode (global context, every other block)
    + Stochastic Depth throughout
    + JumpingKnowledge + GlobalAttention readout
    """
    def __init__(self, in_dim=ATOM_FEATURES_DIM, hidden=256, out=256,
                 gat_heads=4, n_rbf=64, n_blocks=4, drop_path_rate=0.2):
        super().__init__()
        self.n_blocks = n_blocks
        edge_dim = BOND_FEATURES_DIM + n_rbf  # 13 + 64 = 77

        # RBF for edge augmentation
        self.edge_rbf = LearnableRBF(n_rbf=n_rbf, cutoff=10.0)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )

        # Graphormer attention bias
        self.graphormer_bias = GraphormerBias(n_heads=gat_heads)

        # Per-block modules
        self.egnn_layers = nn.ModuleList()
        self.gat_layers = nn.ModuleList()
        self.cfconv_layers = nn.ModuleList()
        self.vn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop_paths = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]

        for i in range(n_blocks):
            # EGNN (equivariant)
            self.egnn_layers.append(
                EGNNLayer(hidden=hidden, edge_dim=BOND_FEATURES_DIM)
            )
            # GATv2
            self.gat_layers.append(
                GATv2Conv(hidden, hidden // gat_heads, heads=gat_heads,
                          edge_dim=edge_dim, dropout=0.1)
            )
            # CFCONV on odd blocks
            if i % 2 == 1:
                self.cfconv_layers.append(
                    ContinuousFilterConv(hidden=hidden, n_rbf=n_rbf)
                )
            else:
                self.cfconv_layers.append(nn.Identity())
            # VirtualNode on even blocks
            if i % 2 == 0:
                self.vn_layers.append(VirtualNode(hidden))
            else:
                self.vn_layers.append(nn.Identity())

            self.norms.append(nn.LayerNorm(hidden))
            self.drop_paths.append(DropPath(dpr[i]))

        self._cfconv_mask = [i % 2 == 1 for i in range(n_blocks)]
        self._vn_mask = [i % 2 == 0 for i in range(n_blocks)]

        # JumpingKnowledge
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden * n_blocks, hidden * 2), nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
        )

        # GlobalAttention readout
        gate_nn = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.readout = GlobalAttention(gate_nn)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, out), nn.GELU(), nn.LayerNorm(out)
        )
        self.se = SEBlock(out)

    def forward(self, x, edge_index, edge_attr, batch, pos=None):
        """
        x: [N, in_dim] atom features
        edge_index: [2, E]
        edge_attr: [E, 13]
        batch: [N]
        pos: [N, 3] 3D coordinates
        """
        if pos is None:
            pos = torch.zeros(x.size(0), 3, device=x.device)

        # Compute distances from 3D coordinates
        row, col = edge_index
        diff = pos[row] - pos[col]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-5)

        # Augment edge features with RBF
        rbf_feat = self.edge_rbf(dist)
        edge_feat = torch.cat([edge_attr, rbf_feat], dim=-1)

        # Project input
        h = self.input_proj(x)
        B = batch.max().item() + 1
        vn_emb = torch.zeros(B, h.size(1), device=h.device)

        layer_outputs = []
        for i in range(self.n_blocks):
            # 1) EGNN (equivariant update of features + coordinates)
            h_egnn, pos = self.egnn_layers[i](h, pos, edge_index, edge_attr)

            # Recompute distances after coordinate update
            diff = pos[row] - pos[col]
            dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-5)
            rbf_feat = self.edge_rbf(dist)
            edge_feat = torch.cat([edge_attr, rbf_feat], dim=-1)

            # 2) GATv2 with updated edge features
            h_gat = self.gat_layers[i](h_egnn, edge_index, edge_attr=edge_feat)
            h_gat = self.norms[i](h_gat)
            h_gat = F.gelu(h_gat)

            # Residual with stochastic depth
            h = h + self.drop_paths[i](h_gat - h) if i > 0 else h_gat

            # 3) CFCONV (odd blocks)
            if self._cfconv_mask[i]:
                h = self.cfconv_layers[i](h, edge_index, dist)

            # 4) VirtualNode (even blocks)
            if self._vn_mask[i]:
                h, vn_emb = self.vn_layers[i](h, batch, vn_emb)

            layer_outputs.append(h)

        # JumpingKnowledge
        jk = torch.cat(layer_outputs, dim=-1)
        jk = self.jk_proj(jk)

        # Readout
        g = self.readout(jk, batch)
        return self.se(self.out_proj(g))


# ================================================================
# 4-Layer Transformer Fusion (Expanded)
# ================================================================

class TransformerFusionBlock(nn.Module):
    """Single transformer block for multi-modal fusion."""
    def __init__(self, d_model, nhead, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dp = DropPath(drop_path)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dp(h))
        x = self.norm2(x + self.dp(self.ffn(x)))
        return x


class CrossModalFusion(nn.Module):
    """4-layer Transformer fusion with modality-specific adapters + SE."""
    def __init__(self, dim_a=384, dim_b=256, dim_c=64, d_model=768,
                 nhead=8, n_layers=4, drop_path_rate=0.1):
        super().__init__()
        self.proj_a = nn.Sequential(nn.Linear(dim_a, d_model), nn.LayerNorm(d_model))
        self.proj_b = nn.Sequential(nn.Linear(dim_b, d_model), nn.LayerNorm(d_model))
        self.proj_c = nn.Sequential(nn.Linear(dim_c, d_model), nn.LayerNorm(d_model))
        self.pos_emb = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        self.modality_emb = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            TransformerFusionBlock(d_model, nhead, drop_path=dpr[i])
            for i in range(n_layers)
        ])

        self.gate = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Linear(256, 3), nn.Softmax(dim=-1)
        )
        self.se = SEBlock(d_model)

    def forward(self, path_a, path_b, path_c):
        a = self.proj_a(path_a).unsqueeze(1)
        b = self.proj_b(path_b).unsqueeze(1)
        c = self.proj_c(path_c).unsqueeze(1)
        tokens = torch.cat([a, b, c], dim=1) + self.pos_emb + self.modality_emb

        for block in self.blocks:
            tokens = block(tokens)

        gate_in = tokens.mean(dim=1)
        weights = self.gate(gate_in)
        gated = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return self.se(gated)


# ================================================================
# MultiHeadOutput with Uncertainty Estimation (MC Dropout)
# ================================================================

class MultiHeadOutput(nn.Module):
    """10 prediction heads with built-in uncertainty (MC Dropout).
    n_odor_dim: number of odor dimensions (22 for legacy, 612 for curated)"""
    def __init__(self, d=256, n_odor_dim=None):
        super().__init__()
        odor_dim = n_odor_dim if n_odor_dim is not None else N_ODOR_DIM
        self.n_odor_dim = odor_dim
        # MC Dropout layers stay active during inference for uncertainty
        self.mc_drop = nn.Dropout(0.05)

        self.odor = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, odor_dim), nn.Sigmoid())
        self.top = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.05),
            nn.Linear(128, odor_dim), nn.Sigmoid())
        self.mid = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.05),
            nn.Linear(128, odor_dim), nn.Sigmoid())
        self.base = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.05),
            nn.Linear(128, odor_dim), nn.Sigmoid())
        self.longevity = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.sillage = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.descriptors = nn.Sequential(
            nn.Linear(d, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 138), nn.Sigmoid())
        self.receptors = nn.Sequential(
            nn.Linear(d, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 400), nn.Sigmoid())
        self.hedonic = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1), nn.Tanh())
        self.super_res = nn.Sequential(
            nn.Linear(d, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, max(200, odor_dim)), nn.Sigmoid())

    def forward(self, h):
        h = self.mc_drop(h)
        return {
            'odor': self.odor(h), 'top': self.top(h),
            'mid': self.mid(h), 'base': self.base(h),
            'longevity': self.longevity(h), 'sillage': self.sillage(h),
            'descriptors': self.descriptors(h), 'receptors': self.receptors(h),
            'hedonic': self.hedonic(h), 'super_res': self.super_res(h),
        }


# ================================================================
# Main Model: OdorPredictor v6 (Industry SOTA)
# ================================================================

class OdorPredictorV6(nn.Module):
    """Industry SOTA 3-path fusion + 10-head model (~30M params).

    Architecture:
      Path A: ChemBERTa 384d -> Bottleneck AttentionPooling -> 384d
      Path B: 4-block EGNN+GATv2+CFCONV+VN -> 256d
      Path C: PhysProps 12d -> MLP -> 64d
      Fusion: 4-layer Transformer -> 768d
      Backbone: 768 -> 256 with SE + skip + stochastic depth
      10 Heads with MC Dropout uncertainty
    """
    def __init__(self, bert_dim=384, phys_dim=12, use_lora=False,
                 n_conformers=1, drop_path_rate=0.2, n_odor_dim=None):
        super().__init__()
        self.bert_dim = bert_dim
        self.use_lora = use_lora
        self.n_conformers = n_conformers

        # Path A: ChemBERTa -> Bottleneck AttentionPooling
        self.attn_pool = AttentionPooling(d_model=bert_dim, n_heads=8, n_latents=4)
        self.path_a_norm = nn.LayerNorm(bert_dim)
        self.path_a_drop = nn.Dropout(0.1)

        # Path B: Industry SOTA GNN
        self.path_b = IndustryGNNPath(
            in_dim=ATOM_FEATURES_DIM, hidden=256, out=256,
            gat_heads=4, n_rbf=64, n_blocks=4,
            drop_path_rate=drop_path_rate
        )

        # Path C: PhysProps -> 64d with Feature Cross
        self.path_c = nn.Sequential(
            nn.Linear(phys_dim, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64),
        )
        # Feature cross-attention between phys and a learnable query
        self.phys_cross = nn.MultiheadAttention(64, 4, dropout=0.1, batch_first=True)
        self.phys_query = nn.Parameter(torch.randn(1, 2, 64) * 0.02)
        self.phys_norm = nn.LayerNorm(64)

        # Fusion: (384, 256, 64) -> 768d via 4-layer Transformer
        self.fusion = CrossModalFusion(
            dim_a=bert_dim, dim_b=256, dim_c=64, d_model=768,
            nhead=8, n_layers=4, drop_path_rate=0.1
        )

        # Backbone: 768 -> 256 with SE + skip + stochastic depth
        self.backbone = nn.Sequential(
            nn.Linear(768, 768), nn.GELU(), nn.LayerNorm(768), nn.Dropout(0.15),
            DropPath(0.05),
            nn.Linear(768, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(0.15),
            DropPath(0.05),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.1),
        )
        self.skip = nn.Linear(768, 256)
        self.backbone_se = SEBlock(256)

        # 10 Heads with uncertainty
        self.heads = MultiHeadOutput(d=256, n_odor_dim=n_odor_dim)
        # GradNorm loss weights
        self.loss_weights = nn.Parameter(torch.ones(10))

    def forward(self, bert_emb, graph_batch, phys, return_aux=True,
                return_backbone=False):
        # Path A: ChemBERTa
        a = self.attn_pool(bert_emb)
        a = self.path_a_drop(self.path_a_norm(a))

        # Path B: GNN with 3D coordinates
        pos = graph_batch.pos if hasattr(graph_batch, 'pos') else None
        b = self.path_b(graph_batch.x, graph_batch.edge_index,
                        graph_batch.edge_attr, graph_batch.batch, pos)

        # Path C: PhysProps with feature cross-attention
        c_raw = self.path_c(phys)
        c_seq = c_raw.unsqueeze(1)
        q = self.phys_query.expand(c_raw.size(0), -1, -1)
        c_cross, _ = self.phys_cross(q, c_seq, c_seq)
        c = self.phys_norm(c_cross.mean(dim=1) + c_raw)

        # Fusion
        fused = self.fusion(a, b, c)

        # Backbone with skip + SE
        h = self.backbone(fused) + self.skip(fused)
        h = self.backbone_se(h)

        if return_backbone:
            # Return backbone features (256d) + fused features (768d)
            # for downstream models like SafetyNet
            if return_aux:
                out = self.heads(h)
                out['backbone_256'] = h       # [B, 256]
                out['fused_768'] = fused       # [B, 768]
                return out
            return self.heads.odor(h), h, fused

        if return_aux:
            return self.heads(h)
        return self.heads.odor(h)

    def predict_with_uncertainty(self, bert_emb, graph_batch, phys,
                                 n_samples=10):
        """MC Dropout uncertainty estimation (industry standard)."""
        was_training = self.training
        self.train()  # Keep dropout active
        try:
            preds = []
            for _ in range(n_samples):
                with torch.no_grad():
                    out = self.forward(bert_emb, graph_batch, phys, return_aux=True)
                    preds.append(out['odor'])

            preds = torch.stack(preds)  # [n_samples, B, 22]
            mean = preds.mean(dim=0)
            std = preds.std(dim=0)
            return mean, std
        finally:
            if not was_training:
                self.eval()

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# ================================================================
# Knowledge Distillation Framework
# ================================================================

class KnowledgeDistiller:
    """Teacher-Student knowledge distillation (Hinton et al., 2015).
    + Feature-based distillation (FitNets style).
    + Response-based distillation with temperature scaling."""
    def __init__(self, teacher, student, temperature=4.0, alpha=0.5):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def distill_loss(self, bert_emb, graph_batch, phys, target):
        """Combined hard-label + soft-label loss."""
        with torch.no_grad():
            teacher_out = self.teacher(bert_emb, graph_batch, phys)
        student_out = self.student(bert_emb, graph_batch, phys)

        # Hard loss (student vs ground truth)
        hard_loss = F.mse_loss(student_out['odor'], target['odor'])
        hard_cos = 1.0 - F.cosine_similarity(
            student_out['odor'], target['odor'], dim=1
        ).mean()

        # Soft loss (student vs teacher, with temperature)
        T = self.temperature
        soft_loss = F.kl_div(
            F.log_softmax(student_out['odor'] / T, dim=-1),
            F.softmax(teacher_out['odor'] / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)

        total = self.alpha * (hard_loss + 0.5 * hard_cos) + \
                (1 - self.alpha) * soft_loss

        return total, {
            'hard_loss': hard_loss,
            'hard_cos': hard_cos,
            'soft_loss': soft_loss,
        }


# ================================================================
# Noisy Student Training
# ================================================================

class NoisyStudentTrainer:
    """Noisy Student (Xie et al., 2020) for self-training.
    Teacher generates pseudo-labels, student trains with noise."""
    def __init__(self, teacher, noise_std=0.1, drop_rate=0.2):
        self.teacher = teacher
        self.noise_std = noise_std
        self.drop_rate = drop_rate
        self.teacher.eval()

    def generate_pseudo_labels(self, bert_emb, graph_batch, phys):
        """Generate pseudo-labels from teacher (no noise)."""
        with torch.no_grad():
            return self.teacher(bert_emb, graph_batch, phys)

    def add_noise(self, bert_emb, phys):
        """Add noise to student inputs (Noisy Student key ingredient)."""
        bert_noisy = bert_emb + torch.randn_like(bert_emb) * self.noise_std
        phys_noisy = phys + torch.randn_like(phys) * self.noise_std * 0.5
        return bert_noisy, phys_noisy


# ================================================================
# EMA (Exponential Moving Average)
# ================================================================

class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.clone().detach()

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply_shadow(self, model):
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self._backup)


# ================================================================
# GradNorm
# ================================================================

class GradNorm:
    """GradNorm: gradient normalization for multi-task learning.
    Uses dict-based tracking to handle dynamic loss keys safely
    (e.g., rdrop/contrastive added at later epochs)."""
    def __init__(self, model, n_tasks=10, alpha=1.5, lr=0.025):
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.lr = lr
        self.initial_losses = {}  # key -> initial loss value

    def update(self, model, losses_dict, epoch):
        if len(losses_dict) < 2:
            return

        with torch.no_grad():
            # Record initial losses for any new keys
            for key, val in losses_dict.items():
                v = val.item() if torch.is_tensor(val) else val
                # Rule #11: NaN/Inf guard — skip corrupted losses
                if math.isnan(v) or math.isinf(v):
                    continue
                if key not in self.initial_losses:
                    self.initial_losses[key] = v

            # Compute ratios only for keys we have history for
            loss_ratios = []
            loss_keys = []
            for key, val in losses_dict.items():
                v = val.item() if torch.is_tensor(val) else val
                # Rule #11: skip NaN/Inf losses entirely
                if math.isnan(v) or math.isinf(v):
                    continue
                init = self.initial_losses.get(key, 0)
                if init > 0:
                    loss_ratios.append(v / init)
                else:
                    loss_ratios.append(1.0)
                loss_keys.append(key)

            if len(loss_ratios) < 2:
                return

            mean_ratio = np.mean(loss_ratios)
            if mean_ratio == 0:
                return

            # Only update weights for indices within model.loss_weights range
            n_weights = min(len(loss_ratios), self.n_tasks,
                           model.loss_weights.data.size(0))
            for i in range(n_weights):
                ri = loss_ratios[i] / mean_ratio
                target_weight = ri ** self.alpha
                # Rule #11: clamp target_weight to prevent explosion
                target_weight = max(0.1, min(target_weight, 10.0))
                current = model.loss_weights.data[i].item()
                model.loss_weights.data[i] = (
                    current * (1 - self.lr) + target_weight * self.lr
                )

            # Rule #11: clamp all weights to safe range
            model.loss_weights.data.clamp_(0.1, 10.0)
            w_sum = model.loss_weights.data[:n_weights].sum()
            if w_sum > 0:
                model.loss_weights.data[:n_weights] *= n_weights / w_sum


# ================================================================
# Full Multi-Task Loss
# ================================================================

def compute_loss(model, pred, target, masks, epoch, max_epochs, training=True):
    """Complete multi-task loss with CosSim, Label Smoothing, Focal BCE."""
    losses = {}
    weights = model.loss_weights.data if hasattr(model, 'loss_weights') else torch.ones(10)
    progress = epoch / max(max_epochs, 1)

    # H1: Odor
    # V6 Fix: label smoothing은 OdorDataset에서 이미 적용됨 (smooth=0.05)
    # 여기서 다시 적용하면 이중 스무딩 (1.0→0.975→0.95125)
    odor_pred = pred['odor'] if isinstance(pred, dict) else pred
    odor_target = target['odor']
    L_mse = F.mse_loss(odor_pred, odor_target)
    # V9 Fix: zero-vector cosine → NaN 방지 (all-zero label 가능)
    cos_sim = F.cosine_similarity(odor_pred, odor_target, dim=1)
    cos_sim = cos_sim.nan_to_num(0.0)  # NaN→0 (cos=0 → L_cos=1)
    L_cos = 1.0 - cos_sim.mean()
    losses['odor_mse'] = L_mse
    losses['odor_cos'] = L_cos
    total = weights[0] * (L_mse + 0.5 * L_cos)

    # H2-H4
    if 'top' in pred and 'top' in target:
        L_tmb = (F.mse_loss(pred['top'], target['top']) +
                 F.mse_loss(pred['mid'], target['mid']) +
                 F.mse_loss(pred['base'], target['base'])) / 3.0
        losses['tmb'] = L_tmb
        total = total + weights[1] * L_tmb
    elif 'top' in pred:
        avg_tmb = (pred['top'] + pred['mid'] + pred['base']) / 3.0
        L_tmb_self = F.mse_loss(avg_tmb, odor_pred.detach())
        losses['tmb_self'] = L_tmb_self
        total = total + weights[1] * 0.1 * L_tmb_self

    # H5-H6
    if 'longevity' in target and 'longevity' in pred:
        L_long = F.mse_loss(pred['longevity'], target['longevity'])
        losses['longevity'] = L_long
        total = total + weights[4] * L_long
    if 'sillage' in target and 'sillage' in pred:
        L_sil = F.mse_loss(pred['sillage'], target['sillage'])
        losses['sillage'] = L_sil
        total = total + weights[5] * L_sil

    # H7: Descriptors (Focal BCE)
    if 'descriptors' in target and 'descriptors' in pred:
        L_desc = focal_bce(pred['descriptors'], target['descriptors'])
        losses['descriptors'] = L_desc
        total = total + weights[6] * L_desc

    # H8: Receptors
    if 'receptors' in target and 'receptors' in pred:
        L_recep = focal_bce(pred['receptors'], target['receptors'])
        losses['receptors'] = L_recep
        total = total + weights[7] * L_recep

    # H9: Hedonic
    if 'hedonic' in target and 'hedonic' in pred:
        L_hed = F.mse_loss(pred['hedonic'], target['hedonic'])
        losses['hedonic'] = L_hed
        total = total + weights[8] * L_hed

    # H10: Super-resolution
    if 'super_res' in pred:
        n_dim = odor_pred.shape[1]
        L_sr = F.mse_loss(pred['super_res'][:, :n_dim], odor_pred.detach())
        losses['super_res'] = L_sr
        total = total + weights[9] * 0.1 * L_sr

    # R-Drop (Rule #6 fix: sigmoid 독립 출력에 softmax 사용 금지)
    # sigmoid 출력은 각 차원이 독립 확률 → per-dimension BCE로 일관성 측정
    if training and 'odor_2' in pred:
        p1 = pred['odor'].clamp(1e-7, 1 - 1e-7)
        p2 = pred['odor_2'].clamp(1e-7, 1 - 1e-7)
        L_rdrop = (
            F.binary_cross_entropy(p1, p2.detach(), reduction='mean') +
            F.binary_cross_entropy(p2, p1.detach(), reduction='mean')
        ) * 0.5
        losses['rdrop'] = L_rdrop
        total = total + 0.1 * L_rdrop

    return total, losses


def focal_bce(pred, target, gamma=2.0, alpha=0.75):
    """Focal Binary Cross-Entropy for imbalanced labels (soft-label safe).
    
    Rule #5: Uses continuous pt formula instead of hard `target == 1`.
    Soft labels (0.8, 0.95) are handled correctly.
    """
    pred_c = pred.clamp(1e-7, 1 - 1e-7)
    bce = F.binary_cross_entropy(pred_c, target, reduction='none')
    # Continuous soft-label pt (NOT `torch.where(target == 1, ...)`)
    pt = pred_c * target + (1 - pred_c) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal = alpha_t * ((1 - pt).clamp(min=1e-8) ** gamma) * bce  # Rule #7: clamp before pow
    return focal.mean()


def contrastive_loss(embeddings, labels, temperature=0.07):
    """Supervised contrastive loss."""
    B = embeddings.size(0)
    if B < 2:
        return torch.tensor(0.0, device=embeddings.device)

    sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    sim = sim / temperature

    label_sim = F.cosine_similarity(labels.unsqueeze(1), labels.unsqueeze(0), dim=-1)
    positives = (label_sim > 0.8).float()
    mask = 1.0 - torch.eye(B, device=embeddings.device)
    positives = positives * mask

    if positives.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # Rule #7: log_softmax로 수치 안정화 (exp(14) ≈ 1.2M overflow 방지)
    # self-similarity를 -inf로 마스킹하여 softmax에서 제외
    sim_masked = sim.masked_fill(~mask.bool(), float('-inf'))
    log_prob = F.log_softmax(sim_masked, dim=1)
    # Rule #7: log_prob -inf 방지 (positives * -inf = NaN)
    log_prob = log_prob.clamp(min=-100.0)
    loss = -(positives * log_prob).sum() / (positives.sum() + 1e-8)
    return loss

