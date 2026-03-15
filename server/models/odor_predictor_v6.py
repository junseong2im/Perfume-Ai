"""OdorPredictor v6 — 3-Path Multi-Modal Fusion + 10-Head Architecture
=====================================================================
설계안 v3 Model 1 구현

Path A: ChemBERTa (77M frozen) + LoRA r=16       → [B, 384]
Path B: GATConv + 3D RBF + Chirality Features     → [B, 128]
Path C: Physical Properties + Concentration (12d)  → [B, 32]

Fusion: Cross-Attention (3 tokens × 512d) → Gated Sum → [B, 512]
Backbone: 512 → 256 → 128 (residual skip)

10 Heads:
  H1: odor_vector [22]     H2-4: top/mid/base [22×3]
  H5-6: longevity/sillage  H7: descriptors [138]
  H8: receptors [400]      H9: hedonic [-1,+1]
  H10: super_res [200]

Training features:
  - GradNorm for automatic loss balancing
  - Progressive Unfreezing (H1 → all)
  - Scaffold Split
  - Curriculum Learning, Mixup, R-Drop
  - EMA, SWA, Deep Ensemble (5 seeds)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, Set2Set
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

# 22d odor space (v5 호환)
ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]
N_ODOR_DIM = 22


# ================================================================
# Graph Featurization (v6): 47d atoms + 13d bonds + 3D + Chirality
# ================================================================

ATOM_FEATURES_DIM = 47

def atom_features_v6(atom):
    """Atom → 47d feature vector (v5 9d + 38d 추가)"""
    # Atomic number one-hot (top 15 + other)
    atom_num = atom.GetAtomicNum()
    common_atoms = [6, 7, 8, 9, 15, 16, 17, 35, 53, 1, 5, 14, 34, 52, 3]
    atom_one_hot = [1.0 if atom_num == a else 0.0 for a in common_atoms]
    atom_one_hot.append(1.0 if atom_num not in common_atoms else 0.0)  # other

    # Degree one-hot (0-5)
    degree = min(atom.GetDegree(), 5)
    degree_oh = [1.0 if degree == d else 0.0 for d in range(6)]

    # Hybridization one-hot (sp, sp2, sp3, sp3d, sp3d2)
    hyb = atom.GetHybridization()
    hyb_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hyb_oh = [1.0 if hyb == h else 0.0 for h in hyb_types]

    # Chirality one-hot (none, R, S, other)
    chiral = atom.GetChiralTag()
    chiral_oh = [
        1.0 if chiral == Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0,
        1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0,
        1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0,
        1.0 if chiral == Chem.rdchem.ChiralType.CHI_OTHER else 0.0,
    ]

    # Ring membership (3,4,5,6)
    ring_info = atom.GetOwningMol().GetRingInfo()
    ring_sizes = [
        1.0 if ring_info.IsAtomInRingOfSize(atom.GetIdx(), s) else 0.0
        for s in [3, 4, 5, 6]
    ]

    # Scalar features
    scalars = [
        atom.GetFormalCharge() / 2.0,
        atom.GetTotalNumHs() / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetMass() / 127.0,  # Normalize by Iodine mass
    ]

    return atom_one_hot + degree_oh + hyb_oh + chiral_oh + ring_sizes + scalars
    # 16 + 6 + 5 + 4 + 4 + 5 = 40... let me pad to 47
    # Actually: 16+6+5+4+4+5 = 40. Need 47 → add more features


def atom_features_v6_full(atom):
    """47d atom features"""
    feats = atom_features_v6(atom)

    # Additional 7 features to reach 47
    mol = atom.GetOwningMol()
    feats.extend([
        atom.GetNumRadicalElectrons() / 2.0,
        float(atom.GetNoImplicit()),
        len(atom.GetNeighbors()) / 4.0,  # Explicit degree
        Descriptors.MolLogP(mol) / 5.0 if mol.GetNumAtoms() < 100 else 0.0,  # Molecule-level
        float(atom.GetAtomicNum() in [7, 8, 16]),  # Is heteroatom
        float(atom.GetAtomicNum() in [9, 17, 35, 53]),  # Is halogen
        float(atom.GetAtomicNum() in [15, 16, 34]),  # Is chalcogen/P
    ])
    return feats  # 47d


BOND_FEATURES_DIM = 13

def bond_features_v6(bond):
    """Bond → 13d feature vector"""
    bt = bond.GetBondType()
    # Bond type one-hot (single, double, triple, aromatic)
    bt_oh = [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
    ]

    # Stereo one-hot (none, Z, E, any)
    stereo = bond.GetStereo()
    stereo_oh = [
        float(stereo == Chem.rdchem.BondStereo.STEREONONE),
        float(stereo == Chem.rdchem.BondStereo.STEREOZ),
        float(stereo == Chem.rdchem.BondStereo.STEREOE),
        float(stereo == Chem.rdchem.BondStereo.STEREOANY),
    ]

    return bt_oh + stereo_oh + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        bond.GetBondTypeAsDouble() / 3.0,
        float(bond.GetIsAromatic()),
        0.0,  # 3D bond length placeholder (filled if 3D coords available)
    ]


class RadialBasisFunction(nn.Module):
    """RBF distance encoding for 3D atomic distances"""
    def __init__(self, n_rbf=16, cutoff=10.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer('centers', centers)
        self.width = (cutoff / n_rbf) * 0.5

    def forward(self, distances):
        """distances: [n_edges] → [n_edges, n_rbf]"""
        d = distances.unsqueeze(-1)  # [E, 1]
        return torch.exp(-((d - self.centers) ** 2) / (2 * self.width ** 2))


def smiles_to_graph_v6(smiles, device='cpu', compute_3d=True):
    """SMILES → PyG Data (v6: 47d atoms + 13d bonds + 3D RBF)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # 3D coordinates
    positions = None
    if compute_3d:
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=200)
            conf = mol.GetConformer()
            positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        except:
            positions = None

    mol = Chem.RemoveHs(mol)

    # Atom features
    atoms = []
    for atom in mol.GetAtoms():
        atoms.append(atom_features_v6_full(atom))
    x = torch.tensor(atoms, dtype=torch.float32)

    # Edge index + features
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features_v6(bond)

        # Add 3D bond length if available
        if positions is not None and i < len(positions) and j < len(positions):
            dist = np.linalg.norm(positions[i] - positions[j])
            bf[-1] = dist / 5.0  # Normalize

        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if not edge_indices:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURES_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data.to(device)


# ================================================================
# Path A: ChemBERTa + LoRA (적용 시 transformers, peft 필요)
# ================================================================

class AttentionPooling(nn.Module):
    """학습 가능한 Attention Pooling (mean pooling 대체)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states, attention_mask=None):
        """[B, seq_len, D] → [B, D]"""
        weights = self.attn(hidden_states).squeeze(-1)  # [B, seq_len]
        if attention_mask is not None:
            weights = weights.masked_fill(~attention_mask.bool(), -1e9)
        weights = F.softmax(weights, dim=-1)
        return (weights.unsqueeze(-1) * hidden_states).sum(dim=1)


class ChemBERTaEncoder(nn.Module):
    """Path A: ChemBERTa 384d (LoRA fine-tune 또는 frozen)"""
    def __init__(self, bert_dim=384, use_lora=False):
        super().__init__()
        self.bert_dim = bert_dim
        self.use_lora = use_lora
        self.pooling = AttentionPooling(bert_dim)

        # Learnable projection so cached embeddings also go through the computation graph
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, bert_dim),
            nn.GELU(),
            nn.LayerNorm(bert_dim),
        )

        # LoRA는 학습 시에만 적용 (peft 라이브러리 필요)
        # 추론 시에는 캐시된 임베딩 사용
        self._bert_model = None

    def load_bert(self, device='cpu'):
        """ChemBERTa 모델 로딩 (학습 시에만)"""
        try:
            from transformers import AutoModel, AutoTokenizer
            self._bert_model = AutoModel.from_pretrained(
                "DeepChem/ChemBERTa-77M-MTR"
            ).to(device)
            self._tokenizer = AutoTokenizer.from_pretrained(
                "DeepChem/ChemBERTa-77M-MTR"
            )
            # Freeze base
            for param in self._bert_model.parameters():
                param.requires_grad = False

            if self.use_lora:
                try:
                    from peft import LoraConfig, get_peft_model
                    lora_config = LoraConfig(
                        r=16, lora_alpha=32,
                        target_modules=["query", "value"],
                        lora_dropout=0.1,
                        modules_to_save=["pooler"],
                    )
                    self._bert_model = get_peft_model(self._bert_model, lora_config)
                    n_train = sum(p.numel() for p in self._bert_model.parameters() if p.requires_grad)
                    print(f"[ChemBERTa] LoRA enabled: {n_train:,} trainable params")
                except ImportError:
                    print("[ChemBERTa] peft not available, using frozen embeddings")
        except ImportError:
            print("[ChemBERTa] transformers not available, use cached embeddings")

    def forward_from_smiles(self, smiles_list, device='cpu'):
        """SMILES list → [B, 384] (실시간 인코딩, 학습 시)"""
        if self._bert_model is None:
            raise RuntimeError("Call load_bert() first")
        tokens = self._tokenizer(
            smiles_list, padding=True, truncation=True,
            max_length=256, return_tensors='pt'
        ).to(device)
        outputs = self._bert_model(**tokens)
        return self.pooling(outputs.last_hidden_state, tokens['attention_mask'])

    def forward(self, bert_embeddings):
        """캐시된 임베딩 [B, 384] → 투영 → [B, 384]
        
        Previously returned `bert_embeddings` unchanged (Ghost LoRA — dead code).
        Now passes through a learnable projection so weights participate in backprop.
        """
        return self.proj(bert_embeddings)


# ================================================================
# Path B: Molecular Graph GNN (3D + Chirality)
# ================================================================

class MolecularGNNv6(nn.Module):
    """GATConv × 3 + JumpingKnowledge + Set2Set (3D RBF 포함)"""
    def __init__(self, atom_dim=47, bond_dim=13, hidden=128, heads=4, n_rbf=16, dropout=0.15):
        super().__init__()
        self.atom_encoder = nn.Linear(atom_dim, hidden)
        self.bond_encoder = nn.Linear(bond_dim, hidden)
        self.rbf = RadialBasisFunction(n_rbf=n_rbf)

        # Edge dim = bond_encoder_out + rbf
        edge_dim = hidden + n_rbf

        self.conv1 = GATConv(hidden, hidden, heads=heads, concat=False,
                             edge_dim=edge_dim, dropout=dropout)
        self.norm1 = nn.BatchNorm1d(hidden)
        self.conv2 = GATConv(hidden, hidden, heads=heads, concat=False,
                             edge_dim=edge_dim, dropout=dropout)
        self.norm2 = nn.BatchNorm1d(hidden)
        self.conv3 = GATConv(hidden, hidden, heads=heads, concat=False,
                             edge_dim=edge_dim, dropout=dropout)
        self.norm3 = nn.BatchNorm1d(hidden)

        # JumpingKnowledge: concat 3 layers → project
        self.jk_proj = nn.Linear(hidden * 3, hidden)
        self.jk_norm = nn.LayerNorm(hidden)

        # Set2Set readout
        self.set2set = Set2Set(hidden, processing_steps=3)
        self.readout_proj = nn.Linear(hidden * 2, hidden)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """PyG Data → [B, 128]"""
        x = self.atom_encoder(data.x)
        batch = data.batch if hasattr(data, 'batch') else \
            torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Edge features: bond encoding + RBF of 3D distances
        if data.edge_attr is not None and data.edge_attr.size(0) > 0:
            bond_feat = self.bond_encoder(data.edge_attr)
            # Extract last column as 3D distance for RBF
            distances = data.edge_attr[:, -1]  # bond length
            rbf_feat = self.rbf(distances)
            edge_feat = torch.cat([bond_feat, rbf_feat], dim=-1)
        else:
            edge_feat = None

        # GATConv layers with residual
        h1 = self.conv1(x, data.edge_index, edge_attr=edge_feat)
        h1 = self.norm1(h1)
        h1 = F.elu(h1) + x  # Residual
        h1 = self.dropout(h1)

        h2 = self.conv2(h1, data.edge_index, edge_attr=edge_feat)
        h2 = self.norm2(h2)
        h2 = F.elu(h2) + h1  # Residual
        h2 = self.dropout(h2)

        h3 = self.conv3(h2, data.edge_index, edge_attr=edge_feat)
        h3 = self.norm3(h3)
        h3 = F.elu(h3) + h2  # Residual

        # JumpingKnowledge
        jk = torch.cat([h1, h2, h3], dim=-1)
        jk = self.jk_proj(jk)
        jk = self.jk_norm(jk)

        # Set2Set pooling
        pooled = self.set2set(jk, batch)
        return self.readout_proj(pooled)  # [B, 128]


# ================================================================
# Path C: Physical Properties + Concentration
# ================================================================

class PhysPropEncoder(nn.Module):
    """12d 물리속성 → 32d (MW, LogP, VP, BP, TPSA, HBD, HBA, rot, rings, aro_rings, fsp3, concentration)"""
    def __init__(self, in_dim=12, hidden=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, out_dim), nn.GELU(), nn.LayerNorm(out_dim),
        )
        # z-score normalization params (training set 기준, 학습 후 세팅)
        self.register_buffer('mean', torch.zeros(in_dim))
        self.register_buffer('std', torch.ones(in_dim))

    def set_normalization(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)

    def forward(self, props):
        """[B, 12] → [B, 32]"""
        normalized = (props - self.mean.to(props.device)) / self.std.to(props.device)
        return self.net(normalized)


# ================================================================
# Cross-Modal Fusion (Cross-Attention + Gated Sum)
# ================================================================

class CrossModalFusion(nn.Module):
    """3 경로 → Cross-Attention → Gated 가중합 → [B, 512]
    
    Learnable modality embeddings added so attention can distinguish
    which token is text (ChemBERTa), graph (GNN), or physics.
    """
    def __init__(self, dim_a=384, dim_b=128, dim_c=32, fused_dim=512, nhead=8):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, fused_dim)
        self.proj_b = nn.Linear(dim_b, fused_dim)
        self.proj_c = nn.Linear(dim_c, fused_dim)

        # ★ Learnable modality positional embeddings (3 modalities)
        self.modality_embeddings = nn.Parameter(
            torch.randn(1, 3, fused_dim) * 0.02
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=nhead, dropout=0.1, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(fused_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(fused_dim, fused_dim),
        )
        self.ffn_norm = nn.LayerNorm(fused_dim)

        # Gated path weighting
        self.gate = nn.Linear(fused_dim, 3)

    def forward(self, feat_a, feat_b, feat_c):
        """[B, 384], [B, 128], [B, 32] → [B, 512]"""
        pa = self.proj_a(feat_a)  # [B, 512]
        pb = self.proj_b(feat_b)
        pc = self.proj_c(feat_c)

        # Stack as sequence: [B, 3, 512]
        tokens = torch.stack([pa, pb, pc], dim=1)

        # ★ Add modality positional embeddings so attention knows which is text/graph/phys
        tokens = tokens + self.modality_embeddings

        # Self-attention over 3 tokens
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.layer_norm(tokens + attn_out)
        tokens = self.ffn_norm(tokens + self.ffn(tokens))

        # Gated weighted sum
        gate_weights = F.softmax(self.gate(tokens.mean(dim=1)), dim=-1)  # [B, 3]
        fused = (gate_weights[:, 0:1] * tokens[:, 0] +
                 gate_weights[:, 1:2] * tokens[:, 1] +
                 gate_weights[:, 2:3] * tokens[:, 2])

        return fused  # [B, 512]


# ================================================================
# Shared Backbone + 10 Heads
# ================================================================

class TaskHead(nn.Module):
    """단일 태스크 헤드: 128 → out_dim"""
    def __init__(self, in_dim, out_dim, activation='sigmoid'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(64, out_dim),
        )
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.net(x))


class OdorPredictorV6(nn.Module):
    """Full 3-path multi-modal predictor with 10 heads

    Total params: ~87M (77M frozen ChemBERTa + ~10M trainable)
    """
    def __init__(self, bert_dim=384, use_lora=False):
        super().__init__()

        # === 3 Encoding Paths ===
        self.path_a = ChemBERTaEncoder(bert_dim, use_lora=use_lora)
        self.path_b = MolecularGNNv6(atom_dim=ATOM_FEATURES_DIM, bond_dim=BOND_FEATURES_DIM)
        self.path_c = PhysPropEncoder(in_dim=12, out_dim=32)

        # === Fusion ===
        self.fusion = CrossModalFusion(dim_a=bert_dim, dim_b=128, dim_c=32, fused_dim=512)

        # === Shared Backbone ===
        self.backbone = nn.Sequential(
            nn.Linear(512, 512), nn.GELU(), nn.LayerNorm(512), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.1),
        )
        self.skip = nn.Linear(512, 128)  # Residual connection

        # === 10 Task Heads ===
        self.heads = nn.ModuleDict({
            'odor':        TaskHead(128,  N_ODOR_DIM, 'sigmoid'),    # H1
            'top':         TaskHead(128,  N_ODOR_DIM, 'sigmoid'),    # H2
            'mid':         TaskHead(128,  N_ODOR_DIM, 'sigmoid'),    # H3
            'base':        TaskHead(128,  N_ODOR_DIM, 'sigmoid'),    # H4
            'longevity':   TaskHead(128,  1,          'sigmoid'),    # H5
            'sillage':     TaskHead(128,  1,          'sigmoid'),    # H6
            'descriptors': TaskHead(128, 138,         'sigmoid'),    # H7
            'receptors':   TaskHead(128, 400,         'sigmoid'),    # H8
            'hedonic':     TaskHead(128,  1,          'tanh'),       # H9
            'super_res':   TaskHead(128, 200,         'sigmoid'),    # H10
        })

        # GradNorm 가중치 (학습 가능)
        self.loss_weights = nn.Parameter(torch.ones(10))

    def forward(self, bert_emb, graph_data, phys_props, return_aux=True):
        """
        Args:
            bert_emb: [B, 384] ChemBERTa embedding (cached or live)
            graph_data: PyG Batch of molecular graphs
            phys_props: [B, 12] physical properties + concentration

        Returns:
            dict of head_name → tensor predictions
        """
        # Encode 3 paths
        feat_a = self.path_a(bert_emb)       # [B, 384]
        feat_b = self.path_b(graph_data)     # [B, 128]
        feat_c = self.path_c(phys_props)     # [B, 32]

        # Fusion
        fused = self.fusion(feat_a, feat_b, feat_c)  # [B, 512]

        # Backbone + skip
        backbone_out = self.backbone(fused) + self.skip(fused)  # [B, 128]

        if not return_aux:
            return self.heads['odor'](backbone_out)

        # All heads
        return {name: head(backbone_out) for name, head in self.heads.items()}

    def get_backbone_features(self, bert_emb, graph_data, phys_props):
        """512d fusion features (SafetyNet backbone용)"""
        feat_a = self.path_a(bert_emb)
        feat_b = self.path_b(graph_data)
        feat_c = self.path_c(phys_props)
        return self.fusion(feat_a, feat_b, feat_c)

    def get_loss_weights(self):
        """GradNorm 정규화된 loss 가중치"""
        with torch.no_grad():
            normalized = F.softmax(self.loss_weights, dim=0) * 10
        return normalized

    def count_parameters(self):
        """학습 가능한 파라미터 수"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


# ================================================================
# EMA (Exponential Moving Average)
# ================================================================

class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + \
                                    (1 - self.decay) * param.data

    def apply_shadow(self, model):
        """EMA weights 적용 (추론 시)"""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """원래 weights 복원 (학습 계속 시)"""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ================================================================
# GradNorm — 자동 loss 밸런싱
# ================================================================

class GradNorm:
    """Multi-task loss balancing via gradient normalization"""
    def __init__(self, model, n_tasks=10, alpha=1.5, lr=0.025):
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.lr = lr
        self.initial_losses = None
        self.model = model

    def update_weights(self, loss_weights, losses, shared_params):
        """
        Args:
            loss_weights: nn.Parameter [n_tasks] (model.loss_weights)
            losses: list of n_tasks scalar losses
            shared_params: last shared layer parameters
        """
        if self.initial_losses is None:
            self.initial_losses = [l.item() for l in losses]

        # Normalized inverse training rates
        with torch.no_grad():
            relative_losses = torch.tensor([
                l.item() / max(l0, 1e-8) for l, l0 in zip(losses, self.initial_losses)
            ])
            avg_rel = relative_losses.mean()
            target_norms = (relative_losses / avg_rel) ** self.alpha

        # Compute gradient norms for each task
        norms = []
        for i, loss in enumerate(losses):
            if loss.requires_grad:
                grads = torch.autograd.grad(
                    loss, shared_params, retain_graph=True, allow_unused=True
                )
                grad_norm = sum(g.norm() for g in grads if g is not None)
                norms.append(grad_norm)
            else:
                norms.append(torch.tensor(0.0))

        if not norms:
            return

        avg_norm = sum(n.item() for n in norms) / len(norms)

        # Update weights
        with torch.no_grad():
            for i in range(min(len(norms), self.n_tasks)):
                target_norm = avg_norm * target_norms[i].item()
                actual_norm = norms[i].item()
                if actual_norm > 0:
                    ratio = target_norm / actual_norm
                    loss_weights.data[i] *= (ratio ** self.lr)

            # Renormalize
            loss_weights.data = loss_weights.data / loss_weights.data.sum() * self.n_tasks


# ================================================================
# Loss Functions
# ================================================================

def focal_bce(pred, target, gamma=2.0, alpha=0.75):
    """Focal Binary Cross Entropy for imbalanced multi-label (soft-label safe)"""
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    # Continuous soft-label pt (avoids hard `target == 1` comparison)
    pt = pred * target + (1 - pred) * (1 - target)
    focal = ((1 - pt) ** gamma) * bce
    if alpha is not None:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal = alpha_t * focal
    return focal.mean()


def contrastive_loss(embeddings, synergy_pairs, clash_pairs, margin=0.3):
    """Compatibility-based contrastive loss"""
    loss = 0.0
    n = 0
    for i, j in synergy_pairs:
        if i < len(embeddings) and j < len(embeddings):
            loss += (1 - F.cosine_similarity(
                embeddings[i:i+1], embeddings[j:j+1]
            )).mean()
            n += 1
    for i, j in clash_pairs:
        if i < len(embeddings) and j < len(embeddings):
            sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1])
            loss += F.relu(sim - margin).mean()
            n += 1
    return loss / max(n, 1)


def compute_loss(model, pred, target, masks, epoch, max_epochs, training=False):
    """v6 전체 loss 계산 (GradNorm 가중치 적용)"""
    weights = model.get_loss_weights()
    losses = []

    # H1: 향 벡터 MSE + CosSim + Label Smoothing
    target_smooth = target['odor'] * 0.95 + 0.025
    L_mse = F.mse_loss(pred['odor'], target_smooth)
    L_cos = 1 - F.cosine_similarity(pred['odor'], target['odor']).mean()
    losses.append(L_mse)
    losses.append(L_cos)

    # H2-4: Top/Mid/Base
    head_names = ['top', 'mid', 'base']
    tmb_losses = []
    for name in head_names:
        if name in target and masks.get('tmb') is not None and masks['tmb'].sum() > 0:
            m = masks['tmb'].unsqueeze(1)
            l = (F.mse_loss(pred[name], target[name], reduction='none') * m).sum() / (m.sum() * N_ODOR_DIM)
            tmb_losses.append(l)
    L_tmb = sum(tmb_losses) / max(len(tmb_losses), 1) if tmb_losses else torch.tensor(0.0)
    losses.append(L_tmb)

    # H5-6: Longevity/Sillage
    prop_losses = []
    for name in ['longevity', 'sillage']:
        if name in target and masks.get('props') is not None and masks['props'].sum() > 0:
            m = masks['props'].unsqueeze(1)
            l = (F.mse_loss(pred[name], target[name], reduction='none') * m).sum() / (m.sum())
            prop_losses.append(l)
    L_props = sum(prop_losses) / max(len(prop_losses), 1) if prop_losses else torch.tensor(0.0)
    losses.append(L_props)

    # H7: Descriptors (Focal BCE)
    if 'descriptors' in target and masks.get('desc') is not None and masks['desc'].sum() > 0:
        L_desc = focal_bce(pred['descriptors'], target['descriptors'], gamma=2.0)
    else:
        L_desc = torch.tensor(0.0)
    losses.append(L_desc)

    # H8: Receptors (Focal BCE)
    if 'receptors' in target and masks.get('recep') is not None and masks['recep'].sum() > 0:
        L_recep = focal_bce(pred['receptors'], target['receptors'], gamma=2.0)
    else:
        L_recep = torch.tensor(0.0)
    losses.append(L_recep)

    # H9: Hedonic
    if 'hedonic' in target and masks.get('hedonic') is not None and masks['hedonic'].sum() > 0:
        L_hedonic = F.mse_loss(pred['hedonic'], target['hedonic'])
    else:
        L_hedonic = torch.tensor(0.0)
    losses.append(L_hedonic)

    # H10: Super Resolution
    if 'super_res' in target and masks.get('superres') is not None and masks['superres'].sum() > 0:
        L_superres = F.mse_loss(pred['super_res'], target['super_res'])
    else:
        L_superres = torch.tensor(0.0)
    losses.append(L_superres)

    # R-Drop (if training)
    L_rdrop = torch.tensor(0.0)
    losses.append(L_rdrop)

    # Contrastive placeholder
    L_contra = torch.tensor(0.0)
    losses.append(L_contra)

    # Weighted sum
    total = sum(w * l for w, l in zip(weights, losses) if l.requires_grad)
    if not isinstance(total, torch.Tensor) or not total.requires_grad:
        total = L_mse + 0.5 * L_cos  # Fallback

    return total, losses


# ================================================================
# Utility: extract physical properties from SMILES
# ================================================================

def extract_phys_props(smiles, concentration_pct=5.0):
    """SMILES → 12d physical property vector"""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        return np.zeros(12, dtype=np.float32)

    try:
        props = np.array([
            Descriptors.MolWt(mol),                    # MW
            Descriptors.MolLogP(mol),                  # LogP
            0.0,  # VP (from ThermodynamicsEngine or PubChem)
            0.0,  # BP (from PubChem)
            Descriptors.TPSA(mol),                     # TPSA
            float(Descriptors.NumHDonors(mol)),        # HBD
            float(Descriptors.NumHAcceptors(mol)),     # HBA
            float(Descriptors.NumRotatableBonds(mol)), # Rotatable
            float(Descriptors.RingCount(mol)),         # Rings
            float(Descriptors.NumAromaticRings(mol)),  # Aromatic rings
            Descriptors.FractionCSP3(mol),             # fsp3
            np.log1p(concentration_pct),               # Weber-Fechner concentration
        ], dtype=np.float32)
    except:
        props = np.zeros(12, dtype=np.float32)

    return props


# ================================================================
# Quick Test
# ================================================================

if __name__ == '__main__':
    print("=== OdorPredictor v6 Architecture Test ===")

    model = OdorPredictorV6(bert_dim=384, use_lora=False)
    params = model.count_parameters()
    print(f"Total params: {params['total']:,}")
    print(f"Trainable params: {params['trainable']:,}")

    # Dummy forward pass
    B = 4
    bert_emb = torch.randn(B, 384)
    phys = torch.randn(B, 12)

    # Create dummy graph batch
    graphs = []
    for _ in range(B):
        n_atoms = np.random.randint(5, 30)
        x = torch.randn(n_atoms, ATOM_FEATURES_DIM)
        n_edges = n_atoms * 2
        edge_index = torch.randint(0, n_atoms, (2, n_edges))
        edge_attr = torch.randn(n_edges, BOND_FEATURES_DIM)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    batch = Batch.from_data_list(graphs)

    preds = model(bert_emb, batch, phys)
    print("\nHead outputs:")
    for name, tensor in preds.items():
        print(f"  {name}: {tensor.shape}")

    print("\nGradNorm weights:", model.get_loss_weights().tolist())
    print("\n✅ Architecture test passed!")
