# competitive_binding.py — Pillar 2: nn.MultiheadAttention (PyTorch CUDA)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MoleculeProjector(nn.Module):
    """분자 벡터 → 임베딩 공간"""
    def __init__(self, input_dim=29, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)


class CompetitiveBinding:
    """nn.MultiheadAttention 기반 수용체 결합 시뮬레이션"""

    RECEPTOR_PROFILES = [
        {'name': 'OR1: 플로럴', 'sensitivity': ['floral', 'rose', 'jasmine', 'violet']},
        {'name': 'OR2: 시트러스', 'sensitivity': ['citrus', 'lemon', 'orange', 'bergamot']},
        {'name': 'OR3: 우디', 'sensitivity': ['woody', 'cedar', 'sandalwood', 'vetiver']},
        {'name': 'OR4: 스파이시', 'sensitivity': ['spicy', 'clove', 'cinnamon', 'pepper']},
        {'name': 'OR5: 스위트', 'sensitivity': ['sweet', 'vanilla', 'caramel', 'honey']},
        {'name': 'OR6: 프레시', 'sensitivity': ['fresh', 'clean', 'aquatic', 'ozonic']},
        {'name': 'OR7: 그린', 'sensitivity': ['green', 'leafy', 'herbal', 'grassy']},
        {'name': 'OR8: 머스크', 'sensitivity': ['musk', 'amber', 'warm', 'powdery']},
    ]

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.embed_dim = 32
        self.num_heads = 4
        self.num_receptors = 8

        # PyTorch 네이티브 Multi-Head Attention
        self.self_attention = None
        self.cross_attention = None
        self.projector = None
        self.receptor_queries = None
        self.binding_model = None
        self.trained = False

    def build_model(self):
        self.projector = MoleculeProjector(29, self.embed_dim).to(self.device)

        # ★ nn.MultiheadAttention — PyTorch 네이티브 (cuDNN 가속)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        ).to(self.device)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        ).to(self.device)

        # 학습 가능한 수용체 쿼리 벡터
        self.receptor_queries = nn.Parameter(
            torch.randn(self.num_receptors, 1, self.embed_dim, device=self.device) * 0.1
        )

        self.binding_model = nn.Sequential(
            nn.Linear(self.embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_receptors),
            nn.Sigmoid()
        ).to(self.device)

        print(f"[CompetitiveBinding] nn.MultiheadAttention on {self.device}")

    def train(self, molecular_engine, epochs=30, on_progress=None):
        if self.projector is None:
            self.build_model()

        molecules = molecular_engine.get_all()
        if not molecules:
            return

        inputs, targets = [], []
        for mol in molecules:
            vec = self._mol_to_vec(mol)
            inputs.append(vec)
            activation = [
                min(1.0, len(set(r['sensitivity']) & set(mol.get('odor_labels', []))) / len(r['sensitivity']))
                for r in self.RECEPTOR_PROFILES
            ]
            targets.append(activation)

        X = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)
        Y = torch.tensor(np.array(targets), dtype=torch.float32).to(self.device)

        # ★ Fix: include self_attention + cross_attention in optimizer
        params = (list(self.projector.parameters()) +
                  list(self.binding_model.parameters()) +
                  list(self.self_attention.parameters()) +
                  list(self.cross_attention.parameters()) +
                  [self.receptor_queries])
        optimizer = torch.optim.Adam(params, lr=0.001)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        for epoch in range(epochs):
            self.projector.train()
            self.binding_model.train()
            self.self_attention.train()
            epoch_loss = 0.0

            for bx, by in loader:
                optimizer.zero_grad()
                # ★ Fix: per-molecule prediction instead of global mean collapse
                embedded = self.projector(bx)  # [B, 32]
                # Self-attention over batch as sequence: [1, B, 32]
                x_seq = embedded.unsqueeze(0)  # [1, B, 32]
                attended, _ = self.self_attention(x_seq, x_seq, x_seq)
                attended = attended.squeeze(0)  # [B, 32]
                out = self.binding_model(attended)  # [B, 8]
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if on_progress:
                on_progress(epoch + 1, epochs, epoch_loss / max(len(loader), 1))

        self.trained = True

    @torch.no_grad()
    def simulate_binding(self, molecules, molecular_engine):
        if self.projector is None:
            self.build_model()

        N = len(molecules)
        if N == 0:
            return None

        # 1. 분자 임베딩
        vecs = [self._mol_to_vec(mol) for mol in molecules]
        X = torch.tensor(np.array(vecs), dtype=torch.float32).to(self.device)
        X_embed = self.projector(X)  # [N, 32]
        X_seq = X_embed.unsqueeze(0)  # [1, N, 32]

        # 2. Self-Attention (분자 간 상호작용)
        self.self_attention.eval()
        attended, self_attn_weights = self.self_attention(X_seq, X_seq, X_seq)
        # self_attn_weights: [1, N, N]

        # 3. 수용체별 Cross-Attention
        self.cross_attention.eval()
        receptor_results = []
        for r in range(self.num_receptors):
            query = self.receptor_queries[r].unsqueeze(0)  # [1, 1, 32]
            _, cross_weights = self.cross_attention(query, attended, attended)
            # cross_weights: [1, 1, N]
            bindings = cross_weights.squeeze().cpu().numpy()
            bindings = bindings[:N] if len(bindings.shape) > 0 else [float(bindings)]

            activation = float(attended.abs().mean())
            receptor_results.append({
                'receptorName': self.RECEPTOR_PROFILES[r]['name'],
                'totalActivation': min(1.0, activation),
                'moleculeBindings': [float(b) for b in bindings],
                'dominantMolecule': int(np.argmax(bindings)) if len(bindings) > 0 else 0
            })

        # 4. 상호작용 분석
        attn = self_attn_weights.squeeze(0).cpu().numpy()  # [N, N]
        interactions = []
        for i in range(N):
            for j in range(i + 1, N):
                mutual = (attn[i][j] + attn[j][i]) / 2
                itype = 'synergy' if mutual > 0.3 else ('masking' if mutual < 0.1 and N > 2 else 'neutral')
                interactions.append({
                    'moleculeA': molecules[i].get('name', f'mol_{i}'),
                    'moleculeB': molecules[j].get('name', f'mol_{j}'),
                    'mutualAttention': float(mutual),
                    'type': itype
                })

        return {
            'receptorResults': receptor_results,
            'interactions': interactions,
            'numMolecules': N
        }

    def _mol_to_vec(self, mol):
        """분자 → 29차원 벡터"""
        labels = mol.get('odor_labels', [])
        all_labels = ['floral', 'citrus', 'woody', 'spicy', 'sweet', 'fresh', 'green',
                      'warm', 'musk', 'fruity', 'rose', 'jasmine', 'cedar', 'vanilla',
                      'amber', 'clean', 'smoky', 'powdery', 'aquatic', 'herbal']
        label_vec = [1.0 if l in labels else 0.0 for l in all_labels]
        props = [
            mol.get('molecular_weight', 200) / 500,
            mol.get('logP', 2) / 10,
            mol.get('vapor_pressure', 1) / 100,
            mol.get('boiling_point', 200) / 500,
            mol.get('hbd', 1) / 5,
            mol.get('hba', 2) / 10,
            mol.get('tpsa', 40) / 200,
            mol.get('rotatable_bonds', 3) / 15,
            mol.get('aromatic_rings', 1) / 5,
        ]
        return label_vec + props
