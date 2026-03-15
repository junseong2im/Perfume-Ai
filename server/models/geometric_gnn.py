# geometric_gnn.py — Pillar 1: MPNN + GRU + Chirality (PyTorch CUDA)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path


class MPNNLayer(nn.Module):
    """메시지 패싱 + GRU 업데이트 (1라운드)"""
    def __init__(self, hidden_dim):
        super().__init__()
        H = hidden_dim
        self.message_fn = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.ReLU()
        )
        self.gru = nn.GRUCell(H, H)

    def forward(self, h, adj):
        # h: [B, N, H], adj: [B, N, N]
        neighbor_sum = torch.bmm(adj, h)  # [B, N, H]
        B, N, H = h.shape
        h_flat = h.reshape(B * N, H)
        n_flat = neighbor_sum.reshape(B * N, H)
        msg = self.message_fn(torch.cat([h_flat, n_flat], dim=1))  # [B*N, H]
        h_new = self.gru(msg, h_flat)  # [B*N, H]
        return h_new.reshape(B, N, H)


class MPNNModel(nn.Module):
    """3라운드 MPNN + Readout MLP"""
    def __init__(self, atom_feat_dim=16, hidden_dim=64, output_dim=20, num_rounds=3):
        super().__init__()
        self.atom_embed = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim),
            nn.ReLU()
        )
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(hidden_dim) for _ in range(num_rounds)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, node_features, adj, mask):
        h = self.atom_embed(node_features)
        for layer in self.mpnn_layers:
            h = layer(h, adj)
            h = h * mask.unsqueeze(-1)
        # Global mean pooling
        h_sum = (h * mask.unsqueeze(-1)).sum(dim=1)
        h_mean = h_sum / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.readout(h_mean)


class GeometricGNN:
    """3D MPNN + Chirality + RBF 거리 인코딩"""

    ODOR_LABELS = [
        'floral', 'citrus', 'woody', 'spicy', 'sweet', 'fresh', 'green', 'warm',
        'musk', 'fruity', 'rose', 'jasmine', 'cedar', 'vanilla', 'amber',
        'clean', 'smoky', 'powdery', 'aquatic', 'herbal'
    ]

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.molecules_3d = []
        self.trained = False

    def load(self, data_path: str):
        path = Path(data_path) / 'molecules-3d.json'
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                self.molecules_3d = json.load(f)
            print(f"[GeometricGNN] Loaded {len(self.molecules_3d)} 3D molecules")

    def build_model(self):
        self.model = MPNNModel(
            atom_feat_dim=16, hidden_dim=64,
            output_dim=len(self.ODOR_LABELS), num_rounds=3
        ).to(self.device)
        print(f"[GeometricGNN] MPNN model on {self.device}")

    def extract_geometric_features(self, mol3d):
        coords = mol3d.get('coordinates', [])
        if not coords or len(coords) < 2:
            return [0.0] * 40

        n = len(coords)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = sum((coords[i][k] - coords[j][k]) ** 2 for k in range(3)) ** 0.5
                dists.append(d)

        mean_d = np.mean(dists) if dists else 0
        max_d = max(dists) if dists else 0
        min_d = min(dists) if dists else 0

        com = np.mean(coords, axis=0)
        rad_gyr = np.sqrt(np.mean([sum((c[k] - com[k]) ** 2 for k in range(3)) for c in coords]))

        # 관성 모멘트
        Ixx = sum((c[1] - com[1]) ** 2 + (c[2] - com[2]) ** 2 for c in coords)
        Iyy = sum((c[0] - com[0]) ** 2 + (c[2] - com[2]) ** 2 for c in coords)
        Izz = sum((c[0] - com[0]) ** 2 + (c[1] - com[1]) ** 2 for c in coords)

        # ★ Fix: Proper chirality via RDKit stereocenters (not arbitrary coords[0:3])
        chiral_sign = 0.0
        smiles = mol3d.get('smiles', '')
        if smiles and n >= 4:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                    if chiral_centers:
                        center_idx = chiral_centers[0][0]  # first stereocenter atom index
                        # Get neighbor indices for this center
                        atom = mol.GetAtomWithIdx(center_idx)
                        nbr_idxs = [nb.GetIdx() for nb in atom.GetNeighbors()]
                        # Need center + 3 neighbors, all within coord range
                        if (len(nbr_idxs) >= 3 and
                                center_idx < len(coords) and
                                all(ni < len(coords) for ni in nbr_idxs[:3])):
                            c0 = coords[center_idx]
                            v1 = [coords[nbr_idxs[0]][k] - c0[k] for k in range(3)]
                            v2 = [coords[nbr_idxs[1]][k] - c0[k] for k in range(3)]
                            v3 = [coords[nbr_idxs[2]][k] - c0[k] for k in range(3)]
                            chiral_sign = (v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
                                           - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
                                           + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]))
            except Exception:
                chiral_sign = 0.0

        # RBF 거리 인코딩
        rbf_centers = np.linspace(0, 5, 8)
        sigma = 0.5
        rbf_mean = [float(np.exp(-((mean_d - c) ** 2) / (2 * sigma ** 2))) for c in rbf_centers]
        rbf_max = [float(np.exp(-((max_d - c) ** 2) / (2 * sigma ** 2))) for c in rbf_centers]
        rbf_min = [float(np.exp(-((min_d - c) ** 2) / (2 * sigma ** 2))) for c in rbf_centers]

        features = [
            n / 50, mean_d / 10, max_d / 15, min_d / 5,
            rad_gyr / 5, Ixx / (n * 100), Iyy / (n * 100), Izz / (n * 100),
            np.sign(chiral_sign), abs(chiral_sign) / 100,
            *([0.0] * 6),  # padding
            *rbf_mean, *rbf_max, *rbf_min
        ]
        return features[:40]

    def train(self, molecular_engine, epochs=40, on_progress=None):
        if self.model is None:
            self.build_model()

        molecules = molecular_engine.get_all()
        if not molecules:
            return

        feats, adjs, masks, targets = [], [], [], []
        for mol in molecules:
            parsed = molecular_engine.parse_smiles(mol.get('smiles', ''))
            if parsed is None:
                continue
            feats.append(parsed['features'])
            adjs.append(parsed['adj'])
            masks.append(parsed['mask'])
            target = [1.0 if l in mol.get('odor_labels', []) else 0.0 for l in self.ODOR_LABELS]
            targets.append(target)

        if not feats:
            return

        X_f = torch.tensor(np.array(feats)).to(self.device)
        X_a = torch.tensor(np.array(adjs)).to(self.device)
        X_m = torch.tensor(np.array(masks)).to(self.device)
        Y = torch.tensor(np.array(targets)).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        dataset = torch.utils.data.TensorDataset(X_f, X_a, X_m, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for bf, ba, bm, by in loader:
                optimizer.zero_grad()
                out = self.model(bf, ba, bm)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if on_progress:
                on_progress(epoch + 1, epochs, total_loss / len(loader))

        self.trained = True

    @torch.no_grad()
    def predict(self, mol, molecular_engine):
        if not self.trained:
            return []
        parsed = molecular_engine.parse_smiles(mol.get('smiles', ''))
        if not parsed:
            return []
        self.model.eval()
        f = torch.tensor(parsed['features']).unsqueeze(0).to(self.device)
        a = torch.tensor(parsed['adj']).unsqueeze(0).to(self.device)
        m = torch.tensor(parsed['mask']).unsqueeze(0).to(self.device)
        out = self.model(f, a, m).squeeze(0).cpu().numpy()
        return [{'label': self.ODOR_LABELS[i], 'score': float(s)} for i, s in enumerate(out) if s > 0.2]

    def get_chiral_pairs(self):
        pairs = []
        for mol in self.molecules_3d:
            if mol.get('chirality_pair'):
                partner = next((m for m in self.molecules_3d if m.get('id') == mol['chirality_pair']), None)
                if partner and mol.get('chirality') == 'R':
                    pairs.append({'name': mol.get('base_name', mol.get('name')), 'R': mol, 'S': partner})
        return pairs

    def get_all_3d(self):
        return self.molecules_3d
