# molecular_engine.py — RDKit SMILES 파서 + PyTorch GNN
# =====================================================
# JS regex → RDKit 네이티브 화학 파서
# =====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

# RDKit 임포트 (실패 시 폴백)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[MolecularEngine] RDKit not found, using fallback parser")


class MoleculeGNN(nn.Module):
    """그래프 기반 분자 냄새 예측 모델"""
    def __init__(self, atom_feat_dim=16, hidden_dim=64, output_dim=55):
        super().__init__()
        self.atom_embed = nn.Linear(atom_feat_dim, hidden_dim)
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, node_features, adj_matrix, mask):
        # node_features: [B, N, F], adj_matrix: [B, N, N], mask: [B, N]
        h = F.relu(self.atom_embed(node_features))  # [B, N, H]

        # GNN 라운드 1: A @ H @ W
        h1 = torch.bmm(adj_matrix, h)  # [B, N, H]
        B, N, H = h1.shape
        h1 = self.bn1(h1.reshape(B * N, H)).reshape(B, N, H)
        h1 = F.relu(self.conv1(h1)) + h  # residual

        # GNN 라운드 2
        h2 = torch.bmm(adj_matrix, h1)
        h2 = self.bn2(h2.reshape(B * N, H)).reshape(B, N, H)
        h2 = F.relu(self.conv2(h2)) + h1  # residual

        # 마스크된 평균 풀링
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        h_masked = h2 * mask_expanded
        graph_vec = h_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, H]

        return self.readout(graph_vec)  # [B, 55]


class MolecularEngine:
    """RDKit + PyTorch 분자 엔진"""

    ODOR_LABELS = [
        'floral', 'rose', 'jasmine', 'lily', 'violet', 'citrus', 'lemon',
        'orange', 'bergamot', 'grapefruit', 'woody', 'cedar', 'sandalwood',
        'vetiver', 'patchouli', 'spicy', 'clove', 'cinnamon', 'pepper',
        'cardamom', 'sweet', 'vanilla', 'caramel', 'honey', 'chocolate',
        'fresh', 'clean', 'ozonic', 'aquatic', 'marine', 'green', 'leafy',
        'herbal', 'grassy', 'tea', 'warm', 'amber', 'balsamic', 'resinous',
        'incense', 'musk', 'powdery', 'animalic', 'skin', 'white_musk',
        'fruity', 'apple', 'peach', 'berry', 'tropical', 'smoky', 'leather',
        'tobacco', 'earthy', 'moss'
    ]

    ATOM_FEATURES_DIM = 16
    MAX_ATOMS = 50

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.molecules = []
        self.model = None
        self.trained = False

    def load(self, data_path: str):
        """분자 데이터 로딩"""
        path = Path(data_path) / 'molecules.json'
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                self.molecules = json.load(f)
            print(f"[MolecularEngine] Loaded {len(self.molecules)} molecules")
        return self.molecules

    # ==========================================================
    # RDKit 기반 SMILES 파싱
    # ==========================================================
    def parse_smiles(self, smiles: str):
        """SMILES → 원자 특성 + 인접행렬 (RDKit 네이티브)"""
        if HAS_RDKIT:
            return self._parse_rdkit(smiles)
        return self._parse_fallback(smiles)

    def _parse_rdkit(self, smiles: str):
        """RDKit 네이티브 파서"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)
        n_atoms = min(mol.GetNumAtoms(), self.MAX_ATOMS)

        # 원자 특성 행렬 [N, 16]
        features = []
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            feat = [
                atom.GetAtomicNum() / 53,                          # 원자번호 (정규화)
                atom.GetDegree() / 4,                              # 결합 수
                atom.GetTotalNumHs() / 4,                          # 수소 수
                atom.GetFormalCharge() / 2,                        # 공식 전하
                atom.GetNumRadicalElectrons() / 2,                 # 라디칼 전자
                1 if atom.GetIsAromatic() else 0,                  # 방향족
                atom.GetMass() / 127,                              # 원자량
                # 혼성화 원-핫 (sp, sp2, sp3, sp3d, sp3d2)
                1 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP else 0,
                1 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2 else 0,
                1 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3 else 0,
                1 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D else 0,
                1 if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D2 else 0,
                # 고리 정보
                1 if atom.IsInRing() else 0,
                1 if atom.IsInRingSize(5) else 0,
                1 if atom.IsInRingSize(6) else 0,
                atom.GetImplicitValence() / 4                      # 묵시적 원자가
            ]
            features.append(feat)

        # 패딩
        while len(features) < self.MAX_ATOMS:
            features.append([0] * self.ATOM_FEATURES_DIM)

        # 인접행렬 [N, N]
        adj = np.zeros((self.MAX_ATOMS, self.MAX_ATOMS), dtype=np.float32)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < self.MAX_ATOMS and j < self.MAX_ATOMS:
                bond_type = bond.GetBondTypeAsDouble()
                adj[i][j] = bond_type
                adj[j][i] = bond_type

        # 자기 연결 + 차수 정규화 (D^{-1/2} A D^{-1/2})
        adj_with_self = adj + np.eye(self.MAX_ATOMS, dtype=np.float32)
        degree = adj_with_self.sum(axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1)))
        adj_norm = d_inv_sqrt @ adj_with_self @ d_inv_sqrt

        # 마스크
        mask = np.zeros(self.MAX_ATOMS, dtype=np.float32)
        mask[:n_atoms] = 1.0

        # 분자 descriptor (RDKit)
        descriptors = {
            'mol_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'n_atoms': n_atoms
        }

        return {
            'features': np.array(features[:self.MAX_ATOMS], dtype=np.float32),
            'adj': adj_norm,
            'mask': mask,
            'descriptors': descriptors
        }

    def _parse_fallback(self, smiles: str):
        """RDKit 없을 때 폴백 (JS 버전과 유사)"""
        ATOM_MAP = {'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'F': 9, 'Cl': 17, 'Br': 35, 'I': 53}

        atoms = []
        bonds = []
        i = 0
        stack = []
        prev = -1

        while i < len(smiles):
            ch = smiles[i]
            if ch == '(':
                stack.append(prev)
                i += 1; continue
            elif ch == ')':
                prev = stack.pop() if stack else prev
                i += 1; continue
            elif ch in '=#/-\\':
                i += 1; continue
            elif ch.isdigit():
                i += 1; continue
            elif ch == '[':
                close = smiles.find(']', i)
                i = close + 1 if close > i else i + 1; continue

            # 2-char atoms
            two = smiles[i:i+2] if i + 1 < len(smiles) else ''
            if two in ('Cl', 'Br', 'Si'):
                atomic_num = ATOM_MAP.get(two, 14)
                atoms.append(atomic_num)
                if prev >= 0:
                    bonds.append((prev, len(atoms) - 1))
                prev = len(atoms) - 1
                i += 2; continue

            if ch.upper() in ATOM_MAP:
                atoms.append(ATOM_MAP[ch.upper()])
                if prev >= 0:
                    bonds.append((prev, len(atoms) - 1))
                prev = len(atoms) - 1
            i += 1

        n = min(len(atoms), self.MAX_ATOMS)
        features = np.zeros((self.MAX_ATOMS, self.ATOM_FEATURES_DIM), dtype=np.float32)
        for idx in range(n):
            features[idx][0] = atoms[idx] / 53

        adj = np.zeros((self.MAX_ATOMS, self.MAX_ATOMS), dtype=np.float32)
        for a, b in bonds:
            if a < self.MAX_ATOMS and b < self.MAX_ATOMS:
                adj[a][b] = 1; adj[b][a] = 1

        adj_self = adj + np.eye(self.MAX_ATOMS, dtype=np.float32)
        degree = adj_self.sum(axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1)))
        adj_norm = d_inv_sqrt @ adj_self @ d_inv_sqrt

        mask = np.zeros(self.MAX_ATOMS, dtype=np.float32)
        mask[:n] = 1.0

        return {'features': features, 'adj': adj_norm, 'mask': mask, 'descriptors': {}}

    # ==========================================================
    # 모델 빌드 + 학습
    # ==========================================================
    def build_model(self):
        self.model = MoleculeGNN(
            atom_feat_dim=self.ATOM_FEATURES_DIM,
            hidden_dim=64,
            output_dim=len(self.ODOR_LABELS)
        ).to(self.device)
        print(f"[MolecularEngine] GNN model on {self.device}")

    def train(self, epochs=60, on_progress=None):
        if self.model is None:
            self.build_model()

        # 학습 데이터 준비
        features_list, adj_list, mask_list, labels_list = [], [], [], []

        for mol in self.molecules:
            parsed = self.parse_smiles(mol.get('smiles', ''))
            if parsed is None:
                continue

            features_list.append(parsed['features'])
            adj_list.append(parsed['adj'])
            mask_list.append(parsed['mask'])

            label_vec = [1.0 if l in mol.get('odor_labels', []) else 0.0 for l in self.ODOR_LABELS]
            labels_list.append(label_vec)

        if len(features_list) == 0:
            return

        # 증강: 5x
        aug_f, aug_a, aug_m, aug_l = list(features_list), list(adj_list), list(mask_list), list(labels_list)
        for _ in range(4):
            for i in range(len(features_list)):
                noise = np.random.randn(*features_list[i].shape).astype(np.float32) * 0.02
                aug_f.append(np.clip(features_list[i] + noise, 0, 1))
                aug_a.append(adj_list[i])
                aug_m.append(mask_list[i])
                aug_l.append(labels_list[i])

        X_feat = torch.tensor(np.array(aug_f)).to(self.device)
        X_adj = torch.tensor(np.array(aug_a)).to(self.device)
        X_mask = torch.tensor(np.array(aug_m)).to(self.device)
        Y = torch.tensor(np.array(aug_l)).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        dataset = torch.utils.data.TensorDataset(X_feat, X_adj, X_mask, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for batch_f, batch_a, batch_m, batch_y in loader:
                optimizer.zero_grad()
                out = self.model(batch_f, batch_a, batch_m)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = (out > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.numel()

            acc = correct / max(total, 1)
            if on_progress:
                on_progress(epoch + 1, epochs, total_loss / len(loader), acc)

        self.trained = True
        print(f"[MolecularEngine] Training complete on {self.device}")

    @torch.no_grad()
    def predict_odor(self, smiles: str):
        if not self.trained:
            return []
        parsed = self.parse_smiles(smiles)
        if parsed is None:
            return []

        feat = torch.tensor(parsed['features']).unsqueeze(0).to(self.device)
        adj = torch.tensor(parsed['adj']).unsqueeze(0).to(self.device)
        mask = torch.tensor(parsed['mask']).unsqueeze(0).to(self.device)

        self.model.eval()
        out = self.model(feat, adj, mask).squeeze(0).cpu().numpy()

        results = []
        for i, score in enumerate(out):
            if score > 0.2:
                results.append({'label': self.ODOR_LABELS[i], 'score': float(score)})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def get_all(self):
        return self.molecules

    def get_by_id(self, mol_id: str):
        return next((m for m in self.molecules if m.get('id') == mol_id), None)

    def find_similar(self, labels: list):
        results = []
        for mol in self.molecules:
            overlap = len(set(labels) & set(mol.get('odor_labels', [])))
            if overlap > 0:
                results.append({**mol, 'similarity': overlap / max(len(labels), 1)})
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:10]

    def generate_variants(self, mol_id: str):
        mol = self.get_by_id(mol_id)
        if not mol:
            return []
        # 간단한 변형: 냄새 예측 + 유사 분자 검색
        predictions = self.predict_odor(mol.get('smiles', ''))
        labels = [p['label'] for p in predictions[:3]]
        similar = self.find_similar(labels)
        return [s for s in similar if s.get('id') != mol_id][:5]
