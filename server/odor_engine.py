# odor_engine.py — Engine 2: AI Nose (GNN → POM → MixtureTransformer)
# ==================================================================
# SMILES → 20차원 냄새 벡터 → 혼합물 상호작용 → 최종 냄새 예측
# ==================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# 냄새 차원 정의 (POM — Principal Odor Map, 22d with fatty/waxy)
ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]
N_ODOR_DIM = len(ODOR_DIMENSIONS)

# SMARTS → 냄새 규칙 (경험적 매핑)
# 각 패턴이 활성화하는 냄새 차원과 강도
STRUCTURE_ODOR_RULES = [
    # (SMARTS, {odor_dim: intensity})
    # === 에스테르 (과일/꽃) ===
    ('[CX3](=O)[OX2]', {'sweet': 0.6, 'fruity': 0.8, 'floral': 0.3}),
    # === 알코올 (플로럴/프레시) ===
    ('[OX2H]', {'floral': 0.4, 'fresh': 0.3, 'green': 0.2}),
    # === 알데히드 (시트러스/그린) ===
    ('[CX3H1](=O)', {'citrus': 0.5, 'green': 0.6, 'fresh': 0.4}),
    # === 케톤 (머스크/파우더리) ===
    ('[CX3](=O)[CX4]', {'musk': 0.5, 'powdery': 0.4, 'warm': 0.3}),
    # === 페놀 (스모키/우디) ===
    ('c1ccccc1O', {'smoky': 0.7, 'woody': 0.5, 'warm': 0.3, 'leather': 0.2}),
    # === 방향족 (sweet/warm) ===
    ('c1ccccc1', {'sweet': 0.2, 'warm': 0.2}),
    # === 테르펜/이소프렌 골격 (우디/허브) ===
    ('CC(=C)C', {'woody': 0.4, 'herbal': 0.5, 'green': 0.3, 'fresh': 0.2}),
    # === 대환 락톤 (머스크) ===
    ('[#6]~1~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~[#6]~1', 
     {'musk': 0.9, 'powdery': 0.5, 'warm': 0.4}),
    # === 티올 (sulfur — metallic/earthy) ===
    ('[SX2H]', {'metallic': 0.6, 'earthy': 0.4, 'sour': 0.3}),
    # === 아민 (earthy/metallic) ===
    ('[NX3;H2,H1]', {'earthy': 0.5, 'metallic': 0.3}),
    # === 쿠마린 (sweet/warm) ===
    ('O=C1Oc2ccccc2C=C1', {'sweet': 0.8, 'warm': 0.6, 'powdery': 0.3}),
    # === 인돌 (floral/earthy) ===
    ('c1ccc2[nH]ccc2c1', {'floral': 0.6, 'earthy': 0.5, 'warm': 0.3}),
    # === 에테르 (ozonic/fresh) ===
    ('[OX2]([CX4])[CX4]', {'ozonic': 0.4, 'fresh': 0.3}),
    # === 카르복실산 (sour) ===
    ('[CX3](=O)[OX2H]', {'sour': 0.7, 'earthy': 0.3}),
    # === 니트릴 (metallic) ===
    ('C#N', {'metallic': 0.5, 'sour': 0.2}),
    # === 바닐린-like (sweet/warm/powdery) ===
    ('OC1=CC(=CC=C1)C=O', {'sweet': 0.9, 'warm': 0.7, 'powdery': 0.5}),
    # === 시트랄-like (citrus/fresh) ===
    ('CC(=CC=O)C', {'citrus': 0.8, 'fresh': 0.5, 'green': 0.3}),
    # === 메톡시벤젠 (spicy/warm) ===
    ('COc1ccccc1', {'spicy': 0.6, 'warm': 0.5, 'smoky': 0.3}),
    # === 락톤 (fruity/sweet) ===
    ('O=C1OCCC1', {'fruity': 0.7, 'sweet': 0.5}),
    # === 멘톨-like (fresh/cool) ===
    ('CC(C)C1CCC(O)CC1', {'fresh': 0.9, 'ozonic': 0.4, 'herbal': 0.3}),
    # === 지방족 알데히드 (fatty) ===
    ('CCCCCCCC=O', {'fatty': 0.8, 'green': 0.3, 'waxy': 0.4}),
    # === 장쇄 알코올 (waxy/fatty) ===
    ('CCCCCCCCO', {'waxy': 0.7, 'fatty': 0.5, 'green': 0.2}),
    # === 장쇄 에스테르 (waxy/fruity) ===
    ('CCCCCCOC(=O)C', {'waxy': 0.6, 'fruity': 0.4, 'fatty': 0.3}),
]

# 프리컴파일
_COMPILED_RULES = []
for smarts_str, odor_map in STRUCTURE_ODOR_RULES:
    pat = Chem.MolFromSmarts(smarts_str)
    if pat:
        _COMPILED_RULES.append((pat, odor_map))


# ================================================================
# OdorGNN: SMILES → 20d Odor Vector
# ================================================================

class MolecularFeatureExtractor:
    """RDKit 분자 디스크립터 → 특성 벡터"""
    
    @staticmethod
    def extract(smiles):
        """SMILES → 분자 특성 딕셔너리"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rot': Descriptors.NumRotatableBonds(mol),
            'rings': Descriptors.RingCount(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'heavy_atoms': mol.GetNumHeavyAtoms(),
            'fraction_csp3': Descriptors.FractionCSP3(mol),
        }


class OdorGNN(nn.Module):
    """분자 구조 → 22차원 냄새 벡터 (v4+v5+D-MPNN 앙상블 지원)"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.feat_extractor = MolecularFeatureExtractor()
        self._trained_model = None
        self._use_trained = False
        self._bert_cache = None  # ChemBERTa 캐시
        self._use_bert = False
        self._model_version = 'v4'  # v4, v5, ensemble, or dmpnn
        
        # Ensemble support: v4 + v5 + D-MPNN
        self._v4_model = None
        self._v5_model = None
        self._dmpnn_model = None    # D-MPNN (chemprop or manual)
        self._dmpnn_type = None      # 'manual_dmpnn' or 'chemprop_cli'
        self._ensemble_mode = False
        self._ensemble_w4 = 0.40     # v4 MLP weight
        self._ensemble_w5 = 0.35     # v5 GATConv weight
        self._ensemble_wd = 0.25     # D-MPNN weight (0 if unavailable)
        self._threshold = 0.0        # prediction thresholding (0=off)
        
        print(f"[OdorGNN] Initialized on {self.device} | {N_ODOR_DIM}d odor space")
    
    def load_trained(self, trained_model, version='v4'):
        """학습된 TrainableOdorNet/V4/V5 로드"""
        self._trained_model = trained_model
        self._trained_model.eval()
        self._use_trained = True
        self._model_version = version
        
        # v4/v5 모델이면 ChemBERTa 캐시 로드
        if version in ('v4', 'v5', 'ensemble'):
            self._load_bert_cache()
        elif hasattr(trained_model, 'input_dim') and trained_model.input_dim == 384:
            self._load_bert_cache()
        
        ver = f"{version} (GATConv+ChemBERTa)" if version == 'v5' else (
            "v4 (ChemBERTa)" if self._use_bert else "v3 (FP)")
        print(f"[OdorGNN] Trained model loaded ({ver})")
    
    def load_ensemble(self, v4_model, v5_model, w4=0.40, w5=0.35, 
                      dmpnn_model=None, wd=0.25):
        """v4 + v5 + D-MPNN 앙상블 모드 로드"""
        self._v4_model = v4_model
        self._v4_model.eval()
        self._v5_model = v5_model
        self._v5_model.eval()
        self._ensemble_mode = True
        
        # D-MPNN가 있으면 3-model 앙상블
        if dmpnn_model is not None:
            self._dmpnn_model = dmpnn_model
            self._ensemble_w4 = w4
            self._ensemble_w5 = w5
            self._ensemble_wd = wd
            dmpnn_label = f" + D-MPNN×{wd:.2f}"
        else:
            # D-MPNN 없으면 v4+v5만 사용, 가중치 재배분
            total = w4 + w5
            self._ensemble_w4 = w4 / total
            self._ensemble_w5 = w5 / total
            self._ensemble_wd = 0
            dmpnn_label = ""
        
        self._use_trained = True
        self._model_version = 'ensemble'
        
        # ChemBERTa 캐시 로드 (양쪽 다 필요)
        self._load_bert_cache()
        
        print(f"[OdorGNN] 🎯 Ensemble loaded: v4×{self._ensemble_w4:.2f} + "
              f"v5×{self._ensemble_w5:.2f}{dmpnn_label}")
    
    def load_dmpnn(self, dmpnn_checkpoint):
        """D-MPNN 모델 로드"""
        self._dmpnn_model = dmpnn_checkpoint
        self._dmpnn_type = dmpnn_checkpoint.get('type', 'manual_dmpnn') if isinstance(dmpnn_checkpoint, dict) else 'chemprop_cli'
        print(f"[OdorGNN] D-MPNN loaded ({self._dmpnn_type})")
    
    def _load_bert_cache(self):
        """ChemBERTa 캐시 파일 로드"""
        try:
            from scripts.precompute_bert import ChemBERTaCache
            self._bert_cache = ChemBERTaCache()
            if self._bert_cache.load():
                self._use_bert = True
            else:
                self._bert_cache = None
        except Exception as e:
            print(f"[OdorGNN] ChemBERTa cache load failed: {e}")
            self._bert_cache = None
    
    def _get_features(self, smiles):
        """SMILES → feature vector (ChemBERTa 우선, FP fallback)"""
        if self._use_bert and self._bert_cache:
            # Canonical SMILES로 변환 (캐시 키와 일치시키기 위해)
            canonical = smiles
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    canonical = Chem.MolToSmiles(mol)
            except:
                pass
            
            emb = self._bert_cache.get(canonical)
            if emb is not None:
                return emb
            
            # v4 모델이지만 캐시 미스 → 384d zero vector (FP 128d 반환 시 차원 불일치)
            return np.zeros(384, dtype=np.float32)
        
        # FP fallback (v3 모델에서만 사용)
        from train_models import _smiles_to_features
        return _smiles_to_features(smiles)
    
    def smarts_encode(self, smiles):
        """SMARTS 규칙 기반 냄새 벡터 (규칙 엔진 — fallback)"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(N_ODOR_DIM)
        
        odor_vec = np.zeros(N_ODOR_DIM)
        for pat, odor_map in _COMPILED_RULES:
            matches = mol.GetSubstructMatches(pat)
            if matches:
                n_matches = min(len(matches), 3)
                for dim_name, intensity in odor_map.items():
                    idx = ODOR_DIMENSIONS.index(dim_name)
                    odor_vec[idx] += intensity * (1 + 0.2 * (n_matches - 1))
        
        mw = Descriptors.MolWt(mol)
        if mw > 200:
            odor_vec[ODOR_DIMENSIONS.index('musk')] += 0.2
            odor_vec[ODOR_DIMENSIONS.index('warm')] += 0.1
        if mw > 250:
            odor_vec[ODOR_DIMENSIONS.index('amber')] += 0.2
        
        logp = Descriptors.MolLogP(mol)
        if logp > 3:
            odor_vec[ODOR_DIMENSIONS.index('woody')] += 0.15
        if logp > 5:
            odor_vec[ODOR_DIMENSIONS.index('leather')] += 0.15
        
        return np.clip(odor_vec, 0, 1)
    
    def encode(self, smiles):
        """SMILES → 20d 냄새 벡터 (앙상블 > v5 > v4 > 규칙 순서)"""
        # === 앙상블 모드 ===
        if self._ensemble_mode and self._v4_model and self._v5_model:
            return self._encode_ensemble(smiles)
        
        if self._use_trained and self._trained_model is not None:
            if self._model_version == 'v5':
                return self._encode_v5(smiles)
            
            feats = self._get_features(smiles)
            with torch.no_grad():
                x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)
                # MultiTaskOdorNet / WideNet: only use main head for inference
                if hasattr(self._trained_model, 'main_head') or hasattr(self._trained_model, 'head'):
                    try:
                        vec = self._trained_model(x, return_aux=False).squeeze(0).cpu().numpy()
                    except TypeError:
                        vec = self._trained_model(x).squeeze(0).cpu().numpy()
                else:
                    vec = self._trained_model(x).squeeze(0).cpu().numpy()
                
                # Thresholding post-processing (zero out noise)
                if hasattr(self, '_threshold') and self._threshold > 0:
                    vec[vec < self._threshold] = 0.0
            return vec
        
        # fallback: 규칙 기반
        return self.smarts_encode(smiles)
    
    def _encode_ensemble(self, smiles):
        """앙상블: v4(MLP) × w4 + v5(GATConv) × w5 + D-MPNN × wd"""
        from models.odor_gat_v5 import smiles_to_graph
        
        def _pad_to_dim(vec, target_dim=N_ODOR_DIM):
            """Pad or truncate vector to target dimension (20d→22d safe)"""
            if len(vec) >= target_dim:
                return vec[:target_dim]
            padded = np.zeros(target_dim)
            padded[:len(vec)] = vec
            return padded
        
        # --- v4 MLP prediction ---
        bert_emb = self._get_features(smiles)  # 384d
        with torch.no_grad():
            x = torch.tensor(bert_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
            v4_vec = _pad_to_dim(self._v4_model(x).squeeze(0).cpu().numpy())
        
        # --- v5 GATConv prediction ---
        graph = smiles_to_graph(smiles, device=self.device)
        if graph is not None:
            with torch.no_grad():
                bert_t = torch.tensor(bert_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
                graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
                v5_vec = _pad_to_dim(self._v5_model(graph, bert_t).squeeze(0).cpu().numpy())
        else:
            v5_vec = v4_vec  # v5 실패 시 v4로 대체
        
        # --- D-MPNN prediction ---
        dmpnn_vec = None
        if self._dmpnn_model is not None and self._ensemble_wd > 0:
            try:
                dmpnn_vec = _pad_to_dim(self._encode_dmpnn(smiles))
            except Exception:
                pass
        
        # --- weighted blend ---
        if dmpnn_vec is not None:
            # 3-model ensemble
            ensemble = (self._ensemble_w4 * v4_vec + 
                       self._ensemble_w5 * v5_vec + 
                       self._ensemble_wd * dmpnn_vec)
        else:
            # 2-model fallback: reweight v4+v5
            w4 = self._ensemble_w4 / (self._ensemble_w4 + self._ensemble_w5)
            w5 = self._ensemble_w5 / (self._ensemble_w4 + self._ensemble_w5)
            ensemble = w4 * v4_vec + w5 * v5_vec
        
        return np.clip(ensemble, 0, 1)
    
    def _encode_dmpnn(self, smiles):
        """D-MPNN: SMILES → 20d odor vector"""
        if self._dmpnn_model is None:
            return None
        
        # Manual D-MPNN (PyTorch)
        if isinstance(self._dmpnn_model, dict) and self._dmpnn_model.get('type') == 'manual_dmpnn':
            from scripts.train_chemprop import _train_manual_dmpnn
            # Reconstruct the model for inference
            from scripts.train_chemprop import DirectedMPNN, mol_to_graph, ATOM_DIM, BOND_DIM
            model = DirectedMPNN(atom_dim=ATOM_DIM, bond_dim=BOND_DIM)
            model.load_state_dict(self._dmpnn_model['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            graph = mol_to_graph(smiles)
            if graph is None:
                return None
            
            with torch.no_grad():
                x = graph['x'].unsqueeze(0).to(self.device) if graph['x'].dim() == 2 else graph['x'].to(self.device)
                edge_index = graph['edge_index'].to(self.device)
                edge_attr = graph['edge_attr'].to(self.device)
                batch_idx = torch.zeros(graph['n_atoms'], dtype=torch.long, device=self.device)
                
                pred = model(x, edge_index, edge_attr, [graph['n_atoms']], batch_idx)
                return pred.squeeze(0).cpu().numpy()
        
        return None
    
    def _encode_v5(self, smiles):
        """v5 GATConv: SMILES → molecular graph + ChemBERTa → 20d"""
        from models.odor_gat_v5 import smiles_to_graph
        
        # Build molecular graph
        graph = smiles_to_graph(smiles, device=self.device)
        if graph is None:
            return self.smarts_encode(smiles)  # fallback
        
        # Get ChemBERTa embedding
        bert_emb = self._get_features(smiles)  # 384d (uses canonical lookup)
        
        with torch.no_grad():
            bert_t = torch.tensor(bert_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Add batch index for single graph
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
            vec = self._trained_model(graph, bert_t).squeeze(0).cpu().numpy()
        return vec
    
    def encode_batch(self, smiles_list):
        """여러 분자 한번에 인코딩"""
        if self._use_trained and self._trained_model is not None:
            feats = np.array([self._get_features(s) for s in smiles_list])
            with torch.no_grad():
                x = torch.tensor(feats, dtype=torch.float32).to(self.device)
                vecs = self._trained_model(x).cpu().numpy()
            return vecs
        
        # fallback
        return np.array([self.smarts_encode(s) for s in smiles_list])
    
    def forward(self, x):
        if self._trained_model:
            return self._trained_model(x)
        return x


# ================================================================
# PrincipalOdorMap: 냄새 좌표계
# ================================================================

class PrincipalOdorMap:
    """20차원 냄새 공간 — 좌표 변환, 유사도 계산, 군집 분석"""
    
    # 대표 냄새 앵커 (참조점) — 22d (fatty, waxy 포함)
    ANCHORS = {
        'rose':       np.array([0.3,0,0,0.9,0,0,0,0.2,0,0.3,0,0,0.2,0,0,0,0,0,0,0,0,0]),
        'lemon':      np.array([0.2,0.3,0,0,0.9,0,0,0.7,0.3,0,0.4,0,0,0,0,0,0,0,0.1,0,0,0]),
        'sandalwood': np.array([0,0,0.9,0,0,0.1,0.3,0,0,0.7,0,0.2,0.3,0,0,0.5,0,0.2,0,0,0,0]),
        'vanilla':    np.array([0.9,0,0,0,0,0,0,0,0,0.7,0,0,0.5,0,0,0.3,0,0,0,0,0,0]),
        'ocean':      np.array([0,0.1,0,0,0,0,0,0.6,0,0,0,0,0,0.8,0,0,0,0,0.5,0,0,0]),
        'musk':       np.array([0.2,0,0.2,0,0,0,0.9,0,0,0.6,0,0,0.5,0,0,0.4,0,0,0,0,0,0]),
        'smoke':      np.array([0,0,0.3,0,0,0.2,0,0,0,0.4,0,0.9,0,0,0,0,0.3,0.2,0,0.1,0,0]),
        'grass':      np.array([0,0,0,0,0,0,0,0.4,0.9,0,0,0,0,0,0.5,0,0,0.3,0,0,0,0]),
        'cinnamon':   np.array([0.3,0,0,0,0,0.9,0,0,0,0.7,0,0,0,0,0.2,0,0,0,0,0,0,0]),
        'rain':       np.array([0,0,0,0,0,0,0,0.5,0.2,0,0,0,0,0.3,0,0,0,0.6,0.7,0.1,0,0]),
    }
    
    def __init__(self):
        self.odor_gnn = None  # lazy init
        print("[POM] Principal Odor Map initialized | 20d space")
    
    def set_gnn(self, odor_gnn):
        self.odor_gnn = odor_gnn
    
    def molecule_to_coord(self, smiles):
        """분자 → POM 좌표"""
        if self.odor_gnn is None:
            raise ValueError("OdorGNN not set")
        return self.odor_gnn.encode(smiles)
    
    def similarity(self, vec_a, vec_b):
        """코사인 유사도"""
        dot = np.dot(vec_a, vec_b)
        norm = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
        return float(dot / norm)
    
    def distance(self, vec_a, vec_b):
        """유클리드 거리"""
        return float(np.linalg.norm(vec_a - vec_b))
    
    def nearest_anchor(self, vec):
        """가장 가까운 앵커 냄새"""
        best_name, best_sim = None, -1
        for name, anchor in self.ANCHORS.items():
            sim = self.similarity(vec, anchor)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name, best_sim
    
    def describe_vector(self, vec, top_k=5):
        """벡터 → 상위 k개 냄새 차원"""
        indices = np.argsort(vec)[::-1][:top_k]
        return [(ODOR_DIMENSIONS[i], float(vec[i])) for i in indices]
    
    def target_from_description(self, description):
        """텍스트 설명 → 목표 벡터 (한국어/영어 키워드 매칭)"""
        target = np.zeros(N_ODOR_DIM)
        desc_lower = description.lower()
        
        # ===== 영어 직접 매칭 (20d 차원명 + 동의어) =====
        en_keywords = {
            # sweet (0)
            'sweet': 'sweet', 'vanilla': 'sweet', 'caramel': 'sweet',
            'honey': 'sweet', 'sugar': 'sweet', 'candy': 'sweet',
            'butterscotch': 'sweet', 'chocolate': 'sweet',
            # sour (1)
            'sour': 'sour', 'acidic': 'sour', 'tart': 'sour', 'vinegar': 'sour',
            # woody (2)
            'woody': 'woody', 'wood': 'woody', 'cedar': 'woody',
            'sandalwood': 'woody', 'pine': 'woody', 'oak': 'woody',
            'balsam': 'woody', 'resinous': 'woody',
            # floral (3)
            'floral': 'floral', 'flower': 'floral', 'rose': 'floral',
            'jasmine': 'floral', 'lily': 'floral', 'violet': 'floral',
            'orchid': 'floral', 'gardenia': 'floral', 'lavender': 'floral',
            # citrus (4)
            'citrus': 'citrus', 'lemon': 'citrus', 'orange': 'citrus',
            'lime': 'citrus', 'grapefruit': 'citrus', 'bergamot': 'citrus',
            'mandarin': 'citrus', 'tangerine': 'citrus',
            # spicy (5)
            'spicy': 'spicy', 'cinnamon': 'spicy', 'clove': 'spicy',
            'pepper': 'spicy', 'ginger': 'spicy', 'nutmeg': 'spicy',
            'cardamom': 'spicy',
            # musk (6)
            'musk': 'musk', 'musky': 'musk', 'animalic': 'musk',
            # fresh (7)
            'fresh': 'fresh', 'clean': 'fresh', 'crisp': 'fresh',
            'cool': 'fresh', 'cooling': 'fresh',
            # green (8)
            'green': 'green', 'grass': 'green', 'leafy': 'green',
            'herbal': 'herbal', 'basil': 'herbal', 'mint': 'herbal',
            # warm (9)
            'warm': 'warm', 'cozy': 'warm', 'toasted': 'warm',
            # fruity (10)
            'fruity': 'fruity', 'fruit': 'fruity', 'apple': 'fruity',
            'peach': 'fruity', 'berry': 'fruity', 'tropical': 'fruity',
            'pineapple': 'fruity', 'banana': 'fruity', 'melon': 'fruity',
            'cherry': 'fruity', 'coconut': 'fruity',
            # smoky (11)
            'smoky': 'smoky', 'smoke': 'smoky', 'roasted': 'smoky',
            'tobacco': 'smoky', 'burnt': 'smoky', 'coffee': 'smoky',
            # powdery (12)
            'powdery': 'powdery', 'powder': 'powdery', 'soft': 'powdery',
            # aquatic (13)
            'aquatic': 'aquatic', 'marine': 'aquatic', 'ocean': 'aquatic',
            'watery': 'aquatic', 'sea': 'aquatic',
            # herbal (14) – already mapped above
            # amber (15)
            'amber': 'amber', 'incense': 'amber', 'oriental': 'amber',
            'resin': 'amber',
            # leather (16)
            'leather': 'leather', 'suede': 'leather',
            # earthy (17)
            'earthy': 'earthy', 'earth': 'earthy', 'mushroom': 'earthy',
            'mossy': 'earthy', 'soil': 'earthy',
            # ozonic (18)
            'ozonic': 'ozonic', 'ozone': 'ozonic', 'petrichor': 'ozonic',
            'rain': 'ozonic',
            # metallic (19)
            'metallic': 'metallic', 'mineral': 'metallic',
        }
        
        for keyword, dim_name in en_keywords.items():
            if keyword in desc_lower:
                idx = ODOR_DIMENSIONS.index(dim_name)
                target[idx] += 0.8
        
        # ===== 한국어 → 영어 키워드 매핑 =====
        kr_to_en = {
            '달콤': 'sweet', '새콤': 'sour', '우디': 'woody', '나무': 'woody',
            '꽃': 'floral', '플로럴': 'floral', '시트러스': 'citrus', '레몬': 'citrus',
            '스파이시': 'spicy', '매운': 'spicy', '계피': 'spicy',
            '머스크': 'musk', '사향': 'musk', '프레시': 'fresh', '신선': 'fresh',
            '풀': 'green', '그린': 'green', '따뜻': 'warm', '온기': 'warm',
            '과일': 'fruity', '과일향': 'fruity', '스모키': 'smoky', '연기': 'smoky',
            '파우더': 'powdery', '바다': 'aquatic', '바닷가': 'aquatic', '오션': 'aquatic',
            '허브': 'herbal', '약초': 'herbal', '앰버': 'amber', '호박': 'amber',
            '가죽': 'leather', '레더': 'leather', '흙': 'earthy', '탈': 'earthy',
            '오존': 'ozonic', '비': 'ozonic', '금속': 'metallic',
            '여름': ['fresh', 'citrus', 'aquatic'],
            '겨울': ['warm', 'spicy', 'woody'],
            '봄': ['floral', 'green', 'fresh'],
            '가을': ['woody', 'smoky', 'amber'],
            '밤': ['musk', 'warm', 'amber'],
            '아침': ['fresh', 'citrus', 'green'],
            '비 온 뒤': ['earthy', 'ozonic', 'green'],
        }
        
        for kr, en in kr_to_en.items():
            if kr in desc_lower:
                if isinstance(en, list):
                    for e in en:
                        idx = ODOR_DIMENSIONS.index(e)
                        target[idx] += 0.6
                else:
                    idx = ODOR_DIMENSIONS.index(en)
                    target[idx] += 0.8
        
        # 앵커 이름 매칭 (POM 앵커 벡터 기반)
        if hasattr(self, 'ANCHORS'):
            for anchor_name, anchor_vec in self.ANCHORS.items():
                if anchor_name in desc_lower:
                    target += anchor_vec * 0.5
        
        # 정규화
        max_val = target.max()
        if max_val > 0:
            target = target / max_val
        
        return target


# ================================================================
# ConcentrationModulator: 농도 의존적 22d 벡터 보정 (V22 후처리)
# ================================================================

# 22차원 각각의 농도-응답 파라미터
# threshold: 최소 감지 농도(%), ec50: 반응 50% 농도, n: Hill 계수
# saturation: 최대 출력 스케일, masking_onset: 마스킹 시작 농도(%)
# inversion: 고농도에서 활성화되는 대체 차원 (예: 인돌 floral→earthy)
DIM_DOSE_PARAMS = {
    'sweet':    {'threshold': 0.005, 'ec50': 3.0, 'n': 1.5, 'saturation': 0.95, 'masking_onset': 12.0},
    'sour':     {'threshold': 0.01,  'ec50': 1.5, 'n': 2.0, 'saturation': 0.80, 'masking_onset':  8.0},
    'woody':    {'threshold': 0.01,  'ec50': 4.0, 'n': 1.2, 'saturation': 0.95, 'masking_onset': 25.0},
    'floral':   {'threshold': 0.001, 'ec50': 3.0, 'n': 1.5, 'saturation': 0.90, 'masking_onset': 12.0},
    'citrus':   {'threshold': 0.01,  'ec50': 2.0, 'n': 2.0, 'saturation': 0.85, 'masking_onset': 10.0},
    'spicy':    {'threshold': 0.005, 'ec50': 1.0, 'n': 2.5, 'saturation': 0.85, 'masking_onset':  5.0},
    'musk':     {'threshold': 0.001, 'ec50': 2.0, 'n': 1.0, 'saturation': 0.90, 'masking_onset': 15.0},
    'fresh':    {'threshold': 0.01,  'ec50': 1.5, 'n': 2.0, 'saturation': 0.85, 'masking_onset':  8.0},
    'green':    {'threshold': 0.005, 'ec50': 1.5, 'n': 2.0, 'saturation': 0.85, 'masking_onset':  8.0},
    'warm':     {'threshold': 0.005, 'ec50': 3.5, 'n': 1.3, 'saturation': 0.90, 'masking_onset': 18.0},
    'fruity':   {'threshold': 0.005, 'ec50': 2.5, 'n': 1.8, 'saturation': 0.85, 'masking_onset': 12.0},
    'smoky':    {'threshold': 0.003, 'ec50': 1.0, 'n': 2.5, 'saturation': 0.80, 'masking_onset':  5.0},
    'powdery':  {'threshold': 0.005, 'ec50': 3.0, 'n': 1.5, 'saturation': 0.90, 'masking_onset': 15.0},
    'aquatic':  {'threshold': 0.005, 'ec50': 1.5, 'n': 2.0, 'saturation': 0.80, 'masking_onset':  8.0},
    'herbal':   {'threshold': 0.008, 'ec50': 2.5, 'n': 1.5, 'saturation': 0.85, 'masking_onset': 10.0},
    'amber':    {'threshold': 0.003, 'ec50': 3.5, 'n': 1.3, 'saturation': 0.95, 'masking_onset': 20.0},
    'leather':  {'threshold': 0.002, 'ec50': 1.5, 'n': 2.0, 'saturation': 0.80, 'masking_onset':  8.0},
    'earthy':   {'threshold': 0.0001,'ec50': 2.0, 'n': 1.8, 'saturation': 0.80, 'masking_onset': 10.0},
    'ozonic':   {'threshold': 0.001, 'ec50': 1.0, 'n': 2.0, 'saturation': 0.75, 'masking_onset':  6.0},
    'metallic': {'threshold': 0.001, 'ec50': 0.5, 'n': 2.5, 'saturation': 0.70, 'masking_onset':  4.0},
    'fatty':    {'threshold': 0.005, 'ec50': 2.0, 'n': 1.5, 'saturation': 0.75, 'masking_onset': 10.0},
    'waxy':     {'threshold': 0.005, 'ec50': 2.5, 'n': 1.5, 'saturation': 0.75, 'masking_onset': 12.0},
}

# 고농도 차원 전환 규칙 (예: 인돌 — 저농도 floral, 고농도 earthy/animalic)
HIGH_CONC_INVERSIONS = {
    # (원래 차원, 목표 차원): (전환 시작 농도, 전환 강도)
    ('floral', 'earthy'):   (5.0, 0.3),   # 인돌 계열
    ('sweet', 'smoky'):     (8.0, 0.2),   # 과도한 sweet → 타는 느낌
    ('fresh', 'metallic'):  (6.0, 0.15),  # 과도한 fresh → 금속 느낌
    ('fruity', 'sour'):     (7.0, 0.25),  # 과도한 fruity → 시큼
    ('musk', 'earthy'):     (10.0, 0.2),  # 과도한 musk → 흙내
}

# 분자별 역치 오버라이드 (molecules.json odor_strength 기반)
# strength가 높을수록 역치가 낮고(더 민감), EC50도 낮음(더 빨리 포화)
# 공식: threshold = 10^(-(strength-1)*0.5) * 0.005, ec50 = 6.0 - strength*0.5
MOLECULE_OVERRIDES = {
    # === strength 9: 극강 감도 (vanillin, damascenone, cinnamaldehyde) ===
    'COc1cc(C=O)ccc1O':           {'threshold': 0.00003, 'ec50': 1.0, 'n': 2.0},  # vanillin
    'O=C/C=C/c1ccccc1':           {'threshold': 0.00005, 'ec50': 1.0, 'n': 2.2},  # cinnamaldehyde
    'CC(=CC1=CC(=O)CC1(C)C)C=CC': {'threshold': 0.00002, 'ec50': 0.8, 'n': 2.0},  # damascenone
    
    # === strength 8: 높은 감도 ===
    'CC(=CCC/C(=C/C=O)/C)C':      {'threshold': 0.0001, 'ec50': 1.5, 'n': 2.0},   # citral
    'COc1cc(CC=C)ccc1O':           {'threshold': 0.0001, 'ec50': 1.2, 'n': 2.2},   # eugenol
    'c1ccc2[nH]ccc2c1':            {'threshold': 0.0001, 'ec50': 1.0, 'n': 1.8},   # indole
    'CC(C)C1CCC(C)CC1O':           {'threshold': 0.0001, 'ec50': 1.5, 'n': 2.0},   # menthol
    'COC(=O)c1ccccc1O':            {'threshold': 0.0001, 'ec50': 1.3, 'n': 2.2},   # methyl salicylate
    'CCC1=C(O)C(=O)C=CO1':         {'threshold': 0.0001, 'ec50': 1.5, 'n': 1.8},   # ethyl maltol
    'CC1(C)C2CCC1(C)C(=O)C2':      {'threshold': 0.0001, 'ec50': 1.5, 'n': 2.0},   # camphor
    'CCC1=CC2=C(C=C1)OCC(=O)O2':   {'threshold': 0.00008, 'ec50': 1.2, 'n': 2.0},  # calone
    'CC(C)C1CCC2(CCCC(=C)C2O)C1':  {'threshold': 0.00008, 'ec50': 1.5, 'n': 1.5},  # vetiverol
    'CC1CCC2(CC3C(CCC2C1(C)C)C3(C)C)O': {'threshold': 0.0001, 'ec50': 1.5, 'n': 1.5},  # patchoulol
    
    # === strength 7: 중상 감도 ===
    'CC(=CCC/C(=C/CO)/C)C':        {'threshold': 0.0003, 'ec50': 2.0, 'n': 1.8},   # geraniol/linalool
    'CC1=CCC(CC1)C(=C)C':          {'threshold': 0.0003, 'ec50': 2.0, 'n': 2.0},   # limonene
    'O=c1ccc2ccccc2o1':            {'threshold': 0.0003, 'ec50': 2.2, 'n': 1.5},   # coumarin
    'CC(=C)C1CC=C(C)C(=O)C1':      {'threshold': 0.0003, 'ec50': 2.0, 'n': 1.8},   # carvone
    'CC1(C)CC(\\C=C\\CO)C=C(C1)C':  {'threshold': 0.0003, 'ec50': 2.5, 'n': 1.5},  # santalol
    'CC(=CC1=CC(=O)CC(C)C1)C':     {'threshold': 0.0003, 'ec50': 2.0, 'n': 1.8},   # alpha-ionone
    'O=Cc1ccccc1':                 {'threshold': 0.0003, 'ec50': 2.0, 'n': 1.8},   # benzaldehyde
    'CCCCCCC1CCC(=O)O1':           {'threshold': 0.0003, 'ec50': 2.5, 'n': 1.8},   # gamma-decalactone
    'CCC=CCCO':                    {'threshold': 0.0003, 'ec50': 2.0, 'n': 2.0},   # cis-3-hexenol
    
    # === strength 6: 중간 감도 ===
    'CC(=CCC/C(=C/CO)/C)C':        {'threshold': 0.0006, 'ec50': 2.5, 'n': 1.5},   # linalool
    'OCCC1=CC=CC=C1':              {'threshold': 0.0006, 'ec50': 2.5, 'n': 1.5},   # phenylethyl alcohol
    'CC(CCO)CCC=C(C)C':            {'threshold': 0.0006, 'ec50': 2.5, 'n': 1.5},   # citronellol
    'CC1=CCC2CC1C2(C)C':           {'threshold': 0.0006, 'ec50': 3.0, 'n': 1.3},   # alpha-pinene
    'CC1CCCCCCCCCCCCC(=O)C1':      {'threshold': 0.0008, 'ec50': 2.5, 'n': 1.0},   # muscone
    'CC(C)(O)CCCC(=C)C':           {'threshold': 0.0006, 'ec50': 2.5, 'n': 1.5},   # dihydromyrcenol
    'CC1=CC(=O)CC(C1)C(C)=CC':     {'threshold': 0.0006, 'ec50': 2.5, 'n': 1.5},   # methyl ionone
    'CC1=CCC(CC1)C(C)(C)O':        {'threshold': 0.0006, 'ec50': 2.5, 'n': 1.5},   # terpineol
    
    # === strength 5: 약한 감도 ===
    'CC1(C)CC(=O)C2(CCC(CC2C1)C(C)C)C': {'threshold': 0.0015, 'ec50': 3.5, 'n': 1.3},  # iso e super
    'CC(=CCC=C(C)C=C)C':           {'threshold': 0.0015, 'ec50': 3.5, 'n': 1.3},   # myrcene
    'CC(=CCC/C(=C/CC(C)(C=C)O)/C)C': {'threshold': 0.0015, 'ec50': 3.5, 'n': 1.3},  # nerolidol
    'CC1CCC2C(C1(C)C)CCC2(C)O':    {'threshold': 0.0015, 'ec50': 3.5, 'n': 1.2},   # cedrol
    'CC1(C)CCCC2(C)C1CCC(C2(O)CO)C(=C)C': {'threshold': 0.0015, 'ec50': 3.5, 'n': 1.3},  # sclareol
    
    # === strength 4: 약 감도 ===
    'CC(=O)OCC(CC1CCCC1=O)C':      {'threshold': 0.003,  'ec50': 4.0, 'n': 1.2},   # hedione
    'CC(=CCC/C(=C/CC/C(=C/CO)/C)/C)C': {'threshold': 0.003, 'ec50': 4.0, 'n': 1.2},  # farnesol
    
    # === 합성 머스크: 매우 낮은 역치, 느린 수용체 결합 ===
    'CC1(C)CC2=CC(=C(C(C2C(O1)(C)C)C)C)C': {'threshold': 0.00005, 'ec50': 2.0, 'n': 0.8},  # galaxolide
    'CC1(C)CCCC2(C)C1CCC1(C)OC(CC12)C':    {'threshold': 0.0003,  'ec50': 3.0, 'n': 1.0},  # ambroxide
}

print(f'[MOLECULE_OVERRIDES] {len(MOLECULE_OVERRIDES)} molecules with specific dose-response params')


class ConcentrationModulator:
    """농도에 따른 22d 냄새 벡터 보정 (Weber-Fechner + Hill)
    
    원리:
    1. Weber-Fechner: perceived ∝ log(C/threshold) — 농도 10배 증가 시 인지 ~2배
    2. Hill equation: 수용체 점유율 = C^n / (EC50^n + C^n)
    3. Saturation: 각 차원별 최대 출력 제한
    4. Inversion: 특정 분자는 고농도에서 향 특성이 전환됨
    """
    
    def __init__(self):
        self._params = DIM_DOSE_PARAMS
        self._inversions = HIGH_CONC_INVERSIONS
        self._mol_overrides = MOLECULE_OVERRIDES
        print(f"[ConcentrationModulator] {len(self._params)} dims, "
              f"{len(self._inversions)} inversions, "
              f"{len(self._mol_overrides)} molecule overrides")
    
    def _hill(self, conc, ec50, n):
        """Hill equation: 0→1 sigmoidal response"""
        if conc <= 0 or ec50 <= 0:
            return 0.0
        return (conc ** n) / (ec50 ** n + conc ** n)
    
    def _weber_fechner(self, conc, threshold):
        """Weber-Fechner: log-scaled intensity perception"""
        if conc <= threshold:
            return 0.0
        return min(1.0, math.log10(conc / threshold) / 3.0)  # normalized to ~1.0 at 1000x threshold
    
    def modulate(self, odor_vec, concentration_pct, smiles=None, return_details=False):
        """V22 예측 벡터 + 농도(%) → 보정된 벡터
        
        Args:
            odor_vec: numpy array [22] — V22 모델 출력
            concentration_pct: float — 원료 농도 (%)
            smiles: str or None — SMILES 제공 시 분자별 파라미터 우선 적용
            return_details: bool — 상세 보정 내역 반환 여부
        
        Returns:
            numpy array [22] — 농도 보정된 벡터
        """
        # 분자별 오버라이드 조회
        mol_override = self._mol_overrides.get(smiles) if smiles else None
        result = np.copy(odor_vec)
        details = {} if return_details else None
        
        for dim_idx, dim_name in enumerate(ODOR_DIMENSIONS):
            if dim_idx >= len(result):
                break
            
            original_val = result[dim_idx]
            if original_val < 0.01:  # 원래 비활성 차원은 스킵
                continue
            
            base_params = self._params.get(dim_name, {
                'threshold': 0.005, 'ec50': 2.5, 'n': 1.5,
                'saturation': 0.85, 'masking_onset': 12.0
            })
            # 분자별 오버라이드가 있으면 threshold/ec50/n을 덮어씀
            if mol_override:
                params = {**base_params, **mol_override}
            else:
                params = base_params
            
            threshold_pct = params['threshold'] * 100
            
            # 1) 역치 필터: 농도가 감지 역치 미만이면 크게 감소
            if concentration_pct < threshold_pct:
                subthreshold_ratio = concentration_pct / max(threshold_pct, 1e-6)
                result[dim_idx] *= subthreshold_ratio * 0.1
                if return_details:
                    details[dim_name] = f'subthreshold ({concentration_pct:.3f}% < {threshold_pct:.3f}%)'
                continue
            
            # 2) Weber-Fechner 인지 강도
            wf = self._weber_fechner(concentration_pct, threshold_pct)
            
            # 3) Hill 수용체 점유율
            hill = self._hill(concentration_pct, params['ec50'], params['n'])
            
            # 4) 결합 보정: 두 모델의 기하 평균
            response = math.sqrt(wf * hill)
            
            # 5) 포화 제한
            response = min(response, params['saturation'])
            
            # 6) 마스킹 감소: 과도한 농도에서는 오히려 인지 감소
            if concentration_pct > params['masking_onset']:
                overshoot = (concentration_pct - params['masking_onset']) / max(params['masking_onset'], 1)
                masking_factor = max(0.2, 1.0 - overshoot * 0.4)
                response *= masking_factor
                if return_details:
                    details[dim_name] = f'masking ({concentration_pct:.1f}% > {params["masking_onset"]:.1f}%)'
            
            result[dim_idx] = original_val * response
            
            if return_details and dim_name not in details:
                details[dim_name] = f'response={response:.3f} (wf={wf:.3f}, hill={hill:.3f})'
        
        # 7) 고농도 차원 전환 (inversion)
        for (src_dim, tgt_dim), (onset_pct, strength) in self._inversions.items():
            if concentration_pct > onset_pct:
                src_idx = ODOR_DIMENSIONS.index(src_dim) if src_dim in ODOR_DIMENSIONS else -1
                tgt_idx = ODOR_DIMENSIONS.index(tgt_dim) if tgt_dim in ODOR_DIMENSIONS else -1
                if src_idx >= 0 and tgt_idx >= 0 and odor_vec[src_idx] > 0.3:
                    transfer = odor_vec[src_idx] * strength * min(1.0, (concentration_pct - onset_pct) / onset_pct)
                    result[tgt_idx] = min(1.0, result[tgt_idx] + transfer)
                    result[src_idx] = max(0.0, result[src_idx] - transfer * 0.5)
        
        result = np.clip(result, 0, 1)
        
        if return_details:
            return result, details
        return result
    
    def batch_modulate(self, odor_vecs, concentrations, smiles_list=None):
        """여러 분자의 벡터를 한번에 농도 보정"""
        results = []
        for i, (vec, conc) in enumerate(zip(odor_vecs, concentrations)):
            sm = smiles_list[i] if smiles_list and i < len(smiles_list) else None
            results.append(self.modulate(vec, conc, smiles=sm))
        return np.array(results)


# ================================================================
# PhysicsMixture: 물리 기반 혼합물 상호작용 모델 (V22 후처리)
# ================================================================

# 22×22 상호작용 매트릭스: 양수=시너지, 음수=길항
# [row_dim][col_dim] = 두 차원이 혼합물에서 만날 때의 상호작용
def _build_interaction_rules():
    """(source_a_idx, source_b_idx, target_idx, strength) rules
    
    86 rules from:
    1. Original 32 perfumery textbook rules 
    2. compatibility.json synergy/clash data (ingredient → dimension conversion)
    3. Additional manual rules for missing dimension pairs
    """
    idx = {d: i for i, d in enumerate(ODOR_DIMENSIONS)}
    
    # ═══════════════════════════════════════════════════
    # SYNERGIES (+): 53 rules
    # ═══════════════════════════════════════════════════
    synergies = [
        # --- Original textbook rules ---
        ('floral', 'citrus',   'fresh',    +0.12),
        ('floral', 'woody',    'warm',     +0.10),
        ('floral', 'musk',     'powdery',  +0.15),
        ('floral', 'green',    'fresh',    +0.08),
        ('woody',  'amber',    'warm',     +0.15),
        ('woody',  'leather',  'smoky',    +0.10),
        ('woody',  'earthy',   'amber',    +0.08),
        ('sweet',  'warm',     'amber',    +0.12),
        ('sweet',  'fruity',   'floral',   +0.08),
        ('spicy',  'warm',     'amber',    +0.12),
        ('spicy',  'woody',    'earthy',   +0.08),
        ('musk',   'warm',     'amber',    +0.10),
        ('citrus', 'fresh',    'green',    +0.10),
        ('citrus', 'fruity',   'fresh',    +0.08),
        ('earthy', 'green',    'herbal',   +0.10),
        ('herbal', 'fresh',    'green',    +0.08),
        ('fruity', 'floral',   'sweet',    +0.10),
        ('smoky',  'leather',  'warm',     +0.12),
        ('amber',  'warm',     'sweet',    +0.08),
        ('powdery','floral',   'sweet',    +0.08),
        
        # --- From compatibility.json synergy pairs ---
        # bergamot(citrus)+rose(floral) = 0.9  →  already covered by floral+citrus
        # rose(floral)+sandalwood(woody) = 0.95  →  floral+woody already, add powdery
        ('floral', 'woody',    'powdery',  +0.08),
        # vanilla(sweet)+amber(amber) = 0.95  →  sweet+amber → warm boost
        ('sweet',  'amber',    'warm',     +0.12),
        # lavender(herbal)+cedarwood(woody) = 0.85  →  herbal+woody → warm
        ('herbal', 'woody',    'warm',     +0.08),
        # musk+cedarwood = 0.8  →  musk+woody → warm
        ('musk',   'woody',    'warm',     +0.10),
        # oud(smoky/woody)+saffron(spicy) = 0.9  →  smoky+spicy → warm
        ('smoky',  'spicy',    'warm',     +0.10),
        # oud(smoky)+rose(floral) = 0.95  →  smoky+floral → warm
        ('smoky',  'floral',   'warm',     +0.08),
        # cinnamon(spicy)+orange(citrus) = 0.9  →  spicy+citrus → warm
        ('spicy',  'citrus',   'warm',     +0.08),
        # patchouli(earthy)+vanilla(sweet) = 0.85  →  earthy+sweet → amber
        ('earthy', 'sweet',    'amber',    +0.08),
        # frankincense(smoky)+amber = 0.85  →  smoky+amber → warm
        ('smoky',  'amber',    'warm',     +0.10),
        # coffee(smoky)+vanilla(sweet) = 0.9  →  smoky+sweet → warm
        ('smoky',  'sweet',    'warm',     +0.08),
        # leather+tobacco(woody) = 0.9  →  leather+woody → smoky boost
        ('leather','woody',    'smoky',    +0.08),
        # honey(sweet)+orange_blossom(floral) = 0.85  →  sweet+floral → powdery
        ('sweet',  'floral',   'powdery',  +0.08),
        # benzoin(sweet/balsamic)+vanilla = 0.9  →  already covered
        # pink_pepper(spicy)+rose(floral) = 0.85  →  spicy+floral → warm
        ('spicy',  'floral',   'warm',     +0.08),
        
        # --- Additional manual rules ---
        ('woody',  'musk',     'warm',     +0.10),  # woody+musk → warm
        ('amber',  'leather',  'smoky',    +0.08),  # amber+leather → smoky
        ('herbal', 'citrus',   'fresh',    +0.10),  # herbal+citrus → fresh
        ('green',  'aquatic',  'fresh',    +0.10),  # green+aquatic → fresh
        ('earthy', 'smoky',    'leather',  +0.08),  # earthy+smoky → leather
        ('woody',  'spicy',    'warm',     +0.08),  # woody+spicy → warm
        ('floral', 'amber',    'powdery',  +0.10),  # floral+amber → powdery
        ('musk',   'powdery',  'sweet',    +0.08),  # musk+powdery → sweet
        ('fruity', 'green',    'fresh',    +0.08),  # fruity+green → fresh
        ('warm',   'leather',  'amber',    +0.08),  # warm+leather → amber
        ('citrus', 'aquatic',  'ozonic',   +0.08),  # citrus+aquatic → ozonic
        ('floral', 'fruity',   'sweet',    +0.10),  # floral+fruity → sweet
        ('sweet',  'powdery',  'floral',   +0.06),  # sweet+powdery → floral
        ('woody',  'green',    'herbal',   +0.08),  # woody+green → herbal
        ('amber',  'spicy',    'warm',     +0.10),  # amber+spicy → warm
    ]
    
    # ═══════════════════════════════════════════════════
    # ANTAGONISMS (-): 33 rules
    # ═══════════════════════════════════════════════════
    antagonisms = [
        # --- Original textbook rules ---
        ('citrus', 'smoky',    'citrus',   -0.25),
        ('citrus', 'smoky',    'smoky',    -0.15),
        ('fresh',  'smoky',    'fresh',    -0.30),
        ('aquatic', 'smoky',   'aquatic',  -0.35),
        ('fresh',  'warm',     'fresh',    -0.10),
        ('citrus', 'leather',  'citrus',   -0.20),
        ('green',  'sweet',    'green',    -0.12),
        ('aquatic','sweet',    'aquatic',  -0.15),
        ('metallic','floral',  'metallic', -0.20),
        ('ozonic', 'warm',     'ozonic',   -0.15),
        ('sour',   'sweet',    'sour',     -0.20),
        ('earthy', 'fresh',    'earthy',   -0.12),
        
        # --- From compatibility.json clash pairs ---
        # mint(fresh/herbal)+oud(smoky/woody) = -0.8  →  fresh+smoky already covered
        # mint(fresh)+leather = -0.7  →  fresh+leather → fresh suppression
        ('fresh',  'leather',  'fresh',    -0.18),
        # mint(fresh)+vanilla(sweet) = -0.6  →  fresh+sweet
        ('fresh',  'sweet',    'fresh',    -0.10),
        # marine(aquatic)+oud(smoky) = -0.7  →  already covered by aquatic+smoky
        # marine(aquatic)+cinnamon(spicy) = -0.65  →  aquatic+spicy → aquatic suppression
        ('aquatic','spicy',    'aquatic',  -0.15),
        # marine(aquatic)+leather = -0.6  →  aquatic+leather → aquatic suppression
        ('aquatic','leather',  'aquatic',  -0.15),
        # lemon(citrus)+vanilla(sweet) = -0.3  →  citrus+sweet mild clash
        ('citrus', 'sweet',    'citrus',   -0.08),
        # lemon(citrus)+amber = -0.4  →  citrus+amber → citrus reduction
        ('citrus', 'amber',    'citrus',   -0.12),
        # grapefruit(citrus)+chocolate(sweet/warm) = -0.5
        ('citrus', 'warm',     'citrus',   -0.10),
        # grass(green)+vanilla(sweet) = -0.5  →  green+sweet already covered
        # grass(green)+amber = -0.6  →  green+amber → green reduction
        ('green',  'amber',    'green',    -0.15),
        # basil(herbal)+chocolate(sweet) = -0.6  →  herbal+sweet → herbal reduction
        ('herbal', 'sweet',    'herbal',   -0.12),
        # bamboo(green)+leather = -0.6  →  green+leather → green reduction
        ('green',  'leather',  'green',    -0.15),
        # marine(aquatic)+castoreum(musk/earthy) = -0.6  →  aquatic+musk
        ('aquatic','musk',     'aquatic',  -0.12),
        
        # --- Additional manual rules ---
        ('metallic','sweet',   'metallic', -0.15),  # metallic+sweet → suppression
        ('metallic','warm',    'metallic', -0.12),  # metallic+warm → suppression
        ('ozonic',  'smoky',   'ozonic',   -0.20),  # ozonic+smoky → clash
        ('ozonic',  'leather', 'ozonic',   -0.18),  # ozonic+leather → clash
        ('waxy',    'fresh',   'waxy',     -0.10),  # waxy+fresh → waxy reduced
        ('fatty',   'floral',  'fatty',    -0.15),  # fatty+floral → fatty reduced
        ('fatty',   'fresh',   'fatty',    -0.12),  # fatty+fresh → fatty reduced
    ]
    
    rules = []
    for a, b, target, strength in synergies + antagonisms:
        ai, bi, ti = idx.get(a, -1), idx.get(b, -1), idx.get(target, -1)
        if ai >= 0 and bi >= 0 and ti >= 0:
            rules.append((ai, bi, ti, strength))
    
    return rules

INTERACTION_RULES = _build_interaction_rules()

# 마스킹 우선순위: 높은 분자가 낮은 분자를 가림
# 값이 높을수록 다른 향을 가리는 힘이 강함
MASKING_POWER = {
    'musk':     0.90,  # 머스크: 최강 마스킹
    'smoky':    0.85,  # 스모키: 강한 마스킹
    'leather':  0.80,  # 레더: 강한 마스킹
    'amber':    0.75,  # 앰버: 중강 마스킹
    'woody':    0.70,  # 우디: 중간 마스킹
    'spicy':    0.65,  # 스파이시: 중간
    'warm':     0.55,  # 따뜻함: 약중간
    'earthy':   0.55,  # 어시: 약중간
    'sweet':    0.45,  # 달콤: 약함
    'sour':     0.40,
    'powdery':  0.40,
    'herbal':   0.35,
    'floral':   0.30,  # 플로럴: 약한 마스킹
    'fruity':   0.25,
    'fresh':    0.20,  # 프레시: 가장 약함 (잘 가려짐)
    'green':    0.20,
    'citrus':   0.20,
    'aquatic':  0.15,
    'ozonic':   0.15,
    'metallic': 0.35,
    'fatty':    0.30,
    'waxy':     0.30,
}


class PhysicsMixture:
    """물리/조향 규칙 기반 혼합물 상호작용 모델
    
    MixtureTransformer(미학습)를 대체하여 조향 지식 기반으로
    정밀한 혼합 냄새 벡터를 예측합니다.
    
    5가지 메커니즘:
    1. 비선형 농도 가중 합산 (power-mean)
    2. 22×22 상호작용 매트릭스 (시너지/길항)
    3. 수용체 경쟁 모델 (경쟁적 결합)
    4. 마스킹 임계값 시스템
    5. 결과 정규화 + 일관성 검증
    """
    
    def __init__(self):
        self._rules = INTERACTION_RULES
        self._masking_power = np.array([MASKING_POWER.get(d, 0.3) for d in ODOR_DIMENSIONS])
        print(f"[PhysicsMixture] Initialized | {len(self._rules)} interaction rules")
    
    def mix(self, odor_vecs, concentrations, return_analysis=False):
        """여러 분자의 보정된 벡터 → 혼합물 최종 벡터
        
        Args:
            odor_vecs: numpy [N, 22] — ConcentrationModulator 통과 후의 벡터들
            concentrations: numpy [N] — 각 분자의 농도 (%)
            return_analysis: bool — 상세 분석 반환 여부
        
        Returns:
            numpy [22] — 혼합물 냄새 벡터
        """
        N = len(odor_vecs)
        if N == 0:
            return np.zeros(N_ODOR_DIM)
        if N == 1:
            if return_analysis:
                return odor_vecs[0], {'type': 'single_molecule'}
            return odor_vecs[0]
        
        odor_vecs = np.array(odor_vecs, dtype=np.float64)
        concentrations = np.array(concentrations, dtype=np.float64)
        conc_weights = concentrations / (concentrations.sum() + 1e-8)
        
        # === Step 1: 비선형 농도 가중 합산 (power-mean, p=0.7) ===
        # 단순 가중 평균보다 강한 분자를 더 부각
        # BUGFIX ⑦: clip to ≥0 before fractional power to prevent NaN
        p = 0.7
        weighted_sum = np.zeros(N_ODOR_DIM)
        for i in range(N):
            weighted_sum += conc_weights[i] * (np.clip(odor_vecs[i], 0, None) ** p)
        base_mixture = np.power(np.clip(weighted_sum, 0, None), 1.0 / p)
        
        # === Step 2: interaction rules ===
        interaction_boost = np.zeros(N_ODOR_DIM)
        for i in range(N):
            for j in range(i + 1, N):
                pair_weight = conc_weights[i] * conc_weights[j] * 8  # boosted
                
                for src_a, src_b, target, rule_strength in self._rules:
                    # check both orderings: mol_i has src_a & mol_j has src_b, or vice versa
                    act_a1 = odor_vecs[i][src_a]
                    act_b1 = odor_vecs[j][src_b]
                    act_a2 = odor_vecs[j][src_a]
                    act_b2 = odor_vecs[i][src_b]
                    
                    strength1 = act_a1 * act_b1 if (act_a1 > 0.05 and act_b1 > 0.05) else 0
                    strength2 = act_a2 * act_b2 if (act_a2 > 0.05 and act_b2 > 0.05) else 0
                    total_strength = max(strength1, strength2)
                    
                    if total_strength > 0:
                        interaction_boost[target] += rule_strength * total_strength * pair_weight
        
        mixture = base_mixture + interaction_boost
        
        # === Step 3: 수용체 경쟁 (Competitive Binding) ===
        # 같은 차원을 강하게 자극하는 분자가 여럿이면 — 가장 강한 것이 지배
        for dim in range(N_ODOR_DIM):
            dim_vals = [odor_vecs[i][dim] * conc_weights[i] for i in range(N)]
            if len(dim_vals) >= 2:
                sorted_vals = sorted(dim_vals, reverse=True)
                if sorted_vals[0] > 0.01 and sorted_vals[1] > 0.01:
                    # 약한 쪽은 경쟁에서 밀림 (20% 감소 — 프로필 보존 개선)
                    competition_factor = 1.0 - 0.2 * (sorted_vals[1] / (sorted_vals[0] + 1e-8))
                    mixture[dim] *= max(0.5, competition_factor)
        
        # === Step 4: 마스킹 시스템 ===
        # 마스킹 파워가 높은 차원이 강하면, 마스킹 파워가 낮은 차원을 억제
        mixture_power = mixture * self._masking_power
        dominant_power = mixture_power.max()
        
        if dominant_power > 0.15:
            for dim in range(N_ODOR_DIM):
                if mixture[dim] > 0.01:
                    dim_power = mixture[dim] * self._masking_power[dim]
                    power_ratio = dim_power / (dominant_power + 1e-8)
                    if power_ratio < 0.03:  # 완화된 마스킹 (프로필 보존)
                        # 마스킹됨: 약하게 감소
                        mask_strength = max(0.2, power_ratio / 0.06)
                        mixture[dim] *= mask_strength
        
        # === Step 5: 정규화 ===
        mixture = np.clip(mixture, 0, 1)
        
        if return_analysis:
            # 상호작용 분석
            interactions = self._analyze_interactions(odor_vecs, concentrations)
            return mixture, {
                'base_mixture': base_mixture.tolist(),
                'interaction_boost': interaction_boost.tolist(),
                'interactions': interactions,
                'dominant_dims': [(ODOR_DIMENSIONS[i], float(mixture[i])) 
                                 for i in np.argsort(mixture)[::-1][:5]],
            }
        
        return mixture
    
    def _analyze_interactions(self, odor_vecs, concentrations):
        """분자 쌍별 상호작용 유형 판정"""
        N = len(odor_vecs)
        conc_weights = concentrations / (concentrations.sum() + 1e-8)
        interactions = []
        
        for i in range(N):
            for j in range(i + 1, N):
                # 공유 활성 차원
                active_i = set(np.where(odor_vecs[i] > 0.2)[0])
                active_j = set(np.where(odor_vecs[j] > 0.2)[0])
                shared = active_i & active_j
                
                # 시너지/길항 점수 합산
                synergy_score = 0
                for src_a, src_b, target, strength in self._rules:
                    act1 = odor_vecs[i][src_a] * odor_vecs[j][src_b]
                    act2 = odor_vecs[j][src_a] * odor_vecs[i][src_b]
                    if max(act1, act2) > 0.04:  # both dims > 0.2
                        synergy_score += strength
                
                # 마스킹 체크: 한쪽이 마스킹 파워 훨씬 높음
                power_i = np.dot(odor_vecs[i], self._masking_power) * conc_weights[i]
                power_j = np.dot(odor_vecs[j], self._masking_power) * conc_weights[j]
                
                if max(power_i, power_j) > 0.01:
                    masking_ratio = min(power_i, power_j) / (max(power_i, power_j) + 1e-8)
                else:
                    masking_ratio = 1.0
                
                # 판정
                if masking_ratio < 0.3:
                    itype = 'masking'
                elif synergy_score > 0.05:
                    itype = 'synergy'
                elif synergy_score < -0.05:
                    itype = 'antagonism'
                else:
                    itype = 'neutral'
                
                interactions.append({
                    'mol_i': i, 'mol_j': j,
                    'type': itype,
                    'synergy_score': round(float(synergy_score), 3),
                    'masking_ratio': round(float(masking_ratio), 3),
                    'shared_dims': len(shared),
                    'masking_prob': round(1 - masking_ratio, 3),
                    'synergy_prob': round(max(0, synergy_score), 3),
                    'neutral_prob': round(max(0, 1 - abs(synergy_score) - (1 - masking_ratio)), 3),
                })
        
        return interactions



# ================================================================
# CategoryCorrector: ingredient 카테고리 기반 GNN 출력 보정
# ================================================================

# 각 ingredient 카테고리의 기대 22d 프로필 (향료학 기반)
# 값은 해당 카테고리에서 기대되는 각 차원의 강도
CATEGORY_PROFILES = {
    'citrus': {
        'citrus': 0.90, 'fresh': 0.55, 'green': 0.20, 'fruity': 0.20,
        'sweet': 0.10, 'ozonic': 0.10,
    },
    'floral': {
        'floral': 0.90, 'sweet': 0.30, 'powdery': 0.25, 'musk': 0.15,
        'green': 0.10, 'fresh': 0.10,
    },
    'woody': {
        'woody': 0.85, 'warm': 0.30, 'smoky': 0.25, 'earthy': 0.20,
        'amber': 0.10, 'musk': 0.05,
    },
    'spicy': {
        'spicy': 1.0, 'warm': 0.50, 'amber': 0.15, 'sweet': 0.10,
        'earthy': 0.10, 'woody': 0.05,
    },
    'fruity': {
        'fruity': 0.85, 'sweet': 0.45, 'fresh': 0.20, 'green': 0.10,
        'floral': 0.10, 'citrus': 0.10,
    },
    'gourmand': {
        'sweet': 0.85, 'warm': 0.45, 'musk': 0.20, 'amber': 0.15,
        'powdery': 0.15, 'fruity': 0.10,
    },
    'musk': {
        'musk': 1.0, 'powdery': 0.35, 'warm': 0.20, 'sweet': 0.15,
        'smoky': 0.10, 'amber': 0.10,
    },
    'amber': {
        'amber': 0.85, 'warm': 0.55, 'sweet': 0.20, 'musk': 0.15,
        'woody': 0.10, 'powdery': 0.10,
    },
    'balsamic': {
        'amber': 0.55, 'warm': 0.45, 'smoky': 0.40, 'sweet': 0.30,
        'woody': 0.15, 'earthy': 0.10,
    },
    'animalic': {
        'leather': 0.90, 'musk': 0.40, 'smoky': 0.30, 'earthy': 0.25,
        'warm': 0.25, 'woody': 0.10,
    },
    'aromatic': {
        'herbal': 0.70, 'fresh': 0.40, 'green': 0.25, 'floral': 0.20,
        'woody': 0.10, 'citrus': 0.10,
    },
    'aquatic': {
        'aquatic': 0.85, 'fresh': 0.55, 'ozonic': 0.35, 'citrus': 0.20,
        'green': 0.10, 'musk': 0.10,
    },
    'green': {
        'green': 0.85, 'fresh': 0.50, 'herbal': 0.20, 'earthy': 0.10,
        'citrus': 0.05, 'woody': 0.05,
    },
    'chypre': {
        'woody': 0.50, 'earthy': 0.45, 'green': 0.35, 'amber': 0.15,
        'floral': 0.10, 'smoky': 0.10,
    },
    'synthetic': {
        'metallic': 0.50, 'waxy': 0.40, 'ozonic': 0.20, 'powdery': 0.15,
        'fresh': 0.10, 'sweet': 0.05,
    },
    # === DB 확장 카테고리 (17개 추가) ===
    'earthy': {
        'earthy': 0.90, 'woody': 0.30, 'warm': 0.15, 'green': 0.10,
        'herbal': 0.10, 'smoky': 0.05,
    },
    'smoky': {
        'smoky': 0.90, 'woody': 0.30, 'warm': 0.25, 'leather': 0.15,
        'earthy': 0.15, 'amber': 0.10,
    },
    'herbal': {
        'herbal': 0.90, 'green': 0.40, 'fresh': 0.30, 'earthy': 0.10,
        'woody': 0.10, 'floral': 0.05,
    },
    'resinous': {
        'amber': 0.60, 'warm': 0.45, 'smoky': 0.35, 'sweet': 0.20,
        'woody': 0.15, 'earthy': 0.10,
    },
    'resin': {
        'amber': 0.55, 'warm': 0.45, 'smoky': 0.35, 'sweet': 0.20,
        'woody': 0.15, 'earthy': 0.10,
    },
    'leather': {
        'leather': 0.90, 'smoky': 0.35, 'warm': 0.25, 'earthy': 0.15,
        'woody': 0.10, 'musk': 0.10,
    },
    'powdery': {
        'powdery': 0.90, 'musk': 0.35, 'sweet': 0.25, 'floral': 0.15,
        'warm': 0.10, 'fresh': 0.05,
    },
    'fresh': {
        'fresh': 0.90, 'citrus': 0.30, 'green': 0.25, 'ozonic': 0.15,
        'aquatic': 0.10, 'herbal': 0.10,
    },
    'aldehyde': {
        'metallic': 0.45, 'waxy': 0.35, 'fresh': 0.20, 'powdery': 0.20,
        'citrus': 0.10, 'floral': 0.10,
    },
    'aldehydic': {
        'metallic': 0.45, 'waxy': 0.35, 'fresh': 0.20, 'powdery': 0.20,
        'citrus': 0.10, 'floral': 0.10,
    },
    'cooling': {
        'fresh': 0.85, 'herbal': 0.30, 'green': 0.20, 'ozonic': 0.20,
        'citrus': 0.10, 'aquatic': 0.10,
    },
    'ozonic': {
        'ozonic': 0.85, 'fresh': 0.50, 'aquatic': 0.25, 'green': 0.15,
        'citrus': 0.10, 'metallic': 0.05,
    },
    'solvent': {
        'metallic': 0.40, 'fresh': 0.30, 'ozonic': 0.25, 'waxy': 0.20,
        'sweet': 0.10, 'citrus': 0.05,
    },
    'waxy': {
        'waxy': 0.80, 'sweet': 0.25, 'powdery': 0.20, 'floral': 0.15,
        'warm': 0.10, 'musk': 0.10,
    },
    'marine': {
        'aquatic': 0.85, 'fresh': 0.50, 'ozonic': 0.30, 'citrus': 0.10,
        'green': 0.10, 'musk': 0.05,
    },
    'warm': {
        'warm': 0.90, 'amber': 0.40, 'sweet': 0.20, 'woody': 0.15,
        'musk': 0.10, 'spicy': 0.10,
    },
    'base': {
        'musk': 0.50, 'woody': 0.30, 'warm': 0.25, 'amber': 0.20,
        'powdery': 0.15, 'earthy': 0.10,
    },
    'carrier': {
        'musk': 0.30, 'sweet': 0.20, 'warm': 0.15, 'powdery': 0.10,
        'fresh': 0.10, 'woody': 0.05,
    },
}

# 특정 ingredient별 커스텀 프로필 (카테고리보다 우선)
INGREDIENT_SPECIFIC_PROFILES = {
    'oud':          {'smoky': 0.80, 'woody': 0.55, 'musk': 0.20, 'earthy': 0.10, 'leather': 0.10, 'warm': 0.05},
    'cinnamon':     {'spicy': 1.0, 'warm': 0.50, 'sweet': 0.10, 'amber': 0.05},
    'oakmoss':      {'green': 0.70, 'earthy': 0.55, 'woody': 0.30, 'herbal': 0.15, 'fresh': 0.10},
    'patchouli':    {'earthy': 0.70, 'woody': 0.40, 'warm': 0.20, 'green': 0.15, 'sweet': 0.10},
    'bergamot':     {'citrus': 0.90, 'fresh': 0.35, 'green': 0.15, 'floral': 0.10, 'fruity': 0.10},
    'lemon':        {'citrus': 0.95, 'fresh': 0.45, 'green': 0.10, 'fruity': 0.10},
    'vetiver':      {'woody': 0.60, 'earthy': 0.45, 'smoky': 0.30, 'green': 0.20, 'warm': 0.10},
    'frankincense': {'smoky': 0.50, 'warm': 0.40, 'amber': 0.30, 'woody': 0.20, 'earthy': 0.10},
    'saffron':      {'spicy': 0.55, 'warm': 0.30, 'smoky': 0.25, 'leather': 0.15, 'earthy': 0.10},
    'benzoin':      {'amber': 0.55, 'sweet': 0.45, 'warm': 0.35, 'smoky': 0.15, 'powdery': 0.10},
    # fresh_aquatic accord support
    'marine':       {'aquatic': 0.90, 'fresh': 0.50, 'citrus': 0.20, 'ozonic': 0.30, 'green': 0.10},
    'white_musk':   {'musk': 0.90, 'fresh': 0.25, 'citrus': 0.15, 'powdery': 0.30, 'sweet': 0.10},
    'bamboo':       {'green': 0.80, 'fresh': 0.40, 'citrus': 0.10, 'aquatic': 0.10, 'woody': 0.10},
    'cedarwood':    {'woody': 0.80, 'warm': 0.20, 'earthy': 0.15, 'fresh': 0.10, 'green': 0.05},
    # oriental_base accord support
    'vanilla':      {'sweet': 0.80, 'warm': 0.50, 'amber': 0.15, 'powdery': 0.10, 'spicy': 0.05},
    'sandalwood':   {'woody': 0.80, 'warm': 0.25, 'musk': 0.15, 'earthy': 0.05, 'sweet': 0.05},
    'amber':        {'amber': 0.90, 'warm': 0.50, 'sweet': 0.15, 'musk': 0.10, 'spicy': 0.10},
}


class CategoryCorrector:
    """ingredient 카테고리 정보로 GNN 출력 벡터를 보정
    
    GNN 모델이 green/woody/fresh/herbal에 편향된 출력을 보정하기 위해
    ingredient의 카테고리 프로필과 블렌딩합니다.
    
    보정 공식: corrected = (1 - blend) * gnn_vec + blend * category_profile
    """
    
    def __init__(self, blend_ratio=0.91):
        self._blend = blend_ratio
        self._dim_idx = {d: i for i, d in enumerate(ODOR_DIMENSIONS)}
        
        # Korean descriptor → V22 dimension mapping
        self._desc_to_dim = {
            '시트러스': 'citrus', '상큼': 'citrus', '레몬': 'citrus',
            '플로럴': 'floral', '로맨틱': 'floral', '꽃': 'floral', '로지': 'floral',
            '우디': 'woody', '나무': 'woody', '드라이': 'woody',
            '스파이시': 'spicy', '매운': 'spicy', '시나몬': 'spicy',
            '머스크': 'musk', '클린': 'musk', '파우더리': 'powdery',
            '달콤': 'sweet', '바닐라': 'sweet', '꿀': 'sweet', '크리미': 'sweet',
            '따뜻한': 'warm', '포근': 'warm', '앰비': 'warm',
            '앰버': 'amber', '레진': 'amber', '발사믹': 'amber',
            '스모키': 'smoky', '연기': 'smoky', '다크': 'smoky', '신비로운': 'smoky',
            '가죽': 'leather', '동물적': 'leather',
            '어시': 'earthy', '미네랄': 'earthy', '흙': 'earthy',
            '아쿠아': 'aquatic', '바다': 'aquatic', '오션': 'aquatic',
            '허벌': 'herbal', '아로마틱': 'herbal', '허브': 'herbal',
            '프루티': 'fruity', '과일': 'fruity', '베리': 'fruity',
            '그린': 'green', '풀': 'green', '내추럴': 'green',
            '메탈릭': 'metallic', '왁시': 'waxy',
            '오존': 'ozonic', '쿨링': 'fresh', '신선': 'fresh',
            '부드러운': 'powdery', '벨벳': 'musk',
        }
        
        # 카테고리별 22d 벡터 미리 계산
        self._profiles = {}
        for cat, dims in CATEGORY_PROFILES.items():
            vec = np.zeros(N_ODOR_DIM)
            for dim_name, value in dims.items():
                if dim_name in self._dim_idx:
                    vec[self._dim_idx[dim_name]] = value
            self._profiles[cat] = vec
        
        # ingredient-specific 프로필 벡터 계산
        self._ing_profiles = {}
        for ing_id, dims in INGREDIENT_SPECIFIC_PROFILES.items():
            vec = np.zeros(N_ODOR_DIM)
            for dim_name, value in dims.items():
                if dim_name in self._dim_idx:
                    vec[self._dim_idx[dim_name]] = value
            self._ing_profiles[ing_id] = vec
        
        # ingredients 로드: JSON 우선 (캘리브레이션 기준), DB 보충
        self._ingredients = {}
        
        # 1) ingredients.json 먼저 (캘리브레이션된 카테고리/디스크립터 우선)
        try:
            import json
            from pathlib import Path
            data_dir = Path(__file__).resolve().parent.parent / 'data'
            ing_path = data_dir / 'ingredients.json'
            if ing_path.exists():
                with open(ing_path, 'r', encoding='utf-8') as f:
                    for ing in json.load(f):
                        self._ingredients[ing['id']] = ing
        except Exception:
            pass
        
        json_count = len(self._ingredients)
        
        # 2) DB에서 추가 향료 보충 (JSON에 없는 것만)
        db_count = 0
        try:
            import database as db_module
            db_ings = db_module.get_all_ingredients()
            for ing in db_ings:
                ing_id = ing.get('id')
                if ing_id and ing_id not in self._ingredients:
                    self._ingredients[ing_id] = {
                        'id': ing_id,
                        'category': ing.get('category'),
                        'descriptors': ing.get('descriptors') or [],
                    }
                    db_count += 1
        except Exception:
            pass
        
        print(f"[CategoryCorrector] {len(self._ingredients)} ingredients "
              f"(JSON={json_count}, DB+={db_count}), "
              f"{len(self._profiles)} category profiles, blend={self._blend}")
    
    def correct(self, gnn_vec, ingredient_id=None, category=None):
        """GNN 벡터를 카테고리 프로필과 블렌딩하여 보정
        
        Args:
            gnn_vec: numpy [22] - GNN 출력
            ingredient_id: str - ingredient ID (ingredients.json)
            category: str - 직접 카테고리 지정
        
        Returns:
            numpy [22] - 보정된 벡터
        """
        # 1) ingredient-specific 프로필 우선 확인
        if ingredient_id and ingredient_id in self._ing_profiles:
            profile = self._ing_profiles[ingredient_id]
        else:
            # 2) 카테고리 프로필 사용
            cat = category
            if cat is None and ingredient_id and ingredient_id in self._ingredients:
                cat = self._ingredients[ingredient_id].get('category')
            
            if cat is None or cat not in self._profiles:
                return gnn_vec  # 보정 불가
            
            profile = self._profiles[cat]
        
        # 블렌딩: GNN 벡터의 강점은 유지하면서 카테고리 특성 주입
        corrected = (1.0 - self._blend) * gnn_vec + self._blend * profile
        
        # descriptor 기반 추가 부스트
        if ingredient_id and ingredient_id in self._ingredients:
            descriptors = self._ingredients[ingredient_id].get('descriptors', [])
            desc_boost = 0.12
            for desc in descriptors:
                mapped = self._desc_to_dim.get(desc)
                if mapped and mapped in self._dim_idx:
                    idx = self._dim_idx[mapped]
                    corrected[idx] = max(corrected[idx], desc_boost)
        
        return np.clip(corrected, 0, 1)
    
    def batch_correct(self, gnn_vecs, ingredient_ids=None, categories=None):
        """여러 벡터를 한번에 보정"""
        results = []
        for i, vec in enumerate(gnn_vecs):
            ing_id = ingredient_ids[i] if ingredient_ids and i < len(ingredient_ids) else None
            cat = categories[i] if categories and i < len(categories) else None
            results.append(self.correct(vec, ingredient_id=ing_id, category=cat))
        return np.array(results)


# 전역 인스턴스
_category_corrector = CategoryCorrector()
_concentration_modulator = ConcentrationModulator()
_physics_mixture = PhysicsMixture()


# ================================================================
# MixtureTransformer: 혼합물 상호작용 예측 (레거시, 학습 모델용)
# ================================================================

class MixtureTransformer(nn.Module):
    """여러 분자의 냄새가 섞일 때의 비선형 상호작용 모델링
    
    학습 목표:
    - Masking: A 분자가 B 분자의 향을 덮음 (Attention weight 편향)
    - Synergy: A+B → 새로운 차원 활성화
    - Suppression: A+B → 특정 차원 상쇄
    """
    
    def __init__(self, d_model=N_ODOR_DIM, nhead=4, num_layers=4, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.d_model = d_model
        
        # 입력 프로젝션: 20d → 64d (Transformer 내부)
        self.input_proj = nn.Linear(d_model + 1, 64)  # +1 for concentration
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 출력 프로젝션: 64d → 20d (mixture odor vector)
        self.output_proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
            nn.Sigmoid(),
        )
        
        # Interaction classifier: 분자 쌍 → 상호작용 유형
        self.interaction_head = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # [masking, synergy, neutral]
        )
        
        self.to(self.device)
        print(f"[MixtureTransformer] {num_layers}-layer, {nhead}-head on {self.device}")
    
    def forward(self, odor_vectors, concentrations=None):
        """
        odor_vectors: [N, 20] — 각 분자의 냄새 벡터
        concentrations: [N] — 농도 (optional)
        Returns: [20] — 혼합물의 최종 냄새 벡터
        """
        N = odor_vectors.shape[0]
        
        if concentrations is None:
            concentrations = torch.ones(N, 1, device=self.device)
        else:
            concentrations = concentrations.unsqueeze(-1)
        
        # 농도를 추가 feature로
        x = torch.cat([odor_vectors, concentrations], dim=-1)  # [N, 21]
        x = self.input_proj(x)  # [N, 64]
        x = x.unsqueeze(0)  # [1, N, 64] — batch dim
        
        # Transformer: 분자 간 상호작용 학습
        attended = self.transformer(x)  # [1, N, 64]
        
        # 농도 가중 풀링: 농도 높은 분자가 최종 향에 더 기여
        weights = F.softmax(concentrations.squeeze(-1), dim=-1)  # [N]
        pooled = (attended.squeeze(0) * weights.unsqueeze(-1)).sum(dim=0)  # [64]
        
        # 최종 냄새 벡터
        mixture_odor = self.output_proj(pooled)  # [20]
        return mixture_odor
    
    def predict_interactions(self, odor_vectors, concentrations=None):
        """분자 쌍별 상호작용 분석"""
        N = odor_vectors.shape[0]
        if concentrations is None:
            concentrations = torch.ones(N, 1, device=self.device)
        else:
            concentrations = concentrations.unsqueeze(-1)
        
        x = torch.cat([odor_vectors, concentrations], dim=-1)
        x = self.input_proj(x).unsqueeze(0)
        attended = self.transformer(x).squeeze(0)  # [N, 64]
        
        interactions = []
        for i in range(N):
            for j in range(i + 1, N):
                pair_feat = torch.cat([attended[i], attended[j]])  # [128]
                logits = self.interaction_head(pair_feat)
                probs = F.softmax(logits, dim=-1)
                itype_idx = probs.argmax().item()
                itype = ['masking', 'synergy', 'neutral'][itype_idx]
                interactions.append({
                    'mol_i': i, 'mol_j': j,
                    'type': itype,
                    'masking_prob': float(probs[0]),
                    'synergy_prob': float(probs[1]),
                    'neutral_prob': float(probs[2]),
                })
        return interactions


# ================================================================
# OdorPredictor: 통합 파이프라인
# ================================================================

class OdorPredictor:
    """Engine 2 통합: 물리엔진 스냅샷 → AI 예측 냄새"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gnn = OdorGNN(device=device)
        self.pom = PrincipalOdorMap()
        self.pom.set_gnn(self.gnn)
        self.transformer = MixtureTransformer(device=device)
        
        # 학습된 가중치 로드
        self._load_trained_weights()
        print(f"[OdorPredictor] Pipeline ready | GNN → POM → Transformer")
    
    def _load_trained_weights(self):
        """train_models.py에서 학습된 가중치 로드 (앙상블 자동 감지)"""
        try:
            import train_models
            result = train_models.load_odor_gnn(str(self.device))
            
            if result[2] == 'ensemble':
                # v4 + v5 둘 다 있음 → 앙상블
                v4_model, v5_model, _ = result
                self.gnn.load_ensemble(v4_model, v5_model)
            elif result[0]:
                # 단일 모델
                model, _, version = result
                self.gnn.load_trained(model, version=version)
            
            mix_model = train_models.load_mixture_transformer(str(self.device))
            if mix_model:
                # MixtureTransformer 내부 가중치를 학습된 것으로 교체
                self._trained_mixer = mix_model
                print(f"[OdorPredictor] ✅ Trained MixtureNet loaded")
            else:
                self._trained_mixer = None
        except Exception as e:
            print(f"[OdorPredictor] ⚠ Trained weights not found: {e}")
            self._trained_mixer = None
    
    @torch.no_grad()
    def predict_single(self, smiles, concentration_pct=None):
        """단일 분자 → 냄새 벡터 + 설명 (농도 보정 포함)"""
        vec = self.gnn.encode(smiles)
        raw_vec = vec.copy()
        
        # 농도 보정 적용
        details = None
        if concentration_pct is not None and concentration_pct > 0:
            vec, details = _concentration_modulator.modulate(
                vec, concentration_pct, return_details=True)
        
        top_dims = self.pom.describe_vector(vec, top_k=5)
        nearest, sim = self.pom.nearest_anchor(vec)
        result = {
            'odor_vector': vec.tolist(),
            'raw_vector': raw_vec.tolist(),
            'top_dimensions': top_dims,
            'nearest_smell': nearest,
            'similarity': round(sim, 3),
        }
        if concentration_pct is not None:
            result['concentration_pct'] = concentration_pct
        if details:
            result['concentration_effects'] = details
        return result
    
    @torch.no_grad()
    def predict_mixture(self, smiles_list, concentrations=None):
        """혼합물 → 최종 냄새 벡터 + 상호작용 (농도 보정 + 물리 혼합)"""
        N = len(smiles_list)
        
        # 기본 농도
        if concentrations is None:
            concentrations = [1.0] * N
        conc_arr = np.array(concentrations, dtype=np.float64)
        
        # 1. 각 분자 V22 인코딩
        raw_vecs = self.gnn.encode_batch(smiles_list)
        
        # 2. 농도 보정: ConcentrationModulator 적용
        modulated_vecs = _concentration_modulator.batch_modulate(raw_vecs, conc_arr, smiles_list=smiles_list)
        
        # 3. 개별 분자 분석
        molecules = []
        for i, (smiles, raw, mod) in enumerate(zip(smiles_list, raw_vecs, modulated_vecs)):
            top_dims = self.pom.describe_vector(mod, top_k=3)
            nearest, sim = self.pom.nearest_anchor(mod)
            molecules.append({
                'smiles': smiles,
                'odor_vector': mod.tolist(),
                'raw_vector': raw.tolist(),
                'concentration_pct': float(conc_arr[i]),
                'dominant_smell': top_dims[0][0] if top_dims else 'unknown',
                'nearest_anchor': nearest,
                'anchor_similarity': round(sim, 3),
            })
        
        # 4. 물리 기반 혼합: PhysicsMixture (학습된 모델보다 우선)
        mixture_vec, analysis = _physics_mixture.mix(
            modulated_vecs, conc_arr, return_analysis=True)
        
        # 5. 혼합물 분석
        mixture_top = self.pom.describe_vector(mixture_vec, top_k=5)
        mixture_nearest, mixture_sim = self.pom.nearest_anchor(mixture_vec)
        
        # 6. 상호작용 분석 (PhysicsMixture 결과 사용)
        interactions = analysis.get('interactions', [])
        
        return {
            'mixture_vector': mixture_vec.tolist(),
            'top_dimensions': mixture_top,
            'nearest_smell': mixture_nearest,
            'similarity': round(mixture_sim, 3),
            'molecules': molecules,
            'interactions': interactions,
            'mixture_method': 'physics_v1',
        }
    
    @torch.no_grad()
    def predict_timeline(self, smiles_list, concentrations, timeline):
        """물리엔진 타임라인 → 시간별 냄새 변화 예측
        
        Args:
            smiles_list: 원래 분자 리스트
            concentrations: 초기 농도
            timeline: ThermodynamicsEngine의 타임라인 출력
        
        Returns:
            각 시점의 냄새 벡터 + 설명
        """
        # 모든 분자 미리 인코딩
        all_vecs = self.gnn.encode_batch(smiles_list)
        
        odor_timeline = []
        for snapshot in timeline:
            t_min = snapshot['time_min']
            
            # 시점별 잔류 농도 가져오기
            remaining = []
            active_vecs = []
            active_smiles = []
            
            for i, mol in enumerate(snapshot['molecules']):
                pct = mol['remaining_pct']
                if pct > 0.01:  # 거의 증발한 분자 제외
                    remaining.append(pct)
                    active_vecs.append(all_vecs[i])
                    active_smiles.append(mol['smiles'])
            
            if not active_vecs:
                odor_timeline.append({
                    'time_min': t_min,
                    'odor_vector': [0.0] * N_ODOR_DIM,
                    'dominant': 'none',
                    'intensity': 0.0,
                })
                continue
            
            # 농도 보정 + 물리 혼합
            conc_arr = np.array(remaining, dtype=np.float64)
            modulated = _concentration_modulator.batch_modulate(
                np.array(active_vecs), conc_arr)
            mixture_vec = _physics_mixture.mix(modulated, conc_arr)
            
            top_dims = self.pom.describe_vector(mixture_vec, top_k=3)
            intensity = float(sum(remaining)) / float(sum(concentrations) + 1e-8)
            
            odor_timeline.append({
                'time_min': t_min,
                'odor_vector': mixture_vec.tolist(),
                'top_dimensions': top_dims,
                'dominant': top_dims[0][0] if top_dims else 'none',
                'intensity': round(intensity, 3),
                'active_molecules': len(active_vecs),
                'note_balance': snapshot.get('note_balance', {}),
            })
        
        return odor_timeline
    
    def target_vector(self, description):
        """텍스트 설명 → 목표 냄새 벡터"""
        return self.pom.target_from_description(description)
    
    def distance_to_target(self, mixture_vec, target_vec):
        """혼합 벡터와 목표 벡터 사이 거리"""
        return self.pom.distance(np.array(mixture_vec), target_vec)


# ================================================================
# 전역 인스턴스 (lazy)
# ================================================================

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = OdorPredictor()
    return _predictor

def predict_single(smiles):
    return get_predictor().predict_single(smiles)

def predict_mixture(smiles_list, concentrations=None):
    return get_predictor().predict_mixture(smiles_list, concentrations)

def predict_timeline(smiles_list, concentrations, timeline):
    return get_predictor().predict_timeline(smiles_list, concentrations, timeline)

def target_from_text(description):
    return get_predictor().target_vector(description)
