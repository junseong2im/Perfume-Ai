# train_models.py — DB 기반 모델 학습 + 가중치 저장/로드
# ================================================================
# v3: 다이어트 OdorGNN (~25K params) + Mixup 증강 + Scaffold Split
# ================================================================

import os, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict

import database as db

WEIGHTS_DIR = Path(__file__).parent / 'weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

# 22차원 냄새 공간 (fatty/waxy 독립 차원 추가)
ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',  # NEW: co-occurrence 분석으로 독립 승격
]
N_DIM = len(ODOR_DIMENSIONS)

# ================================================================
# Co-occurrence 소프트 매핑 (label_mapping.py에서 로드)
# ================================================================
try:
    from scripts.label_mapping import (
        descriptor_to_soft_target, descriptor_to_138d_target,
        get_soft_map, build_all_labels,
        HARD_MAPPING, ODOR_DIMENSIONS_22, N_DIM_22,
    )
except ImportError:
    from label_mapping import (
        descriptor_to_soft_target, descriptor_to_138d_target,
        get_soft_map, build_all_labels,
        HARD_MAPPING, ODOR_DIMENSIONS_22, N_DIM_22,
    )

# Module-level soft map cache (lazy loaded)
_SOFT_MAP = None
_ALL_LABELS_LIST = None

def _get_label_tools(molecules=None):
    """소프트 매핑 + 전체 라벨 리스트 로드 (캐시됨)"""
    global _SOFT_MAP, _ALL_LABELS_LIST
    if _SOFT_MAP is None:
        if molecules is None:
            molecules = db.get_all_molecules()
        _SOFT_MAP = get_soft_map(molecules)
        _ALL_LABELS_LIST = build_all_labels(molecules)
    return _SOFT_MAP, _ALL_LABELS_LIST

def _descriptor_to_target(odor_labels, soft_map=None):
    """DB descriptor 리스트 → 22d soft target vector (co-occurrence 기반)"""
    if soft_map is None:
        soft_map, _ = _get_label_tools()
    return descriptor_to_soft_target(odor_labels, soft_map)

def _descriptor_to_aux_target(odor_labels, all_labels_list=None):
    """DB descriptor 리스트 → 138d 이진 벡터 (Multi-task aux head용)"""
    if all_labels_list is None:
        _, all_labels_list = _get_label_tools()
    return descriptor_to_138d_target(odor_labels, all_labels_list)


# ================================================================
# 학습 데이터 로더 (DB 우선, CSV fallback)
# ================================================================

def load_training_molecules():
    """학습용 분자 데이터 로드.
    
    우선순위:
    1. PostgreSQL DB (db.get_all_molecules())
    2. curated_GS_LF_merged_4983.csv (4,983 GoodScents+Leffingwell)
    3. multi_source_unified.csv (unified 데이터)
    
    Returns:
        list[dict]: [{smiles, odor_labels, name?}, ...]
    """
    # 1) DB first
    try:
        molecules = db.get_all_molecules()
        labeled = [m for m in molecules 
                   if m.get('odor_labels') and m['odor_labels'] != ['odorless']]
        if len(labeled) >= 100:
            print(f"  [DataLoader] DB: {len(labeled)} labeled molecules")
            return molecules
    except Exception as e:
        print(f"  [DataLoader] DB unavailable: {e}")
    
    # 2) CSV fallback: curated GS-LF
    import csv
    csv_paths = [
        Path(__file__).parent / 'data' / 'curated_GS_LF_merged_4983.csv',
        Path(__file__).parent / 'data' / 'multi_source_unified.csv',
    ]
    
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        
        molecules = []
        try:
            with open(csv_path, encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Find SMILES and descriptor columns
                smi_col = None
                desc_col = None
                for i, h in enumerate(header):
                    h_lower = h.lower().strip()
                    if h_lower in ('nonstereosmiles', 'smiles', 'canonical_smiles'):
                        smi_col = i
                    elif h_lower in ('descriptors', 'odor_labels', 'labels'):
                        desc_col = i
                
                if smi_col is None:
                    smi_col = 0  # first column assumed SMILES
                
                for row in reader:
                    if len(row) <= smi_col:
                        continue
                    smiles = row[smi_col].strip()
                    if not smiles:
                        continue
                    
                    # Parse descriptors
                    labels = []
                    if desc_col is not None and desc_col < len(row):
                        raw = row[desc_col].strip()
                        if raw:
                            labels = [l.strip() for l in raw.split(';') if l.strip()]
                    
                    if labels and labels != ['odorless']:
                        molecules.append({
                            'smiles': smiles,
                            'odor_labels': labels,
                        })
            
            if len(molecules) >= 50:
                print(f"  [DataLoader] CSV: {csv_path.name} → {len(molecules)} molecules")
                return molecules
        except Exception as e:
            print(f"  [DataLoader] CSV {csv_path.name} error: {e}")
    
    print("  [DataLoader] ⚠ No training data found!")
    return []


# ================================================================
# SMILES → 분자 특성 벡터 (RDKit)
# ================================================================

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def _smiles_to_features(smiles, n_feat=128):
    """SMILES → 고정 크기 분자 특성 벡터"""
    feats = np.zeros(n_feat, dtype=np.float32)
    
    if not HAS_RDKIT or not smiles:
        return feats
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return feats
    
    # 기본 분자 특성 (0-19)
    try:
        feats[0] = Descriptors.MolWt(mol) / 500.0
        feats[1] = Descriptors.MolLogP(mol) / 10.0
        feats[2] = Descriptors.TPSA(mol) / 200.0
        feats[3] = rdMolDescriptors.CalcNumHBD(mol) / 5.0
        feats[4] = rdMolDescriptors.CalcNumHBA(mol) / 10.0
        feats[5] = rdMolDescriptors.CalcNumRotatableBonds(mol) / 15.0
        feats[6] = rdMolDescriptors.CalcNumAromaticRings(mol) / 4.0
        feats[7] = rdMolDescriptors.CalcNumRings(mol) / 5.0
        feats[8] = mol.GetNumHeavyAtoms() / 50.0
        feats[9] = Descriptors.FractionCSP3(mol)
    except:
        pass
    
    # Morgan fingerprint bits (20-127)
    try:
        from rdkit.Chem import AllChem
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=108)
        for i in range(108):
            feats[20 + i] = fp.GetBit(i)
    except:
        pass
    
    # SMARTS 규칙 기반 특성 (10-19) — 향 관련 작용기
    smarts_rules = {
        10: '[OX2H]',              # hydroxyl → sweet/floral
        11: '[CX3](=O)[OX2H1]',   # carboxylic acid → sour
        12: '[CX3](=O)[#6]',      # ketone/aldehyde → sweet
        13: 'c1ccccc1',           # benzene ring → aromatic
        14: '[SX2]',              # thiol → sulfurous
        15: '[NX3]',              # amine → fishy/earthy
        16: '[OX2]([#6])[#6]',    # ether → ethereal
        17: 'C=C',                # alkene → green
        18: '[CX3](=O)O[#6]',    # ester → fruity
        19: 'C#N',                # nitrile → sharp
    }
    for idx, pattern in smarts_rules.items():
        try:
            pat = Chem.MolFromSmarts(pattern)
            if pat and mol.HasSubstructMatch(pat):
                feats[idx] = 1.0
        except:
            pass
    
    return feats


# ================================================================
# Scaffold Split — 같은 골격은 같은 셋에 할당
# ================================================================

def _get_scaffold(smiles):
    """SMILES → Murcko scaffold (문자열)"""
    try:
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaffold = GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return smiles


def scaffold_split(dataset, val_ratio=0.2, seed=42):
    """Scaffold 기반 train/val split — 새로운 골격을 val에 배치"""
    rng = np.random.RandomState(seed)
    
    # scaffold → indices 매핑 (canonical SMILES로 scaffold 계산)
    scaffold_to_indices = defaultdict(list)
    for i in range(len(dataset.data)):
        smiles = dataset.data[i][2]  # (feat, target, smiles)
        scaffold = _get_scaffold(smiles)
        scaffold_to_indices[scaffold].append(i)
    
    # scaffold 크기 순 정렬 (작은 그룹부터 → val에 할당)
    scaffolds = list(scaffold_to_indices.values())
    rng.shuffle(scaffolds)
    scaffolds.sort(key=len)  # 작은 그룹부터
    
    n_total = len(dataset)
    n_val_target = int(n_total * val_ratio)
    
    train_idx, val_idx = [], []
    
    for group in scaffolds:
        # val이 아직 부족하면 작은 그룹을 val에 배치
        if len(val_idx) < n_val_target:
            val_idx.extend(group)
        else:
            train_idx.extend(group)
    
    print(f"  [Scaffold Split] {len(scaffold_to_indices)} unique scaffolds")
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} "
          f"(val scaffolds are unseen during training)")
    
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    
    return train_set, val_set


# ================================================================
# ChemBERTa Cache Loader
# ================================================================

def _load_bert_cache():
    """ChemBERTa 캐시 파일 로드 (없으면 None)"""
    cache_path = WEIGHTS_DIR / 'chemberta_cache.npz'
    if not cache_path.exists():
        print("  [BERTCache] Not found, using FP fallback")
        return None
    
    data = np.load(cache_path, allow_pickle=True)
    embeddings = data['embeddings']
    smiles_arr = data['smiles']
    hidden_size = int(data['hidden_size'])
    
    smiles_map = {s: i for i, s in enumerate(smiles_arr)}
    print(f"  [BERTCache] Loaded {len(smiles_map)} embeddings ({hidden_size}d)")
    
    return {
        'embeddings': embeddings,
        'map': smiles_map,
        'hidden_size': hidden_size,
    }


# ================================================================
# Dataset
# ================================================================

class OdorDataset(Dataset):
    """DB에서 로드한 (SMILES -> 20d odor) 데이터셋
    
    v5: ChemBERTa 384d 캐시 우선, 미스 시 128d FP fallback
         + SMILES Randomization 증강 지원
    """
    
    def __init__(self, molecules, bert_cache=None, n_augment=0, with_aux=False):
        """OdorDataset 초기화
        
        Args:
            molecules: DB에서 로드한 분자 리스트
            bert_cache: ChemBERTa 캐시 (없으면 자동 로드)
            n_augment: SMILES 랜덤화 변형 수 (0=비활성화, 10=권장)
            with_aux: True면 138d aux target도 저장 (multi-task용)
        """
        self.data = []
        self.aux_targets = []  # 138d raw label vectors
        self.with_aux = with_aux
        skipped = 0
        bert_hits = 0
        fp_fallbacks = 0
        aug_hits = 0
        
        # Soft mapping + all labels 로드
        soft_map, all_labels_list = _get_label_tools(molecules)
        
        # ChemBERTa 캐시 로드
        if bert_cache is None:
            bert_cache = _load_bert_cache()
        
        use_bert = bert_cache is not None
        if use_bert:
            self._input_dim = bert_cache['hidden_size']
        else:
            self._input_dim = 128
        
        # SMILES 증강 준비
        do_augment = n_augment > 0 and use_bert
        if do_augment:
            try:
                from scripts.smiles_augment import randomize_smiles
            except ImportError:
                try:
                    from smiles_augment import randomize_smiles
                except ImportError:
                    do_augment = False
        
        for mol in molecules:
            smiles = mol.get('smiles', '')
            labels = mol.get('odor_labels', [])
            
            if not smiles or not labels or labels == ['odorless']:
                skipped += 1
                continue
            
            target = _descriptor_to_target(labels, soft_map)
            if target.max() == 0:
                skipped += 1
                continue
            
            # 138d aux target (multi-task용)
            aux_target = _descriptor_to_aux_target(labels, all_labels_list) if with_aux else None
            
            # --- 원본 SMILES ---
            if use_bert and smiles in bert_cache['map']:
                feats = bert_cache['embeddings'][bert_cache['map'][smiles]]
                bert_hits += 1
                self.data.append((
                    torch.tensor(feats, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.float32),
                    smiles,
                ))
                if with_aux and aux_target is not None:
                    self.aux_targets.append(torch.tensor(aux_target, dtype=torch.float32))
            elif use_bert:
                fp_fallbacks += 1
                continue
            else:
                feats = _smiles_to_features(smiles)
                self.data.append((
                    torch.tensor(feats, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.float32),
                    smiles,
                ))
                if with_aux and aux_target is not None:
                    self.aux_targets.append(torch.tensor(aux_target, dtype=torch.float32))
            
            # --- SMILES 증강: 랜덤 변형이 캐시에 있으면 추가 ---
            if do_augment:
                variants = randomize_smiles(smiles, n_augment=n_augment)
                for v in variants:
                    if v == smiles:  # 원본은 이미 추가됨
                        continue
                    if v in bert_cache['map']:
                        aug_feats = bert_cache['embeddings'][bert_cache['map'][v]]
                        aug_hits += 1
                        self.data.append((
                            torch.tensor(aug_feats, dtype=torch.float32),
                            torch.tensor(target, dtype=torch.float32),
                            v,
                        ))
                        if with_aux and aux_target is not None:
                            self.aux_targets.append(torch.tensor(aux_target, dtype=torch.float32))
        
        mode = f"ChemBERTa {self._input_dim}d" if use_bert else "FP 128d"
        aug_info = f", aug={aug_hits}" if do_augment else ""
        print(f"  [OdorDataset] {mode} | {len(self.data)} samples"
              f" (bert={bert_hits}, skip={skipped+fp_fallbacks}{aug_info})")
    
    @property
    def input_dim(self):
        return self._input_dim
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.with_aux and idx < len(self.aux_targets):
            return self.data[idx][0], self.data[idx][1], self.aux_targets[idx]
        return self.data[idx][0], self.data[idx][1]


# ================================================================
# Mixup 증강 — Fingerprint 입력에서 유일하게 효과 있는 방법
# ================================================================

def mixup_batch(features, targets, alpha=0.4):
    """배치 내 두 샘플을 랜덤 비율로 섞어 새 데이터 생성
    
    Input(x_i, y_i)와 랜덤 파트너(x_j, y_j)를 λ 비율로 혼합:
        x_mix = λ·x_i + (1-λ)·x_j
        y_mix = λ·y_i + (1-λ)·y_j
    
    Fingerprint가 같아도 라벨이 다른 분자끼리 섞이면
    모델이 '보간 공간'을 학습 → 일반화 능력 향상
    """
    batch_size = features.size(0)
    
    # Beta distribution에서 λ 샘플링
    lam = np.random.beta(alpha, alpha, size=batch_size).astype(np.float32)
    lam = np.maximum(lam, 1 - lam)  # λ >= 0.5 보장 (원본 우세)
    lam = torch.tensor(lam).unsqueeze(1).to(features.device)
    
    # 랜덤 셔플 인덱스
    perm = torch.randperm(batch_size).to(features.device)
    
    mixed_features = lam * features + (1 - lam) * features[perm]
    mixed_targets = lam * targets + (1 - lam) * targets[perm]
    
    return mixed_features, mixed_targets


# ================================================================
# OdorGNN v3 Legacy (128d FP → 20d, ~25K params) — fallback용
# ================================================================

class TrainableOdorNet(nn.Module):
    """v3 Legacy: 128d FP -> 20d (fallback)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.BatchNorm1d(128), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.GELU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, N_DIM), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


# ================================================================
# OdorGNN v4 (384d ChemBERTa → 20d, ~135K params)
# ================================================================

class TrainableOdorNetV4(nn.Module):
    """ChemBERTa(384d) -> 22d odor vector
    
    384 -> 256 -> 128 -> 22  (~135K params)
    """
    
    def __init__(self, input_dim=384):
        super().__init__()
        self.input_dim = input_dim
        
        # Projection: 384 -> 256
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.15),
        )
        
        # Hidden: 256 -> 128
        self.hidden = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
        )
        
        # Output: 128 -> 22
        self.output = nn.Sequential(
            nn.Linear(128, N_DIM),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        h = self.proj(x)    # [B, 256]
        h = self.hidden(h)  # [B, 128]
        return self.output(h)  # [B, 22]


# ================================================================
# OdorGNN v5 (384d ChemBERTa → 22d, ~250K params)
# Residual + deeper network for 17K unified dataset
# ================================================================

class TrainableOdorNetV5(nn.Module):
    """ChemBERTa(384d) -> 22d odor vector (v5)
    
    384 -> 384 (residual) -> 256 -> 128 -> 64 -> 22  (~250K params)
    + Residual connection on first layer
    + Extra hidden layer for more capacity
    """
    def __init__(self, input_dim=384):
        super().__init__()
        self.input_dim = input_dim
        
        # Residual projection: 384 -> 384
        self.res_proj = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(0.15),
        )
        
        # Main path: 384 -> 256
        self.layer1 = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.15),
        )
        
        # Hidden: 256 -> 128
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
        )
        
        # Hidden: 128 -> 64
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
        )
        
        # Output: 64 -> 22
        self.output = nn.Sequential(
            nn.Linear(64, N_DIM),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # Residual on first layer
        h = self.res_proj(x)  # [B, 384]
        if x.shape[-1] == 384:
            h = h + x  # residual connection
        h = self.layer1(h)    # [B, 256]
        h = self.layer2(h)    # [B, 128]
        h = self.layer3(h)    # [B, 64]
        return self.output(h)  # [B, 22]


# ================================================================
# MultiHeadOdorNet (384d → 22d×4 + longevity + sillage, ~300K params)
# 6-head model for complete odor prediction
# ================================================================

class MultiHeadOdorNet(nn.Module):
    """Multi-head odor prediction: 6 outputs from shared backbone
    
    Shared backbone: 384 → 384(res) → 256 → 128
    Head 1: 128 → 22  (기본 향 벡터)
    Head 2: 128 → 22  (Top 노트 강도)
    Head 3: 128 → 22  (Middle 노트 강도)
    Head 4: 128 → 22  (Base 노트 강도)
    Head 5: 128 → 1   (Longevity, 0-1 normalized)
    Head 6: 128 → 1   (Sillage, 0-1 normalized)
    """
    def __init__(self, input_dim=384, n_odor=22):
        super().__init__()
        self.input_dim = input_dim
        self.n_odor = n_odor
        
        # Shared backbone with residual
        self.res_proj = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(0.15),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.15),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
        )
        
        # Head 1: Main odor vector (22d)
        self.head_odor = nn.Sequential(nn.Linear(128, n_odor), nn.Sigmoid())
        # Head 2: Top notes (22d)
        self.head_top = nn.Sequential(nn.Linear(128, n_odor), nn.Sigmoid())
        # Head 3: Middle notes (22d)
        self.head_mid = nn.Sequential(nn.Linear(128, n_odor), nn.Sigmoid())
        # Head 4: Base notes (22d)
        self.head_base = nn.Sequential(nn.Linear(128, n_odor), nn.Sigmoid())
        # Head 5: Longevity (scalar 0-1)
        self.head_longevity = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        # Head 6: Sillage (scalar 0-1)
        self.head_sillage = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
    
    def forward(self, x, return_all=False):
        # Shared backbone
        h = self.res_proj(x)
        if x.shape[-1] == 384:
            h = h + x  # residual
        h = self.layer1(h)
        h = self.layer2(h)  # [B, 128]
        
        odor = self.head_odor(h)  # [B, 22]
        
        if not return_all:
            return odor  # Compatible with existing code
        
        top = self.head_top(h)           # [B, 22]
        mid = self.head_mid(h)           # [B, 22]
        base = self.head_base(h)         # [B, 22]
        longevity = self.head_longevity(h)  # [B, 1]
        sillage = self.head_sillage(h)      # [B, 1]
        
        return odor, top, mid, base, longevity, sillage



class MultiTaskOdorNet(nn.Module):
    """Multi-task: 384d → main(22d) + aux(138d)
    
    Shared backbone (384→256→128) with dual heads:
      - main_head: 128 → 22 (서비스용, soft labels)
      - aux_head:  128 → N_AUX (전체 138 라벨, 학습 보조)
    
    Aux head가 138개 라벨 전부를 학습하므로
    hidden representation이 더 풍부한 분자 구조 특징을 학습.
    """
    
    def __init__(self, input_dim=384, n_aux=138):
        super().__init__()
        self.input_dim = input_dim
        self.n_aux = n_aux
        
        # Shared backbone: 384 → 256 → 128
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.15),
        )
        self.hidden = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
        )
        
        # Main head: 128 → 22 (서비스용)
        self.main_head = nn.Sequential(
            nn.Linear(128, N_DIM),
            nn.Sigmoid(),
        )
        
        # Aux head: 128 → 138 (학습 보조용)
        self.aux_head = nn.Sequential(
            nn.Linear(128, n_aux),
            nn.Sigmoid(),
        )
    
    def forward(self, x, return_aux=False):
        h = self.proj(x)    # [B, 256]
        h = self.hidden(h)  # [B, 128]
        main_out = self.main_head(h)  # [B, 22]
        if return_aux:
            aux_out = self.aux_head(h)  # [B, 138]
            return main_out, aux_out
        return main_out


# ================================================================
# MixtureTransformer (합성 데이터라 크기 유지)
# ================================================================

class TrainableMixtureNet(nn.Module):
    """개별 냄새 벡터들 + 농도 → 혼합 냄새 벡터
    
    합성 데이터 무한 생성 → 큰 모델 OK (6-layer Transformer)
    """
    
    def __init__(self, d_model=N_DIM, nhead=4, num_layers=6):
        super().__init__()
        hidden = d_model * 8  # 160
        self.input_proj = nn.Linear(N_DIM + 1, hidden)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden // 2, N_DIM),
            nn.Sigmoid(),
        )
    
    def forward(self, odor_vecs, concentrations, padding_mask=None):
        """
        odor_vecs: [B, N, N_DIM]
        concentrations: [B, N, 1]
        padding_mask: [B, N] bool, True = padding (Rule #1 방어)
        """
        x = torch.cat([odor_vecs, concentrations], dim=-1)  # [B, N, N_DIM+1]
        x = self.input_proj(x)  # [B, N, hidden]
        
        # Rule #7 guard: 전체 패딩 시 Transformer 크래시 방지
        if padding_mask is not None and padding_mask.all():
            B = odor_vecs.size(0)
            return torch.zeros(B, N_DIM, device=odor_vecs.device)
        
        # Rule #1: padding mask로 O(N²) ghost attention 차단
        attended = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Rule #1/#2: masked concentration-weighted pooling
        conc_logits = concentrations.squeeze(-1)  # [B, N]
        if padding_mask is not None:
            conc_logits = conc_logits.masked_fill(padding_mask, float('-inf'))
        weights = torch.softmax(conc_logits, dim=-1)  # [B, N]
        weights = weights.nan_to_num(0.0)  # Rule #7: 전체 패딩 시 NaN 방지
        
        pooled = (attended * weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden]
        return self.output_proj(pooled)  # [B, N_DIM]


# ================================================================
# BackboneMixtureNet (128d backbone features → 22d, ~500K params)
# B1: Uses v6 backbone features instead of 22d odor vectors
# ================================================================

class BackboneMixtureNet(nn.Module):
    """혼합물 냄새 예측 — backbone features로 학습.
    
    입력: backbone features (128d) + concentration (1d) = 129d
    출력: 22d odor vector
    
    기존 TrainableMixtureNet(22d+1→22d)과 동일한 Transformer 구조이지만,
    128d backbone은 22d보다 훨씬 풍부한 화학적 정보를 포함하므로
    혼합물 상호작용을 더 정확하게 학습 가능.
    
    backward compatible: input_dim으로 22d legacy 모드도 지원.
    """
    
    def __init__(self, input_dim=128, nhead=8, num_layers=6, output_dim=N_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        hidden = max(256, input_dim * 2)  # 256 for 128d, 320 for 160d
        
        self.input_proj = nn.Linear(input_dim + 1, hidden)  # +1 for concentration
        self.input_norm = nn.LayerNorm(hidden)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Deeper output projection: hidden → hidden/2 → output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, output_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, feature_vecs, concentrations, padding_mask=None):
        """
        feature_vecs: [B, N, input_dim] (128d backbone or 22d legacy)
        concentrations: [B, N, 1]
        padding_mask: [B, N] bool tensor, True = padding position (Rule #1 방어)
                      None이면 모든 위치가 유효한 것으로 간주
        
        Tensor flow:
          [B, N, input_dim+1] → proj → [B, N, hidden] → Transformer(mask) → [B, N, hidden]
          → masked concentration softmax → weighted sum → [B, hidden]
          → output_proj → [B, output_dim]
        """
        x = torch.cat([feature_vecs, concentrations], dim=-1)  # [B, N, input_dim+1]
        x = self.input_norm(self.input_proj(x))  # [B, N, hidden]
        
        # Rule #7 guard: 전체 패딩 시 Transformer 크래시 방지
        # PyTorch TransformerEncoder는 src_key_padding_mask가 전부 True이면
        # to_padded_tensor에서 RuntimeError 발생
        if padding_mask is not None and padding_mask.all():
            B = feature_vecs.size(0)
            return torch.zeros(B, self.output_dim, device=feature_vecs.device)
        
        # Rule #1: Transformer에 padding mask 전달 — 패딩 토큰의 O(N²) 상호작용 차단
        attended = self.transformer(x, src_key_padding_mask=padding_mask)  # [B, N, hidden]
        
        # Rule #1/#2: masked concentration-weighted pooling
        # softmax(0) ≈ 1/N ≠ 0 이므로, 패딩 위치를 -inf로 마스킹해야 정확히 0 가중치
        conc_logits = concentrations.squeeze(-1)  # [B, N]
        if padding_mask is not None:
            conc_logits = conc_logits.masked_fill(padding_mask, float('-inf'))
        
        weights = torch.softmax(conc_logits, dim=-1)  # [B, N]
        
        # -inf softmax 결과가 NaN이 되는 edge case 방어 (모든 위치가 패딩인 경우)
        weights = weights.nan_to_num(0.0)  # Rule #7: NaN 발산 방지
        
        pooled = (attended * weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden]
        
        return self.output_proj(pooled)  # [B, output_dim]


# ================================================================
# Asymmetric Loss (ASL) — Sweet 편향 해소
# ================================================================
# 문제: BCE Loss는 모든 차원을 동등하게 취급.
#       'sweet'는 학습 데이터의 60%+ 에서 양성 → 모델이 sweet를 과대예측.
#       RL 에이전트가 이 허점을 탈취: 바닐린 30% "꼼수 레시피" 무한 영속.
#
# 해결: ASL(γ+=1, γ-=4) — 음성 오류(false positive)에 4배 강한 벌점.
#       + 차원별 빈도 역가중치: sweet(빈도 60%) 가중치 ↓, metallic(빈도 2%) ↑

class SoftTargetBCE(nn.Module):
    """Soft-Target-Aware BCE for continuous 0-1 labels.
    
    기존 ASL은 y∈{0,1} 이진 라벨 전용이므로, y=0.4 같은 연속값에서
    pos/neg 항이 동시에 활성화되어 그래디언트가 진동합니다.
    
    SoftTargetBCE는 표준 BCE를 사용하되:
    - focal 항으로 쉬운 샘플 페널티 감소
    - 차원별 빈도 역가중치로 sweet 편향 해소
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, preds, targets):
        eps = 1e-7
        p = preds.clamp(eps, 1 - eps)
        
        # Standard BCE (works correctly with soft targets)
        bce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))
        
        # Focal weighting: down-weight easy predictions
        pt = targets * p + (1 - targets) * (1 - p)  # p_t
        focal = ((1 - pt) ** self.gamma) * bce
        
        return focal.mean()


class DimensionWeightedLoss(nn.Module):
    """SoftTargetBCE + 차원별 빈도 역가중치
    
    학습 데이터에서 각 냄새 차원의 출현 빈도를 계산하고,
    빈도가 높은 차원(sweet, floral)은 가중치를 낮추고
    빈도가 낮은 차원(metallic, ozonic)은 가중치를 높임.
    
    가중치 = 1 / sqrt(frequency + ε)  (역제곱근)
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.bce = SoftTargetBCE(gamma=gamma)
        self.dim_weights = None  # lazy init from data
    
    def compute_weights(self, dataset):
        """학습 데이터에서 차원별 빈도 가중치 계산 (메모리 효율)"""
        n = len(dataset)
        n_dims = None
        freq_sum = None
        
        for i in range(n):
            item = dataset[i]
            target = item[1]  # always 2nd element
            if freq_sum is None:
                n_dims = target.shape[0]
                freq_sum = torch.zeros(n_dims)
            freq_sum += (target > 0.1).float()
        
        freq = freq_sum / max(n, 1)  # [N_DIM]
        # 역제곱근 가중치 (빈도 높을수록 가중치 낮음)
        weights = 1.0 / (torch.sqrt(freq) + 0.1)
        # 정규화: 평균=1
        weights = weights / weights.mean()
        self.dim_weights = weights
        
        # 디버그 출력
        dim_names = ODOR_DIMENSIONS
        print("  [Loss] Dimension frequency weights:")
        for name, f, w in sorted(zip(dim_names, freq.tolist(), weights.tolist()),
                                  key=lambda x: -x[1]):
            bar = '█' * int(w * 5)
            print(f"    {name:10s} freq={f:.2f}  weight={w:.2f} {bar}")
        return weights
    
    def forward(self, preds, targets):
        eps = 1e-7
        p = preds.clamp(eps, 1 - eps)
        
        # Standard BCE with focal weighting
        bce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))
        pt = targets * p + (1 - targets) * (1 - p)
        focal = ((1 - pt) ** self.bce.gamma) * bce  # [B, N_DIM]
        
        # 차원별 가중치 적용
        if self.dim_weights is not None:
            w = self.dim_weights.to(focal.device)
            focal = focal * w.unsqueeze(0)  # [B, N_DIM] * [1, N_DIM]
        
        return focal.mean()


# ================================================================
# 학습 파이프라인
# ================================================================

def train_odor_gnn(epochs=300, lr=0.002, batch_size=64):
    """OdorGNN v4 학습 -- ChemBERTa 384d + Scaffold Split + Mixup
    
    v4 변경:
    - 입력: 384d ChemBERTa (Freeze & Cache)
    - 모델: 384->256->128->20 (~135K params)
    - Split: Scaffold Split
    - 증강: Mixup
    """
    
    weight_path = WEIGHTS_DIR / 'odor_gnn.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ChemBERTa 캐시 로드
    bert_cache = _load_bert_cache()
    
    version = "v4 (ChemBERTa 384d)" if bert_cache else "v3 (FP 128d, fallback)"
    print(f"\n{'='*60}")
    print(f"  OdorGNN Training {version}")
    print(f"  Device: {device}")
    print(f"{'='*60}")
    
    # 1. 학습 데이터 로드 (DB → CSV fallback)
    print("  Loading molecules...")
    molecules = load_training_molecules()
    dataset = OdorDataset(molecules, bert_cache=bert_cache)
    
    if len(dataset) < 10:
        print("  Not enough training data!")
        return None
    
    input_dim = dataset.input_dim
    
    # Scaffold Split
    train_set, val_set = scaffold_split(dataset, val_ratio=0.2)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    # 2. 모델 생성 (ChemBERTa -> V4, else -> V3 legacy)
    if input_dim == 384:
        model = TrainableOdorNetV4(input_dim=input_dim).to(device)
    else:
        model = TrainableOdorNet().to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,} (data ratio 1:{len(dataset)//max(1,n_params//1000)})")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # ★ SoftTargetBCE — Soft label 친화적 로스 (ASL 대체) ★
    criterion = DimensionWeightedLoss(gamma=2.0)
    criterion.compute_weights(train_set)  # 차원별 빈도 가중치 자동 계산
    print(f"  Loss: SoftTargetBCE(γ={criterion.bce.gamma}) + DimWeights")
    
    # Warmup scheduler: 10% warmup → cosine annealing
    warmup_epochs = max(10, epochs // 10)
    
    def get_lr_scale(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)
    
    # 3. 학습 루프
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        # --- Train with Mixup ---
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            # Mixup 증강 (50% 확률로 적용)
            if np.random.random() < 0.5:
                features, targets = mixup_batch(features, targets, alpha=0.4)
            
            preds = model(features)
            
            # Soft targets are already continuous — no additional smoothing needed
            loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # --- Validate (Scaffold Split → 완전 새로운 골격) ---
        model.eval()
        val_loss = 0
        correct_dims = 0
        total_dims = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                preds = model(features)
                val_loss += criterion(preds, targets).item()
                
                # accuracy: threshold 0.3
                pred_active = (preds > 0.3).float()
                target_active = (targets > 0.3).float()
                correct_dims += (pred_active == target_active).sum().item()
                total_dims += target_active.numel()
        
        val_loss /= max(1, len(val_loader))
        accuracy = correct_dims / max(1, total_dims) * 100
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'input_dim': input_dim,
                'n_train': len(train_set),
                'n_val': len(val_set),
                'n_params': n_params,
                'split': 'scaffold',
                'augmentation': 'mixup',
            }, weight_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Acc: {accuracy:.1f}% | Best: {best_val_loss:.4f} | "
                  f"LR: {lr_now:.5f} | {elapsed:.1f}s")
        
        if patience_counter >= 30:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    elapsed = time.time() - start_time
    print(f"  ✅ OdorGNN trained! Best val loss: {best_val_loss:.4f} | "
          f"Params: {n_params:,} | Saved: {weight_path} | {elapsed:.1f}s")
    
    # Best weight 로드
    checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def train_mixture_transformer(odor_model, epochs=200, lr=0.0003, batch_size=32):
    """MixtureTransformer 학습 — self-supervised (합성 데이터)
    
    합성 데이터 무한 생성 가능 → 모델 크기 제약 없음
    6-layer Transformer, 200 에폭, patience 20
    """
    
    weight_path = WEIGHTS_DIR / 'mixture_transformer.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"  MixtureTransformer Training | Device: {device}")
    print(f"{'='*60}")
    
    # 1. DB에서 데이터 로드 + OdorGNN으로 벡터 생성
    print("  Loading molecules and computing odor vectors...")
    molecules = load_training_molecules()
    
    odor_model.eval()
    odor_model.to(device)
    
    # ChemBERTa 캐시 로드 (v4 모델이면 사용)
    bert_cache = _load_bert_cache()
    use_bert = bert_cache is not None and hasattr(odor_model, 'input_dim') and odor_model.input_dim == 384
    
    # 모든 분자의 odor vector 미리 계산
    mol_vectors = []
    for mol in molecules:
        smiles = mol.get('smiles', '')
        if not smiles:
            continue
        
        if use_bert and smiles in bert_cache['map']:
            feats = bert_cache['embeddings'][bert_cache['map'][smiles]]
        else:
            feats = _smiles_to_features(smiles)
        
        with torch.no_grad():
            vec = odor_model(torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device))
            mol_vectors.append(vec.squeeze(0).cpu())
    
    if len(mol_vectors) < 20:
        print("  ⚠ Not enough molecules!")
        return None
    
    mol_vectors = torch.stack(mol_vectors)  # [N, 20]
    print(f"  Computed {len(mol_vectors)} odor vectors")
    
    # 2. 모델 생성
    model = TrainableMixtureNet().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,} (synthetic data → no overfitting risk)")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    # 3. 학습 루프 — 매 배치마다 랜덤 혼합물 생성
    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    n_mols = len(mol_vectors)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = max(50, n_mols // batch_size)
        
        for _ in range(n_batches):
            # 랜덤 혼합물 생성 (2~5개 분자)
            batch_odors = []
            batch_concs = []
            batch_targets = []
            batch_masks = []  # Rule #1: padding mask 리스트
            
            for _ in range(batch_size):
                n_mix = np.random.randint(2, 6)
                indices = np.random.choice(n_mols, n_mix, replace=False)
                
                vecs = mol_vectors[indices]  # [n_mix, 20]
                concs = torch.rand(n_mix, 1) * 10.0 + 0.5  # 0.5 ~ 10.5
                
                # 타겟: 농도 가중 평균 + physics-based 상호작용 (C2 fix)
                weights = torch.softmax(concs.squeeze(-1) * 0.5, dim=-1)
                linear_mix = (vecs * weights.unsqueeze(-1)).sum(dim=0)
                
                # Perfumery synergy/antagonism rules (inspired by odor_engine PhysicsMixture)
                # Format: (dim_a, dim_b, target_dim, strength)
                #   positive = synergy (boost), negative = antagonism (suppress)
                _SYNERGY_RULES = [
                    (0, 3, 10, 0.3),   # sweet + floral → fruity boost
                    (4, 8, 7, 0.25),   # citrus + green → fresh boost
                    (2, 15, 16, 0.2),  # woody + amber → leather boost
                    (5, 9, 15, 0.2),   # spicy + warm → amber boost
                    (0, 10, 0, -0.15), # sweet + fruity → sweet suppress (competition)
                    (3, 12, 3, -0.1),  # floral + powdery → floral suppress
                    (7, 13, 7, 0.15),  # fresh + aquatic → fresh boost
                    (11, 16, 17, 0.2), # smoky + leather → earthy boost
                    (2, 8, 14, 0.15),  # woody + green → herbal boost
                    (6, 9, 12, 0.2),   # musk + warm → powdery boost
                    (0, 5, 9, 0.15),   # sweet + spicy → warm boost
                    (4, 10, 7, 0.2),   # citrus + fruity → fresh boost
                ]
                
                interaction = torch.zeros(N_DIM)
                for a in range(n_mix):
                    for b in range(a + 1, n_mix):
                        pair_weight = weights[a] * weights[b] * 4
                        
                        # Apply synergy/antagonism rules
                        for src_a, src_b, target_dim, strength in _SYNERGY_RULES:
                            act = max(
                                float(vecs[a][src_a] * vecs[b][src_b]),
                                float(vecs[b][src_a] * vecs[a][src_b])
                            )
                            if act > 0.01:
                                interaction[target_dim] += strength * act * pair_weight
                        
                        # Competitive binding: overlapping dimensions get suppressed
                        overlap = torch.min(vecs[a], vecs[b])
                        interaction -= 0.15 * overlap * weights[a] * weights[b]
                        
                        # Contrast effect: differing dimensions get mild boost
                        diff = torch.abs(vecs[a] - vecs[b])
                        interaction += 0.03 * diff * weights[a] * weights[b]
                
                target = (linear_mix + interaction).clamp(0, 1)
                
                # 패딩 (최대 5개까지) + Rule #1: padding mask 생성
                padded_vecs = torch.zeros(5, N_DIM)
                padded_concs = torch.zeros(5, 1)
                pad_mask = torch.ones(5, dtype=torch.bool)  # True = padding
                padded_vecs[:n_mix] = vecs
                padded_concs[:n_mix] = concs
                pad_mask[:n_mix] = False  # 실제 분자 위치만 False
                
                batch_odors.append(padded_vecs)
                batch_concs.append(padded_concs)
                batch_targets.append(target)
                batch_masks.append(pad_mask)
            
            batch_odors = torch.stack(batch_odors).to(device)    # [B, 5, N_DIM]
            batch_concs = torch.stack(batch_concs).to(device)    # [B, 5, 1]
            batch_targets = torch.stack(batch_targets).to(device) # [B, N_DIM]
            batch_pad = torch.stack(batch_masks).to(device)      # [B, 5] Rule #4: device 명시
            
            preds = model(batch_odors, batch_concs, padding_mask=batch_pad)
            loss = criterion(preds, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= n_batches
        scheduler.step()
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }, weight_path)
        else:
            patience += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {epoch_loss:.6f} | Best: {best_loss:.6f} | "
                  f"{elapsed:.1f}s")
        
        if patience >= 20:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    elapsed = time.time() - start_time
    print(f"  ✅ MixtureTransformer trained! Best loss: {best_loss:.6f} | "
          f"Saved: {weight_path} | {elapsed:.1f}s")
    
    return model


# ================================================================
# 가중치 로드
# ================================================================

def load_odor_gnn(device='cuda'):
    """저장된 가중치 로드 (앙상블 → v5 → v4 → v3)"""
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    v5_path = WEIGHTS_DIR / 'odor_gnn_v5.pt'
    v4_path = WEIGHTS_DIR / 'odor_gnn.pt'
    
    v5_model = None
    v4_model = None
    
    # === v5 GATConv 로드 시도 ===
    if v5_path.exists():
        try:
            from models.odor_gat_v5 import OdorGATv5
            cp5 = torch.load(v5_path, map_location=dev, weights_only=False)
            v5_model = OdorGATv5(bert_dim=384).to(dev)
            v5_model.load_state_dict(cp5['model_state_dict'])
            v5_model.eval()
            cos5 = cp5.get('val_cos_sim', 0)
            print(f"  [OdorGNN v5] Loaded (epoch {cp5['epoch']}, cos={cos5:.3f})")
        except Exception as e:
            print(f"  [OdorGNN] v5 load failed: {e}")
            v5_model = None
    
    # === v4 MLP 로드 시도 ===
    if v4_path.exists():
        try:
            cp4 = torch.load(v4_path, map_location=dev, weights_only=False)
            input_dim = cp4.get('input_dim', 128)
            model_type = cp4.get('model_type', '')
            n_aux = cp4.get('n_aux', 0)
            
            if model_type == 'OdorNetV6':
                try:
                    from scripts.train_v6_pipeline import OdorNetV6
                    v4_model = OdorNetV6(input_dim=input_dim, n_aux=n_aux or 138).to(dev)
                    print(f"  [OdorGNN] V6 (Residual+Attn+SE, multitask)")
                except Exception as e:
                    print(f"  [OdorGNN] V6 import failed: {e}, trying fallback")
                    v4_model = None
            elif model_type == 'OdorWideNet':
                n_dim = cp4.get('n_dim', 20)
                # Check if checkpoint is V16 Sequential MLP (net.* keys) 
                # vs V15 residual (b1,b2,b3,b4,skip_in,head keys)
                has_net_keys = any(k.startswith('net.') for k in cp4['model_state_dict'].keys())
                if n_dim == 22 and has_net_keys:
                    # V16: Simple Sequential MLP (384→768→384→192→22)
                    # Checkpoint keys have 'net.' prefix from self.net = Sequential(...)
                    import torch.nn as _nn
                    class _V16WideNet(_nn.Module):
                        def __init__(self, in_dim, out_dim):
                            super().__init__()
                            self.net = _nn.Sequential(
                                _nn.Linear(in_dim, 768), _nn.GELU(), _nn.LayerNorm(768), _nn.Dropout(0.2),
                                _nn.Linear(768, 384), _nn.GELU(), _nn.LayerNorm(384), _nn.Dropout(0.15),
                                _nn.Linear(384, 192), _nn.GELU(), _nn.LayerNorm(192), _nn.Dropout(0.1),
                                _nn.Linear(192, out_dim), _nn.Sigmoid())
                        def forward(self, x, return_aux=False):
                            return self.net(x)
                    v4_model = _V16WideNet(input_dim, n_dim).to(dev)
                    print(f"  [OdorGNN] V16 OdorWideNet ({n_dim}d Sequential MLP)")
                else:
                    # V15 or earlier: Residual WideNet
                    from scripts.retrain_cos09 import OdorWideNet
                    v4_model = OdorWideNet(input_dim=input_dim).to(dev)
            elif model_type == 'MorganFPNet':
                from scripts.retrain_cos09 import MorganFPNet
                v4_model = MorganFPNet(input_dim=input_dim).to(dev)
            elif model_type == 'MultiTaskOdorNet':
                v4_model = MultiTaskOdorNet(input_dim=input_dim, n_aux=n_aux).to(dev)
            elif input_dim == 384:
                v4_model = TrainableOdorNetV4(input_dim=384).to(dev)
            else:
                v4_model = TrainableOdorNet().to(dev)
            
            v4_model.load_state_dict(cp4['model_state_dict'])
            v4_model.eval()
            
            cos4 = cp4.get('cosine_sim', cp4.get('accuracy', cp4.get('val_cos_sim', 0)))
            epoch_info = f"epoch {cp4['epoch']}, " if 'epoch' in cp4 else ""
            threshold = cp4.get('threshold', 0)
            thresh_info = f", thresh={threshold:.2f}" if threshold > 0 else ""
            print(f"  [OdorGNN v4] Loaded ({epoch_info}cos={cos4:.3f}, type={model_type or 'auto'}{thresh_info})")
        except Exception as e:
            print(f"  [OdorGNN] v4 load failed: {e}")
            v4_model = None
    
    # === 앙상블 (둘 다 있으면) ===
    if v4_model and v5_model:
        print(f"  [OdorGNN] 🎯 Both v4+v5 available → Ensemble mode")
        return v4_model, v5_model, 'ensemble'
    
    # === 단일 모델 fallback ===
    if v5_model:
        return v5_model, None, 'v5'
    if v4_model:
        return v4_model, None, 'v4'
    
    return None, None, 'none'


def load_mixture_transformer(device='cuda'):
    """저장된 가중치 로드 → TrainableMixtureNet"""
    weight_path = WEIGHTS_DIR / 'mixture_transformer.pt'
    if not weight_path.exists():
        return None
    
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = TrainableMixtureNet().to(dev)
    checkpoint = torch.load(weight_path, map_location=dev, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  [MixtureTransformer] Loaded weights (epoch {checkpoint['epoch']}, "
          f"loss={checkpoint['loss']:.6f})")
    
    return model


def weights_exist():
    """학습된 가중치가 있는지 확인"""
    return (WEIGHTS_DIR / 'odor_gnn.pt').exists() and \
           (WEIGHTS_DIR / 'mixture_transformer.pt').exists()


# ================================================================
# 전체 학습 + SMILES → 20d 예측 함수 (odor_engine.py에서 사용)
# ================================================================

def ensure_trained():
    """가중치 없으면 학습, 있으면 로드"""
    if weights_exist():
        print("\n[TrainModels] Loading pre-trained weights...")
        odor_model, _, _ = load_odor_gnn()
        mix_model = load_mixture_transformer()
        return odor_model, mix_model
    
    print("\n[TrainModels] No weights - training v4 (ChemBERTa + Scaffold)...")
    odor_model = train_odor_gnn(epochs=300, lr=0.002)
    mix_model = train_mixture_transformer(odor_model, epochs=200, lr=0.0003)
    return odor_model, mix_model


def predict_single_smiles(model, smiles, device='cuda'):
    """학습된 모델로 단일 SMILES → 20d 벡터"""
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    feats = _smiles_to_features(smiles)
    with torch.no_grad():
        x = torch.tensor(feats).unsqueeze(0).to(dev)
        return model(x).squeeze(0).cpu().numpy()


# ================================================================
# 직접 실행 시 학습
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  AI Perfumer — Model Training v3")
    print("  Diet OdorGNN + Mixup + Scaffold Split")
    print("=" * 60)
    
    odor_model, mix_model = ensure_trained()
    
    # 테스트
    if odor_model:
        print("\n--- Test predictions ---")
        test_smiles = [
            ('O=CC1=CC(OC)=C(O)C=C1', 'Vanillin'),
            ('OCC1=CC=CC=C1', 'Benzyl alcohol'),
            ('CC(=O)OCC1=CC=CC=C1', 'Benzyl acetate'),
            ('O=C1CCCCCCCCCCCCC1', 'Muscone'),
        ]
        for smi, name in test_smiles:
            vec = predict_single_smiles(odor_model, smi)
            top_idx = np.argsort(vec)[::-1][:5]
            dims = ', '.join([f"{ODOR_DIMENSIONS[i]}={vec[i]:.3f}" for i in top_idx])
            print(f"  {name:20s} → {dims}")
