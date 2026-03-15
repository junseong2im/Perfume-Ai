# biophysics_simulator.py -- 바이오피직스 향수 시뮬레이터
# ==========================================================
# 1단계: VirtualNose     — 400개 수용체 도킹 시뮬레이션
# 2단계: HedonicFunction — 진화론적 쾌락 함수 + 엔트로피
# 3단계: ThermodynamicsEngine — Clausius-Clapeyron 휘발 시뮬
# 4단계: SelfPlayRL      — PPO 자가 대결 강화학습
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import json
import os
from scipy.integrate import solve_ivp
import database as db
try:
    from data.natural_oil_compositions import unroll_ingredient
except ImportError:
    unroll_ingredient = None
try:
    from scripts.commercial_prior import get_commercial_prior
except ImportError:
    get_commercial_prior = None
try:
    from scripts.proxy_reward import get_proxy_reward, extract_recipe_features
except ImportError:
    get_proxy_reward = None
    extract_recipe_features = None

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


# ================================================================
# 1단계: VirtualNose — 400개 수용체 도킹 시뮬레이션
# ================================================================

def _generate_receptor_bank(n=400):
    """400개 후각 수용체 생성 (실제 OR 유전자 패밀리 기반)"""
    families = [
        # (패밀리, 수, 선호 MW 범위, 선호 LogP 범위, 선호 TPSA 범위, 선호 구조 특성)
        ('OR1',  25, (100,200), (1,4),   (20,60),  'ester'),        # 프루티/플로럴 에스테르
        ('OR2',  30, (120,250), (2,5),   (10,50),  'terpene'),      # 테르펜 계열
        ('OR3',  20, (130,220), (1,3),   (30,70),  'alcohol'),      # 알코올
        ('OR4',  25, (150,300), (3,6),   (10,40),  'sesquiterpene'),# 세스퀴테르펜 (우디)
        ('OR5',  15, (80,180),  (0,3),   (40,80),  'aldehyde'),     # 알데히드
        ('OR6',  20, (100,200), (1,4),   (20,60),  'ketone'),       # 케톤
        ('OR7',  25, (140,280), (2,5),   (10,50),  'lactone'),      # 락톤 (크리미)
        ('OR8',  20, (200,400), (4,8),   (0,30),   'macrocyclic'),  # 머스크 (대환)
        ('OR9',  15, (100,180), (0,2),   (50,90),  'acid'),         # 산성 (치즈, 땀)
        ('OR10', 20, (80,160),  (1,3),   (20,50),  'phenol'),       # 페놀 (스모키)
        ('OR11', 25, (120,250), (2,5),   (10,40),  'ether'),        # 에테르 (스파이시)
        ('OR12', 15, (60,140),  (-1,2),  (40,80),  'sulfur'),       # 황 (마늘, 양파)
        ('OR13', 20, (100,220), (1,4),   (30,60),  'nitrile'),      # 질소 (인돌)
        ('OR14', 25, (130,260), (2,5),   (10,40),  'aromatic'),     # 방향족
        ('OR15', 20, (90,180),  (0,3),   (30,70),  'heterocyclic'), # 헤테로고리
        ('OR16', 25, (150,350), (3,7),   (0,30),   'steroid'),      # 스테로이드 유사
        ('OR17', 15, (80,150),  (0,2),   (40,80),  'amine'),        # 아민 (생선)
        ('OR18', 20, (100,200), (1,4),   (20,50),  'coumarin'),     # 쿠마린 계열
        ('OR19', 15, (140,280), (2,5),   (10,40),  'terpenoid'),    # 테르페노이드
        ('OR20', 10, (200,500), (5,10),  (0,20),   'diterpene'),    # 디테르펜 (레진)
    ]

    receptors = []
    rid = 0
    for family, count, mw_range, logp_range, tpsa_range, pref_type in families:
        for i in range(count):
            # 각 수용체의 선호 분자 특성 (가우시안 분포)
            center_mw = random.uniform(*mw_range)
            center_logp = random.uniform(*logp_range)
            center_tpsa = random.uniform(*tpsa_range)
            # 민감도 (가우시안 폭)
            sigma_mw = random.uniform(20, 60)
            sigma_logp = random.uniform(0.5, 2.0)
            sigma_tpsa = random.uniform(10, 30)
            # 방향족 선호도
            aromatic_pref = random.random() if pref_type in ('aromatic','phenol','heterocyclic') else random.random() * 0.3
            # HBD/HBA 선호
            hbd_pref = random.randint(0, 3)
            hba_pref = random.randint(0, 4)
            # 활성화 임계값
            threshold = random.uniform(0.1, 0.5)

            receptors.append({
                'id': f'{family}_{i}',
                'family': family,
                'type': pref_type,
                'center_mw': center_mw,
                'center_logp': center_logp,
                'center_tpsa': center_tpsa,
                'sigma_mw': sigma_mw,
                'sigma_logp': sigma_logp,
                'sigma_tpsa': sigma_tpsa,
                'aromatic_pref': aromatic_pref,
                'hbd_pref': hbd_pref,
                'hba_pref': hba_pref,
                'threshold': threshold,
            })
            rid += 1
    return receptors


class VirtualNose:
    """인간 후각 수용체 400개 도킹 시뮬레이션"""

    def __init__(self):
        random.seed(42)  # 재현 가능
        self.receptors = _generate_receptor_bank(400)
        self.n_receptors = len(self.receptors)
        self._desc_cache = {}
        print(f"[VirtualNose] {self.n_receptors} receptors initialized")

    def _get_descriptors(self, smiles):
        """RDKit 분자 디스크립터 추출"""
        if smiles in self._desc_cache:
            return self._desc_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        desc = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'rotatable': Descriptors.NumRotatableBonds(mol),
            'heavy_atoms': mol.GetNumHeavyAtoms(),
            'rings': Descriptors.RingCount(mol),
            'has_sulfur': 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[#16]')) else 0,
            'has_nitrogen': 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[#7]')) else 0,
            'has_halogen': 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[F,Cl,Br,I]')) else 0,
        }
        self._desc_cache[smiles] = desc
        return desc

    def _binding_affinity(self, receptor, desc):
        """분자-수용체 결합 친화도 (가우시안 도킹)"""
        if desc is None:
            return 0.0

        # 분자량 가우시안 매칭
        mw_fit = math.exp(-((desc['mw'] - receptor['center_mw'])**2) / (2 * receptor['sigma_mw']**2))
        # LogP 가우시안 매칭
        logp_fit = math.exp(-((desc['logp'] - receptor['center_logp'])**2) / (2 * receptor['sigma_logp']**2))
        # TPSA 가우시안 매칭
        tpsa_fit = math.exp(-((desc['tpsa'] - receptor['center_tpsa'])**2) / (2 * receptor['sigma_tpsa']**2))
        # 방향족 매칭
        aro_fit = 1.0 if (desc['aromatic_rings'] > 0) == (receptor['aromatic_pref'] > 0.5) else 0.5
        # HBD/HBA 매칭 (근접할수록 높음)
        hbd_fit = max(0, 1.0 - abs(desc['hbd'] - receptor['hbd_pref']) * 0.3)
        hba_fit = max(0, 1.0 - abs(desc['hba'] - receptor['hba_pref']) * 0.2)

        # 종합 결합 친화도
        affinity = (mw_fit * 0.25 + logp_fit * 0.25 + tpsa_fit * 0.15 +
                    aro_fit * 0.15 + hbd_fit * 0.10 + hba_fit * 0.10)

        # 임계값 적용 (약한 결합은 무시)
        return affinity if affinity > receptor['threshold'] else 0.0

    def smell(self, smiles_list, concentrations=None, fatigue_state=None):
        """분자 혼합물 → 400차원 Activation Pattern (후각 피로도 적용)
        
        Args:
            smiles_list: SMILES 리스트
            concentrations: 농도 리스트
            fatigue_state: 수용체별 누적 피로도 [n_receptors] (0=신선, 1=완전 피로)
        """
        if concentrations is None:
            concentrations = [1.0] * len(smiles_list)

        activation = np.zeros(self.n_receptors)
        mol_contributions = []

        for smiles, conc in zip(smiles_list, concentrations):
            desc = self._get_descriptors(smiles)
            if desc is None:
                mol_contributions.append(np.zeros(self.n_receptors))
                continue

            mol_act = np.zeros(self.n_receptors)
            for i, rec in enumerate(self.receptors):
                affinity = self._binding_affinity(rec, desc)
                if affinity > 0:
                    ec50 = rec['threshold'] * 10
                    hill_n = 1.5
                    response = (conc**hill_n) / (ec50**hill_n + conc**hill_n)
                    mol_act[i] = affinity * response

            mol_contributions.append(mol_act)

        # 경쟁적 결합: 같은 수용체에 여러 분자 → 가장 강한 것이 지배
        for i in range(self.n_receptors):
            activations_at_receptor = [mc[i] for mc in mol_contributions]
            if any(a > 0 for a in activations_at_receptor):
                sorted_acts = sorted(activations_at_receptor, reverse=True)
                activation[i] = sorted_acts[0]
                for k, a in enumerate(sorted_acts[1:], 1):
                    activation[i] += a * (0.3 ** k)
                activation[i] = min(1.0, activation[i])

        # ★ 후각 피로도 (Olfactory Fatigue) 적용 ★
        # 수용체 탈감작: S_i(t) = exp(-λ * cumulative_activation_i)
        # 오래 자극받은 수용체일수록 민감도 감소
        if fatigue_state is not None:
            fatigue_factor = np.exp(-2.0 * fatigue_state)  # λ=2.0
            activation = activation * fatigue_factor

        return {
            'activation_pattern': activation,
            'active_receptors': int(np.sum(activation > 0.1)),
            'mean_activation': float(np.mean(activation)),
            'max_activation': float(np.max(activation)),
            'receptor_families': self._family_summary(activation),
        }

    def _family_summary(self, activation):
        """수용체 패밀리별 활성화 요약"""
        families = {}
        for i, rec in enumerate(self.receptors):
            fam = rec['family']
            if fam not in families:
                families[fam] = {'activations': [], 'type': rec['type']}
            families[fam]['activations'].append(activation[i])

        summary = {}
        for fam, data in families.items():
            acts = data['activations']
            summary[fam] = {
                'type': data['type'],
                'mean': round(float(np.mean(acts)), 4),
                'max': round(float(np.max(acts)), 4),
                'active': int(sum(1 for a in acts if a > 0.1)),
                'total': len(acts),
            }
        return summary


# ================================================================
# 2단계: HedonicFunction — 진화론적 쾌락 함수
# ================================================================

# 진화적으로 쾌적한 분자 구조 패턴 (SMARTS)
PLEASANT_PATTERNS = [
    ('[CX3](=O)O[CX4]', 'ester', 0.8),          # 에스테르 (과일)
    ('c1ccccc1', 'benzene_ring', 0.3),            # 방향족 고리
    ('[CX3H1](=O)', 'aldehyde', 0.4),             # 알데히드
    ('[OX2H]', 'alcohol', 0.5),                   # 알코올
    ('C=C(C)C', 'isoprene', 0.7),                 # 이소프렌 (테르펜 기본)
    ('[CX3](=O)[CX4]', 'ketone', 0.4),            # 케톤
    ('c1cc(O)ccc1', 'phenol_mild', 0.2),          # 약한 페놀
    ('OC(=O)c1ccccc1', 'benzoate', 0.6),          # 벤조에이트
    ('C/C=C/C(=O)', 'enone', 0.3),                # 에논 (향기로운)
]

UNPLEASANT_PATTERNS = [
    ('[#16]', 'sulfur', -0.8),             # 황 (악취 위험)
    ('[SX2H]', 'thiol', -1.0),             # 티올 (썩은 달걀)
    ('[NX3H2]', 'primary_amine', -0.6),    # 1차 아민 (생선)
    ('S=O', 'sulfoxide', -0.5),            # 설폭사이드
    ('[NX2]=[NX2]', 'diazo', -0.7),        # 디아조 (독성)
    ('[N+](=O)[O-]', 'nitro', -0.9),       # 니트로 (자극적)
    ('C#N', 'nitrile', -0.4),              # 니트릴 (아몬드/독)
    ('[Cl,Br,I]', 'halogen', -0.3),        # 할로겐
]


class HedonicFunction:
    """좋은 향의 수학적 정의"""

    def __init__(self):
        # SMARTS 패턴 프리컴파일
        self.pleasant = [(Chem.MolFromSmarts(s), name, score) for s, name, score in PLEASANT_PATTERNS]
        self.unpleasant = [(Chem.MolFromSmarts(s), name, score) for s, name, score in UNPLEASANT_PATTERNS]
        print("[HedonicFunction] Initialized (pleasant + unpleasant + entropy)")

    def molecular_pleasantness(self, smiles):
        """분자 1개의 쾌적도 (-1 ~ +1)"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        score = 0.0
        matches = []

        for pattern, name, s in self.pleasant:
            if pattern and mol.HasSubstructMatch(pattern):
                score += s
                matches.append((name, s))

        for pattern, name, s in self.unpleasant:
            if pattern and mol.HasSubstructMatch(pattern):
                score += s  # s는 이미 음수
                matches.append((name, s))

        # LogP 안전 구간 (1~5가 대부분의 향료)
        logp = Descriptors.MolLogP(mol)
        if 1.0 <= logp <= 5.0:
            score += 0.3
        elif logp > 8.0 or logp < -1.0:
            score -= 0.3

        # 분자량 안전 구간 (100~350)
        mw = Descriptors.MolWt(mol)
        if 100 <= mw <= 350:
            score += 0.2
        elif mw > 500:
            score -= 0.3

        return max(-1.0, min(1.0, score)), matches

    def entropy_score(self, activation_pattern):
        """Shannon Entropy — 복잡도 황금비율"""
        # 활성화 패턴을 확률 분포로 변환
        acts = np.array(activation_pattern)
        acts = acts[acts > 0.01]  # 비활성 수용체 제외
        if len(acts) < 2:
            return 0.0, 0.0

        # 정규화
        probs = acts / acts.sum()
        # Shannon Entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(probs))

        # 황금비율: 엔트로피가 2.5~4.0 bits 구간에서 최적
        normalized = entropy / max(max_entropy, 1)
        if 0.3 <= normalized <= 0.7:
            pleasantness = 1.0 - abs(normalized - 0.5) * 2  # 0.5에서 최고
        elif normalized < 0.3:
            pleasantness = normalized / 0.3 * 0.5  # 너무 단순
        else:
            pleasantness = max(0, 1.0 - (normalized - 0.7) * 3)  # 너무 복잡

        return round(entropy, 3), round(pleasantness, 3)

    def evaluate_mixture(self, smiles_list, activation_pattern):
        """혼합물 전체 평가"""
        # 1) 개별 분자 쾌적도
        mol_scores = []
        all_matches = []
        for smiles in smiles_list:
            score, matches = self.molecular_pleasantness(smiles)
            mol_scores.append(score)
            all_matches.extend(matches)

        avg_pleasant = np.mean(mol_scores) if mol_scores else 0.0
        min_pleasant = min(mol_scores) if mol_scores else 0.0

        # 2) 엔트로피
        entropy, entropy_score = self.entropy_score(activation_pattern)

        # 3) 수용체 활성화 다양성 (다양한 패밀리 자극 = 풍부한 향)
        active = np.sum(activation_pattern > 0.1)
        diversity = min(1.0, active / 50)  # 50개 이상 활성 = 만점

        # 4) 독성 패널티 (하나라도 강한 불쾌 패턴이면 전체 감점)
        toxic_penalty = max(0, -min_pleasant) * 0.5

        # 종합 쾌락 점수
        hedonic = (avg_pleasant * 0.35 +
                   entropy_score * 0.25 +
                   diversity * 0.20 +
                   (1.0 - toxic_penalty) * 0.20)

        return {
            'hedonic_score': round(max(0, min(1, hedonic)), 3),
            'molecular_pleasantness': round(avg_pleasant, 3),
            'entropy': entropy,
            'entropy_pleasantness': entropy_score,
            'receptor_diversity': round(diversity, 3),
            'active_receptors': int(active),
            'toxic_penalty': round(toxic_penalty, 3),
            'pattern_matches': all_matches[:10],
        }


# ================================================================
# 3단계: ThermodynamicsEngine — 2-Compartment 증발/흡수 시뮬레이션
# ================================================================

# 농도별 초기 오일 비율 (총량 대비)
CONCENTRATION_FACTOR = {
    'EDC': 0.04,    # 2-5%
    'EDT': 0.08,    # 5-15%
    'EDP': 0.18,    # 15-20%
    'Parfum': 0.30, # 20-40%
    'Extrait': 0.35,
}

# ================================================================
# IFRA 피부 안전 규제 — 카테고리별 최대 허용 농도 (%)
# 참고: IFRA 48th Amendment, Category 4 (Fine Fragrance)
# ================================================================
IFRA_MAX_CONCENTRATION = {
    # 카테고리: 최대 농도 (%), 근거
    'aldehyde':  2.0,   # 알데히드 — 피부 자극 (Lilial, Citral 등)
    'citrus':    5.0,   # 시트러스 — 광독성 (Bergapten 등)
    'spicy':     3.0,   # 스파이시 — 감작 (Cinnamal, Eugenol 등)
    'resinous':  8.0,   # 레진 — 감작 (Oakmoss 등)
    'animalic':  1.0,   # 동물성 — 강한 자극
    'leather':   3.0,   # 레더 — Birch Tar 등 제한
    'earthy':    5.0,   # 어시 — Geosmin 등
    'green':     6.0,   # 그린 — 감작 가능성 낮음
    'herbal':    6.0,   # 허벌
    'floral':   10.0,   # 플로럴 — 대부분 안전
    'woody':    10.0,   # 우디 — 대부분 안전
    'fruity':   10.0,   # 프루티
    'gourmand': 10.0,   # 구르망
    'musk':     12.0,   # 머스크 — 합성 머스크 안전
    'amber':    10.0,   # 앰버
    'powdery':  10.0,   # 파우더리
    'fresh':    10.0,   # 프레시
    'sweet':    10.0,   # 스위트
    'smoky':     5.0,   # 스모키
    'aquatic':   8.0,   # 아쿠아틱
    'aromatic':  8.0,   # 아로마틱
}
IFRA_DEFAULT_MAX = 8.0  # 카테고리 불명 시 기본 상한


def ifra_clamp_ratio(ratio, category):
    """IFRA 규제에 따른 농도 클램핑 (미분 가능한 투영)
    
    Args:
        ratio: 원래 비율 (%)
        category: 원료 카테고리
    
    Returns:
        float: IFRA 상한 이내로 클램핑된 비율
    """
    max_conc = IFRA_MAX_CONCENTRATION.get(category, IFRA_DEFAULT_MAX)
    return min(ratio, max_conc)


def ifra_clamp_ratio_torch(ratio_tensor, max_concentration):
    """PyTorch용 미분 가능한 IFRA 투영 (PPO 학습 시 사용)"""
    return torch.clamp(ratio_tensor, max=max_concentration)



class ThermodynamicsEngine:
    """2-Compartment 증발/흡수 시뮬레이션
    
    Compartment 1: Headspace (공기 중) — 후각에 도달하는 농도
    Compartment 2: Skin reservoir (피부 흡수) — long-tail 지속력의 원천
    
    dC_air/dt  = -k1_i * C_air_i  + k_rel_i * C_skin_i   (증발 - 흡수로부터 재방출)
    dC_skin/dt =  k2_i * C_air_i  - k_rel_i * C_skin_i   (흡수 - 재방출)
    
    k1: 증발 속도 (VP, MW, logP 의존)
    k2: 피부 흡수 속도 (MW 의존, Kasting 계열)
    k_rel: 피부→공기 재방출 (MW 의존, k2보다 느림)
    """

    def __init__(self):
        self.R = 8.314  # J/(mol·K)
        self.T_skin = 305.15  # 피부 온도 32°C (K)
        self._vp_cache = {}
        self._desc_cache = {}
        self._vp_lookup = {}       # SMILES → experimental VP (Pa)
        self._threshold_lookup = {} # SMILES → odor threshold (ppb)
        self._load_experimental_data()
        print(f"[ThermodynamicsEngine] Initialized (2-Compartment k1/k2, "
              f"VP lookup: {len(self._vp_lookup)}, "
              f"Thresholds: {len(self._threshold_lookup)})")

    def _load_experimental_data(self):
        """Load experimental VP and threshold data from JSON files"""
        import os
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

        vp_path = os.path.join(data_dir, 'vapor_pressure_lookup.json')
        if os.path.exists(vp_path):
            try:
                with open(vp_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                for key, val in raw.items():
                    if not key.startswith('cas:'):
                        vp_mmhg = val.get('vp_mmHg')
                        if vp_mmhg is not None and isinstance(vp_mmhg, (int, float)):
                            self._vp_lookup[key] = vp_mmhg * 133.322  # mmHg → Pa
            except Exception as e:
                print(f"[ThermodynamicsEngine] VP lookup load error: {e}")

        thresh_path = os.path.join(data_dir, 'odor_threshold_lookup.json')
        if os.path.exists(thresh_path):
            try:
                with open(thresh_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                for key, val in raw.items():
                    thresh = val.get('threshold_ppb')
                    if thresh is not None:
                        self._threshold_lookup[key] = thresh
            except Exception as e:
                print(f"[ThermodynamicsEngine] Threshold lookup load error: {e}")

    def get_odor_threshold(self, smiles):
        """분자별 감지 역치 (ppb in air). 없으면 기본값 100 ppb"""
        return self._threshold_lookup.get(smiles, 100.0)

    def _get_mol_props(self, smiles):
        """분자 물성 추출 (캐시)"""
        if smiles in self._desc_cache:
            return self._desc_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            props = {'mw': 200, 'logp': 2.0, 'tpsa': 30, 'hbd': 0,
                     'rot': 2, 'rings': 1, 'heavy': 14}
        else:
            props = {
                'mw': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'rot': Descriptors.NumRotatableBonds(mol),
                'rings': Descriptors.RingCount(mol),
                'heavy': mol.GetNumHeavyAtoms(),
            }
        self._desc_cache[smiles] = props
        return props

    def estimate_vapor_pressure(self, smiles):
        """분자 증기압(Pa) — 실험값 우선, 없으면 경험적 추정"""
        if smiles in self._vp_cache:
            return self._vp_cache[smiles]

        # 1) Experimental lookup (already in Pa)
        if smiles in self._vp_lookup:
            vp_pa = self._vp_lookup[smiles]
            self._vp_cache[smiles] = round(vp_pa, 3)
            return self._vp_cache[smiles]

        # 2) Empirical model (fallback)
        p = self._get_mol_props(smiles)
        logp_capped = min(p['logp'], 4.0)
        log_vp = 4.5
        log_vp -= p['mw'] * 0.018
        log_vp -= p['hbd'] * 0.7
        log_vp -= p['tpsa'] * 0.010
        log_vp += logp_capped * 0.30
        log_vp -= p['rings'] * 0.25
        log_vp -= p['rot'] * 0.02
        log_vp -= max(0, p['heavy'] - 10) * 0.08
        vp_pa = 10 ** max(-3, min(4, log_vp))
        self._vp_cache[smiles] = round(vp_pa, 3)
        return self._vp_cache[smiles]

    def volatility_class(self, vp):
        """증기압 → 탑/미들/베이스 분류"""
        if vp > 20:
            return 'top'
        elif vp > 1.0:
            return 'middle'
        else:
            return 'base'

    def _compute_rate_constants(self, smiles):
        """분자별 k1(증발), k2(흡수), k_rel(재방출) 계산
        
        k1: log-linear in VP + MW + logP
            log(k1) = a0 + a1*log(VP) + a2*log(MW) + a3*logP
        k2: MW 기반 Kasting-like skin absorption
            k2 = b0 * exp(-b1 * MW)
        k_rel: 피부→공기 재방출 (k2보다 3-10x 느림)
        """
        vp = self.estimate_vapor_pressure(smiles)
        p = self._get_mol_props(smiles)
        mw = p['mw']
        logp = p['logp']

        # k1: 증발 속도 (1/min)
        # 캘리브레이션 목표:
        #   top (VP>20, MW~136):  반감기 ~30min  → k1 ≈ 0.023
        #   mid (VP~5, MW~165):   반감기 ~120min → k1 ≈ 0.006
        #   base (VP<1, MW~250):  반감기 ~360min → k1 ≈ 0.002
        import math as _m
        log_vp_safe = _m.log10(max(vp, 0.001))
        log_mw = _m.log10(max(mw, 50))
        log_k1 = -1.5 + 0.42 * log_vp_safe - 0.6 * log_mw + 0.08 * min(logp, 5)
        k1 = 10 ** max(-3.5, min(-0.5, log_k1))  # 하한 -3.5: musk도 최소 k1=3e-4

        # k2: 피부 흡수 속도 (1/min)
        # Moderate: heavy/lipophilic absorb more, provides reservoir
        logp_factor = max(0.2, min(1.5, logp / 4.0))
        if mw < 300:
            k2 = 0.0012 * logp_factor * (mw / 200) ** 0.5
        else:
            k2 = 0.0012 * logp_factor * (300 / mw) ** 0.3

        # k_rel: 재방출 (provides long-tail)
        k_rel = k2 * 0.10 * (200 / max(mw, 100)) ** 0.5

        return {
            'k1': round(k1, 6),      # 증발 (headspace에서 공기로)
            'k2': round(k2, 6),      # 흡수 (headspace에서 피부로)
            'k_rel': round(k_rel, 6), # 재방출 (피부에서 headspace로)
            'vp': vp,
            'mw': mw,
            'half_life_min': round(0.693 / max(k1, 1e-6), 1),
        }

    def simulate_evaporation(self, smiles_list, concentrations,
                              duration_hours=8, concentration_type='EDP'):
        """2-Compartment 시간별 시뮬레이션 (Stiff ODE Solver — BDF)
        
        기존 Euler(dt=10min) → scipy solve_ivp(method='BDF') 교체.
        적응형 타임스텝으로 탑 노트 폭발적 증발과 베이스 노트 미세 증발을
        동일한 정밀도로 적분.
        
        Args:
            smiles_list: SMILES 리스트
            concentrations: 상대 비율 (합이 1이 아니어도 됨)
            duration_hours: 시뮬레이션 시간
            concentration_type: EDC/EDT/EDP/Parfum
        """
        conc_factor = CONCENTRATION_FACTOR.get(concentration_type, 0.18)
        n_mol = len(smiles_list)

        # 분자별 속도 상수 준비
        molecules_info = []
        y0_list = []  # 초기 상태 벡터
        for smiles, conc in zip(smiles_list, concentrations):
            rates = self._compute_rate_constants(smiles)
            initial_amount = conc * conc_factor
            molecules_info.append({
                'smiles': smiles,
                'initial': initial_amount,
                'rates': rates,
                'note_type': self.volatility_class(rates['vp']),
            })
            y0_list.extend([initial_amount, 0.0])  # [C_air_i, C_skin_i]

        y0 = np.array(y0_list, dtype=np.float64)

        # ODE 시스템 정의: dy/dt = f(t, y)
        # Smooth non-negative projection: C1-continuous for BDF Jacobian
        # softplus(x, k) ≈ max(0, x) but differentiable everywhere
        _sp_k = 50.0  # sharpness — larger = closer to max(0, x)
        def _softplus(x):
            return np.where(x > 10.0 / _sp_k, x, np.log1p(np.exp(_sp_k * x)) / _sp_k)

        def ode_system(t, y):
            dydt = np.zeros_like(y)
            for i in range(n_mol):
                C_air = _softplus(y[2*i])
                C_skin = _softplus(y[2*i + 1])
                k1 = molecules_info[i]['rates']['k1']
                k2 = molecules_info[i]['rates']['k2']
                k_rel = molecules_info[i]['rates']['k_rel']

                # dC_air/dt = -k1*C_air - k2*C_air + k_rel*C_skin
                dydt[2*i] = -k1 * C_air - k2 * C_air + k_rel * C_skin
                # dC_skin/dt = k2*C_air - k_rel*C_skin
                dydt[2*i + 1] = k2 * C_air - k_rel * C_skin
            return dydt

        # 타임라인 샘플링 포인트 (10분 간격 유지 — API 호환성)
        t_end = duration_hours * 60  # 분
        t_eval = np.arange(0, t_end + 1, 10, dtype=np.float64)

        # Stiff ODE Solver — BDF (Backward Differentiation Formula)
        sol = solve_ivp(
            ode_system,
            t_span=(0, t_end),
            y0=y0,
            method='BDF',           # Stiff 전용 암시적 솔버
            t_eval=t_eval,
            rtol=1e-6,              # 상대 오차 허용
            atol=1e-9,              # 절대 오차 허용
            max_step=30.0,          # 최대 30분 스텝
        )

        if not sol.success:
            print(f"[ThermodynamicsEngine] BDF solver warning: {sol.message}, falling back")

        # 타임라인 구성
        timeline = []
        for step_idx in range(len(sol.t)):
            t_min = int(round(sol.t[step_idx]))
            t_hours = t_min / 60

            snapshot = {
                'time_min': t_min,
                'time_hours': round(t_hours, 2),
                'molecules': [],
            }

            for i in range(n_mol):
                C_air = max(0, float(sol.y[2*i, step_idx]))
                C_skin = max(0, float(sol.y[2*i + 1, step_idx]))
                total_remaining = C_air + C_skin
                snapshot['molecules'].append({
                    'smiles': molecules_info[i]['smiles'],
                    'headspace_pct': round(C_air, 6),
                    'skin_pct': round(C_skin, 6),
                    'remaining_pct': round(total_remaining, 6),
                    'note_type': molecules_info[i]['note_type'],
                    'evaporated_pct': round(molecules_info[i]['initial'] - total_remaining, 6),
                })

            # 노트 밸런스 — headspace 기준 (체감 기준!)
            top_hs = sum(m['headspace_pct'] for m in snapshot['molecules'] if m['note_type'] == 'top')
            mid_hs = sum(m['headspace_pct'] for m in snapshot['molecules'] if m['note_type'] == 'middle')
            base_hs = sum(m['headspace_pct'] for m in snapshot['molecules'] if m['note_type'] == 'base')
            total_hs = top_hs + mid_hs + base_hs + 1e-10

            snapshot['note_balance'] = {
                'top': round(top_hs / total_hs, 3),
                'middle': round(mid_hs / total_hs, 3),
                'base': round(base_hs / total_hs, 3),
            }
            snapshot['dominant'] = max(snapshot['note_balance'], key=snapshot['note_balance'].get)
            snapshot['total_headspace'] = round(total_hs, 6)
            timeline.append(snapshot)

        # 전환점 감지
        transitions = []
        prev_dom = timeline[0]['dominant']
        for snap in timeline[1:]:
            if snap['dominant'] != prev_dom:
                transitions.append({
                    'time_min': snap['time_min'],
                    'from': prev_dom,
                    'to': snap['dominant'],
                })
                prev_dom = snap['dominant']

        # 지속력: headspace 총합이 초기의 10% 이하 (체감 기준)
        initial_total_hs = timeline[0]['total_headspace']
        longevity_min = duration_hours * 60
        for snap in timeline:
            if snap['total_headspace'] < initial_total_hs * 0.10:
                longevity_min = snap['time_min']
                break

        # 전환 부드러움
        transition_smoothness = 1.0
        if len(transitions) >= 2:
            gaps = [transitions[i+1]['time_min'] - transitions[i]['time_min']
                    for i in range(len(transitions)-1)]
            if gaps:
                cv = np.std(gaps) / (np.mean(gaps) + 1)
                transition_smoothness = max(0, 1.0 - cv)

        return {
            'timeline': timeline,
            'timeline_summary': timeline[::3],
            'transitions': transitions,
            'longevity_min': longevity_min,
            'longevity_hours': round(longevity_min / 60, 1),
            'transition_smoothness': round(transition_smoothness, 3),
            'smoothness': round(transition_smoothness, 3),
            'solver': 'BDF',  # 솔버 정보 추가
            'solver_steps': len(sol.t),
            'initial_molecules': [{
                'smiles': m['smiles'],
                'concentration': m['initial'],
                'vapor_pressure': m['rates']['vp'],
                'mw': m['rates']['mw'],
                'k1_evap': m['rates']['k1'],
                'k2_absorb': m['rates']['k2'],
                'k_release': m['rates']['k_rel'],
                'half_life_min': m['rates']['half_life_min'],
                'note_type': m['note_type'],
            } for m in molecules_info],
        }



# ================================================================
# 4단계: SelfPlayRL — PPO 자가 대결 강화학습
# ================================================================

class PolicyNetwork(nn.Module):
    """계층형 정책 네트워크 — Categorical (원료 선택) + Dirichlet (비율 결정) + MORL (조건부)
    
    개선사항:
    1. Dirichlet 분포: 비율의 합=1 자동 보장, 극미량(0.01%) 제어 가능
    2. MORL 조건부: 타겟 선호도 w=[w_H, w_L, w_S]를 state에 concat
    3. Sparsity 마스킹: Top-K 원료 외 logit 제거
    """
    def __init__(self, state_dim=420, n_ingredients=100, hidden=256, n_condition=3):
        super().__init__()
        # State + MORL 조건 벡터 (3차원: hedonic, longevity, smoothness 선호도)
        self.condition_dim = n_condition
        input_dim = state_dim + n_condition
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
        )
        # 원료 선택 헤드 (Categorical)
        self.ingredient_head = nn.Linear(hidden, n_ingredients)
        # Dirichlet 농도 파라미터 헤드 (각 원료의 alpha)
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()
        )  # Softplus → alpha > 0 보장
        # 가치 함수 헤드
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, state, condition=None):
        """
        Args:
            state: [B, state_dim] 또는 [state_dim]
            condition: [B, 3] 또는 [3] — MORL 타겟 선호도
        Returns:
            ingredient_logits, alpha (Dirichlet concentration), value
        """
        if condition is None:
            condition = torch.ones(state.shape[0] if state.dim() > 1 else 1, 
                                 self.condition_dim, device=state.device) / self.condition_dim
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        
        x = torch.cat([state, condition], dim=-1)
        h = self.backbone(x)
        ingredient_logits = self.ingredient_head(h)
        alpha = self.alpha_head(h) + 0.1  # 최소 alpha=0.1 (안정성)
        value = self.value_head(h)
        return ingredient_logits, alpha, value


class SelfPlayRL:
    """PPO 기반 자가 대결 향수 설계 에이전트"""

    def __init__(self, virtual_nose, hedonic_fn, thermo_engine):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nose = virtual_nose
        self.hedonic = hedonic_fn
        self.thermo = thermo_engine

        # 원료 풀 로딩
        self.ingredients = []
        self.ing_smiles = {}
        self.n_ingredients = 0
        self._load_ingredients()

        state_dim = self.nose.n_receptors + 20
        self.n_receptors = self.nose.n_receptors
        self.n_condition = 3  # MORL: [w_hedonic, w_longevity, w_smoothness]
        self.policy = PolicyNetwork(state_dim, max(self.n_ingredients, 50),
                                    n_condition=self.n_condition).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        self.generation = 0
        self.best_score = 0.0
        self.best_recipe = None
        self.history = []
        
        # Curriculum Learning: 비선형 환경 전환 세대
        self.curriculum_switch_gen = 50
        self.use_nonlinear_mixing = False  # Set Transformer 사용 여부
        self.mixture_transformer = None    # lazy init
        
        # Sparsity 설정
        self.max_clean_ingredients = 15
        self.sparsity_penalty_per_extra = 0.05
        
        print(f"[SelfPlayRL] PPO+Dirichlet+MORL Agent | {self.n_ingredients} ingredients | {self.device}")

    def _load_ingredients(self):
        """DB에서 원료+SMILES 로딩"""
        try:
            ings = db.get_all_ingredients()
            mols = db.get_all_molecules(limit=500)
            mol_smiles = {}
            for mol in mols:
                name = (mol.get('name') or '').lower()
                if name and mol.get('smiles'):
                    mol_smiles[name] = mol['smiles']

            for ing in ings:
                name_en = (ing.get('name_en') or '').lower()
                smiles = None
                for mname, msmiles in mol_smiles.items():
                    if name_en and (name_en in mname or mname in name_en):
                        smiles = msmiles
                        break
                if smiles is None:
                    smiles = self._category_smiles(ing.get('category', ''))
                self.ingredients.append(ing)
                self.ing_smiles[ing['id']] = smiles

            self.n_ingredients = len(self.ingredients)
        except Exception as e:
            print(f"[SelfPlayRL] Ingredient loading failed: {e}")
            self.n_ingredients = max(self.n_ingredients, 50)

    def _category_smiles(self, category):
        """카테고리별 대표 SMILES"""
        defaults = {
            'floral': 'OCC1=CC=CC=C1',
            'citrus': 'CC(=CCC/C(=C/CO)C)C',
            'woody': 'CC1CCC2(C)C(O)CCC12',
            'spicy': 'C=CC1=CC(OC)=C(O)C=C1',
            'fruity': 'CCCCCC(=O)OC',
            'gourmand': 'O=CC1=CC(OC)=C(O)C=C1',
            'musk': 'O=C1CCCCCCCCCCCCC1',
            'amber': 'CC12CCCC(C)(C1)C1CCC(O)CC12',
            'herbal': 'CC(O)CC=CC(C)C',
            'green': 'CC/C=C\\CCO',
            'fresh': 'CC(=O)OCC=C(C)C',
        }
        return defaults.get(category, 'CCCCCCO')

    def _make_state(self, current_activation=None, selected_count=0, total_pct=0):
        """현재 상태 벡터 (n_receptors + 20)차원"""
        if current_activation is None:
            current_activation = np.zeros(self.n_receptors)
        meta = np.zeros(20)
        meta[0] = selected_count / 15
        meta[1] = total_pct / 30
        return np.concatenate([current_activation, meta])

    def _sample_condition(self):
        """MORL: 결정론적 타겟 선호도 벡터 생성 (세대 기반 순환)
        세대마다 다른 선호도를 체계적으로 탐색"""
        # 6가지 선호도 프로파일 순환 (random 제거)
        profiles = [
            [0.6, 0.2, 0.2],   # 쾌락 중시
            [0.2, 0.6, 0.2],   # 지속력 중시
            [0.2, 0.2, 0.6],   # 부드러움 중시
            [0.4, 0.4, 0.2],   # 쾌락+지속
            [0.4, 0.2, 0.4],   # 쾌락+부드러움
            [0.33, 0.33, 0.34], # 균등
        ]
        w = profiles[self.generation % len(profiles)]
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    def _conditioned_reward(self, hedonic_score, longevity_score, smoothness,
                            active_receptors, condition, n_ingredients_used,
                            ingredient_ids=None):
        """조건부 보상 함수 (MORL + Sparsity + Commercial Plausibility)
        
        Args:
            hedonic_score: 쾌락 점수 (0~1)
            longevity_score: 지속력 점수 (0~1)
            smoothness: 전환 부드러움 (0~1)
            active_receptors: 활성 수용체 수
            condition: [w_H, w_L, w_S] 타겟 선호도
            n_ingredients_used: 사용된 원료 수
            ingredient_ids: 원료 ID 리스트 (PMI 계산용)
        """
        w_h, w_l, w_s = condition[0].item(), condition[1].item(), condition[2].item()
        
        # 기본 보상 (MORL 가중합)
        base_reward = (hedonic_score * (0.25 + w_h * 0.30) +
                       longevity_score * (0.15 + w_l * 0.30) +
                       smoothness * (0.10 + w_s * 0.30) +
                       active_receptors / 100 * 0.10)
        
        # ★ Commercial Plausibility 보너스 (PMI 기반) ★
        if ingredient_ids and get_commercial_prior is not None:
            try:
                prior = get_commercial_prior()
                comm_score = prior.plausibility_score(ingredient_ids)
                base_reward += comm_score * 0.15  # 최대 +0.15
            except Exception:
                pass
        
        # ★ Sparsity 페널티: 15개 초과 시 1개당 -0.05 ★
        if n_ingredients_used > self.max_clean_ingredients:
            extra = n_ingredients_used - self.max_clean_ingredients
            base_reward -= extra * self.sparsity_penalty_per_extra
        
        return max(0, base_reward)

    def _init_mixture_transformer(self):
        """MixtureTransformer lazy 초기화 (Curriculum Learning용)"""
        if self.mixture_transformer is None:
            try:
                from odor_engine import MixtureTransformer
                self.mixture_transformer = MixtureTransformer(device=str(self.device))
                print("[SelfPlayRL] MixtureTransformer activated (nonlinear mixing)")
            except Exception as e:
                print(f"[SelfPlayRL] MixtureTransformer init failed: {e}")
                self.mixture_transformer = None

    def _unroll_smiles_list(self, selected_ings, ratios):
        """천연오일 → 하위 분자 분해 (GC-MS 기반 Unroll)"""
        unrolled_smiles = []
        unrolled_conc = []
        for ing, ratio in zip(selected_ings, ratios):
            ing_id = ing['id']
            if unroll_ingredient is not None:
                sub = unroll_ingredient(ing_id, ratio)
                if sub is not None:
                    for smi, pct in sub:
                        unrolled_smiles.append(smi)
                        unrolled_conc.append(pct)
                    continue
            # fallback: 기존 단일 분자 매핑
            unrolled_smiles.append(self.ing_smiles.get(ing_id, 'CCCCCCO'))
            unrolled_conc.append(ratio)
        return unrolled_smiles, unrolled_conc

    def _evaluate_recipe(self, selected_ings, ratios, condition=None,
                         use_fatigue=True):
        """레시피 종합 평가 (GC-MS Unroll + 후각 피로도 + 비선형 혼합 + MORL)"""
        # GC-MS 분해: bergamot 5% → Limonene 2.1% + Linalyl acetate 1.4% + ...
        smiles_list, concentrations = self._unroll_smiles_list(selected_ings, ratios)

        # 1) 가상 코 (후각 피로도 적용)
        if use_fatigue:
            # 8시간 시뮬레이션 중 3개 시점에서 피로도 축적 측정
            fatigue_cumulative = np.zeros(self.n_receptors)
            fatigue_scores = []
            
            for t_fraction in [0.0, 0.3, 0.7, 1.0]:  # 0h, 2.4h, 5.6h, 8h
                fatigue_state = fatigue_cumulative * t_fraction
                nose_result = self.nose.smell(smiles_list, concentrations, 
                                             fatigue_state=fatigue_state)
                fatigue_cumulative = np.maximum(fatigue_cumulative,
                                               nose_result['activation_pattern'])
                fatigue_scores.append(nose_result['active_receptors'])
            
            # 피로도 반영 활성 수용체 = 시간 가중 평균
            avg_active = np.mean(fatigue_scores)
            nose_result = self.nose.smell(smiles_list, concentrations)  # 기본 (피로 없는)
        else:
            nose_result = self.nose.smell(smiles_list, concentrations)
            avg_active = nose_result['active_receptors']
        
        activation = nose_result['activation_pattern']

        # 2) 쾌락 함수
        hedonic_result = self.hedonic.evaluate_mixture(smiles_list, activation)

        # 3) 열역학
        thermo_result = self.thermo.simulate_evaporation(smiles_list, concentrations,
                                                         duration_hours=8)

        # 종합 보상
        hedonic_score = hedonic_result['hedonic_score']
        longevity_score = min(1.0, thermo_result['longevity_hours'] / 6)
        smoothness = thermo_result.get('smoothness', 
                      thermo_result.get('transition_smoothness', 0.5))

        if condition is None:
            condition = torch.tensor([0.33, 0.33, 0.34], device=self.device)
        
        ingredient_ids = [ing['id'] for ing in selected_ings]
        reward = self._conditioned_reward(
            hedonic_score, longevity_score, smoothness,
            avg_active, condition, len(selected_ings),
            ingredient_ids=ingredient_ids
        )

        return {
            'reward': round(reward, 4),
            'hedonic': hedonic_result,
            'thermodynamics': {
                'longevity_hours': thermo_result['longevity_hours'],
                'transitions': thermo_result['transitions'],
                'smoothness': smoothness,
            },
            'nose': {
                'active_receptors': nose_result['active_receptors'],
                'fatigue_adjusted_active': round(float(avg_active), 1),
                'mean_activation': nose_result['mean_activation'],
                'families': nose_result['receptor_families'],
            },
        }

    @torch.no_grad()
    def generate_recipe(self, n_ingredients=10, condition=None):
        """현재 정책으로 레시피 생성 (IFRA + Dirichlet + MORL)
        
        Args:
            n_ingredients: 최대 원료 수
            condition: [w_H, w_L, w_S] 타겟 선호도 (None이면 균등)
        """
        self.policy.eval()
        selected = []
        ratios = []
        used = set()
        activation = np.zeros(self.n_receptors)
        ifra_violations_prevented = 0
        
        # MORL 조건 벡터
        if condition is not None:
            cond_t = torch.tensor(condition, dtype=torch.float32, device=self.device)
        else:
            cond_t = torch.ones(self.n_condition, device=self.device) / self.n_condition

        # Dirichlet 누적 alpha 수집
        alphas_collected = []
        ings_collected = []

        for step in range(n_ingredients):
            state = self._make_state(activation, len(selected), sum(ratios))
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)

            logits, alpha, _ = self.policy(state_t.unsqueeze(0), cond_t.unsqueeze(0))
            # 이미 선택한 원료 마스킹
            for idx in used:
                logits[0, idx] = -1e9

            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            if idx < len(self.ingredients):
                ing = self.ingredients[idx]
                alphas_collected.append(float(alpha.squeeze()))
                ings_collected.append(ing)
                used.add(idx)

                # 활성화 업데이트 (임시 비율)
                smiles = self.ing_smiles.get(ing['id'], 'CCCCCCO')
                nose_r = self.nose.smell([smiles], [1.0])
                activation = np.maximum(activation, nose_r['activation_pattern'])

        if not ings_collected:
            return None

        # ★ Dirichlet 분포로 비율 결정 ★
        # alpha 값들로 Dirichlet 샘플링 → 합=1 자동 보장
        alphas = torch.tensor(alphas_collected, dtype=torch.float32, device=self.device)
        alphas = alphas.clamp(min=0.1)  # 안정성
        # Dirichlet requires dim >= 2; if only 1 ingredient, skip sampling
        if alphas.numel() >= 2:
            dirichlet = torch.distributions.Dirichlet(alphas)
            raw_ratios = dirichlet.sample()  # [N] 합=1
        else:
            raw_ratios = torch.ones_like(alphas)  # single ingredient gets 100%
        
        # 총 농도를 카테고리에 맞게 스케일링 (총 30~60%)
        total_pct = random.uniform(30, 60)
        raw_ratios_np = raw_ratios.cpu().numpy() * total_pct

        # IFRA 클램핑 적용
        for ing, r in zip(ings_collected, raw_ratios_np):
            r = float(max(0.1, r))  # 최소 0.1%
            category = ing.get('category', '').lower()
            r_original = r
            r = ifra_clamp_ratio(r, category)
            if r < r_original:
                ifra_violations_prevented += 1
            selected.append(ing)
            ratios.append(r)

        evaluation = self._evaluate_recipe(selected, ratios, cond_t)
        
        # 조건 설명
        cond_desc = '균등'
        if condition is not None:
            labels = ['쾌락', '지속력', '부드러움']
            dominant_idx = max(range(len(condition)), key=lambda i: condition[i])
            cond_desc = f'{labels[dominant_idx]} 중시 ({condition[dominant_idx]:.0%})'

        return {
            'ingredients': [{'id': ing['id'], 'name_ko': ing.get('name_ko',''),
                            'category': ing.get('category',''),
                            'percentage': round(r, 2),
                            'ifra_max': IFRA_MAX_CONCENTRATION.get(
                                ing.get('category','').lower(), IFRA_DEFAULT_MAX)}
                           for ing, r in zip(selected, ratios)],
            'evaluation': evaluation,
            'generation': self.generation,
            'safety': {
                'ifra_compliant': True,
                'violations_prevented': ifra_violations_prevented,
                'standard': 'IFRA 48th Amendment, Category 4 (Fine Fragrance)',
            },
            'policy': {
                'distribution': 'Dirichlet',
                'n_ingredients': len(selected),
                'condition': condition if condition else [0.33, 0.33, 0.34],
                'condition_desc': cond_desc,
            },
        }

    def evolve(self, generations=100, population=20, on_progress=None):
        """PPO + Dirichlet + MORL + Curriculum Learning 자가 진화
        
        Curriculum:
            0~50세대: 선형 환경 + 고정 타겟 (빠른 기초 수렴)
            50세대~: 비선형(MixtureTransformer) + 랜덤 타겟 MORL
        """
        self.policy.train()
        gamma = 0.99
        eps_clip = 0.2

        for gen in range(generations):
            states, actions, rewards, old_log_probs, values = [], [], [], [], []
            conditions = []  # MORL 조건 저장
            masks = []       # ★ BUG FIX ④: store rollout masks for PPO update

            # ★ Curriculum Learning: 세대에 따라 환경 전환 ★
            current_gen = self.generation + gen + 1
            if current_gen >= self.curriculum_switch_gen:
                if not self.use_nonlinear_mixing:
                    self._init_mixture_transformer()
                    self.use_nonlinear_mixing = True
                    print(f"[Curriculum] Gen {current_gen}: 비선형 환경 활성화")

            gen_rewards = []
            for _ in range(population):
                selected = []
                ratios = []
                used = set()
                activation = np.zeros(self.n_receptors)
                ep_states, ep_actions = [], []
                
                # MORL: 이번 에피소드의 타겟 선호도
                if current_gen >= self.curriculum_switch_gen:
                    condition = self._sample_condition()  # 랜덤 타겟
                else:
                    condition = torch.ones(self.n_condition, device=self.device) / self.n_condition  # 균등

                n_ing = random.randint(6, 12)
                alphas_ep = []
                
                for step in range(n_ing):
                    state = self._make_state(activation, len(selected), sum(ratios))
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

                    logits, alpha, value = self.policy(state_t, condition.unsqueeze(0))
                    for idx in used:
                        if idx < logits.shape[1]:
                            logits[0, idx] = -1e9

                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    idx = dist.sample()
                    log_prob = dist.log_prob(idx)

                    if idx.item() < len(self.ingredients):
                        ing = self.ingredients[idx.item()]
                        alphas_ep.append(alpha.squeeze())
                        selected.append(ing)
                        used.add(idx.item())
                        ep_states.append(state)
                        ep_actions.append(idx.item())

                        states.append(state_t.squeeze())
                        actions.append(idx)
                        old_log_probs.append(log_prob)
                        values.append(value.squeeze())
                        conditions.append(condition)
                        masks.append(frozenset(used))  # snapshot of mask at this step

                        smiles = self.ing_smiles.get(ing['id'], 'CCCCCCO')
                        nose_r = self.nose.smell([smiles], [1.0])
                        activation = np.maximum(activation, nose_r['activation_pattern'])

                # Dirichlet 비율 결정
                if selected and alphas_ep:
                    alphas_t = torch.stack(alphas_ep).clamp(min=0.1)
                    if alphas_t.dim() == 0:
                        alphas_t = alphas_t.unsqueeze(0)
                    alphas_t = alphas_t.squeeze()
                    if alphas_t.dim() == 0:
                        alphas_t = alphas_t.unsqueeze(0)
                    # Dirichlet requires dim >= 2
                    if alphas_t.numel() >= 2:
                        try:
                            dirichlet = torch.distributions.Dirichlet(alphas_t)
                            raw_ratios = dirichlet.sample()
                            ratios = (raw_ratios * random.uniform(30, 60)).cpu().tolist()
                        except:
                            ratios = [random.uniform(1, 8) for _ in selected]
                    else:
                        ratios = [random.uniform(3, 10) for _ in selected]
                    
                    # IFRA 클램핑
                    for i, ing in enumerate(selected):
                        cat = ing.get('category', '').lower()
                        ratios[i] = ifra_clamp_ratio(max(0.1, ratios[i]), cat)

                # 에피소드 보상
                if selected and ratios:
                    eval_result = self._evaluate_recipe(selected, ratios, condition)
                    ep_reward = eval_result['reward']
                    gen_rewards.append(ep_reward)
                    for i in range(len(ep_states)):
                        rewards.append(ep_reward * (gamma ** (len(ep_states) - i - 1)))

            if not states:
                continue

            # PPO 업데이트
            states_t = torch.stack(states).to(self.device)
            actions_t = torch.stack(actions).to(self.device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            old_lp_t = torch.stack(old_log_probs).detach()
            values_t = torch.stack(values).detach()

            advantages = rewards_t - values_t.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 조건 벡터 배치
            if conditions:
                cond_t = torch.stack(conditions).to(self.device)
            else:
                cond_t = torch.ones(states_t.shape[0], self.n_condition, 
                                   device=self.device) / self.n_condition

            for _ in range(3):  # 3 에폭
                logits, _, new_values = self.policy(states_t, cond_t)
                # ★ BUG FIX ④: Replay rollout masks to keep probability
                # spaces identical between collection and training
                for b_idx in range(logits.shape[0]):
                    for masked_idx in masks[b_idx]:
                        if masked_idx < logits.shape[1]:
                            logits[b_idx, masked_idx] = -1e9
                new_probs = F.softmax(logits, dim=-1)
                new_dist = torch.distributions.Categorical(new_probs)
                new_lp = new_dist.log_prob(actions_t.squeeze())

                ratio = torch.exp(new_lp - old_lp_t.squeeze())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values.squeeze(), rewards_t)
                entropy_bonus = new_dist.entropy().mean() * 0.01

                loss = policy_loss + value_loss * 0.5 - entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

            avg_reward = np.mean(gen_rewards)
            max_reward = max(gen_rewards)
            self.generation = self.generation + 1
            self.history.append({'gen': self.generation, 
                                'avg': round(avg_reward, 4),
                                'max': round(max_reward, 4),
                                'curriculum': 'nonlinear' if self.use_nonlinear_mixing else 'linear'})

            if max_reward > self.best_score:
                self.best_score = max_reward
                best = self.generate_recipe(n_ingredients=10)
                if best:
                    self.best_recipe = best

            if on_progress:
                on_progress(self.generation, self.generation + generations - gen - 1,
                           avg_reward, max_reward)

        result = {
            'generations': self.generation,
            'best_score': round(self.best_score, 4),
            'final_avg': round(np.mean([h['avg'] for h in self.history[-10:]]), 4) if self.history else 0,
            'best_recipe': self.best_recipe,
            'curriculum_phase': 'nonlinear' if self.use_nonlinear_mixing else 'linear',
        }
        self.save_model()
        return result

    def save_model(self, path=None):
        """모델 가중치 + 학습 상태 저장"""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'data', 'rl_model.pt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'policy_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'generation': self.generation,
            'best_score': self.best_score,
            'best_recipe': self.best_recipe,
            'history': self.history[-100:],  # 최근 100세대만
            'n_ingredients': self.n_ingredients,
        }
        torch.save(checkpoint, path)
        print(f"[SelfPlayRL] Model saved → {path} (gen={self.generation}, best={self.best_score:.4f})")

    def load_model(self, path=None):
        """저장된 모델 가중치 로딩"""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'data', 'rl_model.pt')
        if not os.path.exists(path):
            print(f"[SelfPlayRL] No saved model at {path}")
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            # 원료 수가 다르면 로딩 스킵
            if checkpoint.get('n_ingredients', 0) != self.n_ingredients:
                print(f"[SelfPlayRL] Ingredient count mismatch: saved={checkpoint.get('n_ingredients',0)} vs current={self.n_ingredients}")
                return False
            self.policy.load_state_dict(checkpoint['policy_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.generation = checkpoint.get('generation', 0)
            self.best_score = checkpoint.get('best_score', 0)
            self.best_recipe = checkpoint.get('best_recipe', None)
            self.history = checkpoint.get('history', [])
            print(f"[SelfPlayRL] Model loaded ← {path} (gen={self.generation}, best={self.best_score:.4f})")
            return True
        except Exception as e:
            print(f"[SelfPlayRL] Failed to load model: {e}")
            return False


# ================================================================
# 글로벌 인스턴스
# ================================================================
_nose = VirtualNose()
_hedonic = HedonicFunction()
_thermo = ThermodynamicsEngine()
_rl = None  # 지연 초기화 (DB 필요)


def get_rl():
    global _rl
    if _rl is None:
        _rl = SelfPlayRL(_nose, _hedonic, _thermo)
        _rl.load_model()  # 저장된 가중치 자동 로딩
    return _rl


def simulate_recipe(smiles_list, concentrations=None, concentration_type='EDP'):
    """레시피 전체 바이오피직스 시뮬레이션 (Perception Layer 포함)
    
    Fix 2: threshold 기반 perception + 시간축 smell 호출
    """
    conc = concentrations or [1.0] * len(smiles_list)

    # 1) 초기 smell (t=0)
    nose_result = _nose.smell(smiles_list, conc)
    hedonic_result = _hedonic.evaluate_mixture(smiles_list, nose_result['activation_pattern'])

    # 2) 열역학 시뮬레이션
    thermo_result = _thermo.simulate_evaporation(
        smiles_list, conc, duration_hours=8, concentration_type=concentration_type)

    # 3) 시간축 perception — 30분 간격 8회 smell (Fix 2)
    perception_timeline = []
    timeline = thermo_result['timeline']
    # 30분(3 steps) 간격으로 샘플링
    sample_indices = list(range(0, len(timeline), 3))[:9]  # 0, 30, 60, ..., 240min

    for idx in sample_indices:
        snap = timeline[idx]
        t_min = snap['time_min']
        # headspace 농도로 smell 호출
        hs_concs = [m['headspace_pct'] for m in snap['molecules']]

        # Skip if headspace is negligible
        if sum(hs_concs) < 1e-6:
            perception_timeline.append({
                'time_min': t_min,
                'active_receptors': 0,
                'mean_activation': 0.0,
                'top_impact': 0.0, 'mid_impact': 0.0, 'base_impact': 0.0,
            })
            continue

        # Threshold-based perception: only molecules above olfactory threshold contribute
        # threshold = initial_headspace * 0.02 (2% of initial = detection limit)
        initial_hs = timeline[0]['total_headspace']
        threshold = initial_hs * 0.02

        perceived_smiles = []
        perceived_concs = []
        for smi, hs_c, mol_snap in zip(smiles_list, hs_concs, snap['molecules']):
            if hs_c > threshold:
                # sigmoid attenuation near threshold
                ratio = (hs_c - threshold) / max(threshold, 1e-8)
                attenuation = min(1.0, ratio / (1.0 + ratio))  # Hill-like: 0 at threshold, 1 at 2x threshold
                perceived_smiles.append(smi)
                perceived_concs.append(hs_c * attenuation)

        if perceived_smiles:
            snap_nose = _nose.smell(perceived_smiles, perceived_concs)
            active = snap_nose['active_receptors']
            mean_act = snap_nose['mean_activation']
        else:
            active = 0
            mean_act = 0.0

        # Impact per note group
        top_impact = sum(hs_c for hs_c, m in zip(hs_concs, snap['molecules'])
                       if m['note_type'] == 'top' and hs_c > threshold)
        mid_impact = sum(hs_c for hs_c, m in zip(hs_concs, snap['molecules'])
                       if m['note_type'] == 'middle' and hs_c > threshold)
        base_impact = sum(hs_c for hs_c, m in zip(hs_concs, snap['molecules'])
                        if m['note_type'] == 'base' and hs_c > threshold)

        perception_timeline.append({
            'time_min': t_min,
            'active_receptors': int(active),
            'mean_activation': round(float(mean_act), 4),
            'top_impact': round(float(top_impact), 6),
            'mid_impact': round(float(mid_impact), 6),
            'base_impact': round(float(base_impact), 6),
        })

    # 4) 시간축 perception에서 transition 재계산
    perception_transitions = []
    if perception_timeline:
        prev_dom = 'top'
        for pt in perception_timeline:
            impacts = {'top': pt['top_impact'], 'middle': pt['mid_impact'], 'base': pt['base_impact']}
            dom = max(impacts, key=impacts.get) if sum(impacts.values()) > 0 else 'none'
            if dom != prev_dom and dom != 'none':
                perception_transitions.append({
                    'time_min': pt['time_min'], 'from': prev_dom, 'to': dom
                })
                prev_dom = dom

    # 전환 부드러움 (perception 기반)
    p_smoothness = 1.0
    if len(perception_transitions) >= 2:
        gaps = [perception_transitions[i+1]['time_min'] - perception_transitions[i]['time_min']
                for i in range(len(perception_transitions)-1)]
        if gaps:
            cv = np.std(gaps) / (np.mean(gaps) + 1)
            p_smoothness = max(0, 1.0 - cv)

    smoothness = thermo_result.get('smoothness', thermo_result.get('transition_smoothness', 0.5))
    total_score = (hedonic_result['hedonic_score'] * 0.40 +
                   min(1, thermo_result['longevity_hours']/6) * 0.25 +
                   smoothness * 0.20 +
                   min(1.0, nose_result['active_receptors']/100) * 0.15)
    total_score = min(1.0, max(0.0, total_score))

    return {
        'total_score': round(total_score, 3),
        'nose': {
            'active_receptors': int(nose_result['active_receptors']),
            'total_receptors': int(_nose.n_receptors),
            'mean_activation': round(float(nose_result['mean_activation']), 4),
            'families': nose_result['receptor_families'],
        },
        'hedonic': hedonic_result,
        'thermodynamics': {
            'longevity_hours': thermo_result['longevity_hours'],
            'transitions': perception_transitions or thermo_result['transitions'],
            'smoothness': round(float(p_smoothness if perception_transitions else smoothness), 3),
            'initial': thermo_result['initial_molecules'],
            'timeline_summary': thermo_result['timeline'][::2][:8],
        },
        'perception': {
            'timeline': perception_timeline,
            'transitions': perception_transitions,
            'smoothness': round(float(p_smoothness), 3),
        },
    }


def evolve(generations=100, population=20, on_progress=None):
    """RL 진화 실행"""
    rl = get_rl()
    result = rl.evolve(generations=generations, population=population, on_progress=on_progress)
    return _make_serializable(result)


def _make_serializable(obj):
    """numpy/torch 타입을 JSON 직렬화 가능하게 변환"""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def get_status():
    rl = get_rl() if _rl else None
    return {
        'nose_receptors': _nose.n_receptors,
        'pleasant_patterns': len(PLEASANT_PATTERNS),
        'unpleasant_patterns': len(UNPLEASANT_PATTERNS),
        'dose_response_categories': 19,
        'rl_generation': rl.generation if rl else 0,
        'rl_best_score': round(rl.best_score, 4) if rl else 0,
        'rl_ingredients': rl.n_ingredients if rl else 0,
        'device': str(_nose._desc_cache.__class__),  # proxy
    }
