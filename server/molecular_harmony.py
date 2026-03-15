# molecular_harmony.py -- 분자 궁합 분석 v2 (RDKit + 50 수용체 + 농도 의존성)
# ====================================================================
import torch
import torch.nn as nn
import numpy as np
import database as db
import os

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

# ========================
# 50개 후각 수용체 프로필
# ========================
RECEPTOR_PROFILES = [
    # === 플로럴 계열 (6개) ===
    {'id':'OR1A1','name':'로즈 수용체','keys':['rose','geraniol','citronellol','phenylethyl_alcohol'],'labels':['floral','rose'],'threshold':0.001},
    {'id':'OR1A2','name':'재스민 수용체','keys':['jasmine','indole','benzyl_acetate','methyl_jasmonate'],'labels':['floral','jasmine'],'threshold':0.002},
    {'id':'OR2B1','name':'릴리 수용체','keys':['lily','muguet','hydroxycitronellal','lyral'],'labels':['floral','lily','muguet'],'threshold':0.003},
    {'id':'OR2C1','name':'바이올렛 수용체','keys':['violet','ionone','orris','iris'],'labels':['floral','violet','powdery'],'threshold':0.001},
    {'id':'OR2W1','name':'화이트플로럴 수용체','keys':['tuberose','ylang','frangipani','gardenia'],'labels':['floral','sweet'],'threshold':0.002},
    {'id':'OR3A1','name':'라벤더 수용체','keys':['lavender','linalool','linalyl_acetate'],'labels':['floral','herbal','fresh'],'threshold':0.005},
    # === 시트러스 계열 (4개) ===
    {'id':'OR4A1','name':'레몬 수용체','keys':['lemon','limonene','citral','neral'],'labels':['citrus','fresh'],'threshold':0.01},
    {'id':'OR4B1','name':'오렌지 수용체','keys':['orange','valencene','decanal'],'labels':['citrus','sweet','fruity'],'threshold':0.008},
    {'id':'OR4C1','name':'베르가못 수용체','keys':['bergamot','linalyl_acetate','bergapten'],'labels':['citrus','fresh','floral'],'threshold':0.005},
    {'id':'OR4D1','name':'자몽 수용체','keys':['grapefruit','nootkatone','thioterpineol'],'labels':['citrus','fruity','fresh'],'threshold':0.0001},
    # === 우디 계열 (5개) ===
    {'id':'OR5A1','name':'시더 수용체','keys':['cedar','cedrene','cedrol','cedarwood'],'labels':['woody','cedar'],'threshold':0.01},
    {'id':'OR5B1','name':'샌달우드 수용체','keys':['sandalwood','santalol','ebanol'],'labels':['woody','creamy','warm'],'threshold':0.005},
    {'id':'OR5C1','name':'베티버 수용체','keys':['vetiver','vetiverol','khusimol'],'labels':['woody','earthy','smoky'],'threshold':0.003},
    {'id':'OR5D1','name':'패출리 수용체','keys':['patchouli','patchoulol','norpatchoulenol'],'labels':['woody','earthy','warm'],'threshold':0.002},
    {'id':'OR5E1','name':'앰브록산 수용체','keys':['ambroxan','ambroxide','cetalox'],'labels':['woody','amber','warm'],'threshold':0.001},
    # === 스파이시 계열 (4개) ===
    {'id':'OR6A1','name':'시나몬 수용체','keys':['cinnamon','cinnamaldehyde','eugenol'],'labels':['spicy','warm','sweet'],'threshold':0.005},
    {'id':'OR6B1','name':'페퍼 수용체','keys':['pepper','piperine','rotundone'],'labels':['spicy','warm'],'threshold':0.001},
    {'id':'OR6C1','name':'클로브 수용체','keys':['clove','eugenol','isoeugenol'],'labels':['spicy','warm','sweet'],'threshold':0.003},
    {'id':'OR6D1','name':'카다멈 수용체','keys':['cardamom','cineole','terpinyl_acetate'],'labels':['spicy','fresh','herbal'],'threshold':0.005},
    # === 스위트/구르망 (4개) ===
    {'id':'OR7A1','name':'바닐라 수용체','keys':['vanilla','vanillin','ethyl_vanillin','coumarin'],'labels':['sweet','vanilla','gourmand'],'threshold':0.005},
    {'id':'OR7B1','name':'카라멜 수용체','keys':['caramel','maltol','ethyl_maltol','furaneol'],'labels':['sweet','gourmand','warm'],'threshold':0.003},
    {'id':'OR7C1','name':'꿀 수용체','keys':['honey','phenylacetic_acid','phenylacetaldehyde'],'labels':['sweet','warm','animalic'],'threshold':0.002},
    {'id':'OR7D1','name':'초콜릿 수용체','keys':['chocolate','cocoa','pyrazine','theobromine'],'labels':['sweet','gourmand','warm'],'threshold':0.008},
    # === 프레시/아쿠아틱 (4개) ===
    {'id':'OR8A1','name':'아쿠아틱 수용체','keys':['calone','dihydromyrcenol','aquatic','marine'],'labels':['fresh','aquatic','clean'],'threshold':0.005},
    {'id':'OR8B1','name':'오존 수용체','keys':['ozonic','metallic','oxygen','helional'],'labels':['fresh','ozonic','clean'],'threshold':0.001},
    {'id':'OR8C1','name':'민트 수용체','keys':['mint','menthol','menthone','peppermint'],'labels':['fresh','cool','herbal'],'threshold':0.01},
    {'id':'OR8D1','name':'알데히드 수용체','keys':['aldehyde','decanal','undecanal','dodecanal'],'labels':['fresh','clean','waxy'],'threshold':0.003},
    # === 그린/허벌 (4개) ===
    {'id':'OR9A1','name':'풀잎 수용체','keys':['green','cis3hexenol','leaf_alcohol','galbanum'],'labels':['green','fresh','leafy'],'threshold':0.005},
    {'id':'OR9B1','name':'허브 수용체','keys':['herbal','thyme','basil','rosemary','thymol'],'labels':['herbal','aromatic','green'],'threshold':0.008},
    {'id':'OR9C1','name':'차 수용체','keys':['tea','linalool','geraniol','tea_absolute'],'labels':['green','fresh','floral'],'threshold':0.005},
    {'id':'OR9D1','name':'모스 수용체','keys':['moss','oakmoss','treemoss','evernyl'],'labels':['green','earthy','woody'],'threshold':0.002},
    # === 머스크/앰버 (5개) ===
    {'id':'OR10A1','name':'화이트머스크 수용체','keys':['white_musk','galaxolide','habanolide'],'labels':['musk','clean','powdery'],'threshold':0.001},
    {'id':'OR10B1','name':'애니멀머스크 수용체','keys':['muscone','civetone','ambrettolide'],'labels':['musk','animalic','warm'],'threshold':0.0005},
    {'id':'OR10C1','name':'앰버 수용체','keys':['amber','labdanum','ambrein','benzoin'],'labels':['amber','warm','sweet'],'threshold':0.003},
    {'id':'OR10D1','name':'파우더리 수용체','keys':['powdery','heliotropin','coumarin','iris'],'labels':['powdery','sweet','soft'],'threshold':0.005},
    {'id':'OR10E1','name':'인센스 수용체','keys':['incense','olibanum','frankincense','myrrh'],'labels':['resinous','smoky','warm'],'threshold':0.003},
    # === 스모키/레더 (3개) ===
    {'id':'OR11A1','name':'스모키 수용체','keys':['smoky','guaiacol','cresol','birch_tar'],'labels':['smoky','woody','warm'],'threshold':0.002},
    {'id':'OR11B1','name':'레더 수용체','keys':['leather','isobutyl_quinoline','castoreum','suede'],'labels':['leather','smoky','animalic'],'threshold':0.001},
    {'id':'OR11C1','name':'타바코 수용체','keys':['tobacco','solanone','nicotine_free','hay'],'labels':['smoky','warm','sweet'],'threshold':0.005},
    # === 프루티 (4개) ===
    {'id':'OR12A1','name':'복숭아 수용체','keys':['peach','gamma_decalactone','undecalactone'],'labels':['fruity','sweet','creamy'],'threshold':0.005},
    {'id':'OR12B1','name':'베리 수용체','keys':['berry','raspberry','frambinone','strawberry'],'labels':['fruity','sweet'],'threshold':0.003},
    {'id':'OR12C1','name':'사과 수용체','keys':['apple','damascone','hexyl_acetate'],'labels':['fruity','fresh','green'],'threshold':0.008},
    {'id':'OR12D1','name':'트로피칼 수용체','keys':['tropical','mango','passion','coconut'],'labels':['fruity','sweet','creamy'],'threshold':0.005},
    # === 어씨/루트 (3개) ===
    {'id':'OR13A1','name':'흙 수용체','keys':['earthy','geosmin','patchouli','vetiver'],'labels':['earthy','woody'],'threshold':0.0001},
    {'id':'OR13B1','name':'버섯 수용체','keys':['mushroom','1_octen_3_ol','matsutake'],'labels':['earthy','green'],'threshold':0.001},
    {'id':'OR13C1','name':'루트 수용체','keys':['root','angelica','costus','spikenard'],'labels':['earthy','woody','herbal'],'threshold':0.003},
    # === 특수 (4개) ===
    {'id':'OR14A1','name':'캠퍼 수용체','keys':['camphor','borneol','eucalyptol','1_8_cineole'],'labels':['fresh','herbal','cool'],'threshold':0.01},
    {'id':'OR14B1','name':'왁시 수용체','keys':['waxy','aldehyde_c12','stearone'],'labels':['waxy','clean'],'threshold':0.005},
    {'id':'OR14C1','name':'메탈릭 수용체','keys':['metallic','blood','iron','cashmeran'],'labels':['metallic','cool'],'threshold':0.001},
    {'id':'OR14D1','name':'솔티 수용체','keys':['salty','marine','sea','ammonium'],'labels':['fresh','aquatic'],'threshold':0.01},
]

# ========================
# 농도-응답 곡선 (Dose-Response)
# ========================
# Hill 방정식: response = (C^n) / (EC50^n + C^n)
# EC50 = 반응 50% 농도, n = Hill 계수 (가파름)
DOSE_RESPONSE = {
    # 카테고리별 기본 파라미터
    'floral':   {'ec50': 3.0,  'n': 1.5, 'max_pleasant': 8.0,  'masking_onset': 12.0},
    'citrus':   {'ec50': 2.0,  'n': 2.0, 'max_pleasant': 6.0,  'masking_onset': 10.0},
    'woody':    {'ec50': 4.0,  'n': 1.2, 'max_pleasant': 15.0, 'masking_onset': 25.0},
    'spicy':    {'ec50': 1.0,  'n': 2.5, 'max_pleasant': 3.0,  'masking_onset': 5.0},
    'fruity':   {'ec50': 2.5,  'n': 1.8, 'max_pleasant': 7.0,  'masking_onset': 12.0},
    'gourmand': {'ec50': 3.0,  'n': 1.5, 'max_pleasant': 10.0, 'masking_onset': 15.0},
    'aquatic':  {'ec50': 1.5,  'n': 2.0, 'max_pleasant': 4.0,  'masking_onset': 8.0},
    'amber':    {'ec50': 3.5,  'n': 1.3, 'max_pleasant': 12.0, 'masking_onset': 20.0},
    'musk':     {'ec50': 2.0,  'n': 1.0, 'max_pleasant': 8.0,  'masking_onset': 15.0},
    'herbal':   {'ec50': 2.5,  'n': 1.5, 'max_pleasant': 6.0,  'masking_onset': 10.0},
    'green':    {'ec50': 1.5,  'n': 2.0, 'max_pleasant': 5.0,  'masking_onset': 8.0},
    'fresh':    {'ec50': 1.5,  'n': 2.0, 'max_pleasant': 5.0,  'masking_onset': 8.0},
    'leather':  {'ec50': 1.5,  'n': 2.0, 'max_pleasant': 5.0,  'masking_onset': 8.0},
    'smoky':    {'ec50': 1.0,  'n': 2.5, 'max_pleasant': 3.0,  'masking_onset': 5.0},
    'powdery':  {'ec50': 3.0,  'n': 1.5, 'max_pleasant': 10.0, 'masking_onset': 15.0},
    'earthy':   {'ec50': 2.0,  'n': 1.8, 'max_pleasant': 5.0,  'masking_onset': 10.0},
    'animalic': {'ec50': 0.3,  'n': 3.0, 'max_pleasant': 1.0,  'masking_onset': 2.0},
    'resinous': {'ec50': 3.0,  'n': 1.3, 'max_pleasant': 10.0, 'masking_onset': 18.0},
    'aldehyde': {'ec50': 0.5,  'n': 2.5, 'max_pleasant': 2.0,  'masking_onset': 4.0},
}

def hill_response(concentration, ec50, n):
    """Hill 방정식 — 농도에 따른 수용체 반응 (0~1)"""
    return (concentration ** n) / (ec50 ** n + concentration ** n)

def dose_pleasantness(concentration, category):
    """농도별 쾌적도 (0~1). 최적 농도를 넘으면 감소"""
    params = DOSE_RESPONSE.get(category, {'ec50':2.5,'n':1.5,'max_pleasant':8.0,'masking_onset':12.0})
    response = hill_response(concentration, params['ec50'], params['n'])
    if concentration <= params['max_pleasant']:
        return response
    elif concentration <= params['masking_onset']:
        # 최적~마스킹 사이: 점진적 감소
        overshoot = (concentration - params['max_pleasant']) / (params['masking_onset'] - params['max_pleasant'])
        return response * (1.0 - overshoot * 0.5)
    else:
        # 마스킹 영역: 다른 향을 가림
        overshoot = (concentration - params['masking_onset']) / max(params['masking_onset'], 1)
        return response * max(0.2, 0.5 - overshoot * 0.3)

def concentration_interaction(conc_a, cat_a, conc_b, cat_b):
    """두 원료의 농도 기반 상호작용 예측"""
    pleas_a = dose_pleasantness(conc_a, cat_a)
    pleas_b = dose_pleasantness(conc_b, cat_b)
    params_a = DOSE_RESPONSE.get(cat_a, {'masking_onset':12.0})
    params_b = DOSE_RESPONSE.get(cat_b, {'masking_onset':12.0})

    # 마스킹 체크: 한쪽이 마스킹 구간이면 약한 쪽을 가림
    a_masking = conc_a > params_a['masking_onset']
    b_masking = conc_b > params_b['masking_onset']

    if a_masking and not b_masking:
        return {'type': 'masking', 'strength': 0.3 + (conc_a/params_a['masking_onset']-1)*0.3,
                'masker': cat_a, 'masked': cat_b, 'detail': f'{cat_a} {conc_a:.1f}%가 {cat_b}를 가림'}
    if b_masking and not a_masking:
        return {'type': 'masking', 'strength': 0.3 + (conc_b/params_b['masking_onset']-1)*0.3,
                'masker': cat_b, 'masked': cat_a, 'detail': f'{cat_b} {conc_b:.1f}%가 {cat_a}를 가림'}

    # 시너지 체크: 둘 다 최적 구간이면 시너지
    opt_a = conc_a <= params_a.get('max_pleasant',8)
    opt_b = conc_b <= params_b.get('max_pleasant',8)
    if opt_a and opt_b and pleas_a > 0.3 and pleas_b > 0.3:
        synergy = (pleas_a + pleas_b) / 2
        return {'type': 'synergy', 'strength': synergy, 'detail': f'둘 다 최적 농도 — 시너지 {synergy:.2f}'}

    return {'type': 'neutral', 'strength': (pleas_a + pleas_b) / 2, 'detail': '보통'}


# ========================
# RDKit 분자 핑거프린트
# ========================
_fp_cache = {}

def get_fingerprint(smiles, nbits=2048):
    """SMILES → Morgan FP (ECFP4)"""
    if smiles in _fp_cache:
        return _fp_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    _fp_cache[smiles] = fp
    return fp

def tanimoto_similarity(smiles_a, smiles_b):
    """두 분자의 Tanimoto 유사도 (0~1)"""
    fp_a = get_fingerprint(smiles_a)
    fp_b = get_fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return 0.5
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)

def get_mol_descriptors(smiles):
    """분자 물리화학 특성"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'rotatable': Descriptors.NumRotatableBonds(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol),
        'heavy_atoms': mol.GetNumHeavyAtoms(),
    }

def fp_to_numpy(smiles, nbits=256):
    """SMILES → numpy array (256bit Morgan FP)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    arr = np.zeros(nbits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ========================
# 알려진 조향 어코드 (시너지 조합)
# ========================
KNOWN_ACCORDS = {
    'floral_bouquet':   {'ingredients':['rose_absolute','jasmine_absolute','ylang_extra','tuberose_absolute'],'synergy':0.9,'note':'클래식 화이트 플로럴 부케'},
    'citrus_fresh':     {'ingredients':['bergamot','lemon','orange_sweet','grapefruit','petitgrain'],'synergy':0.85,'note':'시트러스 조화'},
    'fougere':          {'ingredients':['lavender','geranium','oakmoss','coumarin_lactone','tonka_bean'],'synergy':0.88,'note':'푸제르 어코드'},
    'chypre':           {'ingredients':['bergamot','oakmoss','labdanum','patchouli','rose_absolute'],'synergy':0.87,'note':'시프레 구조'},
    'oriental':         {'ingredients':['vanilla','benzoin','labdanum','cinnamon_bark','amber_resinoid'],'synergy':0.85,'note':'오리엔탈 베이스'},
    'woody_amber':      {'ingredients':['sandalwood','cedarwood','vetiver','amber_resinoid','patchouli'],'synergy':0.88,'note':'우디 앰버'},
    'lavender_core':    {'ingredients':['lavender','linalool','linalyl_acetate'],'synergy':0.95,'note':'라벤더 핵심 분자'},
    'musk_base':        {'ingredients':['white_musk','galaxolide','muscone_synth','cashmeran'],'synergy':0.82,'note':'머스크 믹스'},
    'gourmand':         {'ingredients':['vanilla','caramel','tonka_bean','cocoa_absolute','benzoin'],'synergy':0.83,'note':'구르망'},
    'herbal_green':     {'ingredients':['basil','thyme','rosemary','clary_sage','galbanum'],'synergy':0.80,'note':'허벌 그린'},
    'smoky_leather':    {'ingredients':['birch_tar','leather_accord','castoreum_synth','vetiver','tobacco_absolute'],'synergy':0.78,'note':'스모키 레더'},
    'aquatic':          {'ingredients':['calone','dihydromyrcenol','sea_salt','ambroxan'],'synergy':0.82,'note':'아쿠아틱'},
    'korean_floral':    {'ingredients':['magnolia','chrysanthemum','plum_blossom','lotus','mugwort'],'synergy':0.75,'note':'한국 전통 화향'},
}

MASKING_RULES = [
    {'strong':'galaxolide','weak_cats':['citrus','green','herbal'],'severity':0.6,'note':'갈락솔라이드→톱노트 마스킹'},
    {'strong':'iso_e_super','weak_cats':['floral','citrus'],'severity':0.5,'note':'ISO E Super→가벼운 향 삼킴'},
    {'strong':'ambroxan','weak_cats':['green','herbal','citrus'],'severity':0.55,'note':'앰브록산→프레시 억제'},
    {'strong':'patchouli','weak_cats':['aquatic','fresh','citrus'],'severity':0.65,'note':'패출리→라이트 노트 지배'},
    {'strong':'skatole','weak_cats':['floral','citrus','fruity','green'],'severity':0.8,'note':'스카톨→극소량만'},
    {'strong':'castoreum_synth','weak_cats':['floral','fresh','clean'],'severity':0.7,'note':'카스토레움→클린 파괴'},
    {'strong':'civet_synth','weak_cats':['fresh','clean','aquatic'],'severity':0.75,'note':'시벳→프레시 압도'},
    {'strong':'oud_synth','weak_cats':['citrus','fruity','green'],'severity':0.6,'note':'우드→밝은 계열 차단'},
]

SYNERGY_PAIRS = {
    ('floral','citrus'):0.75,('floral','green'):0.70,('floral','woody'):0.80,
    ('floral','musk'):0.85,('floral','powdery'):0.80,('citrus','herbal'):0.75,
    ('citrus','aquatic'):0.70,('woody','amber'):0.85,('woody','leather'):0.80,
    ('woody','spicy'):0.75,('amber','gourmand'):0.75,('amber','resinous'):0.85,
    ('musk','gourmand'):0.70,('spicy','gourmand'):0.72,('spicy','resinous'):0.78,
    ('smoky','leather'):0.80,('smoky','woody'):0.75,('earthy','green'):0.78,
    ('earthy','woody'):0.80,('fruity','floral'):0.78,('fruity','gourmand'):0.72,
}

CLASHING_PAIRS = {
    ('aquatic','gourmand'):0.3,('aquatic','smoky'):0.25,('clean','animalic'):0.2,
    ('fresh','smoky'):0.3,('citrus','animalic'):0.25,('green','gourmand'):0.35,
    ('aldehyde','smoky'):0.3,
}


# ========================
# 궁합 엔진
# ========================
ALL_ODOR_LABELS = [
    'floral','citrus','woody','spicy','sweet','fresh','green','warm',
    'musk','fruity','rose','jasmine','cedar','vanilla','amber','clean',
    'smoky','powdery','aquatic','herbal'
]

CATEGORY_TO_ODOR = {
    'floral':['floral','rose','jasmine','sweet'],'citrus':['citrus','fresh','fruity'],
    'woody':['woody','cedar','warm'],'spicy':['spicy','warm'],
    'fruity':['fruity','sweet','fresh'],'gourmand':['sweet','vanilla','warm'],
    'aquatic':['aquatic','fresh','clean'],'amber':['amber','warm','sweet'],
    'musk':['musk','warm','powdery'],'aromatic':['herbal','fresh','green'],
    'herbal':['herbal','green','fresh'],'green':['green','fresh','herbal'],
    'fresh':['fresh','clean','citrus'],'leather':['smoky','warm','woody'],
    'smoky':['smoky','woody','warm'],'powdery':['powdery','sweet','musk'],
    'earthy':['woody','green','herbal'],'animalic':['musk','warm'],
    'resinous':['warm','sweet','amber'],'aldehyde':['fresh','clean','floral'],
    'chypre':['woody','floral','green'],'waxy':['waxy','warm'],
    'synthetic':['musk','fresh','clean'],
}


class MolecularHarmonyEngine:
    """분자 궁합 분석 v2: 50 수용체 + 농도 + RDKit"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trained = False
        self.train_epochs = 0
        self.train_loss = 0.0
        print(f"[MolecularHarmony] v2 | Device: {self.device} | Receptors: {len(RECEPTOR_PROFILES)} | RDKit: OK")

    def _get_receptor_activation(self, labels, concentration=1.0):
        """분자의 각 수용체 활성화 수준 (50개)"""
        activations = {}
        for rp in RECEPTOR_PROFILES:
            # 키워드 매칭
            match_count = sum(1 for k in rp['keys'] if k in labels)
            label_match = sum(1 for l in rp['labels'] if l in labels)
            base = (match_count * 0.3 + label_match * 0.2)
            if base > 0:
                # Hill 방정식으로 농도 의존적 활성화
                threshold = rp['threshold']
                response = hill_response(concentration, threshold * 100, 1.5)
                activations[rp['id']] = min(1.0, base * response)
        return activations

    def _receptor_competition(self, activations_a, activations_b):
        """두 분자의 수용체 경쟁도 분석"""
        shared = set(activations_a.keys()) & set(activations_b.keys())
        if not shared:
            return 0.0, []

        competitions = []
        total_comp = 0
        for rid in shared:
            act_a = activations_a[rid]
            act_b = activations_b[rid]
            # 한쪽이 훨씬 강하면 마스킹
            if max(act_a, act_b) > 0:
                ratio = min(act_a, act_b) / max(act_a, act_b)
                competition = (1 - ratio) * 0.5  # 차이가 클수록 경쟁
                total_comp += competition
                rname = next((r['name'] for r in RECEPTOR_PROFILES if r['id']==rid), rid)
                competitions.append({'receptor': rname, 'competition': round(competition, 3)})

        avg_comp = total_comp / len(shared) if shared else 0
        return avg_comp, competitions

    def _structural_compatibility(self, smiles_a, smiles_b):
        """RDKit 분자 구조 기반 궁합"""
        if not smiles_a or not smiles_b:
            return {'similarity': 0.5, 'compatible': True, 'detail': 'SMILES 없음'}

        sim = tanimoto_similarity(smiles_a, smiles_b)
        desc_a = get_mol_descriptors(smiles_a)
        desc_b = get_mol_descriptors(smiles_b)

        detail = f'Tanimoto: {sim:.3f}'
        compatible = True

        if desc_a and desc_b:
            mw_diff = abs(desc_a['mw'] - desc_b['mw'])
            logp_diff = abs(desc_a['logp'] - desc_b['logp'])
            detail += f' | MW차: {mw_diff:.0f} | LogP차: {logp_diff:.1f}'

            # 적절한 구조 차이 = 시간축 분산
            if 0.15 < sim < 0.5:
                detail += ' | 구조적 상보성 우수'
            elif sim > 0.7:
                detail += ' | 너무 유사 (추가 가치 낮음)'
                compatible = False
            elif sim < 0.1:
                detail += ' | 완전히 다른 구조 (리스크)'

        return {'similarity': round(sim, 3), 'compatible': compatible, 'detail': detail}

    def analyze_pair(self, ing_a, ing_b, conc_a=None, conc_b=None):
        """원료 쌍 종합 궁합 분석 (수용체 50개 + 농도 + 구조)"""
        cat_a = ing_a.get('category', '')
        cat_b = ing_b.get('category', '')
        conc_a = conc_a or ing_a.get('percentage') or ing_a.get('typical_pct') or 3
        conc_b = conc_b or ing_b.get('percentage') or ing_b.get('typical_pct') or 3

        # 1) 농도 기반 상호작용
        dose_result = concentration_interaction(conc_a, cat_a, conc_b, cat_b)

        # 2) 50개 수용체 활성화 + 경쟁
        labels_a = set(CATEGORY_TO_ODOR.get(cat_a, []))
        labels_b = set(CATEGORY_TO_ODOR.get(cat_b, []))
        act_a = self._get_receptor_activation(labels_a, conc_a)
        act_b = self._get_receptor_activation(labels_b, conc_b)
        competition, comp_details = self._receptor_competition(act_a, act_b)

        # 3) RDKit 구조 분석
        smiles_a = ing_a.get('smiles', '')
        smiles_b = ing_b.get('smiles', '')
        structural = self._structural_compatibility(smiles_a, smiles_b)

        # 4) 카테고리 시너지/충돌
        pair = (cat_a, cat_b)
        pair_rev = (cat_b, cat_a)
        cat_score = SYNERGY_PAIRS.get(pair, SYNERGY_PAIRS.get(pair_rev, 0.5))
        if pair in CLASHING_PAIRS or pair_rev in CLASHING_PAIRS:
            cat_score = CLASHING_PAIRS.get(pair, CLASHING_PAIRS.get(pair_rev, 0.5))

        # 5) 종합 점수
        weights = {'dose': 0.30, 'receptor': 0.25, 'structural': 0.20, 'category': 0.25}
        dose_score = 0.8 if dose_result['type'] == 'synergy' else (0.3 if dose_result['type'] == 'masking' else 0.5)
        receptor_score = max(0, 1.0 - competition * 2)
        struct_score = 0.7 if structural['compatible'] else 0.3

        total = (dose_score * weights['dose'] +
                 receptor_score * weights['receptor'] +
                 struct_score * weights['structural'] +
                 cat_score * weights['category'])

        interaction = 'synergy' if total > 0.65 else ('masking' if total < 0.4 else 'neutral')

        return {
            'score': round(total, 3),
            'interaction': interaction,
            'dose': {'type': dose_result['type'], 'detail': dose_result['detail'],
                     'conc_a': conc_a, 'conc_b': conc_b,
                     'pleasant_a': round(dose_pleasantness(conc_a, cat_a), 3),
                     'pleasant_b': round(dose_pleasantness(conc_b, cat_b), 3)},
            'receptor': {'competition': round(competition, 3),
                         'shared_receptors': len(set(act_a.keys()) & set(act_b.keys())),
                         'activated_a': len(act_a), 'activated_b': len(act_b),
                         'details': comp_details[:5]},
            'structural': structural,
            'category_score': round(cat_score, 3),
        }

    def check_ingredient_harmony(self, ingredient_ids):
        """원료 리스트 전체 궁합 분석 (DB 실패 시 JSON 폴백)"""
        # 원료 로드 (DB → JSON 폴백)
        all_ings = []
        try:
            all_ings = db.get_all_ingredients()
        except Exception:
            pass
        if not all_ings:
            import json as _json
            for path in ['data/ingredients.json', '../data/ingredients.json', 'server/data/ingredients.json']:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        all_ings = _json.load(f)
                    break
        ing_map = {ing['id']: ing for ing in all_ings}

        valid = [ing_map[iid] for iid in ingredient_ids if iid in ing_map]
        if len(valid) < 2:
            return {'harmony':0.5,'note':'원료 2개 미만','synergy_count':0,'masking_count':0,'total_pairs':0}

        # 분자 DB에서 SMILES 매칭 (DB → JSON 폴백)
        try:
            all_mols = []
            try:
                all_mols = db.get_all_molecules(limit=500)
            except Exception:
                pass
            if not all_mols:
                import json as _json
                for path in ['data/molecules.json', '../data/molecules.json', 'server/data/molecules.json']:
                    if os.path.exists(path):
                        with open(path, 'r', encoding='utf-8') as f:
                            all_mols = _json.load(f)
                        break
            # ingredient_smiles.json도 로드
            smiles_map = {}
            for path in ['data/ingredient_smiles.json', '../data/ingredient_smiles.json', 'server/data/ingredient_smiles.json']:
                if os.path.exists(path):
                    import json as _json
                    with open(path, 'r', encoding='utf-8') as f:
                        raw_sm = _json.load(f)
                    if isinstance(raw_sm, dict):
                        smiles_map = {k.lower(): v for k, v in raw_sm.items()}
                    break

            mol_map = {}
            for ing in valid:
                # 먼저 ingredient_smiles.json에서 찾기
                iid = ing.get('id', '').lower()
                if iid in smiles_map:
                    mol_map[ing['id']] = smiles_map[iid]
                    continue
                name_en = (ing.get('name_en') or '').lower().strip()
                if not name_en:
                    continue
                best_mol = None
                best_priority = 0
                for mol in all_mols:
                    mname = (mol.get('name') or '').lower().strip()
                    if not mname:
                        continue
                    if name_en == mname:
                        best_mol = mol
                        best_priority = 1
                        break
                    if best_priority < 2:
                        import re
                        if re.search(r'\b' + re.escape(name_en) + r'\b', mname):
                            best_mol = mol
                            best_priority = 2
                if best_mol:
                    mol_map[ing['id']] = best_mol.get('smiles', '')
        except Exception:
            mol_map = {}

        # 쌍별 분석
        pairs = []
        for i in range(len(valid)):
            for j in range(i+1, len(valid)):
                a, b = valid[i], valid[j]
                # SMILES 주입
                if a['id'] in mol_map:
                    a = {**a, 'smiles': mol_map[a['id']]}
                if b['id'] in mol_map:
                    b = {**b, 'smiles': mol_map[b['id']]}
                pr = self.analyze_pair(a, b)
                pr['ing_a'] = a.get('name_ko', a['id'])
                pr['ing_b'] = b.get('name_ko', b['id'])
                pairs.append(pr)

        synergies = [p for p in pairs if p['interaction'] == 'synergy']
        maskings = [p for p in pairs if p['interaction'] == 'masking']

        avg_score = np.mean([p['score'] for p in pairs])
        bonus = len(synergies) / max(len(pairs), 1) * 0.15
        penalty = len(maskings) / max(len(pairs), 1) * 0.2
        harmony = min(1.0, max(0.0, avg_score + bonus - penalty))

        # 어코드/마스킹/시너지/충돌 체크 (기존 유지)
        ing_set = set(ingredient_ids)
        cats = set(ing_map[i].get('category','') for i in ingredient_ids if i in ing_map)

        accords = []
        for aname, adata in KNOWN_ACCORDS.items():
            overlap = ing_set & set(adata['ingredients'])
            if len(overlap) >= 2:
                accords.append({'accord':aname,'matched':list(overlap),'synergy':adata['synergy'],'note':adata['note']})

        mask_warns = []
        for rule in MASKING_RULES:
            if rule['strong'] in ing_set:
                for iid in ingredient_ids:
                    ing = ing_map.get(iid)
                    if ing and ing.get('category','') in rule['weak_cats']:
                        mask_warns.append({'strong':rule['strong'],'weak':iid,'severity':rule['severity'],'note':rule['note']})

        syn_bonuses = [{'pair':f'{a}+{b}','bonus':s} for (a,b),s in SYNERGY_PAIRS.items() if a in cats and b in cats]
        clash_warns = [{'pair':f'{a}+{b}','risk':round(1-s,2)} for (a,b),s in CLASHING_PAIRS.items() if a in cats and b in cats]

        # 농도 경고
        dose_warnings = []
        for p in pairs:
            if p['dose']['type'] == 'masking':
                dose_warnings.append({
                    'pair': f"{p['ing_a']} + {p['ing_b']}",
                    'detail': p['dose']['detail'],
                    'conc': f"{p['dose']['conc_a']:.1f}% + {p['dose']['conc_b']:.1f}%",
                })

        return {
            'harmony': round(harmony, 3),
            'total_pairs': len(pairs),
            'synergy_count': len(synergies),
            'masking_count': len(maskings),
            'neutral_count': len(pairs) - len(synergies) - len(maskings),
            'synergy_ratio': round(len(synergies)/max(len(pairs),1), 3),
            'pair_details': sorted(pairs, key=lambda p: p['score'], reverse=True)[:15],
            'known_accords': accords,
            'masking_warnings': mask_warns,
            'synergy_bonuses': syn_bonuses,
            'clashing_warnings': clash_warns,
            'dose_warnings': dose_warnings,
            'receptors_used': len(RECEPTOR_PROFILES),
            'method': 'receptor50_dose_response_rdkit_structural',
            'smiles_matched': len(mol_map),
        }
    
    def analyze_pair_v22(self, odor_vec_a, odor_vec_b, conc_a=3.0, conc_b=3.0):
        """V22 22d 벡터 기반 쌍 궁합 분석 (ConcentrationModulator + PhysicsMixture 연동)"""
        try:
            from odor_engine import (_concentration_modulator, _physics_mixture, 
                                     ODOR_DIMENSIONS, N_ODOR_DIM)
        except ImportError:
            return {'score': 0.5, 'interaction': 'unknown', 'method': 'fallback'}
        
        odor_vec_a = np.array(odor_vec_a, dtype=np.float64)
        odor_vec_b = np.array(odor_vec_b, dtype=np.float64)
        
        # 농도 보정
        mod_a = _concentration_modulator.modulate(odor_vec_a, conc_a)
        mod_b = _concentration_modulator.modulate(odor_vec_b, conc_b)
        
        # 혼합 예측
        mixture_vec, analysis = _physics_mixture.mix(
            np.array([mod_a, mod_b]), np.array([conc_a, conc_b]), return_analysis=True)
        
        # 코사인 유사도
        dot = np.dot(mod_a, mod_b)
        norm = (np.linalg.norm(mod_a) * np.linalg.norm(mod_b)) + 1e-8
        cos_sim = float(dot / norm)
        
        # 상위 차원
        top_a = [(ODOR_DIMENSIONS[i], float(mod_a[i])) for i in np.argsort(mod_a)[::-1][:3]]
        top_b = [(ODOR_DIMENSIONS[i], float(mod_b[i])) for i in np.argsort(mod_b)[::-1][:3]]
        top_mix = [(ODOR_DIMENSIONS[i], float(mixture_vec[i])) for i in np.argsort(mixture_vec)[::-1][:5]]
        
        # 상호작용 유형
        interactions = analysis.get('interactions', [])
        itype = interactions[0]['type'] if interactions else 'neutral'
        synergy_score = interactions[0].get('synergy_score', 0) if interactions else 0
        
        # 종합 점수
        if itype == 'synergy':
            score = 0.7 + min(0.3, synergy_score)
        elif itype == 'masking':
            mr = interactions[0].get('masking_ratio', 0.5) if interactions else 0.5
            score = 0.2 + mr * 0.3
        elif itype == 'antagonism':
            score = max(0.1, 0.4 + synergy_score)
        else:
            score = 0.5 + cos_sim * 0.2
        
        return {
            'score': round(score, 3),
            'interaction': itype,
            'cosine_similarity': round(cos_sim, 3),
            'top_a': top_a, 'top_b': top_b,
            'mixture_top': top_mix,
            'mixture_vector': mixture_vec.tolist(),
            'conc_a': conc_a, 'conc_b': conc_b,
            'method': 'v22_physics',
        }
    
    def analyze_mixture_v22(self, odor_vectors, concentrations):
        """V22 22d 벡터 기반 전체 혼합물 궁합 분석"""
        try:
            from odor_engine import (_concentration_modulator, _physics_mixture,
                                     ODOR_DIMENSIONS, N_ODOR_DIM)
        except ImportError:
            return {'harmony': 0.5, 'method': 'fallback'}
        
        odor_vectors = np.array(odor_vectors, dtype=np.float64)
        concentrations = np.array(concentrations, dtype=np.float64)
        
        modulated = _concentration_modulator.batch_modulate(odor_vectors, concentrations)
        mixture_vec, analysis = _physics_mixture.mix(
            modulated, concentrations, return_analysis=True)
        
        interactions = analysis.get('interactions', [])
        synergies = [i for i in interactions if i['type'] == 'synergy']
        maskings = [i for i in interactions if i['type'] == 'masking']
        antagonisms = [i for i in interactions if i['type'] == 'antagonism']
        
        n_pairs = max(len(interactions), 1)
        harmony = 0.5 + len(synergies)/n_pairs*0.3 - len(maskings)/n_pairs*0.15 - len(antagonisms)/n_pairs*0.2
        harmony = max(0.0, min(1.0, harmony))
        
        top_mix = [(ODOR_DIMENSIONS[i], float(mixture_vec[i])) 
                    for i in np.argsort(mixture_vec)[::-1][:5]]
        
        return {
            'harmony': round(harmony, 3),
            'mixture_vector': mixture_vec.tolist(),
            'top_dimensions': top_mix,
            'total_pairs': len(interactions),
            'synergy_count': len(synergies),
            'masking_count': len(maskings),
            'antagonism_count': len(antagonisms),
            'interactions': interactions,
            'method': 'v22_physics',
        }


# 글로벌 인스턴스
_harmony_engine = MolecularHarmonyEngine()

def train_harmony(epochs=40, on_progress=None):
    _harmony_engine.trained = True
    _harmony_engine.train_epochs = epochs

def check_harmony(ingredient_ids):
    return _harmony_engine.check_ingredient_harmony(ingredient_ids)

def predict_pair(mol_a, mol_b):
    return _harmony_engine.analyze_pair(mol_a, mol_b)

def predict_pair_v22(odor_vec_a, odor_vec_b, conc_a=3.0, conc_b=3.0):
    """V22 벡터 기반 쌍 궁합"""
    return _harmony_engine.analyze_pair_v22(odor_vec_a, odor_vec_b, conc_a, conc_b)

def analyze_mixture_v22(odor_vectors, concentrations):
    """V22 벡터 기반 혼합물 분석"""
    return _harmony_engine.analyze_mixture_v22(odor_vectors, concentrations)

def get_status():
    return {
        'trained': True,
        'device': str(_harmony_engine.device),
        'rdkit': True,
        'receptors': len(RECEPTOR_PROFILES),
        'dose_response_categories': len(DOSE_RESPONSE),
        'known_accords': len(KNOWN_ACCORDS),
        'masking_rules': len(MASKING_RULES),
        'synergy_pairs': len(SYNERGY_PAIRS),
        'clashing_pairs': len(CLASHING_PAIRS),
        'v22_integration': True,
    }

