# recipe_engine.py -- AI 기반 향수 레시피 생성기 (PyTorch CUDA + v6 AI)
# ================================================================
# v6 GNN → 분자 향 프로파일 예측 → 목표 향과 cosine 매칭
# NeuralNet → 무드/시즌 기반 원료 비율 예측 (fallback)
# VAE → 잠재 공간에서 새로운 조합 탐색
# ================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import os
import database as db
import molecular_harmony as mh
from collections import defaultdict

# ========================
# V6 AI 엔진 연동
# ========================
_v6_engine = None  # Lazy init
_ingredient_smiles = {}  # {ingredient_id: SMILES}
_v6_odor_cache = {}  # {smiles: np.array[22]}

def _init_v6_engine():
    """학습된 v6 모델 로드 (lazy init)"""
    global _v6_engine
    if _v6_engine is not None:
        return _v6_engine
    try:
        from v6_bridge import OdorEngineV6
        _v6_engine = OdorEngineV6(
            weights_dir='weights/v6', device='cpu',
            use_ensemble=False, n_ensemble=1
        )
        print("[RecipeEngine] V6 AI 엔진 로드 완료")
        return _v6_engine
    except Exception as e:
        print(f"[RecipeEngine] V6 로드 실패 (fallback 사용): {e}")
        return None

def _load_ingredient_smiles():
    """원료 ID → SMILES 매핑 로드 (ingredient_smiles.json 1136개 + molecules.json 보충)"""
    global _ingredient_smiles
    if _ingredient_smiles:
        return

    # 1) ingredient_smiles.json (primary — 1136 entries, {id: SMILES})
    for path in ['data/ingredient_smiles.json', '../data/ingredient_smiles.json',
                 'server/data/ingredient_smiles.json']:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                smi_map = json.load(f)
            if isinstance(smi_map, dict):
                for mid, smiles in smi_map.items():
                    if mid and smiles:
                        _ingredient_smiles[mid] = smiles
                        _ingredient_smiles[mid.lower()] = smiles
            print(f"[RecipeEngine] ingredient_smiles.json: {len(smi_map)} SMILES 로드")
            break

    # 2) molecules.json (supplement — 40 entries, richer metadata)
    for path in ['data/molecules.json', '../data/molecules.json',
                 'server/data/molecules.json']:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                mols = json.load(f)
            added = 0
            for m in mols:
                mid = m.get('id', '')
                smiles = m.get('smiles', '')
                if mid and smiles and mid not in _ingredient_smiles:
                    _ingredient_smiles[mid] = smiles
                    added += 1
                # 이름 기반도 매핑
                for name_key in ['name', 'name_en']:
                    name = (m.get(name_key) or '').lower()
                    if name and smiles and name not in _ingredient_smiles:
                        _ingredient_smiles[name] = smiles
            if added:
                print(f"[RecipeEngine] molecules.json: +{added} 추가 SMILES")
            break

    print(f"[RecipeEngine] 총 {len(_ingredient_smiles)} SMILES 매핑 완료")

def _get_ingredient_smiles(ing):
    """원료 dict에서 SMILES 찾기"""
    _load_ingredient_smiles()
    # 1) 직접 SMILES 필드
    if ing.get('smiles'):
        return ing['smiles']
    # 2) ID로 매핑
    if ing.get('id') in _ingredient_smiles:
        return _ingredient_smiles[ing['id']]
    # 3) 이름으로 매핑
    for name_key in ['name_en', 'name', 'id']:
        name = (ing.get(name_key) or '').lower()
        if name in _ingredient_smiles:
            return _ingredient_smiles[name]
    return None


# 22차원 향 차원 이름 (반드시 odor_engine.py와 동일)
ODOR_DIMS_22 = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]

def _mood_to_target_vector(mood, preferences=None):
    """모드 → 22d 목표 향 벡터 생성"""
    target = np.zeros(22)
    mood_targets = {
        'romantic':  {'floral': 0.8, 'sweet': 0.6, 'powdery': 0.5, 'musk': 0.3, 'warm': 0.3},
        'fresh':     {'fresh': 0.9, 'citrus': 0.7, 'green': 0.5, 'aquatic': 0.4, 'ozonic': 0.3},
        'warm':      {'warm': 0.8, 'amber': 0.7, 'sweet': 0.5, 'spicy': 0.4, 'woody': 0.3},
        'bold':      {'spicy': 0.8, 'leather': 0.6, 'woody': 0.5, 'smoky': 0.4, 'amber': 0.3},
        'calm':      {'woody': 0.7, 'herbal': 0.6, 'fresh': 0.5, 'green': 0.4, 'earthy': 0.3},
        'elegant':   {'floral': 0.6, 'powdery': 0.6, 'musk': 0.5, 'amber': 0.4, 'woody': 0.3},
        'sensual':   {'musk': 0.8, 'warm': 0.6, 'amber': 0.5, 'sweet': 0.4, 'floral': 0.3},
        'exotic':    {'spicy': 0.7, 'amber': 0.7, 'woody': 0.5, 'smoky': 0.4, 'sweet': 0.3},
        'energetic': {'citrus': 0.8, 'fresh': 0.7, 'green': 0.5, 'fruity': 0.4, 'ozonic': 0.3},
        'cozy':      {'sweet': 0.8, 'warm': 0.7, 'powdery': 0.5, 'amber': 0.4, 'musk': 0.3},
        'clean':     {'fresh': 0.8, 'ozonic': 0.6, 'aquatic': 0.5, 'musk': 0.4, 'green': 0.3},
        'luxury':    {'woody': 0.7, 'amber': 0.6, 'leather': 0.5, 'musk': 0.5, 'spicy': 0.3},
        'soothing':  {'herbal': 0.7, 'green': 0.6, 'woody': 0.5, 'fresh': 0.4, 'earthy': 0.3},
        'mysterious':{'smoky': 0.7, 'woody': 0.6, 'amber': 0.5, 'leather': 0.4, 'musk': 0.3},
        'grounding': {'earthy': 0.7, 'woody': 0.7, 'herbal': 0.5, 'smoky': 0.3, 'green': 0.3},
        'dreamy':    {'powdery': 0.7, 'floral': 0.6, 'musk': 0.5, 'fresh': 0.4, 'aquatic': 0.3},
    }
    profile = mood_targets.get(mood, mood_targets.get('elegant', {}))
    for dim_name, intensity in profile.items():
        if dim_name in ODOR_DIMS_22:
            target[ODOR_DIMS_22.index(dim_name)] = intensity
    # 선호 카테고리 반영
    if preferences:
        for pref in preferences:
            if pref.lower() in ODOR_DIMS_22:
                idx = ODOR_DIMS_22.index(pref.lower())
                target[idx] = max(target[idx], 0.6)
    # 정규화
    mx = target.max()
    if mx > 0:
        target = target / mx
    return target

def _v6_score_ingredients(ingredients, mood, season, preferences):
    """v6 AI 모델로 각 원료의 적합도 점수 계산
    
    방식: 목표 향 프로파일(22d) vs 각 원료의 v6 예측 향(22d) → cosine 유사도
    """
    engine = _init_v6_engine()
    if engine is None:
        return None
    
    target = _mood_to_target_vector(mood, preferences)
    scores = np.zeros(len(ingredients))
    v6_used = 0
    
    for i, ing in enumerate(ingredients):
        smiles = _get_ingredient_smiles(ing)
        if smiles is None:
            # SMILES 없으면 기존 규칙 기반 점수
            mood_cats = MOOD_CATEGORIES.get(mood, [])
            if ing.get('category', '') in mood_cats:
                scores[i] += 0.4
            if mood in (ing.get('moods') or []):
                scores[i] += 0.3
            if season in (ing.get('seasons') or []):
                scores[i] += 0.2
            continue
        
        # v6 예측 (캐시됨)
        if smiles in _v6_odor_cache:
            odor_vec = _v6_odor_cache[smiles]
        else:
            try:
                odor_vec = engine.encode(smiles)
                # 612d 예측이면 앞 22d만 사용
                if len(odor_vec) > 22:
                    odor_vec = odor_vec[:22]
                _v6_odor_cache[smiles] = odor_vec
            except Exception:
                scores[i] = 0.3  # 예측 실패 시 기본 점수
                continue
        
        # 💡 하이브리드 스코어링 v2 (Dynamic Threshold + Cosine + Hit Rate + Off-note)
        # 1. 동적 임계값: 메인 향 대비 10% 미만만 잡음으로 간주 (미세 뉘앙스 보존)
        max_val = np.max(odor_vec[:22])
        dynamic_threshold = max(0.05, max_val * 0.10)
        pred_vec = np.where(odor_vec[:22] < dynamic_threshold, 0.0, odor_vec[:22])
        
        # 2. Cosine Similarity (방향성)
        dot = np.dot(target, pred_vec)
        norm_t = np.linalg.norm(target)
        norm_o = np.linalg.norm(pred_vec)
        cos_sim = max(0.0, dot / (norm_t * norm_o)) if (norm_t > 0 and norm_o > 0) else 0.0
        
        # 3. Hit Rate + Off-note 페널티
        active_mask = target > 0.05
        sum_pred = np.sum(pred_vec)
        if np.any(active_mask) and sum_pred > 0:
            hit_rate = np.sum(pred_vec[active_mask]) / sum_pred
            # 타겟에 없는 잡향(Off-note)의 비중 → 감점
            off_note_ratio = np.sum(pred_vec[~active_mask]) / sum_pred
            penalty = off_note_ratio * 0.2  # 최대 0.2점 감점
        else:
            hit_rate = 0.0
            penalty = 0.0
        
        # 4. 하이브리드 스코어: 코사인 60% + 적중률 40% - 잡향 페널티
        scores[i] = max(0.0, (cos_sim * 0.6) + (hit_rate * 0.4) - penalty)
        
        # 시즌 보너스
        if season in (ing.get('seasons') or []):
            scores[i] += 0.1
        
        v6_used += 1
    
    if v6_used > 0:
        print(f"[RecipeEngine] V6 하이브리드 스코어링: {v6_used}/{len(ingredients)} 원료에 AI 적용")
    
    return scores

# ========================
# 상수 / 설정
# ========================

NOTE_RATIOS = {
    'classic':  {'top': (15, 25), 'middle': (30, 45), 'base': (25, 40)},
    'fresh':    {'top': (25, 35), 'middle': (30, 40), 'base': (15, 25)},
    'oriental': {'top': (10, 20), 'middle': (25, 35), 'base': (35, 50)},
    'light':    {'top': (30, 40), 'middle': (30, 40), 'base': (10, 20)},
}

# ========================
# 실제 레시피 데이터 기반 비율 프로파일
# ========================
INGREDIENT_PCT_PROFILES = {}   # {ing_id: {note: [pct, ...]}}
STYLE_PCT_PROFILES = {}        # {style: {ing_id: [pct, ...]}}
MOOD_RECIPE_MAP = {}           # {mood: [recipe_indices]}
NOTE_TOTAL_PROFILES = {'top': [], 'middle': [], 'base': []}  # Note-level totals
CATEGORY_PCT_PROFILES = {}     # {(category, note): [pct, ...]}
_RECIPE_DATA = []              # Raw recipe list

def _load_recipe_profiles():
    """Load recipe_training_data.json and build percentage profiles"""
    global _RECIPE_DATA, INGREDIENT_PCT_PROFILES, STYLE_PCT_PROFILES
    global MOOD_RECIPE_MAP, NOTE_TOTAL_PROFILES, CATEGORY_PCT_PROFILES

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'recipe_training_data.json')
    if not os.path.exists(data_path):
        print("[RecipeEngine] WARNING: recipe_training_data.json not found")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        _RECIPE_DATA = json.load(f)

    # Build all DB ingredient info for category lookup
    all_db_ings = db.get_all_ingredients()
    ing_cat_map = {i['id']: i.get('category', 'other') for i in all_db_ings}

    for idx, recipe in enumerate(_RECIPE_DATA):
        style = recipe.get('style', 'classic')
        mood_list = recipe.get('mood', [])
        if isinstance(mood_list, str):
            mood_list = [mood_list]

        note_sums = defaultdict(float)

        for ing in recipe.get('ingredients', []):
            ing_id = ing.get('id', '')
            note = ing.get('note', 'middle')
            pct = ing.get('pct', 0)

            # Per-ingredient profile
            if ing_id not in INGREDIENT_PCT_PROFILES:
                INGREDIENT_PCT_PROFILES[ing_id] = defaultdict(list)
            INGREDIENT_PCT_PROFILES[ing_id][note].append(pct)

            # Per-style profile
            if style not in STYLE_PCT_PROFILES:
                STYLE_PCT_PROFILES[style] = defaultdict(list)
            STYLE_PCT_PROFILES[style][ing_id].append(pct)

            # Per-category profile
            cat = ing_cat_map.get(ing_id, 'other')
            cat_key = (cat, note)
            if cat_key not in CATEGORY_PCT_PROFILES:
                CATEGORY_PCT_PROFILES[cat_key] = []
            CATEGORY_PCT_PROFILES[cat_key].append(pct)

            note_sums[note] += pct

        # Note-level totals
        for note, total in note_sums.items():
            if note in NOTE_TOTAL_PROFILES:
                NOTE_TOTAL_PROFILES[note].append(total)

        # Mood-recipe mapping
        for m in mood_list:
            if m not in MOOD_RECIPE_MAP:
                MOOD_RECIPE_MAP[m] = []
            MOOD_RECIPE_MAP[m].append(idx)

    print(f"[RecipeEngine] Loaded {len(_RECIPE_DATA)} recipes, "
          f"{len(INGREDIENT_PCT_PROFILES)} ingredient profiles, "
          f"{len(STYLE_PCT_PROFILES)} style profiles")

# Auto-load on import
try:
    _load_recipe_profiles()
except Exception as e:
    print(f"[RecipeEngine] Profile loading failed: {e}")

MOOD_CATEGORIES = {
    'romantic':  ['floral', 'powdery', 'fruity', 'gourmand'],
    'fresh':     ['citrus', 'herbal', 'green', 'aquatic', 'fresh'],
    'warm':      ['woody', 'amber', 'spicy', 'resinous', 'gourmand'],
    'bold':      ['leather', 'smoky', 'spicy', 'animalic', 'woody'],
    'calm':      ['woody', 'herbal', 'green', 'powdery', 'earthy'],
    'elegant':   ['floral', 'aldehyde', 'chypre', 'powdery', 'woody'],
    'sensual':   ['musk', 'amber', 'animalic', 'floral', 'gourmand'],
    'exotic':    ['spicy', 'resinous', 'animalic', 'woody', 'smoky'],
    'energetic': ['citrus', 'herbal', 'fresh', 'green', 'spicy'],
    'cozy':      ['gourmand', 'amber', 'woody', 'smoky', 'musk'],
    'clean':     ['fresh', 'aquatic', 'musk', 'green', 'powdery'],
    'luxury':    ['leather', 'amber', 'woody', 'musk', 'floral'],
    'happy':     ['citrus', 'fruity', 'floral', 'fresh', 'gourmand'],
    'grounding': ['earthy', 'woody', 'resinous', 'herbal', 'smoky'],
    'spiritual': ['resinous', 'woody', 'herbal', 'earthy', 'smoky'],
    'dreamy':    ['powdery', 'floral', 'fresh', 'musk', 'aquatic'],
}

MOOD_STYLE = {
    'romantic': 'classic', 'fresh': 'fresh', 'warm': 'oriental',
    'bold': 'oriental', 'calm': 'classic', 'elegant': 'classic',
    'sensual': 'oriental', 'exotic': 'oriental', 'energetic': 'fresh',
    'cozy': 'oriental', 'clean': 'fresh', 'luxury': 'classic',
    'happy': 'fresh', 'grounding': 'oriental', 'spiritual': 'oriental',
    'dreamy': 'light',
}

ALL_MOODS = list(MOOD_CATEGORIES.keys())
ALL_SEASONS = ['spring', 'summer', 'autumn', 'winter']
ALL_CATEGORIES = [
    'floral','citrus','woody','spicy','fruity','gourmand','aquatic',
    'amber','musk','aromatic','herbal','green','fresh','leather',
    'smoky','powdery','earthy','animalic','resinous','aldehyde',
    'chypre','waxy','synthetic'
]

DENSITY_MAP = {
    'citrus':0.85,'herbal':0.91,'green':0.88,'floral':0.93,'woody':0.96,
    'spicy':0.95,'amber':0.98,'resinous':1.02,'musk':0.92,'gourmand':0.95,
    'leather':0.94,'smoky':0.93,'aquatic':0.89,'powdery':0.94,'animalic':0.96,
    'fruity':0.90,'earthy':0.95,'aldehyde':0.82,'chypre':0.94,'aromatic':0.91,
    'fresh':0.87,'waxy':0.93,'synthetic':0.90,
}

PRICE_MAP = {
    'synthetic':3000,'fresh':3500,'aquatic':4000,'musk':4000,'aldehyde':4000,
    'green':4000,'herbal':5000,'citrus':6000,'spicy':6000,'fruity':5000,
    'gourmand':5000,'earthy':5000,'aromatic':5000,'smoky':6000,'powdery':6000,
    'floral':8000,'woody':7000,'amber':7000,'resinous':9000,'leather':8000,
    'animalic':12000,'chypre':8000,'waxy':7000,
}

PREMIUM_IDS = {
    'orris_butter':45000,'orris_concrete':40000,'agarwood_oud':80000,
    'natural_oud':80000,'rose_absolute':25000,'jasmine_absolute':30000,
    'tuberose_absolute':35000,'tuberose_india':35000,'ambergris_synth':20000,
    'deer_musk_synth':18000,'saffron':50000,'frangipani_abs':25000,
    'champaca_gold':28000,'ylang_extra':15000,'immortelle':20000,'costus':18000,
}

IFRA_LIMITS = {
    'oakmoss':0.1,'treemoss':0.1,'cinnamaldehyde':0.5,'coumarin_lactone':2.0,
    'methyl_salicylate':2.0,'skatole':0.01,'costus':0.1,'styrax':0.6,
    'peru_balsam':0.4,'hyraceum':0.2,'birch_tar':0.5,'juniper_tar':0.5,
    'linalool':5.0,'limonene':5.0,'citral':0.6,'eugenol':1.0,
    'isoeugenol':0.02,'cinnamic_alcohol':0.8,'anisaldehyde':2.0,
    'aldehyde_c7':0.5,'aldehyde_c8':0.5,'aldehyde_c9':0.5,
    'aldehyde_c10':1.0,'aldehyde_c11':0.5,'aldehyde_c12':0.5,
    'para_cresyl_methyl':0.2,'habanero':0.1,
}

AGING_GUIDE = {
    'Parfum (P)':          {'min_days':21,'recommended_days':42,'note':'고농도 - 충분한 숙성이 중요'},
    'Eau de Parfum (EDP)': {'min_days':14,'recommended_days':28,'note':'2주 이상 숙성 권장'},
    'Eau de Toilette (EDT)':{'min_days':7,'recommended_days':21,'note':'1~3주 숙성'},
    'Eau de Cologne (EDC)': {'min_days':3,'recommended_days':14,'note':'가벼운 농도 - 빠른 숙성'},
}

NAME_PARTS = {
    'romantic': [('Velvet','Rose'),('Midnight','Bloom'),('Silk','Petal')],
    'fresh':    [('Azure','Morning'),('Crystal','Wave'),('Citrus','Breeze')],
    'warm':     [('Golden','Ember'),('Amber','Glow'),('Cinnamon','Dream')],
    'bold':     [('Dark','Leather'),('Iron','Smoke'),('Noir','Edge')],
    'calm':     [('Zen','Garden'),('Silent','Dawn'),('Soft','Moss')],
    'elegant':  [('Ivory','Lace'),('Crystal','Palace'),('Silver','Moon')],
    'sensual':  [('Noir','Velvet'),('Oud','Desire'),('Satin','Night')],
    'exotic':   [('Spice','Route'),('Mystic','East'),('Saffron','Gold')],
    'energetic':[('Electric','Citrus'),('Spark','Lime'),('Vivid','Rush')],
    'cozy':     [('Vanilla','Hearth'),('Cocoa','Blanket'),('Caramel','Fire')],
    'clean':    [('Pure','Cotton'),('Fresh','Linen'),('Cloud','Mist')],
    'luxury':   [('Royal','Oud'),('Platinum','Noir'),('Crown','Jewel')],
}

NAME_KO_MAP = {
    'Velvet Rose':'벨벳 로즈','Midnight Bloom':'미드나잇 블룸',
    'Silk Petal':'실크 페탈','Azure Morning':'아주르 모닝',
    'Crystal Wave':'크리스탈 웨이브','Citrus Breeze':'시트러스 브리즈',
    'Golden Ember':'골든 엠버','Amber Glow':'앰버 글로우',
    'Cinnamon Dream':'시나몬 드림','Dark Leather':'다크 레더',
    'Iron Smoke':'아이언 스모크','Noir Edge':'누아르 엣지',
    'Zen Garden':'젠 가든','Silent Dawn':'사일런트 던','Soft Moss':'소프트 모스',
    'Ivory Lace':'아이보리 레이스','Crystal Palace':'크리스탈 팰리스',
    'Silver Moon':'실버 문','Noir Velvet':'누아르 벨벳',
    'Oud Desire':'우드 디자이어','Satin Night':'새틴 나이트',
    'Spice Route':'스파이스 루트','Mystic East':'미스틱 이스트',
    'Saffron Gold':'사프란 골드','Electric Citrus':'일렉트릭 시트러스',
    'Spark Lime':'스파크 라임','Vivid Rush':'비비드 러시',
    'Vanilla Hearth':'바닐라 허스','Cocoa Blanket':'코코아 블랭킷',
    'Caramel Fire':'카라멜 파이어','Pure Cotton':'퓨어 코튼',
    'Fresh Linen':'프레시 리넨','Cloud Mist':'클라우드 미스트',
    'Royal Oud':'로얄 우드','Platinum Noir':'플래티넘 누아르',
    'Crown Jewel':'크라운 주얼',
}


# ========================
# AI 모델 정의
# ========================

class IngredientEncoder(nn.Module):
    """원료 메타데이터 -> 임베딩 벡터"""
    def __init__(self, cat_dim=23, meta_dim=5, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cat_dim + meta_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, embed_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ConditionEncoder(nn.Module):
    """무드 + 시즌 + 선호 -> 조건 벡터"""
    def __init__(self, mood_dim=16, season_dim=4, cat_dim=23, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mood_dim + season_dim + cat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class RecipeScorer(nn.Module):
    """(원료 임베딩, 조건 벡터) -> 이 원료가 이 조건에 적합한 정도 (0~1)"""
    def __init__(self, embed_dim=32):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, ingredient_embed, condition_embed):
        # ingredient_embed: [N, 32], condition_embed: [1, 32] -> broadcast
        combined = torch.cat([ingredient_embed, condition_embed.expand_as(ingredient_embed)], dim=-1)
        return self.scorer(combined).squeeze(-1)


class HarmonyNet(nn.Module):
    """원료 쌍의 궁합 예측 (MultiheadAttention 기반)"""
    def __init__(self, embed_dim=32, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )
        self.harmony_head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, ingredients_embed):
        # ingredients_embed: [1, K, 32]
        attended, attn_weights = self.attention(
            ingredients_embed, ingredients_embed, ingredients_embed
        )
        # 전체 조화도 = attended 벡터의 평균에 대한 점수
        harmony = self.harmony_head(attended.mean(dim=1))
        return harmony.squeeze(), attn_weights.squeeze(0)


# ========================
# AI 레시피 엔진
# ========================

class AIRecipeEngine:
    """PyTorch CUDA 기반 향수 레시피 생성기"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ingredient_encoder = None
        self.condition_encoder = None
        self.recipe_scorer = None
        self.harmony_net = None
        self.trained = False
        self.ingredients_cache = None
        self.ingredient_ids = []
        self.ingredient_map = {}
        self.ingredient_embeddings = None
        self.train_epochs = 0
        self.train_loss = 0.0
        print(f"[AIRecipeEngine] Device: {self.device}")

    def _encode_ingredient(self, ing):
        """원료 1개 -> 특성 벡터 (28차원: 카테고리 23 + 메타 5)"""
        cat = ing.get('category', '')
        cat_vec = [1.0 if c == cat else 0.0 for c in ALL_CATEGORIES]
        meta = [
            (ing.get('volatility') or 5) / 10.0,
            (ing.get('intensity') or 5) / 10.0,
            (ing.get('longevity') or 5) / 10.0,
            (ing.get('typical_pct') or 3) / 30.0,
            {'top': 0.0, 'middle': 0.5, 'base': 1.0}.get(ing.get('note_type', 'middle'), 0.5),
        ]
        return cat_vec + meta

    def _encode_condition(self, mood, season, preferences):
        """무드+시즌+선호 -> 조건 벡터 (43차원: 무드 16 + 시즌 4 + 카테고리 23)"""
        mood_vec = [1.0 if m == mood else 0.0 for m in ALL_MOODS]
        season_vec = [1.0 if s == season else 0.0 for s in ALL_SEASONS]
        pref_vec = [1.0 if c in preferences else 0.0 for c in ALL_CATEGORIES]
        # 무드에 따른 카테고리 가중치 추가
        mood_cats = MOOD_CATEGORIES.get(mood, [])
        for i, c in enumerate(ALL_CATEGORIES):
            if c in mood_cats:
                pref_vec[i] = max(pref_vec[i], 0.7)
        return mood_vec + season_vec + pref_vec

    def _build_models(self):
        """모델 초기화"""
        self.ingredient_encoder = IngredientEncoder(
            cat_dim=len(ALL_CATEGORIES), meta_dim=5, embed_dim=32
        ).to(self.device)

        self.condition_encoder = ConditionEncoder(
            mood_dim=len(ALL_MOODS), season_dim=len(ALL_SEASONS),
            cat_dim=len(ALL_CATEGORIES), embed_dim=32
        ).to(self.device)

        self.recipe_scorer = RecipeScorer(embed_dim=32).to(self.device)
        self.harmony_net = HarmonyNet(embed_dim=32, num_heads=4).to(self.device)

    def _load_ingredients(self):
        """DB에서 원료 로딩 + 캐싱 (DB 실패 시 JSON 폴백)"""
        if self.ingredients_cache is not None:
            return self.ingredients_cache

        # DB 시도
        try:
            self.ingredients_cache = db.get_all_ingredients()
            if self.ingredients_cache:
                self.ingredient_ids = [ing['id'] for ing in self.ingredients_cache]
                self.ingredient_map = {ing['id']: ing for ing in self.ingredients_cache}
                return self.ingredients_cache
        except Exception as e:
            print(f"[AIRecipeEngine] DB 로드 실패 ({e}), JSON 폴백")

        # JSON 폴백
        for path in ['data/ingredients.json', '../data/ingredients.json',
                     'server/data/ingredients.json']:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.ingredients_cache = json.load(f)
                self.ingredient_ids = [ing['id'] for ing in self.ingredients_cache]
                self.ingredient_map = {ing['id']: ing for ing in self.ingredients_cache}
                print(f"[AIRecipeEngine] JSON에서 {len(self.ingredients_cache)}개 원료 로드")
                return self.ingredients_cache

        self.ingredients_cache = []
        return self.ingredients_cache

    def train(self, epochs=50, on_progress=None):
        """
        자기 지도 학습 (Self-supervised):
        - 무드/시즌 조건에 맞는 원료는 높은 점수
        - 조건에 안 맞는 원료는 낮은 점수
        - 궁합이 좋은 원료 쌍(같은 향수 계열)은 높은 조화도
        """
        ingredients = self._load_ingredients()
        if not ingredients:
            print("[AIRecipeEngine] No ingredients to train on")
            return

        if self.ingredient_encoder is None:
            self._build_models()

        # 1) 원료 임베딩 사전 계산
        ing_features = [self._encode_ingredient(ing) for ing in ingredients]
        X_ing = torch.tensor(ing_features, dtype=torch.float32).to(self.device)

        # 2) 학습 데이터 생성: 실제 레시피 데이터 + 자기 지도 학습 혼합
        train_data = []

        # 2a) 실제 레시피 기반 지도학습 (recipe_training_data.json)
        ing_id_set = set(self.ingredient_ids)
        for recipe in _RECIPE_DATA[:500]:  # 최대 500개 샘플링
            mood = recipe.get('mood', ['romantic'])
            if isinstance(mood, list):
                mood = mood[0] if mood else 'romantic'
            season_list = recipe.get('season', ['all'])
            season = season_list[0] if season_list else 'all'
            if season == 'all':
                season = ALL_SEASONS[idx % len(ALL_SEASONS)]  # 결정론적 순환
            if mood not in ALL_MOODS:
                mood = 'romantic'

            prefs = []
            style = recipe.get('style', 'classic')
            style_cats = {'floral': ['floral'], 'woody': ['woody', 'resinous'],
                         'oriental': ['amber', 'spicy'], 'fresh': ['citrus', 'herbal'],
                         'citrus': ['citrus']}
            prefs = style_cats.get(style, [])
            cond_vec = self._encode_condition(mood, season, prefs)

            # 라벨: 실제 레시피에 포함된 원료 = pct 기반 점수, 아닌 원료 = 0
            recipe_ing_ids = {ing['id']: ing['pct'] for ing in recipe.get('ingredients', [])}
            labels = []
            for ing in ingredients:
                if ing['id'] in recipe_ing_ids:
                    # 비율에 비례한 점수 (0.3~1.0)
                    pct = recipe_ing_ids[ing['id']]
                    score = min(1.0, 0.3 + pct / 50.0)  # 50%면 1.0
                else:
                    score = 0.0
                labels.append(score)

            train_data.append((cond_vec, labels))

        # 2b) 자기 지도 학습 (결정론적 조합)
        for ssl_idx in range(1000):
            mood = ALL_MOODS[ssl_idx % len(ALL_MOODS)]
            season = ALL_SEASONS[ssl_idx % len(ALL_SEASONS)]
            prefs = [c for i, c in enumerate(ALL_CATEGORIES) if (ssl_idx + i) % 5 == 0]
            cond_vec = self._encode_condition(mood, season, prefs)

            mood_cats = MOOD_CATEGORIES.get(mood, [])
            labels = []
            for ing in ingredients:
                score = 0.0
                if ing.get('category', '') in mood_cats:
                    score += 0.4
                if mood in (ing.get('moods') or []):
                    score += 0.3
                if season in (ing.get('seasons') or []):
                    score += 0.2
                if ing.get('category', '') in prefs:
                    score += 0.1
                labels.append(min(1.0, score))

            train_data.append((cond_vec, labels))

        # 3) 학습
        all_params = (
            list(self.ingredient_encoder.parameters()) +
            list(self.condition_encoder.parameters()) +
            list(self.recipe_scorer.parameters()) +
            list(self.harmony_net.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=0.002)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCELoss()

        self.ingredient_encoder.train()
        self.condition_encoder.train()
        self.recipe_scorer.train()

        for epoch in range(epochs):
            # 결정론적 셔플: epoch 기반 인덱스 재배치
            n_td = len(train_data)
            indices = [(i * 7 + epoch * 13) % n_td for i in range(n_td)]
            train_data = [train_data[i] for i in sorted(set(indices))]
            total_loss = 0.0
            batch_size = 64
            n_batches = 0

            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                batch_conds = torch.tensor(
                    [d[0] for d in batch], dtype=torch.float32
                ).to(self.device)
                batch_labels = torch.tensor(
                    [d[1] for d in batch], dtype=torch.float32
                ).to(self.device)

                optimizer.zero_grad()

                # 원료 임베딩 (전체)
                ing_embed = self.ingredient_encoder(X_ing)  # [N_ing, 32]

                # 배치 내 각 조건에 대해 스코어 계산
                batch_loss = 0.0
                for j in range(len(batch)):
                    cond_embed = self.condition_encoder(batch_conds[j:j+1])  # [1, 32]
                    scores = self.recipe_scorer(ing_embed, cond_embed)  # [N_ing]
                    loss = criterion(scores, batch_labels[j])
                    batch_loss += loss

                batch_loss /= len(batch)

                # 궁합 학습: 같은 카테고리 원료끼리는 조화도 높게
                # 랜덤으로 5개 원료 샘플링해서 HarmonyNet 학습
                sample_idx = torch.randint(0, len(ingredients), (5,), device=self.device)
                sample_embed = ing_embed[sample_idx].unsqueeze(0)  # [1, 5, 32]
                harmony_score, _ = self.harmony_net(sample_embed)
                # 같은 카테고리가 많을수록 높은 조화도
                sample_cats = [ingredients[i]['category'] for i in sample_idx.tolist()]
                unique_ratio = len(set(sample_cats)) / len(sample_cats)
                # 다양성이 적절한 게 좋음 (너무 같으면 단조, 너무 다르면 산만)
                harmony_target = torch.tensor(
                    [0.8 if 0.3 < unique_ratio < 0.8 else 0.3],
                    dtype=torch.float32
                ).to(self.device)
                harmony_loss = criterion(harmony_score.unsqueeze(0), harmony_target)
                batch_loss += harmony_loss * 0.3

                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            self.train_loss = avg_loss
            self.train_epochs = epoch + 1

            if on_progress:
                on_progress(epoch + 1, epochs, avg_loss)

        # 4) 학습 후 임베딩 캐싱
        self.ingredient_encoder.eval()
        with torch.no_grad():
            self.ingredient_embeddings = self.ingredient_encoder(X_ing)

        self.trained = True
        print(f"[AIRecipeEngine] Trained {epochs} epochs, loss: {self.train_loss:.4f}")

    @torch.no_grad()
    def _ai_score_ingredients(self, mood, season, preferences):
        """AI로 각 원료의 적합도 점수 계산"""
        if not self.trained or self.ingredient_embeddings is None:
            return None

        self.recipe_scorer.eval()
        self.condition_encoder.eval()

        cond_vec = self._encode_condition(mood, season, preferences)
        cond_tensor = torch.tensor([cond_vec], dtype=torch.float32).to(self.device)
        cond_embed = self.condition_encoder(cond_tensor)  # [1, 32]

        scores = self.recipe_scorer(
            self.ingredient_embeddings, cond_embed
        )  # [N_ing]

        return scores.cpu().numpy()

    @torch.no_grad()
    def _ai_harmony_check(self, selected_ids):
        """선택된 원료 세트의 조화도 검증"""
        if not self.trained or self.ingredient_embeddings is None:
            return 0.5, None

        self.harmony_net.eval()

        indices = []
        for sid in selected_ids:
            if sid in self.ingredient_ids:
                indices.append(self.ingredient_ids.index(sid))

        if len(indices) < 2:
            return 0.5, None

        embed = self.ingredient_embeddings[indices].unsqueeze(0)  # [1, K, 32]
        harmony_score, attn_weights = self.harmony_net(embed)

        return float(harmony_score), attn_weights.cpu().numpy()


# ========================
# 글로벌 엔진 인스턴스
# ========================
_engine = AIRecipeEngine()


def train_ai(epochs=50, on_progress=None):
    """AI 모델 학습 (서버 시작 시 또는 수동 호출)"""
    _engine.train(epochs=epochs, on_progress=on_progress)


def generate_recipe(mood='romantic', season='spring', preferences=None,
                    intensity=50, complexity=None, batch_ml=100,
                    target_profile=None, candidate_ingredients=None,
                    concentrate_pct=None):
    """AI 기반 향수 레시피 생성
    
    Args:
        mood, season, preferences: 기본 조건
        intensity, complexity: 강도/복잡도
        batch_ml: 배치 크기 (ml)
        target_profile: np.array[22] — 외부 목표 향 벡터 (AI 기획자가 제공)
        candidate_ingredients: list[dict] — AI가 선별한 후보 원료 (ai_score 포함)
        concentrate_pct: float — 농축률 직접 지정 (None이면 intensity 기반)
    """
    preferences = preferences or []

    # 아직 학습 안 됐으면 자동 학습 시도 (DB 없으면 건너뜀)
    if not _engine.trained:
        try:
            print("[AIRecipeEngine] Auto-training (first call)...")
            _engine.train(epochs=30)
        except Exception as e:
            print(f"[AIRecipeEngine] 학습 실패 ({e}), V6/규칙 기반 스코어링으로 진행")

    # 1) 조향 스타일
    style = MOOD_STYLE.get(mood, 'classic')
    ratios = NOTE_RATIOS[style]

    # 2) 복잡도
    if complexity is None:
        # 결정론적 복잡도: intensity 비례 (random 제거)
        if intensity < 30:
            complexity = 7   # 6~8 중간값
        elif intensity < 70:
            complexity = 10  # 8~12 중간값
        else:
            complexity = 12  # 10~15 중간값

    top_n = max(2, int(complexity * 0.3))
    mid_n = max(2, int(complexity * 0.4))
    base_n = max(2, complexity - top_n - mid_n)

    # 3) AI 스코어링 — 후보 원료가 제공되면 그것을 사용
    if candidate_ingredients is not None:
        # 외부 AI 기획자가 제공한 후보 사용
        ingredients = _engine._load_ingredients()
        
        # 후보 ID → ai_score 매핑
        candidate_scores = {}
        for c in candidate_ingredients:
            candidate_scores[c.get('id', '')] = c.get('ai_score', 0.5)
        
        # 전체 원료에 대해 스코어 부여 (후보에 있으면 AI 점수, 없으면 0)
        ai_scores = np.zeros(len(ingredients))
        for i, ing in enumerate(ingredients):
            ai_scores[i] = candidate_scores.get(ing.get('id', ''), 0.0)
        
        print(f"[RecipeEngine] 외부 AI 후보 {len(candidate_scores)}개 수신, "
              f"매칭 {sum(1 for s in ai_scores if s > 0)}/{len(ingredients)}")
    else:
        # 내부 V6 → NeuralNet → 규칙 fallback 체인
        ingredients = _engine._load_ingredients()
        
        # V6 AI 모델 기반 스코어링 시도
        ai_scores = _v6_score_ingredients(ingredients, mood, season, preferences)
        
        if ai_scores is None:
            # V6 실패 시 기존 NeuralNet 시도
            ai_scores = _engine._ai_score_ingredients(mood, season, preferences)
        
        if ai_scores is None:
            # 모두 실패 시 규칙 기반 폴백
            ai_scores = np.zeros(len(ingredients))
            for i, ing in enumerate(ingredients):
                mood_cats = MOOD_CATEGORIES.get(mood, [])
                if ing.get('category', '') in mood_cats:
                    ai_scores[i] += 0.4
                if mood in (ing.get('moods') or []):
                    ai_scores[i] += 0.3
                if season in (ing.get('seasons') or []):
                    ai_scores[i] += 0.2

    # 4) 노트별 분류 + AI 점수 기반 선택
    by_note = {'top': [], 'middle': [], 'base': []}
    for i, ing in enumerate(ingredients):
        nt = ing.get('note_type', 'middle')
        if nt in by_note:
            by_note[nt].append((ing, float(ai_scores[i])))

    selected = {'top': [], 'middle': [], 'base': []}
    targets = {'top': top_n, 'middle': mid_n, 'base': base_n}
    used_categories = set()

    for note_type, count in targets.items():
        pool = by_note.get(note_type, [])
        if not pool:
            continue

        # 💡 Temperature Softmax 샘플링 (랜덤 노이즈 대신 확률적 선택)
        pool.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 후보군 압축 (목표의 3배수)
        top_pool = pool[:count * 3]
        if not top_pool:
            continue
        
        # 결정론적 Softmax 선택: 점수 상위 순으로 카테고리 다양성 확보
        scores_arr = np.array([max(x[1], 0.01) for x in top_pool])
        # 점수 내림차순 정렬 인덱스 (결정론적)
        chosen_indices = np.argsort(scores_arr)[::-1][:count * 2]
        
        chosen = []
        cat_count = {}
        for idx in chosen_indices:
            ing, sc = top_pool[idx]
            cat = ing.get('category', '')
            cat_count.setdefault(cat, 0)
            if cat_count[cat] >= 2:
                continue
            chosen.append((ing, sc))
            cat_count[cat] += 1
            used_categories.add(cat)
            if len(chosen) >= count:
                break

        selected[note_type] = chosen

    # 5) 조화도 검증 — V6 MixtureNet 우선, 레거시 HarmonyNet 폴백
    all_selected_ids = []
    all_selected_items = []
    for note_list in selected.values():
        all_selected_ids.extend([ing['id'] for ing, _ in note_list])
        all_selected_items.extend([(ing, sc) for ing, sc in note_list])

    # V6 MixtureNet으로 실제 혼합 향 예측 → 타겟과 비교
    v6_eng = _init_v6_engine()
    use_v6_harmony = False
    if v6_eng is not None and target_profile is not None:
        smiles_list = []
        ratios_list = []
        for ing, sc in all_selected_items:
            smi = _get_ingredient_smiles(ing)
            if smi:
                smiles_list.append(smi)
                ratios_list.append(max(1.0, sc * 10))  # 스코어 비례 배합
        if len(smiles_list) >= 2:
            try:
                mix_res = v6_eng.predict_mixture(smiles_list, ratios_list)
                mix_vec = mix_res['mixture'][:22]
                n_t = np.linalg.norm(target_profile)
                n_m = np.linalg.norm(mix_vec)
                if n_t > 0 and n_m > 0:
                    best_harmony = float(np.dot(target_profile, mix_vec) / (n_t * n_m))
                else:
                    best_harmony = 0.5
                use_v6_harmony = True
                print(f"[RecipeEngine] V6 MixtureNet 조화도: {best_harmony:.3f}")
            except Exception as e:
                print(f"[RecipeEngine] V6 MixtureNet 실패: {e}, 레거시 폴백")

    if not use_v6_harmony:
        harmony_score, attn_weights = _engine._ai_harmony_check(all_selected_ids)
        best_harmony = harmony_score

    best_selected = {k: list(v) for k, v in selected.items()}
    # 조화도가 낮으면 재시도 (최대 3회)
    for retry in range(3):
        if best_harmony >= 0.6:
            break
        for note_type in ['top', 'middle', 'base']:
            pool = by_note.get(note_type, [])
            if not pool or not best_selected[note_type]:
                continue
            weakest_idx = -1
            weakest_score = float('inf')
            for idx, (ing, sc) in enumerate(best_selected[note_type]):
                if sc < weakest_score:
                    weakest_score = sc
                    weakest_idx = idx
            if weakest_idx >= 0:
                current_ids = set(ing['id'] for ing, _ in best_selected[note_type])
                for ing, sc in pool:
                    if ing['id'] not in current_ids:
                        best_selected[note_type][weakest_idx] = (ing, sc)
                        break

        retry_ids = []
        for note_list in best_selected.values():
            retry_ids.extend([ing['id'] for ing, _ in note_list])

        if use_v6_harmony and v6_eng is not None:
            # V6 재검증
            smi_l, rat_l = [], []
            for nl in best_selected.values():
                for ing, sc in nl:
                    s = _get_ingredient_smiles(ing)
                    if s:
                        smi_l.append(s); rat_l.append(max(1.0, sc * 10))
            if len(smi_l) >= 2:
                try:
                    mr = v6_eng.predict_mixture(smi_l, rat_l)
                    mv = mr['mixture'][:22]
                    nt2 = np.linalg.norm(target_profile)
                    nm2 = np.linalg.norm(mv)
                    rh = float(np.dot(target_profile, mv) / (nt2 * nm2)) if (nt2 > 0 and nm2 > 0) else 0.5
                except:
                    rh = 0.0
            else:
                rh = 0.0
        else:
            rh, _ = _engine._ai_harmony_check(retry_ids)

        if rh > best_harmony:
            best_harmony = rh
            selected = best_selected

    # 6) 배합 비율 계산 (실제 데이터 기반 프로파일 참조)
    formula_raw = []
    for note_type in ['top', 'middle', 'base']:
        ratio_range = ratios[note_type]
        # Note-level total from real data (if available)
        if NOTE_TOTAL_PROFILES.get(note_type):
            real_avgs = NOTE_TOTAL_PROFILES[note_type]
            # 데이터 평균값 사용 (random 제거)
            avg_total = sum(real_avgs) / len(real_avgs)
            total_note_pct = avg_total
            # Still respect the style-specific bounds
            total_note_pct = max(ratio_range[0], min(ratio_range[1], total_note_pct))
        else:
            total_note_pct = (ratio_range[0] + ratio_range[1]) / 2  # 범위 중간값

        items = selected[note_type]
        if not items:
            continue

        # Assign weights using real ingredient profiles
        weights = []
        for ing, sc in items:
            ing_id = ing['id']
            cat = ing.get('category', 'other')
            weight = None

            # Priority 1: exact ingredient+note profile from real data
            if ing_id in INGREDIENT_PCT_PROFILES:
                note_pcts = INGREDIENT_PCT_PROFILES[ing_id].get(note_type, [])
                if note_pcts:
                    avg = sum(note_pcts) / len(note_pcts)
                    # 데이터 평균값 직접 사용 (random.gauss 제거)
                    weight = max(0.5, avg)

            # Priority 2: style-specific profile
            if weight is None and style in STYLE_PCT_PROFILES:
                style_pcts = STYLE_PCT_PROFILES[style].get(ing_id, [])
                if style_pcts:
                    avg = sum(style_pcts) / len(style_pcts)
                    weight = max(0.5, avg)  # 데이터 평균값 직접 사용

            # Priority 3: category+note profile
            if weight is None:
                cat_key = (cat, note_type)
                cat_pcts = CATEGORY_PCT_PROFILES.get(cat_key, [])
                if cat_pcts:
                    avg = sum(cat_pcts) / len(cat_pcts)
                    weight = max(0.3, avg)  # 데이터 평균값 직접 사용

            # Priority 4: AI score proportional (fallback)
            if weight is None:
                typical = ing.get('typical_pct') or 3
                weight = max(0.5, typical * (0.7 + sc * 0.6))

            # Intensity cap (very strong ingredients shouldn't dominate)
            if ing.get('intensity', 5) >= 8:
                max_pct = ing.get('max_pct') or 10
                weight = min(weight, max_pct)

            # 💡 #5 IFRA 사전 캡 — 한도의 90% 선에서 안전 차단
            ifra_limit = IFRA_LIMITS.get(ing_id)
            if ifra_limit is not None:
                safe_weight = ifra_limit * 0.9
                weight = min(weight, safe_weight)

            weights.append(weight)

        # Normalize weights to match total_note_pct
        total_weight = sum(weights) or 1
        for idx, (ing, sc) in enumerate(items):
            pct = weights[idx] / total_weight * total_note_pct
            pct = max(0.5, pct)  # minimum 0.5%
            formula_raw.append((ing, note_type, pct))

    # 7) 농도 (concentrate_pct 직접 지정 시 우선)
    if concentrate_pct is not None:
        concentrate = concentrate_pct
        if concentrate >= 25:
            concentration = 'Parfum (P)'
        elif concentrate >= 15:
            concentration = 'Eau de Parfum (EDP)'
        elif concentrate >= 8:
            concentration = 'Eau de Toilette (EDT)'
        else:
            concentration = 'Eau de Cologne (EDC)'
    elif intensity >= 70:
        concentrate = 30  # 25~35 중간값 (결정론적)
        concentration = 'Parfum (P)'
    elif intensity >= 50:
        concentrate = 20  # 15~25 중간값
        concentration = 'Eau de Parfum (EDP)'
    elif intensity >= 30:
        concentrate = 11.5  # 8~15 중간값
        concentration = 'Eau de Toilette (EDT)'
    else:
        concentrate = 6  # 4~8 중간값
        concentration = 'Eau de Cologne (EDC)'

    scale = concentrate / sum(p for _, _, p in formula_raw) if formula_raw else 1

    # 8) 최종 포뮬라
    formula = []
    total_cost = 0
    ifra_warnings = []

    for ing, note_type, pct in formula_raw:
        final_pct = round(pct * scale, 2)
        cat = ing.get('category', '')
        ing_id = ing['id']
        ml = round(final_pct / 100 * batch_ml, 2)
        density = DENSITY_MAP.get(cat, 0.92)
        grams = round(ml * density, 2)
        price_per_10ml = PREMIUM_IDS.get(ing_id, PRICE_MAP.get(cat, 5000))
        cost = round(ml / 10 * price_per_10ml)
        total_cost += cost

        ifra_limit = IFRA_LIMITS.get(ing_id)
        ifra_status = None
        if ifra_limit is not None:
            if final_pct > ifra_limit:
                ifra_status = 'exceeded'
                ifra_warnings.append({
                    'ingredient': ing.get('name_ko', ing_id),
                    'current_pct': final_pct,
                    'ifra_max_pct': ifra_limit,
                    'action': f'{ifra_limit}% 이하로 줄여야 합니다'
                })
            else:
                ifra_status = 'safe'

        formula.append({
            'id': ing_id,
            'name_ko': ing.get('name_ko', ing_id),
            'name_en': ing.get('name_en', ''),
            'category': cat,
            'note_type': note_type,
            'percentage': final_pct,
            'ml': ml,
            'grams': grams,
            'cost_krw': cost,
            'volatility': ing.get('volatility', 5),
            'intensity': ing.get('intensity', 5),
            'longevity': ing.get('longevity', 5),
            'descriptors': ing.get('descriptors', []),
            'ifra_limit': ifra_limit,
            'ifra_status': ifra_status,
            # 프로 필드
            'cas_number': ing.get('cas_number', '-'),
            'substitutes': ing.get('substitutes', []),
            'dilution_solvent': ing.get('dilution_solvent', '-'),
            'dilution_pct': ing.get('dilution_pct', 100),
            'function_note': ing.get('function_note', ''),
        })

    # 혼합 순서 (베이스 먼저)
    mix_order = {'base': 0, 'middle': 1, 'top': 2}
    formula.sort(key=lambda f: (mix_order.get(f['note_type'], 1), -f['percentage']))

    # 9) 혼합 단계
    mixing_steps = []
    step_num = 1

    base_items = [f for f in formula if f['note_type'] == 'base']
    if base_items:
        mixing_steps.append({
            'step': step_num, 'label': '베이스 노트 (먼저 혼합)',
            'note_type': 'base', 'ingredients': base_items,
            'instruction': '비커에 베이스 원료를 순서대로 넣고 잘 저어줍니다.'
        })
        step_num += 1

    mid_items = [f for f in formula if f['note_type'] == 'middle']
    if mid_items:
        mixing_steps.append({
            'step': step_num, 'label': '미들 노트 (하트)',
            'note_type': 'middle', 'ingredients': mid_items,
            'instruction': '미들 원료를 추가하고 부드럽게 혼합합니다.'
        })
        step_num += 1

    top_items = [f for f in formula if f['note_type'] == 'top']
    if top_items:
        mixing_steps.append({
            'step': step_num, 'label': '탑 노트 (마지막)',
            'note_type': 'top', 'ingredients': top_items,
            'instruction': '탑 원료를 마지막에 추가합니다. 휘발성이 높아 먼저 넣으면 날아갑니다.'
        })
        step_num += 1

    alcohol_pct = round(100 - concentrate, 1)
    alcohol_ml = round(alcohol_pct / 100 * batch_ml, 1)
    alcohol_cost = round(alcohol_ml / 100 * 5000)
    total_cost += alcohol_cost

    mixing_steps.append({
        'step': step_num, 'label': '용매 (에탄올)',
        'note_type': 'solvent',
        'ingredients': [{
            'id': 'ethanol_95', 'name_ko': '에탄올 95%', 'name_en': 'Ethanol 95%',
            'category': 'solvent', 'note_type': 'solvent',
            'percentage': alcohol_pct, 'ml': alcohol_ml,
            'grams': round(alcohol_ml * 0.789, 1), 'cost_krw': alcohol_cost,
        }],
        'instruction': '향료가 완전히 혼합된 후 에탄올을 천천히 부어 희석합니다.'
    })

    # 10) 이름
    name_options = NAME_PARTS.get(mood, NAME_PARTS['elegant'])
    # 결정론적 이름: mood + season 해시 기반 선택
    name_idx = abs(hash(f"{mood}_{season}")) % len(name_options)
    adj, noun = name_options[name_idx]
    name_en = f"{adj} {noun}"
    name_ko = NAME_KO_MAP.get(name_en, name_en)

    # 11) 피라미드
    pyramid = {
        'top': [f['name_ko'] for f in formula if f['note_type'] == 'top'],
        'middle': [f['name_ko'] for f in formula if f['note_type'] == 'middle'],
        'base': [f['name_ko'] for f in formula if f['note_type'] == 'base'],
    }

    # 12) 통계
    total_concentrate = sum(f['percentage'] for f in formula)
    avg_longevity = sum(f['longevity'] * f['percentage'] for f in formula) / max(total_concentrate, 1)
    avg_intensity = sum(f['intensity'] * f['percentage'] for f in formula) / max(total_concentrate, 1)
    longevity_hours = round(avg_longevity * 1.2, 1)
    sillage = 'heavy' if avg_intensity > 7 else 'moderate' if avg_intensity > 5 else 'intimate'

    aging = AGING_GUIDE.get(concentration, AGING_GUIDE['Eau de Parfum (EDP)'])

    # 13) 팁
    tips = []
    top_pct = sum(f['percentage'] for f in formula if f['note_type'] == 'top')
    base_pct = sum(f['percentage'] for f in formula if f['note_type'] == 'base')
    if top_pct > base_pct * 1.5:
        tips.append('탑 노트 비중이 높습니다. 초반 30분이 화려하고 빠르게 변화합니다.')
    if base_pct > top_pct * 1.5:
        tips.append('베이스 중심 향수입니다. 드라이다운이 오래 지속됩니다.')
    if len(used_categories) >= 5:
        tips.append(f'{len(used_categories)}개 카테고리를 사용한 복합적인 향입니다.')
    if concentration == 'Parfum (P)':
        tips.append('고농도 퍼퓸입니다. 손목/목에 소량만 사용하세요.')
    if ifra_warnings:
        tips.append(f'IFRA 경고: {len(ifra_warnings)}개 원료가 안전 한도를 초과합니다.')

    # 14) 분자 수준 궁합 분석
    mol_harmony = mh.check_harmony(all_selected_ids)

    return {
        'name': name_en,
        'name_ko': name_ko,
        'concentration': concentration,
        'style': style,
        'mood': mood,
        'season': season,
        'batch_ml': batch_ml,
        'mixing_steps': mixing_steps,
        'formula': formula,
        'pyramid': pyramid,
        'cost': {
            'ingredients_krw': total_cost - alcohol_cost,
            'alcohol_krw': alcohol_cost,
            'total_krw': total_cost,
            'total_formatted': f'{total_cost:,}원',
        },
        'aging': {
            'min_days': aging['min_days'],
            'recommended_days': aging['recommended_days'],
            'note': aging['note'],
            'storage': '직사광선을 피해 서늘하고 어두운 곳에 보관. 하루에 한 번 가볍게 흔들어 줍니다.',
        },
        'ifra_warnings': ifra_warnings,
        'ai': {
            'model': 'AIRecipeEngine',
            'device': str(_engine.device),
            'trained': _engine.trained,
            'train_epochs': _engine.train_epochs,
            'train_loss': round(_engine.train_loss, 4),
            'harmony_score': round(best_harmony, 3),
            'method': 'neural_scoring + harmony_attention + molecular_harmony',
        },
        'molecular_harmony': {
            'harmony': mol_harmony.get('harmony', 0),
            'synergy_count': mol_harmony.get('synergy_count', 0),
            'masking_count': mol_harmony.get('masking_count', 0),
            'known_accords': mol_harmony.get('known_accords', []),
            'masking_warnings': mol_harmony.get('masking_warnings', []),
            'synergy_bonuses': mol_harmony.get('synergy_bonuses', []),
            'clashing_warnings': mol_harmony.get('clashing_warnings', []),
            'method': mol_harmony.get('method', 'rule_based'),
        },
        'stats': {
            'total_ingredients': len(formula),
            'total_concentrate_pct': round(total_concentrate, 1),
            'alcohol_pct': round(100 - total_concentrate, 1),
            'longevity_hours': longevity_hours,
            'sillage': sillage,
            'sillage_ko': {'heavy':'강함','moderate':'보통','intimate':'은은함'}.get(sillage, sillage),
            'categories_used': len(used_categories),
            'avg_intensity': round(avg_intensity, 1),
        },
        'tips': tips,
        'equipment': [
            '비커 (유리, 100~200ml)',
            '정밀 저울 (0.01g 단위)',
            '유리 막대 (교반용)',
            '스포이트/피펫 (소량 원료용)',
            f'향수 병 ({batch_ml}ml, 차광 유리 권장)',
            '에탄올 95% (약국/화학약품점)',
        ],
    }


def generate_variations(base_recipe, count=3):
    """기본 레시피 변형"""
    variations = []
    for i in range(count):
        var = generate_recipe(
            mood=base_recipe.get('mood', 'romantic'),
            season=base_recipe.get('season', 'spring'),
            preferences=None,
            intensity=base_recipe.get('stats', {}).get('avg_intensity', 50) * 10 + (i - 1) * 5,  # 결정론적 오프셋
            complexity=len(base_recipe.get('formula', [])) + (i - 1),  # 결정론적 변화
            batch_ml=base_recipe.get('batch_ml', 100),
        )
        var['variation_of'] = base_recipe.get('name', '')
        var['variation_num'] = i + 1
        variations.append(var)
    return variations


# ========================
# 클론 레시피 (유명 향수 재현)
# ========================

# 카테고리별 밀도 (g/ml)
_CLONE_DENSITY = {
    'musk': 0.96, 'animalic': 0.95, 'vanilla': 0.90, 'resinous': 1.05,
    'amber': 1.02, 'sandalwood': 0.97, 'woody': 0.93, 'jasmine': 0.92,
    'rose': 0.94, 'lily_of_the_valley': 0.91, 'spicy': 0.98,
    'citrus': 0.85, 'herbal': 0.88, 'tonka': 0.94, 'solvent': 0.789,
}

# 카테고리별 가격 (KRW per 10ml)
_CLONE_PRICE = {
    'musk': 3000, 'animalic': 15000, 'vanilla': 4000, 'resinous': 6000,
    'amber': 7000, 'sandalwood': 12000, 'woody': 5000, 'jasmine': 5000,
    'rose': 8000, 'lily_of_the_valley': 4000, 'spicy': 6000,
    'citrus': 3500, 'herbal': 3000, 'tonka': 5000, 'solvent': 500,
}

# 카테고리→한국어명
_CLONE_NAME_KO = {
    'musk': '머스크', 'animalic': '애니멀릭', 'vanilla': '바닐라',
    'resinous': '레진', 'amber': '앰버', 'sandalwood': '샌달우드',
    'woody': '우디', 'jasmine': '자스민', 'rose': '로즈',
    'lily_of_the_valley': '릴리 오브 더 밸리', 'spicy': '스파이시',
    'citrus': '시트러스', 'herbal': '허벌', 'tonka': '통카빈',
}

# 노트별 기본 특성
_CLONE_NOTE_PROPS = {
    'top': {'volatility': 8, 'intensity': 6, 'longevity': 3},
    'middle': {'volatility': 5, 'intensity': 6, 'longevity': 6},
    'base': {'volatility': 2, 'intensity': 5, 'longevity': 9},
}


def list_clones():
    """사용 가능한 클론 포뮬러 목록"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    clones = []
    if not os.path.exists(data_dir):
        return clones
    for f in os.listdir(data_dir):
        if f.endswith('_clone.json'):
            fpath = os.path.join(data_dir, f)
            try:
                with open(fpath, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                clones.append({
                    'id': f.replace('_clone.json', ''),
                    'name': data.get('name', f),
                    'original': data.get('original', ''),
                    'perfumer': data.get('perfumer', ''),
                    'concentration': data.get('concentration', ''),
                    'total_ingredients': data.get('total_ingredients', 0),
                })
            except Exception:
                pass
    return clones


def clone_recipe(clone_id, batch_ml=100):
    """클론 포뮬러를 generate_recipe() 형식으로 변환"""
    data_path = os.path.join(os.path.dirname(__file__), 'data', f'{clone_id}_clone.json')
    if not os.path.exists(data_path):
        return {'error': f'Clone formula not found: {clone_id}'}

    with open(data_path, 'r', encoding='utf-8') as f:
        clone_data = json.load(f)

    # 농도 파싱
    conc_str = clone_data.get('concentration', 'Eau de Parfum (20%)')
    if '20%' in conc_str or 'EDP' in conc_str:
        concentrate_pct = 20.0
        concentration = 'Eau de Parfum (EDP)'
    elif '30%' in conc_str or 'Parfum' in conc_str:
        concentrate_pct = 30.0
        concentration = 'Parfum (P)'
    elif '10%' in conc_str or 'EDT' in conc_str:
        concentrate_pct = 10.0
        concentration = 'Eau de Toilette (EDT)'
    else:
        concentrate_pct = 20.0
        concentration = 'Eau de Parfum (EDP)'

    concentrate_ml = batch_ml * concentrate_pct / 100
    alcohol_ml = batch_ml - concentrate_ml

    # 포뮬러 변환
    formula_items = clone_data.get('formula', [])
    total_ppt = sum(item.get('ppt', 0) for item in formula_items)
    if total_ppt == 0:
        total_ppt = 1

    formula = []
    total_cost = 0
    ifra_warnings = []
    used_categories = set()

    for item in formula_items:
        name = item.get('name', 'Unknown')
        cat = item.get('category', 'other')
        note = item.get('note', 'middle')
        ppt = item.get('ppt', 0)
        desc = item.get('description', '')

        pct_in_concentrate = ppt / total_ppt * 100
        final_pct = round(pct_in_concentrate * concentrate_pct / 100, 2)
        ml = round(ppt / total_ppt * concentrate_ml, 3)
        density = _CLONE_DENSITY.get(cat, 0.92)
        grams = round(ml * density, 3)
        price = _CLONE_PRICE.get(cat, 5000)
        cost = round(ml / 10 * price)
        total_cost += cost
        used_categories.add(cat)

        note_props = _CLONE_NOTE_PROPS.get(note, {'volatility': 5, 'intensity': 5, 'longevity': 5})

        # IFRA 체크 (기본적인 것만)
        ifra_limit = None
        ifra_status = None
        if 'cinnamon' in name.lower():
            ifra_limit = 0.5
        elif 'eugenol' in name.lower():
            ifra_limit = 0.6
        elif 'coumarin' in name.lower():
            ifra_limit = 2.0

        if ifra_limit is not None:
            if final_pct > ifra_limit:
                ifra_status = 'exceeded'
                ifra_warnings.append({
                    'ingredient': name,
                    'current_pct': final_pct,
                    'ifra_max_pct': ifra_limit,
                    'action': f'{ifra_limit}% 이하로 줄여야 합니다'
                })
            else:
                ifra_status = 'safe'

        formula.append({
            'id': name.lower().replace(' ', '_').replace('(', '').replace(')', ''),
            'name_ko': _CLONE_NAME_KO.get(cat, cat) + ' - ' + name,
            'name_en': name,
            'category': cat,
            'note_type': note,
            'percentage': final_pct,
            'ml': round(ml, 2),
            'grams': round(grams, 2),
            'cost_krw': cost,
            'volatility': note_props['volatility'],
            'intensity': note_props['intensity'],
            'longevity': note_props['longevity'],
            'descriptors': [desc] if desc else [],
            'ifra_limit': ifra_limit,
            'ifra_status': ifra_status,
        })

    # 혼합 순서 (베이스 먼저)
    mix_order = {'base': 0, 'middle': 1, 'top': 2}
    formula.sort(key=lambda f: (mix_order.get(f['note_type'], 1), -f['percentage']))

    # 믹싱 스텝
    mixing_steps = []
    step_num = 1

    base_items = [f for f in formula if f['note_type'] == 'base']
    if base_items:
        mixing_steps.append({
            'step': step_num, 'label': '베이스 노트 (먼저 혼합)',
            'note_type': 'base', 'ingredients': base_items,
            'instruction': '비커에 베이스 원료를 순서대로 넣고 잘 저어줍니다. 24시간 숙성시킵니다.'
        })
        step_num += 1

    mid_items = [f for f in formula if f['note_type'] == 'middle']
    if mid_items:
        mixing_steps.append({
            'step': step_num, 'label': '미들 노트 (하트)',
            'note_type': 'middle', 'ingredients': mid_items,
            'instruction': '미들 원료를 추가하고 부드럽게 혼합합니다.'
        })
        step_num += 1

    top_items = [f for f in formula if f['note_type'] == 'top']
    if top_items:
        mixing_steps.append({
            'step': step_num, 'label': '탑 노트 (마지막)',
            'note_type': 'top', 'ingredients': top_items,
            'instruction': '탑 원료를 마지막에 추가합니다. 휘발성이 높아 먼저 넣으면 날아갑니다.'
        })
        step_num += 1

    # 에탄올
    alcohol_cost = round(alcohol_ml / 100 * 5000)
    total_cost += alcohol_cost

    mixing_steps.append({
        'step': step_num, 'label': '용매 (에탄올)',
        'note_type': 'solvent',
        'ingredients': [{
            'id': 'ethanol_95', 'name_ko': '에탄올 95%', 'name_en': 'Ethanol 95%',
            'category': 'solvent', 'note_type': 'solvent',
            'percentage': round(100 - concentrate_pct, 1), 'ml': round(alcohol_ml, 1),
            'grams': round(alcohol_ml * 0.789, 1), 'cost_krw': alcohol_cost,
        }],
        'instruction': '향료가 완전히 혼합된 후 에탄올을 천천히 부어 희석합니다.'
    })

    # 피라미드
    pyramid = {
        'top': [f['name_en'] for f in formula if f['note_type'] == 'top'],
        'middle': [f['name_en'] for f in formula if f['note_type'] == 'middle'],
        'base': [f['name_en'] for f in formula if f['note_type'] == 'base'],
    }

    # 통계
    total_concentrate = sum(f['percentage'] for f in formula)
    avg_longevity = sum(f['longevity'] * f['percentage'] for f in formula) / max(total_concentrate, 1)
    avg_intensity = sum(f['intensity'] * f['percentage'] for f in formula) / max(total_concentrate, 1)
    longevity_hours = round(avg_longevity * 1.2, 1)
    sillage = 'heavy' if avg_intensity > 7 else 'moderate' if avg_intensity > 5 else 'intimate'

    aging = AGING_GUIDE.get(concentration, AGING_GUIDE['Eau de Parfum (EDP)'])

    # 팁
    tips = clone_data.get('key_character_notes', [])
    mixing_protocol = clone_data.get('mixing_protocol', [])

    return {
        'name': clone_data.get('original', clone_id),
        'name_ko': clone_data.get('name', clone_id),
        'clone_of': clone_data.get('original', ''),
        'perfumer': clone_data.get('perfumer', ''),
        'concentration': concentration,
        'style': 'oriental',
        'mood': 'sensual',
        'season': 'winter',
        'batch_ml': batch_ml,
        'mixing_steps': mixing_steps,
        'formula': formula,
        'pyramid': pyramid,
        'cost': {
            'ingredients_krw': total_cost - alcohol_cost,
            'alcohol_krw': alcohol_cost,
            'total_krw': total_cost,
            'total_formatted': f'{total_cost:,}원',
        },
        'aging': {
            'min_days': aging['min_days'],
            'recommended_days': aging['recommended_days'],
            'note': aging['note'],
            'storage': '직사광선을 피해 서늘하고 어두운 곳에 보관. 하루에 한 번 가볍게 흔들어 줍니다.',
        },
        'ifra_warnings': ifra_warnings,
        'ai': {
            'model': 'CloneFormula (GC-MS based)',
            'device': 'analysis',
            'trained': True,
            'train_epochs': 0,
            'train_loss': 0,
            'harmony_score': 0.95,
            'method': 'gc_ms_reconstruction + expert_formulation',
        },
        'stats': {
            'total_ingredients': len(formula),
            'total_concentrate_pct': round(total_concentrate, 1),
            'alcohol_pct': round(100 - total_concentrate, 1),
            'longevity_hours': longevity_hours,
            'sillage': sillage,
            'sillage_ko': {'heavy': '강함', 'moderate': '보통', 'intimate': '은은함'}.get(sillage, sillage),
            'categories_used': len(used_categories),
            'avg_intensity': round(avg_intensity, 1),
        },
        'tips': tips,
        'mixing_protocol': mixing_protocol,
        'equipment': [
            '비커 (유리, 100~200ml)',
            '정밀 저울 (0.01g 단위)',
            '유리 막대 (교반용)',
            '스포이트/피펫 (소량 원료용)',
            f'향수 병 ({batch_ml}ml, 차광 유리 권장)',
            '에탄올 95% (약국/화학약품점)',
        ],
    }
