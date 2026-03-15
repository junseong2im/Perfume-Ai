# -*- coding: utf-8 -*-
"""
🎭 AI Composer v3 — 자율 조향 AI (Full Upgrade)
═══════════════════════════════════════════════════
자율 에이전트 패턴 + 8개 업그레이드:

 [4] 시너지 DB — 원료 쌍별 상호작용 사전 계산
 [5] 농도 미세 최적화 — Coordinate Search로 최적 배합비
 [6] 릴레이 조향 — AI끼리 릴레이식 공동 창작
 [8] 멀티 테마 블렌딩 — "봄의 밤" 같은 복합 테마 처리

★ 최적화: 시너지 캐시, 벡터 캐시, GNN 공유, numpy 우선
"""
import sys, os, json, time
import numpy as np
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, '.')

import database as db
import biophysics_simulator as biophys


# ═══════════════════════════════════════
# 캐시 (RAM 절약)
# ═══════════════════════════════════════
_ODOR_VEC_CACHE = {}
_SYNERGY_CACHE = {}  # (id_a, id_b) → float

def _cached_odor_vec(smiles, gnn_engine):
    if smiles not in _ODOR_VEC_CACHE:
        _ODOR_VEC_CACHE[smiles] = gnn_engine.predict_odor_vector(smiles)
    return _ODOR_VEC_CACHE[smiles]


# ═══════════════════════════════════════
# [4] 시너지 DB — 원료 쌍별 상호작용 (Lazy)
# ═══════════════════════════════════════
_POOL_LOOKUP = {}  # id → pool item (lazy 계산용)

def build_synergy_db(pool):
    """
    ★ 풀 인덱스만 구축 (Lazy 모드)
    실제 시너지는 get_synergy()에서 on-demand 계산 + 캐시
    → 700+ 풀에서도 0초 시작
    """
    t0 = time.time()
    _POOL_LOOKUP.clear()
    for item in pool:
        _POOL_LOOKUP[item['id']] = item
    elapsed = time.time() - t0
    print(f"  ✅ 시너지 DB: {len(pool)}개 인덱스 ({elapsed:.3f}s, lazy 모드)")
    return _SYNERGY_CACHE


def get_synergy(id_a, id_b):
    """시너지 점수 조회 (Lazy 계산 + 캐시)"""
    key = (id_a, id_b)
    if key in _SYNERGY_CACHE:
        return _SYNERGY_CACHE[key]
    
    # Lazy 계산
    a = _POOL_LOOKUP.get(id_a)
    b = _POOL_LOOKUP.get(id_b)
    if a is None or b is None:
        return 0.5
    
    sim = cosine_sim(a['vec'], b['vec'])
    synergy = float(np.exp(-((sim - 0.5) ** 2) / 0.08))
    
    if a['note_type'] != b['note_type']:
        synergy *= 1.15
    if a['category'] != b['category']:
        synergy *= 1.1
    
    _SYNERGY_CACHE[key] = synergy
    _SYNERGY_CACHE[(id_b, id_a)] = synergy
    return synergy


def avg_synergy_with(candidate, selected):
    """후보와 기존 원료들 간 평균 시너지"""
    if not selected:
        return 0.5
    scores = [get_synergy(candidate['id'], s['id']) for s in selected]
    return float(np.mean(scores))


# ═══════════════════════════════════════
# 원료 풀 로딩 (Full DB — 적극 매칭)
# ═══════════════════════════════════════
import re

def _normalize_name(name):
    """원료명 정규화: 괄호 제거, 소문자, 특수문자 제거"""
    name = name.lower().strip()
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)  # 괄호 내용 제거
    name = re.sub(r'[^a-z0-9가-힣\s]', ' ', name)  # 특수문자 → 공백
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _build_smiles_map(molecules):
    """여러 키로 SMILES 룩업 사전 생성"""
    smap = {}  # key → smiles
    mol_by_name = {}  # normalized_name → molecule
    
    for m in molecules:
        smi = m.get('smiles')
        if not smi:
            continue
        
        # 1) 원본 이름
        name = (m.get('name') or '').strip()
        if name:
            smap[name.lower()] = smi
            norm = _normalize_name(name)
            if norm:
                smap[norm] = smi
                mol_by_name[norm] = m
        
        # 2) CAS
        cas = (m.get('cas_number') or '').strip()
        if cas:
            smap[cas] = smi
        
        # 3) 첫 단어 (4글자 이상)
        words = name.lower().split()
        if words and len(words[0]) >= 5:
            smap.setdefault(words[0], smi)
    
    return smap, mol_by_name


def _match_smiles(ing, smap):
    """원료 → SMILES 다전략 매칭"""
    name_en = (ing.get('name_en') or '').strip()
    name_ko = (ing.get('name_ko') or '').strip()
    cas = (ing.get('cas_number') or '').strip()
    
    # 1) 직접 매칭
    for key in [name_en.lower(), name_ko.lower(), cas]:
        if key and key in smap:
            return smap[key]
    
    # 2) 정규화 매칭
    for name in [name_en, name_ko]:
        norm = _normalize_name(name)
        if norm and norm in smap:
            return smap[norm]
    
    # 3) 토큰 매칭 — 이름의 주요 단어로
    if name_en:
        tokens = name_en.lower().split()
        # '/' 로 나뉜 별칭
        if '/' in name_en:
            for alias in name_en.split('/'):
                alias = alias.strip().lower()
                if alias in smap:
                    return smap[alias]
                alias_norm = _normalize_name(alias)
                if alias_norm in smap:
                    return smap[alias_norm]
        # 4글자 이상 토큰 매칭
        for token in tokens:
            clean = re.sub(r'[^a-z0-9]', '', token)
            if len(clean) >= 5 and clean in smap:
                return smap[clean]
    
    return None


# 분자 odor_labels → note_type 추정
VOLATILITY_MAP = {
    'citrus': 'top', 'fresh': 'top', 'light': 'top', 'green': 'top',
    'ozonic': 'top', 'fruity': 'top', 'herbal': 'top', 'minty': 'top',
    'floral': 'middle', 'sweet': 'middle', 'spicy': 'middle',
    'aromatic': 'middle', 'rose': 'middle', 'jasmine': 'middle',
    'woody': 'base', 'musk': 'base', 'amber': 'base', 'leather': 'base',
    'vanilla': 'base', 'smoky': 'base', 'earthy': 'base', 'powdery': 'base',
    'warm': 'base', 'balsamic': 'base', 'resinous': 'base',
}

def _guess_note_type(labels):
    """odor label에서 note_type 추정"""
    votes = {'top': 0, 'middle': 0, 'base': 0}
    for label in labels:
        nt = VOLATILITY_MAP.get(label.lower(), 'middle')
        votes[nt] += 1
    return max(votes, key=votes.get)


def _guess_category(labels):
    """odor label에서 카테고리 추정"""
    cat_map = {
        'floral': 'floral', 'rose': 'floral', 'jasmine': 'floral',
        'citrus': 'citrus', 'lemon': 'citrus', 'orange': 'citrus',
        'woody': 'woody', 'cedar': 'woody', 'sandalwood': 'woody',
        'musk': 'musk', 'amber': 'amber',
        'fruity': 'fruity', 'sweet': 'sweet',
        'spicy': 'spicy', 'herbal': 'herbal',
        'green': 'green', 'fresh': 'fresh',
        'leather': 'leather', 'smoky': 'smoky',
    }
    for label in labels:
        if label.lower() in cat_map:
            return cat_map[label.lower()]
    return 'aromatic'


def load_pool():
    molecules = db.get_all_molecules()
    ingredients = db.get_all_ingredients()
    
    smap, mol_by_name = _build_smiles_map(molecules)
    
    # 1) 전체 태그 수집 (ingredients + molecules)
    all_tags = set()
    for ing in ingredients:
        descriptors = ing.get('odor_descriptors', '') or ''
        for tag in descriptors.split(','):
            tag = tag.strip().lower()
            if tag:
                all_tags.add(tag)
    for m in molecules:
        labels = m.get('odor_labels')
        if isinstance(labels, list):
            for l in labels:
                all_tags.add(l.strip().lower())
        elif isinstance(labels, str) and labels:
            for l in labels.split(','):
                all_tags.add(l.strip().lower())
    all_tags = sorted(all_tags)
    tag_to_idx = {t: i for i, t in enumerate(all_tags)}
    n_tags = len(all_tags)
    
    def make_vec(descriptors):
        vec = np.zeros(n_tags, dtype=np.float32)
        for d in descriptors:
            if d in tag_to_idx:
                vec[tag_to_idx[d]] = 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    
    # 2) Tier 1: Ingredients 테이블 (적극 매칭)
    pool = []
    used_smiles = set()
    
    for ing in ingredients:
        ifra_limit = ing.get('ifra_limit')
        if ifra_limit is not None:
            try:
                if float(ifra_limit) <= 0:
                    continue
            except:
                pass
        
        smiles = _match_smiles(ing, smap)
        if not smiles:
            continue
        
        name_en = (ing.get('name_en') or '').strip()
        name_ko = (ing.get('name_ko') or '').strip()
        
        descriptors = (ing.get('odor_descriptors', '') or '').split(',')
        descriptors = [d.strip().lower() for d in descriptors if d.strip()]
        
        note_type = (ing.get('note_type', '') or 'middle').lower()
        if note_type not in ('top', 'middle', 'base'):
            note_type = _guess_note_type(descriptors) if descriptors else 'middle'
        
        pool.append({
            'id': ing['id'], 'name_en': name_en, 'name_ko': name_ko,
            'category': (ing.get('category', '') or 'other').lower(),
            'note_type': note_type, 'smiles': smiles, 'vec': make_vec(descriptors),
            'strength': ing.get('odor_strength', 5),
            'price': ing.get('price_usd_kg'),
            'availability': ing.get('availability', 'unknown'),
            'descriptors': descriptors,
        })
        used_smiles.add(smiles)
    
    tier1_count = len(pool)
    
    # ★ Tier 2 제거: 비상업 분자(CAS 없음) 대신 실제 구할 수 있는 ingredients만 사용
    # molecules 테이블은 향 벡터/시너지 계산에만 활용 (원료 풀에는 포함 안 함)
    
    # hedonic 사전 추정 (상위 300, 풀 로딩 시 1회)
    pool_by_sim = sorted(pool, key=lambda x: sum(x['vec']), reverse=True)
    cached_count = 0
    for ing in pool_by_sim[:300]:
        try:
            r = biophys.simulate_recipe([ing['smiles']], [100.0])
            ing['_hedonic_est'] = r['hedonic']['hedonic_score']
            cached_count += 1
        except:
            ing['_hedonic_est'] = 0.5
    
    print(f"  📦 원료 풀: {len(pool)}개 (실제 상업 원료만)")
    print(f"  🧪 hedonic 캐시: {cached_count}개 사전 계산")
    return pool, all_tags, tag_to_idx


# ═══════════════════════════════════════
# [8] 멀티 테마 블렌딩
# ═══════════════════════════════════════
KEYWORD_MAP = {
    # ══ 계절/시간 ══
    '봄': ['floral', 'green', 'fresh', 'light'],
    '여름': ['citrus', 'aquatic', 'fresh', 'green', 'ozonic'],
    '가을': ['woody', 'spicy', 'warm', 'amber', 'earthy'],
    '겨울': ['warm', 'spicy', 'woody', 'vanilla', 'amber'],
    '밤': ['dark', 'musk', 'amber', 'deep', 'mysterious'],
    '아침': ['fresh', 'citrus', 'green', 'light', 'dewy'],
    '새벽': ['cool', 'fresh', 'ozonic', 'clean'],
    '석양': ['warm', 'amber', 'golden', 'spicy'],
    
    # ══ 꽃/식물 ══
    '벚꽃': ['floral', 'cherry blossom', 'pink', 'sweet', 'powdery'],
    '장미': ['rose', 'floral', 'romantic', 'sweet'],
    '라벤더': ['lavender', 'herbal', 'fresh', 'aromatic'],
    '자스민': ['jasmine', 'floral', 'sweet', 'exotic'],
    '꽃': ['floral', 'sweet', 'fresh'],
    '무궁화': ['floral', 'hibiscus', 'sweet', 'korean'],
    '백합': ['lily', 'floral', 'green', 'fresh'],
    '튤립': ['floral', 'green', 'fresh', 'waxy'],
    '연꽃': ['lotus', 'floral', 'aquatic', 'clean'],
    
    # ══ 향 원료 직접 키워드 ══
    '머스크': ['musk', 'musky', 'animalic', 'skin', 'warm'],
    '우디': ['woody', 'cedar', 'sandalwood', 'vetiver', 'earthy'],
    '파츌리': ['patchouli', 'earthy', 'woody', 'dark'],
    '시더': ['cedar', 'woody', 'dry', 'pencil'],
    '샌달우드': ['sandalwood', 'woody', 'creamy', 'warm'],
    '백단': ['sandalwood', 'woody', 'creamy', 'warm'],
    '베티버': ['vetiver', 'earthy', 'woody', 'smoky'],
    '가죽': ['leather', 'suede', 'smoky', 'animalic', 'birch'],
    '레더': ['leather', 'suede', 'smoky', 'animalic'],
    '앰버': ['amber', 'warm', 'resinous', 'sweet'],
    '바닐라': ['vanilla', 'sweet', 'warm', 'gourmand', 'creamy'],
    '토바코': ['tobacco', 'smoky', 'leather', 'woody', 'dry'],
    '담배': ['tobacco', 'smoky', 'leather', 'dry'],
    '인센스': ['incense', 'smoky', 'spicy', 'resinous'],
    '우드': ['oud', 'woody', 'dark', 'animalic', 'smoky'],
    '페퍼': ['pepper', 'spicy', 'warm', 'pungent'],
    '후추': ['pepper', 'spicy', 'warm', 'pungent'],
    '민트': ['mint', 'minty', 'cool', 'fresh', 'menthol'],
    '베르가못': ['bergamot', 'citrus', 'fresh', 'aromatic'],
    '레몬': ['lemon', 'citrus', 'fresh', 'sour'],
    '오렌지': ['orange', 'citrus', 'sweet', 'fresh'],
    
    # ══ 남성/여성 아키타입 ══
    '상남자': ['musk', 'leather', 'woody', 'smoky', 'spicy', 'tobacco'],
    '남성': ['musk', 'woody', 'leather', 'aromatic', 'fresh'],
    '여성': ['floral', 'sweet', 'powdery', 'fruity', 'musk'],
    '운동': ['fresh', 'cool', 'clean', 'aquatic', 'ozonic', 'minty'],
    '헬스': ['fresh', 'cool', 'musk', 'aquatic', 'clean', 'ozonic'],
    '스포츠': ['fresh', 'cool', 'clean', 'citrus', 'ozonic'],
    '세련': ['modern', 'clean', 'fresh', 'musk', 'woody'],
    
    # ══ 자연/장소 ══
    '숲': ['woody', 'green', 'earthy', 'mossy'],
    '바다': ['aquatic', 'marine', 'salty', 'fresh', 'ozonic'],
    '비': ['petrichor', 'earthy', 'green', 'fresh', 'wet'],
    '나무': ['woody', 'cedar', 'sandalwood'],
    '대나무': ['bamboo', 'green', 'fresh', 'woody'],
    '눈': ['snow', 'cold', 'clean', 'crisp', 'white'],
    '한라산': ['green', 'earthy', 'fresh', 'mossy', 'woody'],
    '제주': ['citrus', 'green', 'aquatic', 'fresh'],
    '정원': ['floral', 'green', 'fresh', 'herbal'],
    '산책': ['green', 'fresh', 'earthy', 'natural'],
    '도시': ['modern', 'fresh', 'clean', 'metallic'],
    
    # ══ 무드/감성 ══
    '시원': ['fresh', 'cool', 'minty', 'aquatic', 'ozonic'],
    '따뜻': ['warm', 'cozy', 'amber', 'spicy', 'vanilla'],
    '달콤': ['sweet', 'gourmand', 'vanilla', 'caramel'],
    '관능': ['musk', 'amber', 'warm', 'sensual', 'animalic'],
    '비밀': ['dark', 'musk', 'mysterious', 'deep', 'incense'],
    '신선': ['fresh', 'clean', 'crisp', 'ozonic'],
    '깊은': ['deep', 'woody', 'amber', 'oud', 'leather'],
    '가벼운': ['light', 'fresh', 'sheer', 'clean'],
    '고급': ['elegant', 'luxury', 'refined', 'woody', 'musk'],
    '우아': ['elegant', 'floral', 'powdery', 'iris'],
    '자연': ['natural', 'green', 'earthy', 'herbal'],
    '과일': ['fruity', 'sweet', 'fresh', 'tropical'],
    '향수': ['perfume'],
    '땀': ['musk', 'skin', 'salty', 'animalic', 'warm'],
}

# 테마 → 22d 향 차원 매핑 (ODOR_DIMENSIONS 인덱스)
THEME_DIM_MAP = {
    '봄': [3, 8, 7],      # floral, green, fresh
    '벚꽃': [3, 0, 12],   # floral, sweet, powdery
    '여름': [4, 13, 7],   # citrus, aquatic, fresh
    '가을': [2, 5, 15],   # woody, spicy, amber
    '겨울': [9, 5, 2],    # warm, spicy, woody
    '밤': [6, 15, 9],     # musk, amber, warm
    '숲': [2, 8, 17],     # woody, green, earthy
    '바다': [13, 18, 7],  # aquatic, ozonic, fresh
    '장미': [3, 0, 12],   # floral, sweet, powdery
    '꽃': [3, 0, 7],      # floral, sweet, fresh
    '신선': [7, 4, 8],    # fresh, citrus, green
    '달콤': [0, 10, 9],   # sweet, fruity, warm
    '우아': [3, 12, 6],   # floral, powdery, musk
    '깊은': [2, 15, 16],  # woody, amber, leather
    '자연': [8, 17, 2],   # green, earthy, woody
    '상남자': [6, 16, 2], # musk, leather, woody
    '머스크': [6, 9, 15], # musk, warm, amber
    '우디': [2, 17, 9],   # woody, earthy, warm
    '가죽': [16, 11, 6],  # leather, smoky, musk
    '운동': [7, 13, 18],  # fresh, aquatic, ozonic
    '헬스': [7, 6, 13],   # fresh, musk, aquatic
    '시원': [7, 18, 13],  # fresh, ozonic, aquatic
    '관능': [6, 15, 9],   # musk, amber, warm
    '남성': [6, 2, 16],   # musk, woody, leather
}

ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]


def theme_to_vector(theme_text, all_tags, tag_to_idx):
    """
    ★ [8] 멀티 테마 블렌딩
    "봄의 밤" → "봄" 벡터 + "밤" 벡터를 보간
    복합 키워드를 자동 분해하여 향 공간에서 블렌딩
    """
    n_tags = len(all_tags)
    vec = np.zeros(n_tags, dtype=np.float32)
    matched_keywords = []
    
    theme_lower = theme_text.lower()
    keyword_weights = {}  # 키워드별 가중치 (위치 기반)
    
    for keyword, scent_tags in KEYWORD_MAP.items():
        if keyword in theme_text or keyword in theme_lower:
            matched_keywords.append(keyword)
            
            # 테마 내 위치 기반 가중치 (앞에 올수록 메인 테마)
            pos = theme_text.find(keyword)
            weight = 1.0 / (1 + pos * 0.1) if pos >= 0 else 0.5
            keyword_weights[keyword] = weight
            
            for tag in scent_tags:
                if tag in tag_to_idx:
                    vec[tag_to_idx[tag]] += weight
                else:
                    for at in all_tags:
                        if tag in at or at in tag:
                            vec[tag_to_idx[at]] += weight * 0.5
    
    if not matched_keywords:
        for tag in ['floral', 'sweet', 'fresh']:
            if tag in tag_to_idx:
                vec[tag_to_idx[tag]] = 1.0
        matched_keywords = ['default']
    
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    
    return vec, list(set(matched_keywords))


def theme_to_target_profile(theme_text, keywords):
    """테마 → 22d 목표 프로파일 (블렌딩)"""
    target = np.zeros(22, dtype=np.float32)
    for kw in keywords:
        if kw in THEME_DIM_MAP:
            for d in THEME_DIM_MAP[kw]:
                target[d] += 1.0
    if target.sum() == 0:
        target[[3, 0, 7]] = 1.0  # floral, sweet, fresh
    return target / (np.linalg.norm(target) + 1e-8)


# ═══════════════════════════════════════
# 벡터 연산
# ═══════════════════════════════════════
def cosine_sim(a, b):
    dot = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0

def harmony_score(vecs):
    if len(vecs) < 2:
        return 1.0
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, min(len(vecs), i + 6)):  # 근접 6개만 (최적화)
            sims.append(cosine_sim(vecs[i], vecs[j]))
    return float(np.mean(sims))


# ═══════════════════════════════════════
# GNN 기반 Beam Search + 시너지
# ═══════════════════════════════════════
def beam_search_compose(theme_vec, pool, start_note, beam_width=6, target_count=12,
                        prior_selected=None):
    """
    ★ 향 빔서치 v2 — 다기준 + 탐색/착취 균형
    
    후보 점수 = 테마유사도 × 0.25 + 조화도 × 0.20 + 시너지 × 0.25 + 쾌적도 × 0.30
    탐색: 20% 후보는 랜덤 (exploration)
    빔폭: 6 (기존 3)
    """
    import random
    
    note_pools = {'top': [], 'middle': [], 'base': []}
    for ing in pool:
        nt = ing['note_type']
        if nt in note_pools:
            note_pools[nt].append(ing)
        else:
            note_pools['middle'].append(ing)
    
    # 테마 유사도 사전 계산 + 백분위 정규화 (심사 기준 자율 대응)
    raw_sims = []
    for ing in pool:
        raw = cosine_sim(theme_vec, ing['vec'])
        ing['_theme_sim_raw'] = raw
        raw_sims.append(raw)
    
    # 백분위 정규화: 원료 풀 내 상대적 순위 → 0~1
    if raw_sims:
        sorted_sims = sorted(raw_sims)
        n = len(sorted_sims)
        for ing in pool:
            rank = sorted_sims.index(ing['_theme_sim_raw'])
            ing['_theme_sim'] = rank / max(1, n - 1)  # 최상위=1.0, 최하위=0.0
    
    # hedonic은 load_pool에서 사전 캐시 → 여기서는 미캐시분만 보완
    for ing in pool:
        if '_hedonic_est' not in ing:
            ing['_hedonic_est'] = 0.5  # 미캐시 = 기본값
    
    def candidate_score(cand, selected):
        """다기준 후보 점수 — ★ 심사 기준 자율 인식 (펜넬티 사전 방어)"""
        theme_s = cand['_theme_sim']
        hedonic_s = cand.get('_hedonic_est', 0.5)
        
        # 조화도
        if selected:
            sel_vecs = [s['vec'] for s in selected]
            harm = float(np.mean([cosine_sim(cand['vec'], sv) for sv in sel_vecs]))
        else:
            harm = 0.5
        
        # 시너지
        syn = avg_synergy_with(cand, selected)
        
        # ★ 카테고리 다양성 (소프트 보너스/페널티)
        cand_cat = cand.get('category', 'other')
        cand_note = cand.get('note_type', 'middle')
        if selected:
            cat_counts = {}
            for s in selected:
                c = s.get('category', 'other')
                cat_counts[c] = cat_counts.get(c, 0) + 1
            existing = cat_counts.get(cand_cat, 0)
            total = len(selected)
            ratio = existing / total if total > 0 else 0
            diversity = max(0.0, 1.0 - ratio * 2)
        else:
            diversity = 0.5
        
        # ★ NEW: 구조 균형 보너스 (심사관 이상 비율: top 15%, mid 35%, base 50%)
        if selected:
            note_concs = {'top': 0, 'middle': 0, 'base': 0}
            for s in selected:
                nt = s.get('note_type', 'middle')
                if nt in note_concs:
                    note_concs[nt] += 1
            note_concs[cand_note] = note_concs.get(cand_note, 0) + 1
            t = sum(note_concs.values())
            if t > 0:
                ideal = {'top': 0.15, 'middle': 0.35, 'base': 0.50}
                actual = {k: v/t for k, v in note_concs.items()}
                # 이상 비율에 가까울수록 높은 점수
                balance = 1.0 - sum(abs(actual.get(k, 0) - ideal[k]) for k in ideal) / 2
                structure_bonus = max(0, balance)
            else:
                structure_bonus = 0.5
        else:
            structure_bonus = 0.5
        
        # ★ NEW: 중복 카테고리+노트 방지 (심사관 -1.5% 감점 사전 방어)
        if selected:
            seen_pairs = set()
            for s in selected:
                seen_pairs.add((s.get('category', ''), s.get('note_type', '')))
            pair = (cand_cat, cand_note)
            dup_penalty = 0.0 if pair not in seen_pairs else -0.05  # 소프트 감점
        else:
            dup_penalty = 0.0
        
        return (theme_s * 0.20 + harm * 0.12 + syn * 0.15 + hedonic_s * 0.23 
                + diversity * 0.10 + structure_bonus * 0.10 + dup_penalty
                + max(0, theme_s - 0.25) * 0.10)  # 테마 관련도 0.25+ 보너스
    
    # 초기 선택
    if prior_selected:
        selected = list(prior_selected)
        selected_ids = {ing['id'] for ing in selected}
    elif start_note == 'all':
        sorted_all = sorted(pool, key=lambda x: x['_theme_sim'], reverse=True)
        beam = []
        chosen_notes = set()
        for ing in sorted_all:
            if ing['note_type'] not in chosen_notes:
                beam.append(ing)
                chosen_notes.add(ing['note_type'])
                if len(beam) >= 3:
                    break
        if not beam:
            beam = sorted_all[:3]
        selected = list(beam)
        selected_ids = {ing['id'] for ing in selected}
    else:
        start_pool = note_pools.get(start_note, note_pools['middle'])
        # 상위 후보 + 랜덤 탐색 후보
        sorted_start = sorted(start_pool, key=lambda x: x['_theme_sim'], reverse=True)
        top_picks = sorted_start[:beam_width]
        
        # 탐색: beam_width 중 20%는 랜덤
        explore_count = max(1, beam_width // 5)
        exploit_count = beam_width - explore_count
        
        exploit = sorted_start[:exploit_count]
        remaining = sorted_start[exploit_count:]
        if remaining and explore_count > 0:
            explore = random.sample(remaining[:50], min(explore_count, len(remaining[:50])))
        else:
            explore = []
        
        selected = exploit + explore
        selected_ids = {ing['id'] for ing in selected}
    
    # 확장 순서
    if start_note == 'top':
        order = ['middle', 'base']
    elif start_note == 'base':
        order = ['middle', 'top']
    elif start_note == 'middle':
        order = ['top', 'base']
    elif start_note == 'relay':
        note_counts = {}
        for s in selected:
            note_counts[s['note_type']] = note_counts.get(s['note_type'], 0) + 1
        order = sorted(['top', 'middle', 'base'], key=lambda n: note_counts.get(n, 0))
    else:
        note_counts = {}
        for s in selected:
            note_counts[s['note_type']] = note_counts.get(s['note_type'], 0) + 1
        order = sorted(['top', 'middle', 'base'], key=lambda n: note_counts.get(n, 0))
    
    note_targets = {
        'top': max(2, target_count // 4),
        'middle': max(3, target_count // 3),
        'base': max(3, target_count // 3),
    }
    
    for note in order:
        current_count = sum(1 for s in selected if s['note_type'] == note)
        need = note_targets[note] - current_count
        if need <= 0:
            continue
        candidates = [ing for ing in note_pools[note] if ing['id'] not in selected_ids]
        
        # 후보가 많으면 상위 100 + 랜덤 20으로 축소 (6k 전체 평가 방지)
        if len(candidates) > 150:
            sorted_cands = sorted(candidates, key=lambda x: x['_theme_sim'], reverse=True)
            top_cands = sorted_cands[:100]
            rest = sorted_cands[100:]
            random_cands = random.sample(rest, min(20, len(rest)))
            candidates = top_cands + random_cands
        
        for _ in range(need):
            if not candidates:
                break
            
            # 탐색/착취: 20% 확률로 랜덤 선택
            if random.random() < 0.2 and len(candidates) > 5:
                best_ing = random.choice(candidates[:20])  # 상위 20 중 랜덤
            else:
                best_score = -1
                best_ing = None
                for cand in candidates:
                    total = candidate_score(cand, selected)
                    if total > best_score:
                        best_score = total
                        best_ing = cand
            
            if best_ing:
                selected.append(best_ing)
                selected_ids.add(best_ing['id'])
                candidates = [c for c in candidates if c['id'] != best_ing['id']]
    
    while len(selected) < target_count:
        remaining = [ing for ing in pool if ing['id'] not in selected_ids]
        if not remaining:
            break
        # 상위 50만 평가
        remaining_sorted = sorted(remaining, key=lambda x: x['_theme_sim'], reverse=True)[:50]
        best_ing = max(remaining_sorted, key=lambda c: candidate_score(c, selected))
        selected.append(best_ing)
        selected_ids.add(best_ing['id'])
    
    return selected


def optimize_concentrations(selected, theme_vec):
    """배합비 최적화 (결정론적) + ★ Fix #5: 소프트 농도 캡"""
    NOTE_PCT = {'top': 15, 'middle': 35, 'base': 50}
    MAX_SINGLE_PCT = 20.0  # ★ 개별 원료 최대 20%
    groups = {'top': [], 'middle': [], 'base': []}
    for ing in selected:
        nt = ing.get('note_type', 'middle')
        if nt in groups:
            groups[nt].append(ing)
        else:
            groups['middle'].append(ing)
    
    result = []
    for note, note_ings in groups.items():
        if not note_ings:
            continue
        target_pct = NOTE_PCT[note]
        weights = []
        for ing in note_ings:
            theme_w = max(0.1, ing['_theme_sim'])
            strength = max(1, ing.get('strength') or 5)
            w = theme_w * (10.0 / strength)
            weights.append(w)
        total_w = sum(weights)
        
        # 초기 농도 계산
        pcts = []
        for w in weights:
            pct = (w / total_w) * target_pct if total_w > 0 else target_pct / len(note_ings)
            pcts.append(pct)
        
        # ★ 소프트 캡: 20% 초과분 재분배
        excess = 0.0
        capped_count = 0
        for i, pct in enumerate(pcts):
            if pct > MAX_SINGLE_PCT:
                excess += pct - MAX_SINGLE_PCT
                pcts[i] = MAX_SINGLE_PCT
                capped_count += 1
        if excess > 0 and len(pcts) > capped_count:
            uncapped = [i for i, p in enumerate(pcts) if p < MAX_SINGLE_PCT]
            share = excess / len(uncapped) if uncapped else 0
            for i in uncapped:
                pcts[i] += share
        
        for ing, pct in zip(note_ings, pcts):
            result.append({
                'id': ing['id'], 'name_en': ing['name_en'], 'name_ko': ing['name_ko'],
                'category': ing['category'], 'note_type': note,
                'concentration': round(pct, 2), 'smiles': ing['smiles'],
                'theme_relevance': round(ing['_theme_sim'], 3),
                'price_usd_kg': ing.get('price'),
                'availability': ing.get('availability', 'unknown'),
            })
    return result


# ═══════════════════════════════════════
# [5] 농도 미세 최적화 (Fine Coordinate Search)
# ═══════════════════════════════════════
def optimize_concentrations_fine(recipe_ings, max_rounds=3):
    """
    ★ 좌표 탐색으로 농도 미세 조정 (개선판)
    - ±0.3% 단계 (기존 ±1% → 3배 미세)
    - 다기준 목적함수: hedonic × 0.4 + longevity × 0.3 + smoothness × 0.3
    - 최대 3라운드
    """
    ings = [dict(i) for i in recipe_ings]  # 복사
    smiles = [i['smiles'] for i in ings]
    STEP = 0.3  # 미세 단계
    
    def eval_score(concs):
        try:
            r = biophys.simulate_recipe(smiles, concs)
            h = r['hedonic']['hedonic_score']
            l = min(1.0, r['thermodynamics']['longevity_hours'] / 8)
            s = r['thermodynamics'].get('smoothness', 0.5)
            return h * 0.4 + l * 0.3 + s * 0.3
        except:
            return 0.0
    
    concs = [i['concentration'] for i in ings]
    best_score = eval_score(concs)
    improvements = []
    
    for rnd in range(max_rounds):
        improved = False
        for idx in range(len(concs)):
            original = concs[idx]
            best_local = best_score
            best_concs = None
            best_dir = None
            
            # 여러 단계 시도: ±0.3%, ±0.6%, ±1.0%
            for delta in [STEP, STEP * 2, STEP * 3.3]:
                for direction, sign in [('↑', 1), ('↓', -1)]:
                    trial = list(concs)
                    trial[idx] = max(0.3, min(60, original + sign * delta))
                    total = sum(trial)
                    trial = [c * 100 / total for c in trial]
                    
                    score = eval_score(trial)
                    if score > best_local:
                        best_local = score
                        best_concs = trial
                        best_dir = direction
            
            if best_concs is not None:
                improvement = best_local - best_score
                concs = best_concs
                best_score = best_local
                improved = True
                name = ings[idx]['name_en'] or ings[idx]['name_ko']
                improvements.append(f"{name} {best_dir} (+{improvement:.4f})")
        
        if not improved:
            break
    
    for i, c in enumerate(concs):
        ings[i]['concentration'] = round(c, 2)
    
    return ings, improvements


# ═══════════════════════════════════════
# 자기 성찰 (Reflect + Refine)
# ═══════════════════════════════════════
def reflect_on_recipe(recipe_ings, target_profile, gnn_engine):
    """GNN 자체 평가 → 약점 진단"""
    reasoning = []
    smiles_list = [i['smiles'] for i in recipe_ings]
    
    odor_vecs = np.array([_cached_odor_vec(s, gnn_engine) for s in smiles_list])
    concs = np.array([i['concentration'] for i in recipe_ings], dtype=np.float32)
    weights = concs / (concs.sum() + 1e-8)
    mixture_vec = np.average(odor_vecs, weights=weights, axis=0)
    
    gap = target_profile - mixture_vec
    weak_dims = []
    for i in np.argsort(gap)[::-1]:
        if gap[i] > 0.05:
            weak_dims.append((i, ODOR_DIMENSIONS[i], float(gap[i])))
        if len(weak_dims) >= 3:
            break
    
    top3 = [(ODOR_DIMENSIONS[i], round(float(mixture_vec[i]), 3)) for i in np.argsort(mixture_vec)[-3:][::-1]]
    reasoning.append(f"혼합향: {', '.join(f'{n}({v})' for n, v in top3)}")
    
    if weak_dims:
        reasoning.append(f"약점: {', '.join(f'{n}(부족 {g:.2f})' for _, n, g in weak_dims)}")
    else:
        reasoning.append("약점 없음")
    
    return {'weak_dims': weak_dims, 'mixture_vec': mixture_vec, 'reasoning': reasoning}


def refine_recipe(selected, recipe_ings, pool, theme_vec, reflection, reasoning_log):
    """약점 보강"""
    weak_dims = reflection['weak_dims']
    if not weak_dims:
        reasoning_log.append("수정 불필요")
        return selected, recipe_ings, []
    
    selected_ids = {ing['id'] for ing in selected}
    changes = []
    
    for dim_idx, dim_name, gap in weak_dims[:2]:
        candidates = [i for i in pool if i['id'] not in selected_ids
                       and dim_name in ' '.join(i.get('descriptors', []))]
        if not candidates:
            continue
        
        worst_idx, worst_ing = min(
            enumerate(selected), key=lambda x: x[1]['_theme_sim']
        )
        best_cand = max(candidates, key=lambda c: c['_theme_sim'] + avg_synergy_with(c, selected) * 0.5)
        
        selected_ids.discard(worst_ing['id'])
        selected_ids.add(best_cand['id'])
        selected[worst_idx] = best_cand
        
        old_n = worst_ing['name_en'] or worst_ing['name_ko']
        new_n = best_cand['name_en'] or best_cand['name_ko']
        changes.append(f"{dim_name} 보강: {old_n} → {new_n}")
    
    recipe_ings = optimize_concentrations(selected, theme_vec)
    return selected, recipe_ings, changes


# ═══════════════════════════════════════
# [6] 릴레이 조향
# ═══════════════════════════════════════
def relay_compose(theme_vec, pool, all_tags, tag_to_idx):
    """
    ★ 릴레이 조향: 3개 서브 AI가 순차 구축
    Sub-A: Top 노트 4개 선택
    Sub-B: Top을 보고 Middle 4개 선택
    Sub-C: Top+Middle을 보고 Base 4개 선택
    """
    reasoning = ["릴레이 조향 시작"]
    
    # Sub-A: Top 노트만
    top_pool = [i for i in pool if i['note_type'] == 'top']
    for i in pool:
        i['_theme_sim'] = cosine_sim(theme_vec, i['vec'])
    
    top_sorted = sorted(top_pool, key=lambda x: x['_theme_sim'], reverse=True)
    top_selected = top_sorted[:4]
    reasoning.append(f"Sub-A: Top {len(top_selected)}개 선택")
    
    # Sub-B: Middle (Top과 시너지 높은 것)
    mid_pool = [i for i in pool if i['note_type'] == 'middle' and i['id'] not in {s['id'] for s in top_selected}]
    mid_scored = sorted(mid_pool, key=lambda c:
        c['_theme_sim'] * 0.3 + avg_synergy_with(c, top_selected) * 0.7,
        reverse=True
    )
    mid_selected = mid_scored[:4]
    reasoning.append(f"Sub-B: Middle {len(mid_selected)}개 선택 (Top과 시너지 기반)")
    
    # Sub-C: Base (Top+Middle과 조화)
    prior = top_selected + mid_selected
    base_pool = [i for i in pool if i['note_type'] == 'base' and i['id'] not in {s['id'] for s in prior}]
    base_scored = sorted(base_pool, key=lambda c:
        c['_theme_sim'] * 0.25 + avg_synergy_with(c, prior) * 0.75,
        reverse=True
    )
    base_selected = base_scored[:4]
    reasoning.append(f"Sub-C: Base {len(base_selected)}개 선택 (전체 조화 기반)")
    
    selected = top_selected + mid_selected + base_selected
    return selected, reasoning


# ═══════════════════════════════════════
# ★ 자율 전략 선택 (테마 분석 기반)
# ═══════════════════════════════════════
def _auto_select_strategy(theme_vec, keywords, composer_id):
    """
    ★ 테마 벡터 분석으로 조향 전략을 자율 결정
    하드코딩 없이 데이터에서 전략 도출
    """
    import random
    
    # 테마 벡터에서 주요 차원 분석
    # 각 노트 타입에 관련된 odor 차원 인덱스
    top_dims = {'fresh': 7, 'citrus': 4, 'green': 8, 'herbal': 14}   # 휘발성 높은 노트
    base_dims = {'woody': 2, 'musk': 6, 'amber': 15, 'leather': 16}  # 지속력 높은 노트
    mid_dims = {'floral': 3, 'spicy': 5, 'powdery': 12, 'fruity': 10} # 중심 노트
    
    # 각 노트 그룹의 테마 강도 계산
    n_tags = len(theme_vec)
    
    def group_strength(dim_map):
        total = 0
        for dim_name, dim_idx in dim_map.items():
            if dim_idx < n_tags:
                total += theme_vec[dim_idx]
        return total
    
    top_score = group_strength(top_dims)
    base_score = group_strength(base_dims)
    mid_score = group_strength(mid_dims)
    
    # 점수로 전략 순위 결정
    strategies = [
        ('top', top_score),
        ('base', base_score),
        ('middle', mid_score),
    ]
    strategies.sort(key=lambda x: x[1], reverse=True)
    
    # 4개 조향사 각각 다른 전략
    if composer_id <= 3:
        # 상위 3개는 순위 기반 (but avoiding ties)
        if composer_id <= len(strategies):
            return strategies[composer_id - 1][0]
        return strategies[0][0]
    else:
        # 4번째: relay 또는 랜덤 탐색
        all_options = ['top', 'middle', 'base', 'relay', 'all']
        return random.choice(all_options)


# ═══════════════════════════════════════
# 조향 AI 실행
# ═══════════════════════════════════════
def run_composer(composer_id, theme_text, pool, all_tags, tag_to_idx, output_dir,
                 gnn_engine=None, do_fine_optimize=False):
    """자율 조향 AI — Think → Compose → Reflect → Refine → [5] Fine-Optimize"""
    import random
    REFINE_ROUNDS = 2
    
    t0 = time.time()
    reasoning_log = []
    
    # THINK — 테마 분석
    theme_vec, keywords = theme_to_vector(theme_text, all_tags, tag_to_idx)
    target_profile = theme_to_target_profile(theme_text, keywords)
    reasoning_log.append(f"테마 '{theme_text}' → 키워드: {', '.join(keywords)}")
    
    # ★ 자율 전략 선택 (테마 기반, 하드코딩 X)
    start_note = _auto_select_strategy(theme_vec, keywords, composer_id)
    
    print(f"\n  🎭 조향 AI #{composer_id} ({start_note} 전략 — 자율 선택)")
    reasoning_log.append(f"자율 전략: {start_note} (테마 분석 기반)")
    
    # COMPOSE
    if start_note == 'relay':
        selected, relay_reasons = relay_compose(theme_vec, pool, all_tags, tag_to_idx)
        reasoning_log.extend(relay_reasons)
    else:
        selected = beam_search_compose(theme_vec, pool, start_note, beam_width=6, target_count=12)
    
    recipe_ings = optimize_concentrations(selected, theme_vec)
    reasoning_log.append(f"초기 구성: {len(selected)}개 원료")
    print(f"     📝 {len(selected)}개 원료 구성")
    
    # REFLECT → REFINE 루프
    round_scores = []
    all_changes = []
    
    for rnd in range(REFINE_ROUNDS):
        if gnn_engine is None:
            break
        reflection = reflect_on_recipe(recipe_ings, target_profile, gnn_engine)
        reasoning_log.extend(reflection['reasoning'])
        
        smiles_list = [i['smiles'] for i in recipe_ings]
        concs = [i['concentration'] for i in recipe_ings]
        try:
            sim = biophys.simulate_recipe(smiles_list, concs)
            score = sim['hedonic']['hedonic_score']
        except:
            score = 0.5
        round_scores.append(round(score, 4))
        
        if reflection['weak_dims']:
            selected, recipe_ings, changes = refine_recipe(
                selected, recipe_ings, pool, theme_vec, reflection, reasoning_log
            )
            all_changes.extend(changes)
            if changes:
                print(f"     🔄 라운드 {rnd+1}: {', '.join(changes)}")
            else:
                break
        else:
            print(f"     ✅ 라운드 {rnd+1}: 자체 평가 통과")
            break
    
    # [5] 농도 미세 최적화 (위너 후보만)
    if do_fine_optimize:
        print(f"     🔬 농도 미세 최적화...")
        recipe_ings, fine_improvements = optimize_concentrations_fine(recipe_ings, max_rounds=1)
        if fine_improvements:
            reasoning_log.extend(fine_improvements)
            print(f"     ✅ 최적화: {len(fine_improvements)}건 조정")
    
    # 최종 시뮬레이션
    smiles_list = [i['smiles'] for i in recipe_ings]
    conc_list = [i['concentration'] for i in recipe_ings]
    try:
        sim_result = biophys.simulate_recipe(smiles_list, conc_list)
        scores = {
            'hedonic_score': sim_result['hedonic']['hedonic_score'],
            'longevity_hours': sim_result['thermodynamics']['longevity_hours'],
            'smoothness': sim_result['thermodynamics'].get('smoothness', 0.5),
            'active_receptors': sim_result['nose']['active_receptors'],
        }
        scores['total_score'] = (
            scores['hedonic_score'] * 0.35 +
            min(1.0, scores['longevity_hours'] / 6) * 0.25 +
            scores['smoothness'] * 0.15 +
            min(1.0, scores['active_receptors'] / 120) * 0.25
        )
    except:
        scores = {'hedonic_score': 0.5, 'longevity_hours': 4.0, 'smoothness': 0.5,
                  'active_receptors': 60, 'total_score': 0.5}
    
    round_scores.append(round(scores['hedonic_score'], 4))
    overall_harmony = harmony_score([s['vec'] for s in selected])
    
    elapsed = time.time() - t0
    
    # 시너지 통계
    syn_scores = []
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            syn_scores.append(get_synergy(selected[i]['id'], selected[j]['id']))
    avg_syn = float(np.mean(syn_scores)) if syn_scores else 0.5
    
    reasoning_log.append(f"최종: 점수 {scores['total_score']:.4f} | 조화 {overall_harmony:.3f} | 시너지 {avg_syn:.3f}")
    
    recipe = {
        'composer_id': composer_id,
        'strategy': start_note,
        'theme': theme_text, 'theme_keywords': keywords,
        'ingredients': recipe_ings,
        'scores': {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()},
        'harmony': round(overall_harmony, 4),
        'avg_synergy': round(avg_syn, 4),
        'reasoning_trace': reasoning_log,
        'refinement_rounds': len(round_scores) - 1,
        'round_scores': round_scores,
        'changes_made': all_changes,
        'elapsed_sec': round(elapsed, 2),
        'ingredient_count': len(recipe_ings),
        'note_distribution': {
            'top': sum(1 for i in recipe_ings if i['note_type'] == 'top'),
            'middle': sum(1 for i in recipe_ings if i['note_type'] == 'middle'),
            'base': sum(1 for i in recipe_ings if i['note_type'] == 'base'),
        },
        'total_cost_estimate': round(sum(
            float(i.get('price_usd_kg') or 0) * i['concentration'] / 100
            for i in recipe_ings
        ), 2),
    }
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'recipe_{composer_id}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"     점수: {scores['total_score']:.3f} | 조화: {overall_harmony:.3f} | 시너지: {avg_syn:.3f}")
    print(f"     시간: {elapsed:.2f}s")
    
    return recipe


# ═══════════════════════════════════════
# 메인
# ═══════════════════════════════════════
def main(theme_file=None, theme_text=None, output_dir=None, gnn_engine=None):
    if output_dir is None:
        output_dir = os.path.join('weights', 'ai_perfumer', 'composers')
    if theme_text is None:
        if theme_file and os.path.exists(theme_file):
            with open(theme_file, 'r', encoding='utf-8') as f:
                theme_text = f.read().strip()
        else:
            theme_text = "봄날의 벚꽃 향수"
    
    pool, all_tags, tag_to_idx = load_pool()
    
    # [4] 시너지 DB 구축
    build_synergy_db(pool)
    
    recipes = []
    for cid in range(1, 5):
        recipe = run_composer(cid, theme_text, pool, all_tags, tag_to_idx, output_dir, gnn_engine)
        recipes.append(recipe)
    
    return recipes, pool, all_tags, tag_to_idx


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    main(theme_text=args.theme, output_dir=args.output)
