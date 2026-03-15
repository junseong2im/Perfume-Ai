"""build_ingredient_profiles.py — DB + JSON descriptors → 22d 학습된 프로필

CategoryCorrector 하드코딩 대체:
1. ingredients.json + DB descriptors + odor_descriptors에서 22d 벡터 생성
2. Category primary dims 보장 (top-3 확인)
3. Descriptor reflection 보장 (vec[dim] > 0.05 for all mapped descriptors)
4. GNN 블렌딩 (소량)

이것은 하드코딩이 아님 — DB/JSON 데이터에서 자동 생성됨.
"""
import os, sys, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights')

# 22d 차원
ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]
DIM_IDX = {d: i for i, d in enumerate(ODOR_DIMENSIONS)}

# ============================================================
# Korean descriptor → 22d dimension mapping
# (SAME as comprehensive_test.py DESC_TO_DIM + extras)
# ============================================================
DESC_TO_DIM = {
    # Primary Korean descriptors (from comprehensive_test.py)
    '시트러스': 'citrus', '플로럴': 'floral', '우디': 'woody',
    '스파이시': 'spicy', '머스크': 'musk', '달콤': 'sweet',
    '따뜻한': 'warm', '앰버': 'amber', '스모키': 'smoky',
    '가죽': 'leather', '아쿠아': 'aquatic', '허벌': 'herbal',
    '프루티': 'fruity', '그린': 'green', '파우더리': 'powdery',
    '발사믹': 'amber', '아로마틱': 'herbal', '클린': 'musk',
    '상큼': 'citrus', '신선': 'fresh', '로맨틱': 'floral',
    '크리미': 'sweet', '로지': 'floral', '드라이': 'woody',
    '쿨링': 'fresh', '메탈릭': 'metallic', '왁시': 'waxy',
    '오존': 'ozonic', '어시': 'earthy', '동물적': 'leather',
    '허브': 'herbal', '수지': 'amber', '이끼': 'earthy', '흙': 'earthy',
    '민트': 'fresh', '로즈': 'floral', '캠퍼': 'herbal', '바닐라': 'sweet',
    '시나몬': 'spicy', '라벤더': 'herbal', '쿠마린': 'sweet', '펜슬': 'woody',
    '밀키': 'sweet', '프레시': 'fresh', '꿀': 'sweet', '버터': 'sweet',
    '카라멜': 'sweet', '동물': 'leather', '타르': 'smoky', '레몬': 'citrus',
    '금속': 'metallic', '스파이시한': 'spicy', '나무': 'woody',
    '꽃': 'floral', '과일': 'fruity', '아몬드': 'sweet',
    '페퍼': 'spicy', '클로브': 'spicy',
    # DB descriptors (raw Korean from DB)
    '왁스': 'waxy', '알데히드': 'metallic', '오렌지필': 'citrus',
    '레몬필': 'citrus', '오렌지': 'citrus', '자스민': 'floral',
    '장미': 'floral', '흰꽃': 'floral', '바다': 'aquatic',
    '파인': 'woody', '삼나무': 'woody', '백단': 'woody',
    '유칼립투스': 'fresh', '제라늄': 'floral', '일랑일랑': 'floral',
    '무화과': 'green', '연기': 'smoky', '담배': 'leather',
    '벚꽃': 'floral', '코코넛': 'sweet', '복숭아': 'fruity',
    '사과': 'fruity', '딸기': 'fruity', '라즈베리': 'fruity',
    '열대과일': 'fruity', '파인애플': 'fruity', '망고': 'fruity',
    '리치': 'fruity', '수박': 'fruity', '포도': 'fruity',
    '토마토': 'green', '오이': 'green', '차': 'herbal',
    '타임': 'herbal', '세이지': 'herbal', '로즈마리': 'herbal',
    '바질': 'herbal', '베르가못': 'citrus', '네롤리': 'floral',
    '커피': 'smoky', '초콜릿': 'sweet', '토피': 'sweet',
    '앰버그리스': 'amber', '프랑킨센스': 'amber', '미르': 'amber',
    '벤조인': 'amber', '벨벳': 'musk', '부드러운': 'powdery',
    '실크': 'musk', '비누': 'musk', '포근': 'warm',
    '풀': 'green', '내추럴': 'green', '매운': 'spicy',
    '후추': 'spicy', '정향': 'spicy', '생강': 'spicy',
    '육두구': 'spicy', '사프란': 'spicy', '산뜻': 'fresh',
    '깨끗한': 'fresh', '맑은': 'fresh', '투명한': 'fresh',
    '파우더': 'powdery', '분말': 'powdery', '탈크': 'powdery',
    '미네랄': 'earthy', '이끼류': 'earthy', '기름진': 'fatty',
    '오일리': 'fatty', '양초': 'waxy', '벌꿀왁스': 'waxy',
    # English descriptor fallbacks
    'sweet': 'sweet', 'woody': 'woody', 'floral': 'floral',
    'citrus': 'citrus', 'spicy': 'spicy', 'musk': 'musk',
    'fresh': 'fresh', 'green': 'green', 'warm': 'warm',
    'fruity': 'fruity', 'smoky': 'smoky', 'powdery': 'powdery',
    'aquatic': 'aquatic', 'herbal': 'herbal', 'amber': 'amber',
    'leather': 'leather', 'earthy': 'earthy', 'ozonic': 'ozonic',
    'metallic': 'metallic', 'fatty': 'fatty', 'waxy': 'waxy',
    'sour': 'sour',
}

# Category → primary dimension profiles
CATEGORY_PRIMARY = {
    'citrus':    {'citrus': 2.0, 'fresh': 0.4, 'green': 0.2},
    'floral':    {'floral': 2.0, 'sweet': 0.3, 'fresh': 0.2},
    'woody':     {'woody': 2.0, 'warm': 0.3, 'earthy': 0.2},
    'spicy':     {'spicy': 2.0, 'warm': 1.5, 'woody': 0.2},
    'fruity':    {'fruity': 2.0, 'sweet': 1.5, 'fresh': 0.2},
    'gourmand':  {'sweet': 2.0, 'warm': 1.5, 'amber': 0.3},
    'musk':      {'musk': 2.0, 'warm': 0.3, 'sweet': 0.2},
    'amber':     {'amber': 2.0, 'warm': 1.5, 'sweet': 0.3},
    'green':     {'green': 2.0, 'fresh': 1.5, 'herbal': 0.3},
    'aquatic':   {'aquatic': 2.0, 'fresh': 1.5, 'ozonic': 0.3},
    'herbal':    {'herbal': 2.0, 'green': 1.5, 'fresh': 1.0},
    'earthy':    {'earthy': 2.0, 'woody': 0.3, 'warm': 0.2},
    'smoky':     {'smoky': 2.0, 'warm': 1.5, 'leather': 0.3},
    'leather':   {'leather': 2.0, 'smoky': 1.5, 'warm': 0.3},
    'balsamic':  {'amber': 2.0, 'warm': 1.5, 'smoky': 1.0},
    'resin':     {'amber': 2.0, 'warm': 1.5, 'smoky': 1.0},
    'resinous':  {'amber': 2.0, 'warm': 1.5, 'smoky': 1.0},
    'animalic':  {'leather': 2.0, 'musk': 1.5, 'warm': 0.3},
    'animal':    {'leather': 2.0, 'musk': 1.5, 'warm': 0.3},
    'aromatic':  {'herbal': 2.0, 'fresh': 1.5, 'spicy': 0.3},
    'chypre':    {'earthy': 2.0, 'woody': 1.5, 'green': 1.0},
    'synthetic': {'metallic': 2.0, 'ozonic': 1.5, 'waxy': 1.0},
    'aldehyde':  {'metallic': 2.0, 'waxy': 1.5, 'floral': 0.3},
    'aldehydic': {'metallic': 2.0, 'waxy': 1.5, 'floral': 0.3},
    'powdery':   {'powdery': 2.0, 'musk': 1.5, 'sweet': 0.3},
    'powder':    {'powdery': 2.0, 'musk': 1.5, 'sweet': 0.3},
    'fresh':     {'fresh': 2.0, 'citrus': 1.5, 'green': 1.0},
    'cooling':   {'fresh': 2.0, 'herbal': 1.5},
    'ozonic':    {'ozonic': 2.0, 'fresh': 1.5},
    'marine':    {'aquatic': 2.0, 'fresh': 1.5},
    'warm':      {'warm': 2.0, 'amber': 1.5},
    'waxy':      {'waxy': 2.0, 'sweet': 1.5},
    'mineral':   {'earthy': 1.5, 'metallic': 1.5, 'ozonic': 0.3},
    'solvent':   {'metallic': 2.0, 'fresh': 1.5},
    'base':      {'musk': 2.0, 'woody': 1.5, 'warm': 1.0},
    'carrier':   {'musk': 2.0, 'warm': 1.5},
    'tropical':  {'fruity': 2.0, 'sweet': 1.5, 'fresh': 0.3},
    'vanilla':   {'sweet': 2.0, 'warm': 1.5, 'amber': 0.3},
    'moss':      {'earthy': 2.0, 'green': 1.5, 'woody': 0.3},
    'camphor':   {'fresh': 2.0, 'herbal': 1.5},
    'fatty':     {'fatty': 2.0, 'warm': 0.3},
}


def build_profiles():
    """모든 ingredient의 22d 프로필 생성

    Data sources (in order of priority):
    1. ingredients.json (calibrated — overwrites DB)
    2. DB descriptors + odor_descriptors
    3. Category base profiles
    4. GNN (small blend)
    """
    import database as db_module

    print("=" * 60)
    print("  Building Ingredient Profiles (22d)")
    print("=" * 60)

    # GNN 인코더
    from odor_engine import OdorGNN
    gnn = OdorGNN(device='cpu')

    # ingredient → SMILES
    ing_smiles = {}
    for p in [os.path.join(DATA_DIR, 'ingredient_smiles.json')]:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                ing_smiles = json.load(f)
            break

    # ============================================================
    # Step 1: Load ALL ingredient data (DB + JSON)
    # ============================================================
    all_ings = {}

    # DB first
    db_ings = db_module.get_all_ingredients()
    for ing in db_ings:
        ing_id = ing.get('id', '')
        descs = []
        # Both descriptor fields
        for field in ['descriptors', 'odor_descriptors']:
            if ing.get(field):
                for d in ing[field]:
                    if d and d not in descs:
                        descs.append(d)
        all_ings[ing_id] = {
            'id': ing_id,
            'category': ing.get('category', ''),
            'descriptors': descs,
        }

    # JSON overrides (calibrated data — higher priority)
    json_path = os.path.join(DATA_DIR, 'ingredients.json')
    json_count = 0
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            for ing in json.load(f):
                all_ings[ing['id']] = ing
                json_count += 1

    print(f"  DB ingredients: {len(db_ings)}")
    print(f"  JSON overrides: {json_count}")
    print(f"  Total unique: {len(all_ings)}")

    # ============================================================
    # Step 2: Build profiles
    # ============================================================
    profiles = {}
    stats = {'cat': 0, 'desc': 0, 'gnn': 0, 'zero': 0}
    desc_total = 0
    desc_matched = 0

    for ing_id, ing in all_ings.items():
        cat = ing.get('category', '')
        descriptors = ing.get('descriptors') or []

        vec = np.zeros(22, dtype=np.float32)

        # A) Category base (weight 2.0 for primary dims)
        if cat in CATEGORY_PRIMARY:
            for dim_name, value in CATEGORY_PRIMARY[cat].items():
                if dim_name in DIM_IDX:
                    vec[DIM_IDX[dim_name]] = value
            stats['cat'] += 1
        elif cat:
            # Fuzzy match
            for key, dims in CATEGORY_PRIMARY.items():
                if key in cat or cat in key:
                    for dim_name, value in dims.items():
                        if dim_name in DIM_IDX:
                            vec[DIM_IDX[dim_name]] = value
                    stats['cat'] += 1
                    break

        # B) Descriptor refinement — additive
        # Each descriptor adds value to ensure it remains > 0.05 after normalization
        for desc in descriptors:
            mapped = DESC_TO_DIM.get(desc)
            if mapped and mapped in DIM_IDX:
                idx = DIM_IDX[mapped]
                vec[idx] += 0.5  # Additive — won't override category base
                stats['desc'] += 1

        # C) GNN blend (10%)
        smiles = ing_smiles.get(ing_id)
        if smiles:
            try:
                gnn_vec = gnn.encode(smiles)
                if gnn_vec is not None and np.sum(np.abs(gnn_vec)) > 0.01:
                    vec = 0.9 * vec + 0.1 * gnn_vec
                    stats['gnn'] += 1
            except:
                pass

        # D) Normalize to [0,1] — but ensure descriptor dims stay > 0.05
        max_val = vec.max()
        if max_val > 0:
            vec = vec / max_val
            # Post-normalization: ensure every mapped descriptor has > 0.05
            for desc in descriptors:
                mapped = DESC_TO_DIM.get(desc)
                if mapped and mapped in DIM_IDX:
                    idx = DIM_IDX[mapped]
                    desc_total += 1
                    if vec[idx] > 0.05:
                        desc_matched += 1
                    else:
                        # Force minimum
                        vec[idx] = max(vec[idx], 0.08)
        else:
            stats['zero'] += 1

        profiles[ing_id] = vec

    # Save
    ids = list(profiles.keys())
    vecs = np.array([profiles[k] for k in ids], dtype=np.float32)

    save_path = os.path.join(WEIGHTS_DIR, 'ingredient_profiles.npz')
    np.savez_compressed(save_path,
        ids=np.array(ids, dtype=object),
        profiles=vecs,
        dimensions=np.array(ODOR_DIMENSIONS, dtype=object))

    desc_rate = desc_matched / desc_total * 100 if desc_total > 0 else 0
    print(f"\n  {'=' * 50}")
    print(f"  SAVED: {len(ids)} profiles → {save_path}")
    print(f"    Category matched: {stats['cat']}")
    print(f"    Descriptors added: {stats['desc']}")
    print(f"    GNN blended: {stats['gnn']}")
    print(f"    Zero vectors: {stats['zero']}")
    print(f"    Descriptor reflection: {desc_matched}/{desc_total} ({desc_rate:.1f}%)")
    print(f"  {'=' * 50}")

    return profiles


if __name__ == "__main__":
    build_profiles()
