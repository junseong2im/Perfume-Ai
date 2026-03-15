"""
Biophysics Simulation Training Pipeline
========================================
300회 시뮬레이션: 유명 향수 → SMILES 매핑 → simulate_recipe → 유사도/신뢰도 검사
Option A: Count FP + ChemBERTa embedding + Metric Learning
"""
import json, os, sys, time, math, csv, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import biophysics_simulator as biophys
from precompute_bert import ChemBERTaCache

# RDKit imports for Morgan FP
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ChemBERTa 캐시 (384d molecular embeddings)
_bert_cache = None
def _get_bert():
    global _bert_cache
    if _bert_cache is None:
        _bert_cache = ChemBERTaCache()
        _bert_cache.load()
    return _bert_cache

# Lazy DB import
db = None
def _get_db():
    global db
    if db is None:
        try:
            import database as _db
            db = _db
        except Exception:
            db = False
    return db if db else None

# ==================
# 1) Data Loading
# ==================
def load_famous_perfumes():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'famous_perfumes.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_smiles_map():
    """DB ingredient → SMILES mapping (same as SelfPlayRL._load_ingredients)"""
    _db = _get_db()
    if _db is None:
        print("[WARNING] DB unavailable, using category-based SMILES only")
        return {}
    try:
        ings = _db.get_all_ingredients()
        mols = _db.get_all_molecules(limit=1000)
    except Exception as e:
        print(f"[WARNING] DB unavailable ({e}), using category-based SMILES only")
        return {}
    mol_smiles = {}
    for mol in mols:
        name = (mol.get('name') or '').lower()
        if name and mol.get('smiles'):
            mol_smiles[name] = mol['smiles']

    ing_map = {}  # id → {smiles, category, note_type}
    for ing in ings:
        name_en = (ing.get('name_en') or '').lower()
        iid = ing.get('id', '')
        smiles = None
        for mname, msmiles in mol_smiles.items():
            if name_en and (name_en in mname or mname in name_en):
                smiles = msmiles
                break
        if smiles is None:
            smiles = _category_smiles(ing.get('category', ''))
        ing_map[iid] = {
            'smiles': smiles,
            'category': ing.get('category', ''),
            'note_type': ing.get('note_type', 'middle'),
        }
    return ing_map

# Category fallback SMILES — multi-modal prototypes (Fix 6)
# Each category has 2-5 scaffolds for chemical diversity
CATEGORY_SMILES = {
    'citrus':   ['CC(=CCC/C(=C\\C)C)C',         # limonene
                 'CC(=O)OCC=C(C)C',              # linalyl acetate
                 'CC(C)=CCCC(C)=CC=O'],           # citral
    'floral':   ['OCC=C(C)CCC=C(C)C',            # linalool
                 'OCC1=CC=CC=C1',                 # phenylethyl alcohol (rose)
                 'O=C(C)c1ccccc1'],               # acetophenone
    'woody':    ['CC1=CCC(CC1)C(C)=C',            # terpinolene
                 'CC1CCC2(C)C(O)CCC12',           # cedrol
                 'CC(C)c1ccc(C)cc1C'],             # iso E super approx
    'oriental': ['O=C/C=C/c1ccccc1',              # cinnamaldehyde
                 'O=Cc1ccc(O)c(OC)c1',            # vanillin
                 'C1CC(=O)OC1c1ccccc1'],           # coumarin
    'musk':     ['O=C1CCCCCCCCCCCCC1',             # cyclopentadecanone (macrocyclic)
                 'CC1(C)CC(=O)c2cc(C)ccc21',       # galaxolide/HHCB approx (polycyclic)
                 'CCCCCCCCCCCCCCCC(=O)OCC',        # ethylene brassylate (macrocyclic)
                 'CC1(C)CCCC2(C)C1CCC1=CC(=O)CCC12'], # ambroxide (woody-ambery)
    'fresh':    ['CC(=CCC/C(=C\\C)C)C',            # limonene
                 'CC(=O)OCC=C(C)C',               # linalyl acetate
                 'OCC=CC'],                        # leaf alcohol (green)
    'spicy':    ['C=CCc1ccc(O)c(OC)c1',            # eugenol
                 'O=C/C=C/c1ccccc1',              # cinnamaldehyde
                 'CC(C)C1CCC(=O)C(C)C1'],          # carvone (mint-spicy)
    'green':    ['CC/C=C\\CCO',                     # cis-3-hexenol
                 'CC(=O)OC/C=C\\CC',               # cis-3-hexenyl acetate
                 'CC(C)=CCCC(C)=CCO'],             # nerol
    'fruity':   ['CCCCOC(=O)CC',                   # ethyl butyrate (pear)
                 'CC(=O)OCCCC',                    # butyl acetate (apple)
                 'CCCCCC(=O)OC',                   # methyl hexanoate (berry)
                 'CCCCC(=O)OCC',                   # ethyl pentanoate (tropical)
                 'CC(=O)OCCCCC'],                  # pentyl acetate (banana)
    'amber':    ['CC1(C)CCCC2(C)C1CCC1=CC(=O)CCC12', # ambroxan-like
                 'c1cc2c(cc1O)OCO2',               # heliotropin
                 'CC(=O)c1ccc(OC)cc1'],             # anisyl acetone
    'vanilla':  ['O=Cc1ccc(O)c(OC)c1',             # vanillin
                 'O=Cc1ccc(OCC)c(OC)c1',           # ethyl vanillin
                 'OC(=O)c1ccccc1O'],               # benzoic acid (benzoin)
    'leather':  ['c1ccc2c(c1)c(=O)[nH]c2=O',      # isatin
                 'c1ccc(c(c1)O)N',                 # 2-aminophenol
                 'Oc1cccc2ccccc12'],               # beta-naphthol
    'gourmand': ['CC1=CC(=O)OC1',                  # sotolon-like
                 'O=C(CC)c1ccccc1',                # propiophenone
                 'O=Cc1ccc(O)c(OC)c1'],            # vanillin
    'aquatic':  ['O=CC1=CC(C)CC1',                 # calone-like
                 'CC1=CCC(C(=O)C)CC1',             # dihydroterpineol (marine)
                 'OCCCCCC=O'],                     # 6-hydroxyhexanal
}

# 키워드 → 프로토타입 인덱스 매핑
_KEYWORD_MAP = {
    'apple': ('fruity', 1), 'pear': ('fruity', 0), 'banana': ('fruity', 4),
    'peach': ('fruity', 3), 'berry': ('fruity', 2), 'raspberry': ('fruity', 2),
    'strawberry': ('fruity', 2), 'pineapple': ('fruity', 3),
    'cherry': ('fruity', 2), 'plum': ('fruity', 2),
    'rose': ('floral', 1), 'jasmine': ('floral', 0), 'lily': ('floral', 2),
    'lemon': ('citrus', 2), 'orange': ('citrus', 0), 'bergamot': ('citrus', 0),
    'lime': ('citrus', 0), 'grapefruit': ('citrus', 0),
    'vanilla': ('vanilla', 0), 'chocolate': ('gourmand', 2),
    'cinnamon': ('oriental', 0), 'tonka': ('oriental', 2),
    'cedar': ('woody', 1), 'sandal': ('woody', 0), 'oak': ('woody', 2),
    'musk': ('musk', 0), 'galaxolide': ('musk', 1), 'ambrox': ('musk', 3),
    'leather': ('leather', 0), 'suede': ('leather', 2),
    'salt': ('aquatic', 0), 'sea': ('aquatic', 0), 'ocean': ('aquatic', 0),
}

def _category_smiles(cat):
    """Multi-modal category SMILES with keyword matching"""
    cat_lower = cat.lower()
    # 1) Keyword match
    for kw, (cat_name, idx) in _KEYWORD_MAP.items():
        if kw in cat_lower:
            protos = CATEGORY_SMILES.get(cat_name, ['CCCCCCO'])
            return protos[min(idx, len(protos) - 1)]
    # 2) Category match — use first prototype
    for key, protos in CATEGORY_SMILES.items():
        if key in cat_lower:
            return protos[0]
    return 'CCCCCCO'  # hexanol fallback

# ==================
# 2) Note → Concentration mapping
# ==================
NOTE_CONC = {'top': 0.15, 'middle': 0.35, 'base': 0.50}

def perfume_to_simulation(perfume, ing_map):
    """Convert a famous perfume record to SMILES list + concentrations"""
    smiles_list = []
    concentrations = []
    matched = 0
    category_matched = 0
    total = 0
    
    for note_type, note_key in [('top', 'top_notes'), ('middle', 'middle_notes'), ('base', 'base_notes')]:
        notes = perfume.get(note_key, [])
        n = len(notes)
        if n == 0:
            continue
        base_conc = NOTE_CONC[note_type]
        per_note = base_conc / n
        
        for note_id in notes:
            total += 1
            if note_id in ing_map:
                smiles_list.append(ing_map[note_id]['smiles'])
                matched += 1
            else:
                # Use category-based fallback
                smi = _category_smiles(note_id)
                smiles_list.append(smi)
                if smi != 'CCCCCCO':  # Not the generic fallback
                    category_matched += 1
            concentrations.append(per_note)
    
    # Coverage: DB match = 1.0, category match = 0.7, generic fallback = 0.0
    if total > 0:
        coverage = (matched * 1.0 + category_matched * 0.7) / total
    else:
        coverage = 0
    return smiles_list, concentrations, coverage

# ==================
# 3) Similarity Metrics (Option A: Count FP + ChemBERTa + Metric Learning)
# ==================
def longevity_similarity(sim_hours, real_longevity_1to5):
    """Real longevity 1-5 → estimated hours, compare with simulation"""
    real_hours = {1: 1.5, 2: 3, 3: 5, 4: 7, 5: 10}[real_longevity_1to5]
    delta = abs(sim_hours - real_hours)
    return max(0, 1.0 - delta / real_hours)

def note_transition_similarity(transitions, perfume):
    """Check if transitions follow expected top→middle→base order"""
    if not transitions:
        return 0.5  # No transitions = neutral
    expected_order = ['top', 'middle', 'base']
    actual_order = [t['from'] for t in transitions] + [transitions[-1]['to']]
    # Remove duplicates maintaining order
    seen = set()
    unique = []
    for n in actual_order:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    # Score based on order correctness
    score = 0
    for i, note in enumerate(unique):
        if i < len(expected_order) and note == expected_order[i]:
            score += 1
    return score / max(len(expected_order), 1)

def hedonic_similarity(hedonic_score, real_rating):
    """Compare hedonic score (0-1) vs real rating (1-5 → 0-1)"""
    normalized_rating = (real_rating - 1) / 4  # 1→0, 5→1
    return 1.0 - abs(hedonic_score - normalized_rating)

# ==================
# Option A ① — Morgan Count FP Tanimoto Similarity
# ==================
def count_fp_mixture(smiles_list, concentrations=None):
    """Generate concentration-weighted count fingerprint for a mixture"""
    if not HAS_RDKIT:
        return None
    n = len(smiles_list)
    if n == 0:
        return None
    concs = concentrations or [1.0 / n] * n
    total_conc = sum(concs)
    if total_conc == 0:
        total_conc = 1.0
    
    fp_sum = None
    for smi, c in zip(smiles_list, concs):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=2048)
        fp_dict = fp.GetNonzeroElements()
        w = c / total_conc
        if fp_sum is None:
            fp_sum = {k: v * w for k, v in fp_dict.items()}
        else:
            for k, v in fp_dict.items():
                fp_sum[k] = fp_sum.get(k, 0) + v * w
    return fp_sum

def count_fp_tanimoto(fp1, fp2):
    """Tanimoto similarity between two count fingerprints (dicts)"""
    if fp1 is None or fp2 is None:
        return 0.5  # neutral fallback
    all_keys = set(fp1.keys()) | set(fp2.keys())
    if not all_keys:
        return 0.5
    dot = sum(fp1.get(k, 0) * fp2.get(k, 0) for k in all_keys)
    norm1 = sum(v * v for v in fp1.values())
    norm2 = sum(v * v for v in fp2.values())
    denom = norm1 + norm2 - dot
    if denom <= 0:
        return 1.0
    return dot / denom

# ==================
# Option A ② — ChemBERTa Mixture Embedding Cosine Similarity
# ==================
def bert_mixture_embedding(smiles_list, concentrations=None):
    """Compute concentration-weighted ChemBERTa embedding for mixture"""
    bert = _get_bert()
    if bert is None or bert.size == 0:
        return None
    n = len(smiles_list)
    if n == 0:
        return None
    concs = concentrations or [1.0 / n] * n
    total_conc = sum(concs)
    if total_conc == 0:
        total_conc = 1.0
    
    emb_sum = np.zeros(bert.hidden_size, dtype=np.float32)
    weight_sum = 0.0
    for smi, c in zip(smiles_list, concs):
        emb = bert.get(smi)
        if emb is not None:
            w = c / total_conc
            emb_sum += emb * w
            weight_sum += w
    
    if weight_sum == 0:
        return None
    emb_sum /= weight_sum  # re-normalize 
    # L2 normalize for cosine
    norm = np.linalg.norm(emb_sum)
    if norm > 0:
        emb_sum /= norm
    return emb_sum

def cosine_similarity(v1, v2):
    """Cosine similarity between two L2-normalized vectors"""
    if v1 is None or v2 is None:
        return 0.5
    return float(np.dot(v1, v2))

# ==================
# Option A ③ — Reweighted Compute Similarity
# ==================
def compute_similarity(sim_result, perfume, coverage, fp_sim_score=0.5, bert_sim_score=0.5):
    """Compute overall similarity with structural metrics (Option A)
    
    Weights:
      longevity     0.20  (was 0.30)
      note_trans    0.15  (was 0.25)
      hedonic       0.20  (was 0.25)
      coverage      0.10  (was 0.20)
      count_fp_sim  0.15  (NEW — vs style centroid)
      bert_sim      0.20  (NEW — vs style centroid)
    """
    # Longevity
    long_sim = longevity_similarity(
        sim_result['thermodynamics']['longevity_hours'],
        perfume['longevity']
    )
    # Note transitions
    note_sim = note_transition_similarity(
        sim_result['thermodynamics'].get('transitions', []),
        perfume
    )
    # Hedonic
    hed_sim = hedonic_similarity(
        sim_result['hedonic']['hedonic_score'],
        perfume['rating']
    )
    
    # Overall (reweighted with structural metrics)
    overall = (long_sim * 0.20 +
               note_sim * 0.15 +
               hed_sim * 0.20 +
               coverage * 0.10 +
               fp_sim_score * 0.15 +
               bert_sim_score * 0.20)
    
    return {
        'overall': round(overall, 4),
        'longevity_sim': round(long_sim, 4),
        'note_transition_sim': round(note_sim, 4),
        'hedonic_sim': round(hed_sim, 4),
        'smiles_coverage': round(coverage, 4),
        'count_fp_sim': round(fp_sim_score, 4),
        'bert_sim': round(bert_sim_score, 4),
    }

# ==================
# 4) Confidence Metrics
# ==================
def compute_confidence(sim_result, coverage):
    """Confidence — 연속 점수 (가우시안 기반, 0~1)"""
    import math

    def gaussian(x, mu, sigma):
        return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

    scores = []

    # 1) SMILES coverage (0~1, 선형)
    scores.append(('smiles_coverage', min(1.0, coverage), coverage))

    # 2) Active receptors — 적정 범위 100±50
    active = sim_result['nose']['active_receptors']
    receptor_score = gaussian(active, 100, 50) if active > 10 else active / 100
    scores.append(('receptor_range', receptor_score, active))

    # 3) Longevity — 적정 범위 4±3h
    hours = sim_result['thermodynamics']['longevity_hours']
    longevity_score = gaussian(hours, 4.0, 3.0) if hours > 0.1 else 0.05
    scores.append(('longevity_range', longevity_score, hours))

    # 4) Hedonic — 적정 범위 0.5±0.3
    hedonic = sim_result['hedonic']['hedonic_score']
    hedonic_score = gaussian(hedonic, 0.55, 0.25)
    scores.append(('hedonic_range', hedonic_score, hedonic))

    # 5) Transition smoothness (0~1, 직접 사용)
    smooth = sim_result['thermodynamics'].get('smoothness', 0)
    scores.append(('smoothness', float(smooth), smooth))

    # 가중 평균
    weights = [0.20, 0.25, 0.25, 0.15, 0.15]
    confidence = sum(w * s for w, (_, s, _) in zip(weights, scores))

    return {
        'confidence': round(float(confidence), 4),
        'passed': sum(1 for _, s, _ in scores if s > 0.5),
        'total': len(scores),
        'checks': {name: {'score': round(float(s), 4), 'value': round(float(val), 4)}
                   for name, s, val in scores},
    }

# ==================
# 5) Main Training Loop
# ==================
def run_training(n_iterations=300, verbose=True):
    """Run n_iterations of simulation training with similarity/confidence checks"""
    print("=" * 70)
    print("BIOPHYSICS SIMULATION TRAINING PIPELINE (Option A)")
    print("=" * 70)
    
    # Load data
    perfumes = load_famous_perfumes()
    print(f"Loaded {len(perfumes)} famous perfumes")
    
    ing_map = build_smiles_map()
    print(f"Built SMILES map: {len(ing_map)} ingredients")
    
    n = min(n_iterations, len(perfumes))
    
    # ============================================================
    # PASS 1: Precompute SMILES, Count FP, ChemBERTa embeddings
    # ============================================================
    print(f"\n  [Pass 1] Precomputing structural embeddings for {n} perfumes...")
    precomputed = []
    style_embeddings = {}   # style → list of BERT embeddings
    style_fps = {}          # style → list of Count FPs
    
    for i in range(n):
        perfume = perfumes[i]
        smiles_list, concentrations, coverage = perfume_to_simulation(perfume, ing_map)
        
        fp = count_fp_mixture(smiles_list, concentrations) if smiles_list else None
        emb = bert_mixture_embedding(smiles_list, concentrations) if smiles_list else None
        
        precomputed.append({
            'smiles': smiles_list,
            'concs': concentrations,
            'coverage': coverage,
            'fp': fp,
            'bert_emb': emb,
        })
        
        style = perfume['style']
        if emb is not None:
            style_embeddings.setdefault(style, []).append(emb)
        if fp is not None:
            style_fps.setdefault(style, []).append(fp)
    
    # Build style centroids (BERT)
    style_centroids = {}
    for style, embs in style_embeddings.items():
        centroid = np.mean(embs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        style_centroids[style] = centroid
    
    # Build style average FP (for Count FP comparison)
    style_avg_fps = {}
    for style, fps_list in style_fps.items():
        merged = {}
        for fp in fps_list:
            for k, v in fp.items():
                merged[k] = merged.get(k, 0) + v / len(fps_list)
        style_avg_fps[style] = merged
    
    bert_cache = _get_bert()
    bert_hits = sum(1 for p in precomputed if p['bert_emb'] is not None)
    fp_hits = sum(1 for p in precomputed if p['fp'] is not None)
    print(f"  BERT embeddings: {bert_hits}/{n} ({bert_cache.size} in cache)")
    print(f"  Count FP: {fp_hits}/{n}")
    print(f"  Style centroids: {len(style_centroids)} styles")
    
    # ============================================================
    # PASS 2: Run simulations + compute similarity with structural metrics
    # ============================================================
    print(f"\n  [Pass 2] Running simulations...")
    
    # Results storage
    results = []
    cumulative_similarity = 0
    cumulative_confidence = 0
    best_score = 0
    worst_score = 1
    
    start_time = time.time()
    
    for i in range(n):
        perfume = perfumes[i]
        pre = precomputed[i]
        iter_start = time.time()
        
        smiles_list = pre['smiles']
        concentrations = pre['concs']
        coverage = pre['coverage']
        
        if not smiles_list:
            if verbose:
                print(f"[{i+1:3d}] SKIP: {perfume['name']} — no valid ingredients")
            continue
        
        # Run simulation
        try:
            sim_result = biophys.simulate_recipe(smiles_list, concentrations)
        except Exception as e:
            if verbose:
                print(f"[{i+1:3d}] ERROR: {perfume['name']} — {e}")
            continue
        
        # Structural similarities vs style centroid
        style = perfume['style']
        fp_sim = count_fp_tanimoto(pre['fp'], style_avg_fps.get(style))
        bert_sim = cosine_similarity(pre['bert_emb'], style_centroids.get(style))
        
        # Similarity check (pass structural scores directly)
        similarity = compute_similarity(
            sim_result, perfume, coverage,
            fp_sim_score=fp_sim, bert_sim_score=bert_sim
        )

        
        # Confidence check
        confidence = compute_confidence(sim_result, coverage)
        
        # Track stats
        cumulative_similarity += similarity['overall']
        cumulative_confidence += confidence['confidence']
        avg_sim = cumulative_similarity / (len(results) + 1)
        avg_conf = cumulative_confidence / (len(results) + 1)
        
        if similarity['overall'] > best_score:
            best_score = similarity['overall']
        if similarity['overall'] < worst_score:
            worst_score = similarity['overall']
        
        iter_time = time.time() - iter_start
        
        result = {
            'iteration': i + 1,
            'name': perfume['name'],
            'brand': perfume['brand'],
            'style': perfume['style'],
            'total_score': sim_result['total_score'],
            'similarity': similarity,
            'confidence': confidence,
            'sim_longevity_hours': sim_result['thermodynamics']['longevity_hours'],
            'sim_hedonic': round(sim_result['hedonic']['hedonic_score'], 4),
            'sim_active_receptors': sim_result['nose']['active_receptors'],
            'time_sec': round(iter_time, 3),
        }
        results.append(result)
        
        # Print progress
        if verbose:
            status = "✅" if similarity['overall'] > 0.5 else "⚠️" if similarity['overall'] > 0.3 else "❌"
            conf_str = "🟢" if confidence['confidence'] >= 0.8 else "🟡" if confidence['confidence'] >= 0.6 else "🔴"
            print(f"[{i+1:3d}/{n_iterations}] {status} {perfume['name'][:30]:30s} | "
                  f"Sim={similarity['overall']:.3f} Conf={confidence['confidence']:.2f}{conf_str} | "
                  f"L={sim_result['thermodynamics']['longevity_hours']:.1f}h "
                  f"H={sim_result['hedonic']['hedonic_score']:.2f} "
                  f"R={sim_result['nose']['active_receptors']:3d} | "
                  f"{iter_time:.2f}s")
        
        # Milestone reports
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"\n{'─'*70}")
            print(f"  MILESTONE: {i+1}/{n_iterations} iterations")
            print(f"  Avg Similarity: {avg_sim:.4f} | Avg Confidence: {avg_conf:.4f}")
            print(f"  Best: {best_score:.4f} | Worst: {worst_score:.4f}")
            print(f"  Time: {elapsed:.1f}s ({elapsed/(i+1):.2f}s/iter)")
            print(f"{'─'*70}\n")
    
    total_time = time.time() - start_time
    
    # Final report
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE: {len(results)}/{n_iterations} iterations")
    print(f"{'='*70}")
    
    if results:
        sims = [r['similarity']['overall'] for r in results]
        confs = [r['confidence']['confidence'] for r in results]
        
        print(f"\n  Similarity:")
        print(f"    Mean:   {sum(sims)/len(sims):.4f}")
        print(f"    Best:   {max(sims):.4f}")
        print(f"    Worst:  {min(sims):.4f}")
        print(f"    Median: {sorted(sims)[len(sims)//2]:.4f}")
        
        print(f"\n  Confidence:")
        print(f"    Mean:   {sum(confs)/len(confs):.4f}")
        print(f"    ≥80%:   {sum(1 for c in confs if c >= 0.8)}/{len(confs)}")
        print(f"    ≥60%:   {sum(1 for c in confs if c >= 0.6)}/{len(confs)}")
        
        # Style breakdown
        style_scores = {}
        for r in results:
            s = r['style']
            if s not in style_scores:
                style_scores[s] = []
            style_scores[s].append(r['similarity']['overall'])
        
        print(f"\n  By Style:")
        for style, scores in sorted(style_scores.items(), key=lambda x: -sum(x[1])/len(x[1])):
            print(f"    {style:15s}: avg={sum(scores)/len(scores):.4f} (n={len(scores)})")
        
        # Sub-metric breakdown
        long_sims = [r['similarity']['longevity_sim'] for r in results]
        note_sims = [r['similarity']['note_transition_sim'] for r in results]
        hed_sims = [r['similarity']['hedonic_sim'] for r in results]
        cov_sims = [r['similarity']['smiles_coverage'] for r in results]
        
        print(f"\n  Sub-metrics (mean):")
        print(f"    Longevity Match:    {sum(long_sims)/len(long_sims):.4f}")
        print(f"    Note Transition:    {sum(note_sims)/len(note_sims):.4f}")
        print(f"    Hedonic Correlation:{sum(hed_sims)/len(hed_sims):.4f}")
        print(f"    SMILES Coverage:    {sum(cov_sims)/len(cov_sims):.4f}")
    
    print(f"\n  Total Time: {total_time:.1f}s ({total_time/max(len(results),1):.2f}s/iter)")
    
    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'biophysics_training_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_iterations': len(results),
            'total_time_sec': round(total_time, 2),
            'summary': {
                'mean_similarity': round(sum(sims)/len(sims), 4) if results else 0,
                'mean_confidence': round(sum(confs)/len(confs), 4) if results else 0,
                'best_similarity': round(max(sims), 4) if results else 0,
                'worst_similarity': round(min(sims), 4) if results else 0,
            },
            'results': results,
        }, f, indent=1, ensure_ascii=False)
    print(f"\n  Results saved → {out_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=300, help='Number of iterations')
    parser.add_argument('-q', action='store_true', help='Quiet mode')
    args = parser.parse_args()
    run_training(n_iterations=args.n, verbose=not args.q)
