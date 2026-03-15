# -*- coding: utf-8 -*-
"""
⚖ AI Judge v3 — 자율 심사 AI (Full Upgrade)
═══════════════════════════════════════════════
 [2] 적대적 비평가 — 레시피 결함을 공격적으로 탐색
 [3] 비교 Elo 랭킹 — 1:1 대결 후 Elo 점수로 순위
 + 동적 가중치, 다단계 심사, 토론

★ 최적화: GNN 공유, 향 벡터 캐시, numpy 경량 연산
"""
import sys, os, json, time
import numpy as np
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, '.')

import biophysics_simulator as biophys
import torch
import torch.nn as nn
from pathlib import Path

WEIGHTS_DIR = Path('weights')

ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]
N_DIM = len(ODOR_DIMENSIONS)


# ═══════════════════════════════════════
# GNN 모델 (경량)
# ═══════════════════════════════════════
class TrainableOdorNetV4(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.input_dim = input_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.1),
        )
        self.output = nn.Sequential(nn.Linear(128, N_DIM), nn.Sigmoid())
    def forward(self, x):
        return self.output(self.backbone(x))


class TrainableMixtureNet(nn.Module):
    def __init__(self, d_model=N_DIM, nhead=4, num_layers=6):
        super().__init__()
        hidden = d_model * 8
        self.input_proj = nn.Linear(N_DIM + 1, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden // 2, N_DIM), nn.Sigmoid(),
        )
    def forward(self, odor_vecs, concentrations, pooling_mode='concentration'):
        x = torch.cat([odor_vecs, concentrations], dim=-1)
        x = self.input_proj(x)
        attended = self.transformer(x)
        if pooling_mode == 'concentration':
            w = torch.softmax(concentrations.squeeze(-1), dim=-1)
            pooled = (attended * w.unsqueeze(-1)).sum(dim=1)
        elif pooling_mode == 'max':
            pooled = attended.max(dim=1).values
        elif pooling_mode == 'top_k':
            k = min(5, attended.shape[1])
            _, idx = concentrations.squeeze(-1).topk(k, dim=1)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, attended.shape[-1])
            pooled = torch.gather(attended, 1, idx_exp).mean(dim=1)
        elif pooling_mode == 'uniform':
            pooled = attended.mean(dim=1)
        else:
            w = torch.softmax(concentrations.squeeze(-1), dim=-1)
            pooled = (attended * w.unsqueeze(-1)).sum(dim=1)
        return self.output_proj(pooled)


class GNNJudgeEngine:
    _odor_cache = {}
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.odor_model = None
        self.mixture_model = None
        self.bert_cache = None
        self._fp_func = None
        self._load_models()
    
    def _load_models(self):
        cache_path = WEIGHTS_DIR / 'chemberta_cache.pt'
        if cache_path.exists():
            self.bert_cache = torch.load(cache_path, map_location='cpu', weights_only=True)
            print(f"  ✅ ChemBERTa 캐시: {len(self.bert_cache)}개")
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            def fp_func(smi):
                mol = Chem.MolFromSmiles(smi)
                if mol is None: return np.zeros(384, dtype=np.float32)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=384)
                return np.array(fp, dtype=np.float32)
            self._fp_func = fp_func
        except ImportError:
            self._fp_func = lambda smi: np.zeros(384, dtype=np.float32)
        
        odor_path = WEIGHTS_DIR / 'odor_gnn.pt'
        if odor_path.exists():
            try:
                state = torch.load(odor_path, map_location=self.device, weights_only=True)
                input_dim = 384
                for key in state:
                    if 'backbone.0.weight' in key:
                        input_dim = state[key].shape[1]; break
                self.odor_model = TrainableOdorNetV4(input_dim=input_dim).to(self.device)
                self.odor_model.load_state_dict(state, strict=False)
                self.odor_model.eval()
                print(f"  ✅ OdorGNN ({input_dim}d → 22d)")
            except Exception as e:
                print(f"  ⚠ OdorGNN: {e}")
        
        mix_path = WEIGHTS_DIR / 'mixture_transformer.pt'
        if mix_path.exists():
            try:
                state = torch.load(mix_path, map_location=self.device, weights_only=True)
                self.mixture_model = TrainableMixtureNet().to(self.device)
                self.mixture_model.load_state_dict(state, strict=False)
                self.mixture_model.eval()
                print(f"  ✅ MixtureTransformer")
            except Exception as e:
                print(f"  ⚠ MixtureTransformer: {e}")
    
    def smiles_to_embedding(self, smiles):
        if self.bert_cache and smiles in self.bert_cache:
            return self.bert_cache[smiles].numpy()
        return self._fp_func(smiles)
    
    def predict_odor_vector(self, smiles):
        if smiles in GNNJudgeEngine._odor_cache:
            return GNNJudgeEngine._odor_cache[smiles]
        if self.odor_model is None:
            vec = np.full(N_DIM, 0.5, dtype=np.float32)
            GNNJudgeEngine._odor_cache[smiles] = vec
            return vec
        emb = self.smiles_to_embedding(smiles)
        with torch.no_grad():
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred = self.odor_model(x)
        vec = pred.squeeze(0).cpu().numpy()
        GNNJudgeEngine._odor_cache[smiles] = vec
        return vec
    
    def predict_mixture(self, smiles_list, concentrations, pooling_mode='concentration'):
        odor_vecs = np.array([self.predict_odor_vector(s) for s in smiles_list])
        concs = np.array(concentrations, dtype=np.float32)
        if self.mixture_model is None:
            if pooling_mode == 'max': return np.max(odor_vecs, axis=0)
            elif pooling_mode == 'top_k':
                k = min(5, len(concs)); return np.mean(odor_vecs[np.argsort(concs)[-k:]], axis=0)
            elif pooling_mode == 'uniform': return np.mean(odor_vecs, axis=0)
            else:
                w = concs / max(concs.sum(), 1e-8); return np.average(odor_vecs, weights=w, axis=0)
        with torch.no_grad():
            ov = torch.tensor(odor_vecs, dtype=torch.float32).unsqueeze(0).to(self.device)
            cc = torch.tensor(concs, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
            pred = self.mixture_model(ov, cc, pooling_mode=pooling_mode)
        return pred.squeeze(0).cpu().numpy()


# ═══════════════════════════════════════
# 심사 AI 설정
# ═══════════════════════════════════════
JUDGE_POOLING = {1: 'concentration', 2: 'max', 3: 'top_k', 4: 'uniform'}
JUDGE_PERSONA = {
    1: {'name': '균형 심사관', 'focus': 'balance', 'desc': '모든 기준 균형'},
    2: {'name': '감성 심사관', 'focus': 'hedonic', 'desc': '쾌적도/첫인상'},
    3: {'name': '기술 심사관', 'focus': 'technical', 'desc': 'GNN/구조 중시'},
    4: {'name': '조화 심사관', 'focus': 'harmony', 'desc': '원료 간 조화'},
}


# ═══════════════════════════════════════
# 평가 함수들 (GNN, 조화, 시뮬, 구조)
# ═══════════════════════════════════════
def evaluate_gnn_quality(mixture_vec):
    eps = 1e-8
    mean_v, std_v = np.mean(mixture_vec), np.std(mixture_vec) + eps
    z = (mixture_vec - mean_v) / std_v
    
    mask = z > 0.5
    prominence = float(np.clip((np.mean(z[mask]) if mask.any() else np.max(z)) / 2.0, 0, 1))
    
    pleasant = [0, 2, 3, 4, 7, 10]
    unpleasant = [11, 16, 19]
    hedonic = float(np.clip(0.5 + (np.mean(z[pleasant]) - np.mean(z[unpleasant])) * 0.25, 0, 1))
    
    active = np.mean(z > 0)
    complexity = 1.0 if 0.35 <= active <= 0.65 else (active/0.35 if active < 0.35 else 1.0-(active-0.65)/0.35)
    complexity = float(np.clip(complexity, 0, 1))
    
    top5 = np.argsort(mixture_vec)[-5:][::-1]
    dom = np.sum(mixture_vec[top5]) / (np.sum(mixture_vec) + eps)
    dominance = float(np.clip(1.0 - abs(dom - 0.30) * 5, 0, 1))
    
    total = prominence * 0.25 + hedonic * 0.30 + complexity * 0.20 + dominance * 0.25
    return {
        'prominence_score': round(prominence, 4), 'hedonic_score': round(hedonic, 4),
        'complexity_score': round(complexity, 4), 'dominance_score': round(dominance, 4),
        'dominant_notes': [ODOR_DIMENSIONS[i] for i in top5],
        'score': round(float(total), 4),
    }

def evaluate_harmony(ingredients):
    cats = [i.get('category', '') for i in ingredients]
    unique = len(set(cats))
    diversity = min(1.0, unique / max(5, len(cats) / 2))
    relevances = [i.get('theme_relevance', 0.5) for i in ingredients]
    avg_rel = float(np.mean(relevances)) if relevances else 0.5
    score = diversity * 0.5 + avg_rel * 0.5
    return {'score': round(score, 4), 'diversity': round(diversity, 3),
            'avg_relevance': round(avg_rel, 3), 'unique_categories': unique}

def evaluate_simulation(smiles_list, concentrations):
    try:
        r = biophys.simulate_recipe(smiles_list, concentrations)
        h, l = r['hedonic']['hedonic_score'], r['thermodynamics']['longevity_hours']
        s, a = r['thermodynamics'].get('smoothness', 0.5), r['nose']['active_receptors']
    except:
        h, l, s, a = 0.5, 4.0, 0.5, 60
    return {
        'hedonic_score': round(h, 4), 'longevity_hours': round(l, 1),
        'longevity_score': round(min(1.0, l/6), 4), 'smoothness': round(s, 4),
        'active_receptors': a, 'receptor_score': round(min(1.0, a/120), 4),
    }

def evaluate_structure(ingredients):
    notes = {'top': [], 'middle': [], 'base': []}
    for i in ingredients:
        nt = i.get('note_type', 'middle')
        if nt in notes: notes[nt].append(i['concentration'])
    ideal = {'top': 15, 'middle': 35, 'base': 50}
    ratio_scores = {}
    for n in ['top', 'middle', 'base']:
        actual = sum(notes[n])
        ratio_scores[n] = round(max(0, 1.0 - abs(actual - ideal[n]) / ideal[n]), 3)
    coverage = sum(1 for v in notes.values() if v) / 3.0
    count = len(ingredients)
    count_score = 1.0 if 8 <= count <= 15 else max(0, 1.0 - abs(count - 12) / 12)
    score = np.mean(list(ratio_scores.values())) * 0.4 + coverage * 0.3 + count_score * 0.3
    return {'score': round(float(score), 4), 'ratio_scores': ratio_scores,
            'coverage': round(coverage, 2), 'count_score': round(count_score, 3)}


# ═══════════════════════════════════════
# [2] 적대적 비평가 (Adversarial Critic)
# ═══════════════════════════════════════
def adversarial_critique(recipe, gnn_engine):
    """
    ★ 악마의 변호인 — 레시피의 모든 결함을 공격적으로 찾기
    
    일반 심사와 반대 관점:
    - 모든 것에 의심을 갖고 약점을 파고듦
    - 패널티 점수로 최종 점수에서 차감
    """
    ings = recipe['ingredients']
    smiles = [i['smiles'] for i in ings]
    concs = [i['concentration'] for i in ings]
    
    flaws = []
    penalty = 0.0
    
    # 1) 카테고리 과잉 집중
    cats = [i.get('category', '') for i in ings]
    cat_counts = {}
    for c in cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    max_cat = max(cat_counts.values()) if cat_counts else 0
    if max_cat > len(ings) * 0.5:
        dominant_cat = max(cat_counts, key=cat_counts.get)
        flaws.append(f"⚠ 카테고리 과잉: '{dominant_cat}' {max_cat}/{len(ings)}개 ({max_cat/len(ings):.0%})")
        penalty += 0.03 * (max_cat / len(ings))
    
    # 2) 농도 독점
    max_conc = max(concs)
    if max_conc > 40:
        dominant_ing = ings[concs.index(max_conc)]
        name = dominant_ing['name_en'] or dominant_ing['name_ko']
        flaws.append(f"⚠ 농도 독점: '{name}' {max_conc:.1f}% → 다른 원료 매몰 위험")
        penalty += 0.02
    
    # 3) 유사 원료 중복 (같은 카테고리+같은 노트)
    seen = set()
    for i in ings:
        key = (i.get('category', ''), i.get('note_type', ''))
        if key in seen:
            name = i['name_en'] or i['name_ko']
            flaws.append(f"⚠ 유사 중복: '{name}' ({key[0]}/{key[1]}) 이미 존재")
            penalty += 0.015
            break  # 하나만
        seen.add(key)
    
    # 4) 테마 관련도 낮은 원료
    low_rel = [i for i in ings if i.get('theme_relevance', 0.5) < 0.3]
    if len(low_rel) >= 3:
        names = ', '.join((i['name_en'] or i['name_ko']) for i in low_rel[:3])
        flaws.append(f"⚠ 테마 무관 원료 {len(low_rel)}개: {names}")
        penalty += 0.02 * min(len(low_rel), 3)
    
    # 5) GNN 프로파일 편향
    mixture = gnn_engine.predict_mixture(smiles, concs, 'uniform')
    z = (mixture - np.mean(mixture)) / (np.std(mixture) + 1e-8)
    extreme = np.sum(np.abs(z) > 2.0)
    if extreme >= 3:
        flaws.append(f"⚠ 향 프로파일 극단 편향: {extreme}개 차원이 ±2σ 초과")
        penalty += 0.015 * extreme
    
    # 6) 비용 효율
    cost = sum(float(i.get('price_usd_kg') or 0) * i['concentration'] / 100 for i in ings)
    if cost > 50:
        flaws.append(f"⚠ 비용 과다: ${cost:.0f}/kg (업계 평균 대비 높음)")
        penalty += 0.01
    
    penalty = min(0.15, penalty)  # 최대 15% 패널티
    
    if not flaws:
        flaws.append("✅ 주요 결함 발견 안됨 — 방어 성공")
    
    return {
        'flaws': flaws,
        'penalty': round(float(penalty), 4),
        'flaw_count': len([f for f in flaws if f.startswith('⚠')]),
    }


# ═══════════════════════════════════════
# 동적 가중치 + 의견 생성
# ═══════════════════════════════════════
def compute_dynamic_weights(judge_id, theme_text, recipe):
    persona = JUDGE_PERSONA[judge_id]
    weights = {'harmony': 0.15, 'gnn_quality': 0.25, 'simulation': 0.15, 'hedonic': 0.25, 'structure': 0.20}
    reasoning = []
    
    if persona['focus'] == 'hedonic':
        weights['hedonic'] += 0.08; weights['gnn_quality'] -= 0.04; weights['structure'] -= 0.04
        reasoning.append("감성 중시: 쾌적도↑")
    elif persona['focus'] == 'technical':
        weights['gnn_quality'] += 0.06; weights['structure'] += 0.04
        weights['hedonic'] -= 0.05; weights['harmony'] -= 0.05
        reasoning.append("기술 중시: GNN/구조↑")
    elif persona['focus'] == 'harmony':
        weights['harmony'] += 0.08; weights['simulation'] += 0.04
        weights['gnn_quality'] -= 0.06; weights['structure'] -= 0.06
        reasoning.append("조화 중시: 조화도↑")
    else:
        reasoning.append("균형 심사")
    
    total = sum(weights.values())
    return {k: round(v/total, 4) for k, v in weights.items()}, reasoning


def generate_opinion(total, harmony, gnn_q, sim, structure, ings, weights, judge_id, critique):
    persona = JUDGE_PERSONA[judge_id]
    comments = []
    
    dominant = ', '.join(gnn_q.get('dominant_notes', [])[:3])
    if gnn_q['score'] >= 0.7: comments.append(f"GNN: 혼합 향 우수 ({dominant}).")
    elif gnn_q['score'] >= 0.5: comments.append(f"GNN: 양호 ({dominant}), 복잡도 보강 가능.")
    else: comments.append("GNN: 향 단조로움.")
    
    if harmony['score'] >= 0.8: comments.append("조화 우수.")
    elif harmony['score'] >= 0.6: comments.append("조화 양호.")
    else: comments.append(f"다양성 부족 ({harmony['unique_categories']}개).")
    
    if sim['longevity_hours'] >= 6: comments.append(f"지속력 우수 ({sim['longevity_hours']}h).")
    elif sim['longevity_hours'] >= 4: comments.append(f"지속력 적정 ({sim['longevity_hours']}h).")
    else: comments.append(f"지속력 보강 필요 ({sim['longevity_hours']}h).")
    
    if structure['score'] >= 0.8: comments.append("구조 안정.")
    
    if sim['hedonic_score'] >= 0.7: comments.append("쾌적.")
    elif sim['hedonic_score'] >= 0.5: comments.append("쾌적도 보통.")
    else: comments.append("쾌적도 개선 필요.")
    
    # 비평 반영
    for flaw in critique.get('flaws', []):
        if flaw.startswith('⚠'):
            comments.append(f"[비평] {flaw}")
    
    if total >= 0.80: verdict = "강력 추천"
    elif total >= 0.75: verdict = "추천"
    elif total >= 0.6: verdict = "보통"
    else: verdict = "재작업 필요"
    
    return {'verdict': verdict, 'comments': comments, 'total_score': round(total, 4)}


# ═══════════════════════════════════════
# 레시피 평가 (Multi-Pass)
# ═══════════════════════════════════════
def evaluate_recipe(judge_id, recipe, gnn_engine):
    ings = recipe['ingredients']
    smiles = [i['smiles'] for i in ings]
    concs = [i['concentration'] for i in ings]
    
    pooling_mode = JUDGE_POOLING[judge_id]
    mixture_vec = gnn_engine.predict_mixture(smiles, concs, pooling_mode)
    gnn_q = evaluate_gnn_quality(mixture_vec)
    harmony = evaluate_harmony(ings)
    sim = evaluate_simulation(smiles, concs)
    structure = evaluate_structure(ings)
    
    # [2] 적대적 비평
    critique = adversarial_critique(recipe, gnn_engine)
    
    # 동적 가중치
    weights, w_reasoning = compute_dynamic_weights(judge_id, recipe.get('theme', ''), recipe)
    
    # Pass 2 조정
    if sim['hedonic_score'] < 0.4:
        weights['hedonic'] = min(0.40, weights['hedonic'] + 0.05)
        total_w = sum(weights.values())
        weights = {k: v/total_w for k, v in weights.items()}
    
    sim_score = (sim['longevity_score'] + sim['smoothness'] + sim['receptor_score']) / 3
    total = (
        harmony['score'] * weights['harmony'] +
        gnn_q['score'] * weights['gnn_quality'] +
        sim_score * weights['simulation'] +
        sim['hedonic_score'] * weights['hedonic'] +
        structure['score'] * weights['structure']
    )
    
    # 비평 패널티 적용
    total = max(0, total - critique['penalty'])
    
    opinion = generate_opinion(total, harmony, gnn_q, sim, structure, ings, weights, judge_id, critique)
    
    return {
        'composer_id': recipe['composer_id'], 'judge_id': judge_id,
        'pooling_mode': pooling_mode,
        'weights_used': {k: round(v, 4) for k, v in weights.items()},
        'scores': {
            'harmony': harmony, 'gnn_quality': gnn_q,
            'simulation': sim, 'hedonic': {'score': sim['hedonic_score']},
            'structure': structure,
        },
        'critique': critique,
        'total_score': round(float(total), 4),
        'opinion': opinion,
    }


# ═══════════════════════════════════════
# [3] Elo 랭킹 시스템
# ═══════════════════════════════════════
def compute_elo_ranking(all_evaluations):
    """
    ★ Fix #8: Swiss-round Elo 랭킹
    O(n²) 라운드 로빈 → O(n log n) 스위스 라운드
    유사 레이팅끼리 매칭하여 효율적 순위 결정
    """
    import math
    
    # 레시피별 평균 점수 수집
    recipe_avg = {}
    recipe_detail = {}
    
    for jid, evals in all_evaluations.items():
        for ev in evals:
            cid = ev['composer_id']
            if cid not in recipe_avg:
                recipe_avg[cid] = []
                recipe_detail[cid] = {}
            recipe_avg[cid].append(ev['total_score'])
            recipe_detail[cid][jid] = ev['scores']
    
    composers = list(recipe_avg.keys())
    n = len(composers)
    elo = {cid: 1500 for cid in composers}
    K = 32
    
    # 매치 1:1 비교 함수
    def compare(a, b):
        wins_a, wins_b = 0, 0
        for jid in all_evaluations:
            a_evals = [e for e in all_evaluations[jid] if e['composer_id'] == a]
            b_evals = [e for e in all_evaluations[jid] if e['composer_id'] == b]
            if a_evals and b_evals:
                if a_evals[0]['total_score'] > b_evals[0]['total_score']:
                    wins_a += 1
                elif b_evals[0]['total_score'] > a_evals[0]['total_score']:
                    wins_b += 1
        return wins_a, wins_b
    
    # 스위스 라운드: ceil(log2(n)) + 1 라운드
    num_rounds = min(math.ceil(math.log2(max(n, 2))) + 1, n - 1)
    played = set()  # (min, max) 쌍 → 이미 대결한 쌍
    matches = []
    
    for rnd in range(num_rounds):
        # Elo 기준 정렬 → 인접 페어링
        sorted_by_elo = sorted(composers, key=lambda c: elo[c], reverse=True)
        
        paired = set()
        round_pairs = []
        
        for i in range(len(sorted_by_elo)):
            a = sorted_by_elo[i]
            if a in paired:
                continue
            # 인접한 미대결 상대 찾기
            for j in range(i + 1, len(sorted_by_elo)):
                b = sorted_by_elo[j]
                if b in paired:
                    continue
                pair_key = (min(a, b), max(a, b))
                if pair_key not in played:
                    round_pairs.append((a, b))
                    paired.add(a)
                    paired.add(b)
                    played.add(pair_key)
                    break
        
        if not round_pairs:
            break  # 더 이상 새 매치 없음
        
        for a, b in round_pairs:
            wins_a, wins_b = compare(a, b)
            
            # Elo 업데이트
            expected_a = 1.0 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
            expected_b = 1.0 - expected_a
            
            if wins_a > wins_b:
                actual_a, actual_b = 1.0, 0.0
            elif wins_b > wins_a:
                actual_a, actual_b = 0.0, 1.0
            else:
                actual_a, actual_b = 0.5, 0.5
            
            elo[a] += K * (actual_a - expected_a)
            elo[b] += K * (actual_b - expected_b)
            
            matches.append({
                'a': a, 'b': b,
                'wins_a': wins_a, 'wins_b': wins_b,
                'result': 'A' if wins_a > wins_b else ('B' if wins_b > wins_a else 'Draw'),
                'round': rnd + 1,
            })
    
    elo_ranking = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'elo_scores': {str(cid): round(score) for cid, score in elo_ranking},
        'matches': matches,
        'elo_ranking': [cid for cid, _ in elo_ranking],
        'swiss_rounds': num_rounds,
    }


# ═══════════════════════════════════════
# 토론 라운드
# ═══════════════════════════════════════
def debate_round(preliminary_results, gnn_engine):
    print(f"\n  🗣  토론 라운드")
    
    recipe_scores = {}
    for jid, evals in preliminary_results.items():
        for ev in evals:
            cid = ev['composer_id']
            if cid not in recipe_scores: recipe_scores[cid] = {}
            recipe_scores[cid][jid] = ev['total_score']
    
    final_results = {}
    debate_log = []
    
    for jid, evals in preliminary_results.items():
        final_results[jid] = []
        for ev in evals:
            cid = ev['composer_id']
            my_score = ev['total_score']
            others = [s for j, s in recipe_scores[cid].items() if j != jid]
            avg_others = float(np.mean(others)) if others else my_score
            diff = my_score - avg_others
            
            debate_reasoning = []
            adjusted = False
            
            if abs(diff) > 0.02:
                adjustment = -diff * 0.3
                new_score = round(my_score + adjustment, 4)
                debate_reasoning.append(f"평균({avg_others:.4f}) 대비 {'높음' if diff > 0 else '낮음'} → {new_score:.4f}")
                ev['total_score'] = new_score
                adjusted = True
                debate_log.append(f"     심사#{jid}→조향#{cid}: {my_score:.4f}→{new_score:.4f}")
            else:
                debate_reasoning.append(f"의견 일치 (차이 {abs(diff):.4f})")
            
            ev['debate'] = {
                'original_score': round(my_score, 4), 'final_score': ev['total_score'],
                'adjusted': adjusted, 'other_avg': round(avg_others, 4), 'reasoning': debate_reasoning,
            }
            final_results[jid].append(ev)
    
    if debate_log:
        for msg in debate_log: print(msg)
    else:
        print("     의견 일치 — 변동 없음")
    
    return final_results


# ═══════════════════════════════════════
# 심사 AI 실행
# ═══════════════════════════════════════
def run_judge(judge_id, recipe_files, gnn_engine):
    persona = JUDGE_PERSONA[judge_id]
    print(f"\n  ⚖ #{judge_id} {persona['name']}")
    
    evaluations = []
    for rp in recipe_files:
        with open(rp, 'r', encoding='utf-8') as f:
            recipe = json.load(f)
        ev = evaluate_recipe(judge_id, recipe, gnn_engine)
        evaluations.append(ev)
        cid = recipe['composer_id']
        cr = ev['critique']
        print(f"     #{cid}: {ev['total_score']:.4f} (결함 {cr['flaw_count']}개, 패널티 -{cr['penalty']:.3f})")
    
    evaluations.sort(key=lambda x: x['total_score'], reverse=True)
    for rank, ev in enumerate(evaluations, 1):
        ev['rank'] = rank
    return evaluations


# ═══════════════════════════════════════
# 종합 집계 (올림픽 + Elo)
# ═══════════════════════════════════════
def aggregate_results(final_results, output_dir, final_dir):
    print(f"\n{'='*60}")
    print(f"  🏆 최종 심사 (GNN + 비평 + Elo)")
    print(f"{'='*60}")
    
    recipe_scores = {}
    for jid, evals in final_results.items():
        output = {
            'judge_id': jid, 'persona': JUDGE_PERSONA[jid], 'evaluations': evals,
            'ranking': [ev['composer_id'] for ev in evals],
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'evaluation_{jid}.json'), 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        for ev in evals:
            cid = ev['composer_id']
            if cid not in recipe_scores: recipe_scores[cid] = []
            recipe_scores[cid].append(ev['total_score'])
    
    # 올림픽 채점
    final_ranking = []
    for cid, scores in recipe_scores.items():
        ss = sorted(scores)
        trimmed = ss[1:-1] if len(ss) >= 4 else ss
        avg = float(np.mean(trimmed))
        final_ranking.append({
            'composer_id': cid,
            'all_scores': [round(s, 4) for s in scores],
            'trimmed_scores': [round(s, 4) for s in trimmed],
            'final_score': round(avg, 4),
            'score_variance': round(float(np.var(scores)), 6),
            'score_range': round(max(scores) - min(scores), 4),
        })
    
    final_ranking.sort(key=lambda x: x['final_score'], reverse=True)
    for rank, fr in enumerate(final_ranking, 1):
        fr['rank'] = rank
        print(f"  #{rank} 조향#{fr['composer_id']}: {fr['final_score']:.4f} [{', '.join(f'{s:.3f}' for s in fr['all_scores'])}]")
    
    # [3] Elo 랭킹
    elo_result = compute_elo_ranking(final_results)
    elo_scores = elo_result['elo_scores']
    elo_parts = [f'#{c}({elo_scores[str(c)]})' for c in elo_result['elo_ranking']]
    print(f"\n  📊 Elo 랭킹: {' > '.join(elo_parts)}")
    
    # Elo와 올림픽 순위 비교
    olympic_order = [fr['composer_id'] for fr in final_ranking]
    elo_order = elo_result['elo_ranking']
    rank_agree = olympic_order == elo_order
    
    rankings_by_judge = [[ev['composer_id'] for ev in evals] for evals in final_results.values()]
    first_choices = [r[0] for r in rankings_by_judge]
    consensus = len(set(first_choices)) == 1
    
    result = {
        'final_ranking': final_ranking,
        'winner': final_ranking[0],
        'consensus': consensus,
        'first_choices_by_judge': first_choices,
        'judge_rankings': rankings_by_judge,
        'avg_score_variance': round(float(np.mean([fr['score_variance'] for fr in final_ranking])), 6),
        'elo': elo_result,
        'olympic_elo_agree': rank_agree,
        'gnn_integrated': True,
        'debate_applied': True,
        'adversarial_critic': True,
    }
    
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(final_dir, 'ranking.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n  🏅 최종 1위: 조향#{final_ranking[0]['composer_id']} ({final_ranking[0]['final_score']:.4f})")
    if not rank_agree:
        print(f"  ⚡ 올림픽 vs Elo 순위 불일치!")
    
    return result


def main(composer_dir=None, output_dir=None, gnn_engine=None):
    if composer_dir is None: composer_dir = os.path.join('weights', 'ai_perfumer', 'composers')
    if output_dir is None: output_dir = os.path.join('weights', 'ai_perfumer', 'judges')
    final_dir = os.path.join('weights', 'ai_perfumer', 'final')
    
    recipe_files = sorted([
        os.path.join(composer_dir, f) for f in os.listdir(composer_dir)
        if f.startswith('recipe_') and f.endswith('.json')
    ])
    if not recipe_files:
        print("❌ 레시피 없음!"); return
    
    print("=" * 60)
    print("  ⚖ AI Judge v3 — 비평 + Elo + 토론")
    print("=" * 60)
    
    if gnn_engine is None:
        print("\n  🧠 GNN 로딩...")
        gnn_engine = GNNJudgeEngine()
    else:
        print("\n  ✅ GNN 공유")
    
    # Phase 1: 독립 심사
    print(f"\n{'─'*60}\n  Phase 1: 독립 심사 + 적대적 비평\n{'─'*60}")
    preliminary = {}
    for jid in range(1, 5):
        preliminary[jid] = run_judge(jid, recipe_files, gnn_engine)
    
    # Phase 2: 토론
    print(f"\n{'─'*60}\n  Phase 2: 토론\n{'─'*60}")
    final_results = debate_round(preliminary, gnn_engine)
    
    # Phase 3: 집계 + Elo
    ranking = aggregate_results(final_results, output_dir, final_dir)
    
    return ranking, gnn_engine


if __name__ == '__main__':
    main()
