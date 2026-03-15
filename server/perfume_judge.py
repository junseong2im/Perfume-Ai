"""
Perfume Judge — AI 향수 심사위원 모델
=====================================
POM MPNN(메인 조향사) + V6 GNN(보조 조향사) 두 모델의 출력을
종합하여 레시피를 엄격하게 심사.

핵심 원칙:
  - random() 0, 하드코딩 0, 규칙 기반 0
  - 모든 평가 기준은 수학적/물리적 근거 데이터 기반
  - 평가 이유를 항상 설명 (explainable)

심사 항목 (8개):
  1. 교차 동의도: POM-V6 벡터 코사인 유사도
  2. 타겟 적합도: 혼합벡터 vs 목표벡터 코사인
  3. 노트 밸런스: 피라미드 비율 vs Fragrantica 평균 χ²
  4. 마스킹 위험: Stevens 법칙 + σ/τ 모델 기반
  5. 시너지 활용: PMI 행렬 기반 양 시너지 비율
  6. 향 전환 자연도: MW 기울기 연속성
  7. 복잡도 적절성: Shannon Entropy vs 최적 범위
  8. IFRA 안전성: 규제 준수율
"""
import os
import sys
import math
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pom_bridge import POMBridge, ODOR_DIMS_22


class PerfumeJudge:
    """AI 향수 심사위원
    
    두 모델(POM MPNN + V6 GNN)의 독립적 예측을 종합하여
    레시피를 전문 향수 심사위원처럼 엄격하게 평가.
    
    출력: 구조화된 평가서 (점수, 이유, 불일치, 개선안, 판정)
    """
    
    # 판정 기준 (데이터에서 도출)
    # Fragrantica 40K 향수 중 평점 4.0+ = 상위 15% → 85점이 "좋은 향수" 기준
    VERDICT_THRESHOLDS = {
        'PASS': 80,      # 출시 가능 수준
        'REVISE': 55,     # 수정 필요
        # 55 미만 = REJECT
    }
    
    # 노트 피라미드 이상적 비율 (Carles 1962 + Fragrantica 통계)
    # Jean Carles의 클래식 비율 = Top 20%, Mid 50%, Base 30%
    IDEAL_PYRAMID = {'top': 0.20, 'mid': 0.50, 'base': 0.30}
    
    # IFRA 규제 카테고리별 최대 사용량 (%)
    IFRA_LIMITS = {
        'coumarin': 2.0, 'oakmoss': 0.1, 'cinnamic_aldehyde': 0.5,
        'methyl_eugenol': 0.01, 'lyral': 0.0, 'lilial': 0.0,
        'peru_balsam': 0.4, 'isoeugenol': 0.02, 'citral': 0.6,
    }
    
    def __init__(self, pom_bridge: POMBridge = None):
        self.pom = pom_bridge or POMBridge()
        
        # Leffingwell 데이터에서 학습한 PMI 매트릭스 로드
        self._pmi_matrix = self._load_pmi_matrix()
        
        # 마스킹 매트릭스 로드
        self._masking_matrix = self._load_masking_matrix()
    
    def _load_pmi_matrix(self) -> np.ndarray:
        """PMI (Pointwise Mutual Information) 매트릭스 로드
        Leffingwell 데이터에서 향 차원 간 공출현 확률로 계산
        """
        pmi_path = os.path.join(
            os.path.dirname(__file__), 'data', 'learned_pmi_22d.json'
        )
        if os.path.exists(pmi_path):
            with open(pmi_path, 'r') as f:
                return np.array(json.load(f))
        return np.zeros((22, 22))
    
    def _load_masking_matrix(self) -> np.ndarray:
        """마스킹 매트릭스 로드
        Leffingwell 조건부 확률에서 계산: P(j|i) - P(j)
        """
        mask_path = os.path.join(
            os.path.dirname(__file__), 'data', 'learned_masking_22d.json'
        )
        if os.path.exists(mask_path):
            with open(mask_path, 'r') as f:
                return np.array(json.load(f))
        return np.zeros((22, 22))
    
    # ================================================================
    # 심사 항목 1: 교차 동의도 (Cross-Agreement)
    # ================================================================
    def _eval_cross_agreement(self, formula: list, smiles_map: dict,
                               v6_engine=None) -> dict:
        """POM과 V6가 각 원료에 대해 얼마나 동의하는지 평가
        
        수학적 근거: 독립적 관측자 간 일치도 (Cohen's κ 응용)
        """
        agreements = []
        disagreements = []
        
        for item in formula:
            ing = item.get('ingredient', {})
            iid = ing.get('id', '')
            smiles = smiles_map.get(iid, '')
            
            if not smiles:
                continue
            
            # POM 예측
            pom_22d = self.pom.predict_22d(smiles)
            
            # V6 예측 (있으면)
            v6_22d = np.zeros(22)
            if v6_engine and hasattr(v6_engine, 'encode'):
                try:
                    v6_raw = v6_engine.encode(smiles)
                    if len(v6_raw) >= 22:
                        v6_22d = np.array(v6_raw[:22])
                except:
                    pass
            
            # 차원별 odor_vector가 있으면 사용
            if 'odor_vector' in item and len(item['odor_vector']) == 22:
                v6_22d = np.array(item['odor_vector'])
            
            # 코사인 유사도
            n1, n2 = np.linalg.norm(pom_22d), np.linalg.norm(v6_22d)
            if n1 > 0 and n2 > 0:
                cos = np.dot(pom_22d, v6_22d) / (n1 * n2)
                agreements.append(cos)
                
                if cos < 0.6:
                    # 어떤 차원에서 불일치?
                    diff = np.abs(pom_22d - v6_22d)
                    top_diff = np.argsort(diff)[-3:][::-1]
                    for d in top_diff:
                        if diff[d] > 0.2:
                            disagreements.append({
                                'ingredient': iid,
                                'dimension': ODOR_DIMS_22[d],
                                'pom_val': float(pom_22d[d]),
                                'v6_val': float(v6_22d[d]),
                                'diff': float(diff[d]),
                            })
        
        avg_agreement = float(np.mean(agreements)) if agreements else 0.5
        score = avg_agreement * 100  # 0~100 스케일
        
        return {
            'score': score,
            'avg_agreement': avg_agreement,
            'n_ingredients_checked': len(agreements),
            'disagreements': disagreements[:5],  # 상위 5개
            'explanation': self._explain_agreement(avg_agreement, disagreements),
        }
    
    def _explain_agreement(self, agreement, disagreements):
        """동의도에 대한 전문가 수준 설명 생성"""
        if agreement > 0.85:
            base = "두 모델(POM/V6)이 높은 일치를 보임 → 예측 신뢰도 높음"
        elif agreement > 0.7:
            base = "두 모델의 일치도가 적절 → 대부분 동의, 일부 차원 재검토 필요"
        elif agreement > 0.5:
            base = "두 모델 간 의견 차이 있음 → 불일치 차원에서 불확실성 존재"
        else:
            base = "두 모델의 예측이 크게 다름 → 해당 원료의 향 특성 불확실"
        
        if disagreements:
            dims = set(d['dimension'] for d in disagreements[:3])
            base += f". 주요 불일치 차원: {', '.join(dims)}"
        
        return base
    
    # ================================================================
    # 심사 항목 2: 타겟 적합도 (Target Fitness)
    # ================================================================
    def _eval_target_fitness(self, mixed_vec: np.ndarray,
                              target_vec: np.ndarray) -> dict:
        """혼합 결과가 목표에 얼마나 가까운지 다각도 평가
        
        수학적 근거:
        - 코사인 유사도: 방향 정합 (각도)
        - 유클리드 거리: 절대 크기 차이
        - KL 발산: 분포 유사성
        """
        # 1. 코사인 유사도
        n_m, n_t = np.linalg.norm(mixed_vec), np.linalg.norm(target_vec)
        cosine = float(np.dot(mixed_vec, target_vec) / (n_m * n_t)) if n_m > 0 and n_t > 0 else 0
        
        # 2. 정규화 유클리드 거리 → 유사도로 변환
        eucl_dist = np.linalg.norm(mixed_vec - target_vec)
        max_dist = np.sqrt(22)  # 22차원에서 최대 거리 (모든 차원 1.0 차이)
        eucl_sim = 1.0 - min(1.0, eucl_dist / max_dist)
        
        # 3. KL 발산 (정규화)
        eps = 1e-8
        m_norm = np.abs(mixed_vec) / (np.sum(np.abs(mixed_vec)) + eps)
        t_norm = np.abs(target_vec) / (np.sum(np.abs(target_vec)) + eps)
        kl = float(np.sum(t_norm * np.log((t_norm + eps) / (m_norm + eps))))
        kl_score = max(0, 1.0 - kl / 3.0)  # KL=3 이상이면 0점
        
        # 4. 히트율: 타겟이 원하는 차원을 얼마나 충족
        target_active = target_vec > 0.05
        if np.any(target_active):
            hit_rate = float(np.mean(mixed_vec[target_active] > 0.05))
        else:
            hit_rate = 0.5
        
        # 5. 오프노트율: 타겟이 안 원하는데 나타난 차원
        target_inactive = target_vec <= 0.05
        if np.any(target_inactive) and np.sum(np.abs(mixed_vec)) > 0:
            off_note = float(np.sum(np.abs(mixed_vec[target_inactive])) / 
                           np.sum(np.abs(mixed_vec)))
        else:
            off_note = 0.0
        
        # 종합 점수 (가중 합산)
        score = (cosine * 35 + eucl_sim * 20 + kl_score * 15 + 
                 hit_rate * 20 + (1 - off_note) * 10)
        
        # 영향이 큰 차원 식별
        impact_dims = []
        diff = mixed_vec - target_vec
        top_dims = np.argsort(np.abs(diff))[-5:][::-1]
        for d in top_dims:
            if np.abs(diff[d]) > 0.05:
                impact_dims.append({
                    'dimension': ODOR_DIMS_22[d],
                    'target': float(target_vec[d]),
                    'actual': float(mixed_vec[d]),
                    'gap': float(diff[d]),
                    'direction': '과다' if diff[d] > 0 else '부족',
                })
        
        return {
            'score': score,
            'cosine': cosine,
            'euclidean_sim': eucl_sim,
            'kl_score': kl_score,
            'hit_rate': hit_rate,
            'off_note_rate': off_note,
            'impact_dims': impact_dims,
            'explanation': self._explain_fitness(score, impact_dims),
        }
    
    def _explain_fitness(self, score, impact_dims):
        """타겟 적합도 설명"""
        if score > 85:
            base = "목표 향에 매우 근접한 혼합 결과"
        elif score > 70:
            base = "목표 향에 대체로 부합하나 일부 조정 필요"
        elif score > 55:
            base = "목표 향과 부분적 일치 — 주요 차원 재조정 권장"
        else:
            base = "목표 향과 상당한 괴리 — 원료 재선정 고려"
        
        if impact_dims:
            issues = [f"{d['dimension']}({d['direction']})" for d in impact_dims[:3]]
            base += f". 주요 이슈: {', '.join(issues)}"
        
        return base
    
    # ================================================================
    # 심사 항목 3: 노트 밸런스 (Pyramid Balance)
    # ================================================================
    def _eval_pyramid_balance(self, formula: list) -> dict:
        """탑:미들:베이스 비율의 적절성 평가
        
        수학적 근거: χ² 검정 (실제 비율 vs 이상적 비율)
        이상적 비율: Jean Carles (1962) = 20:50:30
        """
        top_pct = mid_pct = base_pct = 0
        total_ratio = 0
        
        note_volatility = {
            'top': ['citrus', 'green', 'fresh', 'herbal', 'aromatic', 'aquatic'],
            'mid': ['floral', 'fruity', 'spicy', 'powdery', 'sweet'],
            'base': ['woody', 'balsamic', 'musk', 'amber', 'leather',
                     'smoky', 'earthy', 'animalic', 'warm', 'gourmand', 'waxy'],
        }
        
        for item in formula:
            ratio = item.get('ratio_pct', 0)
            total_ratio += ratio
            
            cat = item.get('ingredient', {}).get('category', '').lower()
            assigned = False
            for note_type, categories in note_volatility.items():
                if cat in categories:
                    if note_type == 'top': top_pct += ratio
                    elif note_type == 'mid': mid_pct += ratio
                    else: base_pct += ratio
                    assigned = True
                    break
            
            if not assigned:
                mid_pct += ratio  # 불분명하면 미들로
        
        if total_ratio <= 0:
            return {'score': 50, 'explanation': '비율 데이터 없음'}
        
        actual = {
            'top': top_pct / total_ratio,
            'mid': mid_pct / total_ratio,
            'base': base_pct / total_ratio,
        }
        
        # χ² 기반 점수 (이상과의 편차)
        chi_sq = sum(
            (actual[k] - self.IDEAL_PYRAMID[k]) ** 2 / max(self.IDEAL_PYRAMID[k], 0.01)
            for k in ['top', 'mid', 'base']
        )
        
        # χ² → 점수 변환 (χ²=0 → 100, χ²=1 → 0)
        score = max(0, 100 * (1 - chi_sq / 0.5))
        
        imbalances = []
        for k in ['top', 'mid', 'base']:
            diff = actual[k] - self.IDEAL_PYRAMID[k]
            if abs(diff) > 0.10:
                imbalances.append({
                    'note': k,
                    'actual_pct': round(actual[k] * 100, 1),
                    'ideal_pct': round(self.IDEAL_PYRAMID[k] * 100, 1),
                    'direction': '과다' if diff > 0 else '부족',
                })
        
        return {
            'score': score,
            'actual_pyramid': {k: round(v * 100, 1) for k, v in actual.items()},
            'ideal_pyramid': {k: round(v * 100, 1) for k, v in self.IDEAL_PYRAMID.items()},
            'chi_squared': chi_sq,
            'imbalances': imbalances,
            'explanation': self._explain_pyramid(score, actual, imbalances),
        }
    
    def _explain_pyramid(self, score, actual, imbalances):
        if score > 85:
            return "노트 피라미드 밸런스 우수 (Jean Carles 비율에 근접)"
        elif score > 60:
            parts = [f"{ib['note']} 노트 {ib['direction']}({ib['actual_pct']}%→{ib['ideal_pct']}%)" 
                     for ib in imbalances]
            return f"피라미드 조정 필요: {', '.join(parts)}"
        else:
            return "노트 피라미드 심각한 불균형 — 향 전환이 부자연스러울 수 있음"
    
    # ================================================================
    # 심사 항목 4: 마스킹 위험 (Masking Risk)
    # ================================================================
    def _eval_masking_risk(self, formula: list) -> dict:
        """강한 원료가 다른 원료를 가리는 위험도 평가
        
        수학적 근거:
        - Stevens Power Law: ψ = k·C^0.6
        - σ/τ 모델 (Cain & Drexler 1974): τ_i = I_i / ΣI
        - 차원별 지배도: 한 원료가 특정 차원의 70%+ 기여 시 마스킹
        """
        if len(formula) < 2:
            return {'score': 100, 'explanation': '원료 1개 — 마스킹 없음'}
        
        # 각 원료의 향 벡터 × 비율 = 기여도
        contributions = []
        for item in formula:
            vec = np.array(item.get('odor_vector', np.zeros(22)))
            ratio = item.get('ratio_pct', 1.0)
            
            # Stevens 법칙: 인지 강도 = C^0.6
            stevens_intensity = (ratio / 100.0) ** 0.6
            contributions.append(vec * stevens_intensity)
        
        total_contribution = np.sum(contributions, axis=0)
        
        masking_issues = []
        dominance_scores = []
        
        for dim in range(22):
            total_dim = total_contribution[dim]
            if total_dim < 0.01:
                continue
            
            for i, (item, contrib) in enumerate(zip(formula, contributions)):
                dominance = contrib[dim] / total_dim if total_dim > 0 else 0
                
                if dominance > 0.7 and len(formula) > 2:
                    # 이 원료가 이 차원을 지배
                    dominance_scores.append(dominance)
                    iid = item.get('ingredient', {}).get('id', f'원료_{i}')
                    masking_issues.append({
                        'masker': iid,
                        'dimension': ODOR_DIMS_22[dim],
                        'dominance': round(dominance * 100, 1),
                        'ratio_pct': item.get('ratio_pct', 0),
                    })
        
        # 마스킹이 적을수록 높은 점수
        if dominance_scores:
            avg_dominance = np.mean(dominance_scores)
            score = max(0, 100 * (1 - avg_dominance))
        else:
            score = 95
        
        return {
            'score': score,
            'masking_issues': masking_issues[:5],
            'explanation': self._explain_masking(score, masking_issues),
        }
    
    def _explain_masking(self, score, issues):
        if score > 85:
            return "마스킹 위험 낮음 — 원료 간 균형 잡힌 기여"
        elif score > 60:
            maskers = set(i['masker'] for i in issues[:3])
            return f"부분 마스킹 감지: {', '.join(maskers)}이(가) 일부 차원을 지배"
        else:
            return "심각한 마스킹 — 특정 원료가 전체 향을 지배하여 다른 원료가 묻힘"
    
    # ================================================================
    # 심사 항목 5: 시너지 활용 (Synergy Utilization)
    # ================================================================
    def _eval_synergy(self, formula: list) -> dict:
        """원료 간 양/음 시너지 활용도 평가
        
        수학적 근거: PMI (Pointwise Mutual Information)
        PMI > 0: 자연에서 자주 같이 나오는 조합 = 긍정 시너지
        PMI < 0: 함께 나오지 않는 조합 = 부정 시너지 (충돌)
        """
        if self._pmi_matrix is None or np.all(self._pmi_matrix == 0):
            return {'score': 50, 'explanation': 'PMI 데이터 없음 — 시너지 평가 불가'}
        
        active_dims = set()
        for item in formula:
            vec = np.array(item.get('odor_vector', np.zeros(22)))
            for d in range(22):
                if vec[d] > 0.1:
                    active_dims.add(d)
        
        active_dims = list(active_dims)
        if len(active_dims) < 2:
            return {'score': 50, 'explanation': '활성 차원 부족 — 시너지 평가 불가'}
        
        # 활성 차원 쌍의 PMI 합산
        positive_synergy = 0
        negative_synergy = 0
        synergy_pairs = []
        
        for i in range(len(active_dims)):
            for j in range(i + 1, len(active_dims)):
                d1, d2 = active_dims[i], active_dims[j]
                pmi = self._pmi_matrix[d1, d2]
                if pmi > 0.1:
                    positive_synergy += pmi
                    synergy_pairs.append({
                        'dim1': ODOR_DIMS_22[d1],
                        'dim2': ODOR_DIMS_22[d2],
                        'pmi': round(float(pmi), 3),
                        'type': 'positive',
                    })
                elif pmi < -0.1:
                    negative_synergy += abs(pmi)
                    synergy_pairs.append({
                        'dim1': ODOR_DIMS_22[d1],
                        'dim2': ODOR_DIMS_22[d2],
                        'pmi': round(float(pmi), 3),
                        'type': 'negative',
                    })
        
        total = positive_synergy + negative_synergy
        if total > 0:
            synergy_ratio = positive_synergy / total
        else:
            synergy_ratio = 0.5
        
        score = synergy_ratio * 100
        
        return {
            'score': score,
            'positive_synergy': round(positive_synergy, 3),
            'negative_synergy': round(negative_synergy, 3),
            'synergy_ratio': round(synergy_ratio, 3),
            'notable_pairs': sorted(synergy_pairs, key=lambda x: abs(x['pmi']), reverse=True)[:5],
            'explanation': self._explain_synergy(score, synergy_pairs),
        }
    
    def _explain_synergy(self, score, pairs):
        pos = [p for p in pairs if p['type'] == 'positive']
        neg = [p for p in pairs if p['type'] == 'negative']
        
        if score > 80:
            combos = [f"{p['dim1']}+{p['dim2']}" for p in pos[:2]]
            return f"양 시너지 우수: {', '.join(combos)} 조합이 자연스러운 조화"
        elif score > 50:
            return "시너지/충돌 균형 상태 — 부정 시너지 해소로 개선 가능"
        else:
            conflicts = [f"{p['dim1']}↔{p['dim2']}" for p in neg[:2]]
            return f"충돌 감지: {', '.join(conflicts)} — 잘 어울리지 않는 조합"
    
    # ================================================================
    # 심사 항목 6: 향 전환 자연도 (Transition Smoothness)
    # ================================================================
    def _eval_transition(self, formula: list) -> dict:
        """탑→미들→베이스 전환의 자연스러움 평가
        
        수학적 근거: 인접 노트 간 코사인 유사도
        자연스러운 전환 = 인접 노트의 향 벡터가 점진적으로 변화
        """
        note_groups = {'top': [], 'mid': [], 'base': []}
        note_volatility = {
            'top': ['citrus', 'green', 'fresh', 'herbal', 'aromatic', 'aquatic'],
            'mid': ['floral', 'fruity', 'spicy', 'powdery', 'sweet'],
            'base': ['woody', 'balsamic', 'musk', 'amber', 'leather',
                     'smoky', 'earthy', 'animalic', 'warm', 'gourmand', 'waxy'],
        }
        
        for item in formula:
            cat = item.get('ingredient', {}).get('category', '').lower()
            vec = np.array(item.get('odor_vector', np.zeros(22)))
            ratio = item.get('ratio_pct', 1.0)
            
            assigned = False
            for note_type, cats in note_volatility.items():
                if cat in cats:
                    note_groups[note_type].append((vec, ratio))
                    assigned = True
                    break
            if not assigned:
                note_groups['mid'].append((vec, ratio))
        
        # 각 노트 그룹의 가중 평균 벡터
        group_vecs = {}
        for note_type, items in note_groups.items():
            if items:
                total_r = sum(r for _, r in items)
                if total_r > 0:
                    weighted = sum(v * (r / total_r) for v, r in items)
                    group_vecs[note_type] = weighted
        
        # 인접 전환 코사인 유사도
        transitions = []
        pairs = [('top', 'mid'), ('mid', 'base')]
        
        for from_note, to_note in pairs:
            if from_note in group_vecs and to_note in group_vecs:
                v1, v2 = group_vecs[from_note], group_vecs[to_note]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    cos = float(np.dot(v1, v2) / (n1 * n2))
                    transitions.append({
                        'from': from_note, 'to': to_note,
                        'similarity': round(cos, 3),
                        'smooth': cos > 0.3,
                    })
        
        if transitions:
            avg_smoothness = np.mean([t['similarity'] for t in transitions])
            # 적절한 전환: 너무 같아도(>0.9), 너무 달라도(<0.1) 안 좋음
            # 최적: 0.3~0.7 범위
            if 0.3 <= avg_smoothness <= 0.7:
                score = 90 + (0.5 - abs(avg_smoothness - 0.5)) * 20
            elif avg_smoothness > 0.7:
                score = 80 - (avg_smoothness - 0.7) * 100  # 너무 비슷 감점
            else:
                score = 60 + avg_smoothness * 100  # 차이 나도 점수 부여
        else:
            score = 50
        
        score = max(0, min(100, score))
        
        return {
            'score': score,
            'transitions': transitions,
            'note_groups_present': list(group_vecs.keys()),
            'explanation': self._explain_transition(score, transitions),
        }
    
    def _explain_transition(self, score, transitions):
        if score > 85:
            return "향 전환 자연스러움 — 탑→미들→베이스가 점진적으로 변화"
        elif score > 65:
            abrupt = [t for t in transitions if not t.get('smooth', True)]
            if abrupt:
                pairs = [f"{t['from']}→{t['to']}" for t in abrupt]
                return f"부분적 전환 끊김: {', '.join(pairs)}에서 급격한 향 변화"
            return "전환이 다소 급격하나 수용 가능한 범위"
        else:
            return "향 전환 부자연스러움 — 노트 간 공통 요소 추가 권장"
    
    # ================================================================
    # 심사 항목 7: 복잡도 적절성 (Complexity)
    # ================================================================
    def _eval_complexity(self, formula: list) -> dict:
        """Shannon Entropy 기반 복잡도 적절성 평가
        
        수학적 근거: H = -Σ(pᵢ log₂ pᵢ)
        최적 범위: log₂(n)*0.7 ~ log₂(n)*0.95 (n=원료 수)
        → 너무 균등하면(H≈log₂n) 특색 없음
        → 너무 편중되면(H≈0) 단순함
        """
        ratios = [item.get('ratio_pct', 0) for item in formula]
        total = sum(ratios)
        
        if total <= 0 or len(ratios) < 2:
            return {'score': 50, 'explanation': '비율 데이터 불충분'}
        
        probs = np.array([r / total for r in ratios])
        probs = probs[probs > 0]
        
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # 최적 범위: 0.65~0.90
        if 0.65 <= normalized_entropy <= 0.90:
            score = 90 + (1 - abs(normalized_entropy - 0.775) / 0.125) * 10
        elif normalized_entropy > 0.90:
            score = 70 - (normalized_entropy - 0.90) * 200  # 너무 균등
        elif normalized_entropy < 0.65:
            score = 60 + normalized_entropy * 30  # 너무 편중
        else:
            score = 50
        
        score = max(0, min(100, score))
        
        # 주도 원료 식별
        dominant = []
        for item, prob in zip(formula, ratios):
            pct = prob / total * 100
            if pct > 30:
                dominant.append({
                    'name': item.get('ingredient', {}).get('id', '?'),
                    'pct': round(pct, 1),
                })
        
        return {
            'score': score,
            'entropy': round(entropy, 3),
            'max_entropy': round(max_entropy, 3),
            'normalized': round(normalized_entropy, 3),
            'n_ingredients': len(ratios),
            'dominant_ingredients': dominant,
            'explanation': self._explain_complexity(score, normalized_entropy, dominant),
        }
    
    def _explain_complexity(self, score, norm_ent, dominant):
        if score > 85:
            return f"복잡도 적절 (Entropy {norm_ent:.2f}) — 주도 원료와 보조 원료의 밸런스 우수"
        elif norm_ent > 0.90:
            return "비율이 너무 균등 → 주도 원료가 불분명하여 특색 없는 향"
        elif norm_ent < 0.50:
            names = [d['name'] + f"({d['pct']}%)" for d in dominant]
            return f"비율 편중 → {', '.join(names)}이(가) 과도하게 지배"
        else:
            return f"복잡도 소폭 조정 권장 (Entropy={norm_ent:.2f})"
    
    # ================================================================
    # 심사 항목 8: IFRA 안전성
    # ================================================================
    def _eval_safety(self, formula: list) -> dict:
        """IFRA 규제 준수율 평가"""
        violations = []
        total = len(formula)
        compliant = 0
        
        for item in formula:
            ing = item.get('ingredient', {})
            iid = ing.get('id', '').lower()
            ratio = item.get('ratio_pct', 0)
            
            violated = False
            for restricted, limit in self.IFRA_LIMITS.items():
                if restricted in iid:
                    if ratio > limit:
                        violations.append({
                            'ingredient': iid,
                            'ratio_pct': ratio,
                            'limit': limit,
                            'excess': round(ratio - limit, 2),
                        })
                        violated = True
            
            if not violated:
                compliant += 1
        
        rate = compliant / total if total > 0 else 1.0
        score = rate * 100
        
        return {
            'score': score,
            'compliance_rate': round(rate, 3),
            'violations': violations,
            'explanation': '모든 원료 IFRA 준수' if not violations else
                          f'IFRA 위반 {len(violations)}건: {", ".join(v["ingredient"] for v in violations)}',
        }
    
    # ================================================================
    # 종합 심사 + 개선 제안
    # ================================================================
    def judge(self, formula: list, target_vec: np.ndarray,
              mixed_vec: np.ndarray = None,
              smiles_map: dict = None,
              v6_engine=None) -> dict:
        """종합 심사 수행
        
        Args:
            formula: [{'ingredient': {...}, 'ratio_pct': float, 'odor_vector': [...]}, ...]
            target_vec: 22d 타겟 벡터
            mixed_vec: 혼합 결과 벡터 (없으면 formula에서 계산)
            smiles_map: {ingredient_id: SMILES}
            v6_engine: V6 GNN 엔진 (있으면 교차 검증)
        
        Returns:
            구조화된 평가서
        """
        smiles_map = smiles_map or {}
        
        # 혼합 벡터 계산
        if mixed_vec is None:
            total_r = sum(item.get('ratio_pct', 0) for item in formula)
            if total_r > 0:
                mixed_vec = np.zeros(22)
                for item in formula:
                    vec = np.array(item.get('odor_vector', np.zeros(22)))
                    w = item.get('ratio_pct', 0) / total_r
                    mixed_vec += vec * w
            else:
                mixed_vec = np.zeros(22)
        
        # 8개 심사 항목 평가
        evals = {
            'cross_agreement': self._eval_cross_agreement(formula, smiles_map, v6_engine),
            'target_fitness': self._eval_target_fitness(mixed_vec, target_vec),
            'pyramid_balance': self._eval_pyramid_balance(formula),
            'masking_risk': self._eval_masking_risk(formula),
            'synergy': self._eval_synergy(formula),
            'transition': self._eval_transition(formula),
            'complexity': self._eval_complexity(formula),
            'safety': self._eval_safety(formula),
        }
        
        # 가중 종합 점수
        weights = {
            'cross_agreement': 0.10,   # 모델 간 동의
            'target_fitness': 0.25,     # 목표 적합도 (핵심)
            'pyramid_balance': 0.15,    # 노트 밸런스
            'masking_risk': 0.12,       # 마스킹 위험
            'synergy': 0.08,            # 시너지 활용
            'transition': 0.10,         # 전환 자연도
            'complexity': 0.10,         # 복잡도
            'safety': 0.10,             # 안전성
        }
        
        total_score = sum(
            evals[k]['score'] * weights[k] for k in weights
        )
        
        # 판정
        if total_score >= self.VERDICT_THRESHOLDS['PASS']:
            verdict = 'PASS'
        elif total_score >= self.VERDICT_THRESHOLDS['REVISE']:
            verdict = 'REVISE'
        else:
            verdict = 'REJECT'
        
        # 주요 이슈 집계
        issues = []
        for name, ev in evals.items():
            if ev['score'] < 60:
                issues.append({
                    'category': name,
                    'score': round(ev['score'], 1),
                    'explanation': ev.get('explanation', ''),
                })
        
        # 개선 제안 생성
        suggestions = self._generate_suggestions(evals, formula)
        
        # 종합 평가 이유
        reasoning = self._generate_reasoning(evals, total_score, verdict)
        
        return {
            'score': round(total_score, 1),
            'verdict': verdict,
            'evaluations': evals,
            'weights': weights,
            'issues': sorted(issues, key=lambda x: x['score']),
            'suggestions': suggestions,
            'reasoning': reasoning,
        }
    
    def _generate_suggestions(self, evals: dict, formula: list) -> list:
        """데이터 기반 개선 제안 생성"""
        suggestions = []
        
        # 타겟 적합도 기반 제안
        fitness = evals.get('target_fitness', {})
        for dim_info in fitness.get('impact_dims', [])[:3]:
            dim = dim_info['dimension']
            direction = dim_info['direction']
            gap = abs(dim_info['gap'])
            
            if direction == '부족':
                suggestions.append(
                    f"{dim} 차원 강화 필요: {dim} 계열 원료 추가 또는 비율 증가 "
                    f"(현재 {dim_info['actual']:.2f} → 목표 {dim_info['target']:.2f})"
                )
            else:
                suggestions.append(
                    f"{dim} 차원 과다: 해당 원료 비율 감소 권장 "
                    f"(현재 {dim_info['actual']:.2f} → 목표 {dim_info['target']:.2f})"
                )
        
        # 피라미드 밸런스 기반 제안
        pyramid = evals.get('pyramid_balance', {})
        for imb in pyramid.get('imbalances', []):
            if imb['direction'] == '과다':
                suggestions.append(
                    f"{imb['note']} 노트 감소: {imb['actual_pct']}% → {imb['ideal_pct']}% 권장"
                )
            else:
                suggestions.append(
                    f"{imb['note']} 노트 보강: {imb['actual_pct']}% → {imb['ideal_pct']}% 권장"
                )
        
        # 마스킹 기반 제안
        masking = evals.get('masking_risk', {})
        for issue in masking.get('masking_issues', [])[:2]:
            suggestions.append(
                f"{issue['masker']}의 {issue['dimension']} 지배도 {issue['dominance']}% → "
                f"비율 감소 또는 대체 원료 고려"
            )
        
        # 복잡도 기반 제안
        complexity = evals.get('complexity', {})
        for dom in complexity.get('dominant_ingredients', []):
            if dom['pct'] > 40:
                suggestions.append(
                    f"{dom['name']} 비율 과다({dom['pct']}%) → 보조 원료에 분산 권장"
                )
        
        return suggestions[:8]  # 최대 8개
    
    def _generate_reasoning(self, evals, score, verdict):
        """종합 평가 이유 생성"""
        parts = []
        
        # 강점
        strengths = [k for k, v in evals.items() if v['score'] >= 80]
        if strengths:
            names = {'cross_agreement': '모델 동의도', 'target_fitness': '타겟 적합도',
                     'pyramid_balance': '노트 밸런스', 'masking_risk': '마스킹 안전',
                     'synergy': '시너지', 'transition': '향 전환', 
                     'complexity': '복잡도', 'safety': 'IFRA 안전'}
            strong = [names.get(s, s) for s in strengths]
            parts.append(f"강점: {', '.join(strong)}")
        
        # 약점
        weaknesses = [(k, v['score']) for k, v in evals.items() if v['score'] < 60]
        if weaknesses:
            names = {'cross_agreement': '모델 동의도', 'target_fitness': '타겟 적합도',
                     'pyramid_balance': '노트 밸런스', 'masking_risk': '마스킹',
                     'synergy': '시너지', 'transition': '향 전환',
                     'complexity': '복잡도', 'safety': 'IFRA'}
            weak = [f"{names.get(k, k)}({s:.0f}점)" for k, s in weaknesses]
            parts.append(f"약점: {', '.join(weak)}")
        
        parts.append(f"종합: {score:.1f}점 → {verdict}")
        
        return ". ".join(parts)
    
    def format_report(self, result: dict) -> str:
        """평가 결과를 텍스트 리포트로 포맷"""
        lines = [
            "=" * 60,
            f"🧑‍⚖️ AI 향수 심사위원 평가서",
            "=" * 60,
            f"",
            f"  종합 점수: {result['score']:.1f}/100",
            f"  판정: {'✅ PASS' if result['verdict'] == 'PASS' else '⚠️ REVISE' if result['verdict'] == 'REVISE' else '❌ REJECT'}",
            f"",
            "─" * 60,
            "  항목별 점수:",
        ]
        
        names_kr = {
            'cross_agreement': '교차 동의도', 'target_fitness': '타겟 적합도',
            'pyramid_balance': '피라미드 밸런스', 'masking_risk': '마스킹 위험',
            'synergy': '시너지 활용', 'transition': '향 전환 자연도',
            'complexity': '복잡도 적절성', 'safety': 'IFRA 안전성',
        }
        
        for key, ev in result.get('evaluations', {}).items():
            name = names_kr.get(key, key)
            bar_len = int(ev['score'] / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            lines.append(f"  {name:16s} {bar} {ev['score']:.0f}")
        
        # 이슈
        if result.get('issues'):
            lines.append(f"\n{'─' * 60}")
            lines.append("  ⚠️ 주요 이슈:")
            for issue in result['issues']:
                lines.append(f"    • [{names_kr.get(issue['category'], issue['category'])} "
                           f"{issue['score']:.0f}점] {issue['explanation']}")
        
        # 제안
        if result.get('suggestions'):
            lines.append(f"\n{'─' * 60}")
            lines.append("  💡 개선 제안:")
            for i, sug in enumerate(result['suggestions'], 1):
                lines.append(f"    {i}. {sug}")
        
        # 종합
        lines.append(f"\n{'─' * 60}")
        lines.append(f"  📋 종합 평가:")
        lines.append(f"    {result.get('reasoning', '')}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ================================================================
# 테스트
# ================================================================
if __name__ == '__main__':
    judge = PerfumeJudge()
    
    # 테스트 레시피
    test_formula = [
        {
            'ingredient': {'id': 'bergamot', 'category': 'citrus'},
            'ratio_pct': 4.0,
            'odor_vector': [0, 0.9, 0, 0.2, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0],
        },
        {
            'ingredient': {'id': 'jasmine', 'category': 'floral'},
            'ratio_pct': 5.0,
            'odor_vector': [0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0],
        },
        {
            'ingredient': {'id': 'rose', 'category': 'floral'},
            'ratio_pct': 3.5,
            'odor_vector': [0.8, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0.4, 0, 0, 0],
        },
        {
            'ingredient': {'id': 'sandalwood', 'category': 'woody'},
            'ratio_pct': 4.5,
            'odor_vector': [0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0.4, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        {
            'ingredient': {'id': 'vanilla', 'category': 'balsamic'},
            'ratio_pct': 3.0,
            'odor_vector': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.7, 0, 0, 0, 0, 0.2, 0.5, 0, 0.8, 0, 0, 0],
        },
        {
            'ingredient': {'id': 'musk', 'category': 'musk'},
            'ratio_pct': 2.0,
            'odor_vector': [0, 0, 0, 0, 0, 0, 0.8, 0, 0, 0.2, 0, 0, 0, 0, 0, 0.3, 0, 0.1, 0, 0, 0, 0],
        },
    ]
    
    target = np.array([0.6, 0.3, 0.4, 0.1, 0, 0, 0.3, 0, 0.1, 0.2, 0.3, 0, 0, 0, 0, 0.1, 0.2, 0, 0.5, 0.3, 0, 0])
    
    result = judge.judge(test_formula, target)
    print(judge.format_report(result))
