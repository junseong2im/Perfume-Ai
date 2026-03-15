"""
Commercial Prior — 원료 공기출현(Co-occurrence) PMI 행렬 + KL 패널티
====================================================================
698개 레시피 데이터에서 "성공한 향수의 뼈대"를 학습하여
AI가 생성한 레시피의 상업적 그럴듯함(Commercial Plausibility)을 평가.

사용법:
    from scripts.commercial_prior import CommercialPrior
    prior = CommercialPrior()
    score = prior.plausibility_score(['bergamot', 'jasmine', 'sandalwood', 'musk'])
    # → 0.82 (그럴듯한 조합) vs 0.15 (이상한 조합)
"""
import json
import os
import math
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations


class CommercialPrior:
    """성공한 향수 레시피의 통계적 사전 확률 모델"""
    
    def __init__(self, data_path=None):
        if data_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base, 'data', 'recipe_training_data.json')
        
        self.pmi_matrix = {}       # (a, b) → PMI score
        self.ingredient_freq = {}  # ingredient → frequency
        self.note_dist = {}        # note_type distribution per ingredient
        self.category_dist = {}    # category distribution in successful perfumes
        self.n_recipes = 0
        self.cooccurrence = defaultdict(int)
        
        self._build(data_path)
    
    def _build(self, data_path):
        """698개 레시피에서 통계 모델 구축"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                recipes = json.load(f)
        except Exception as e:
            print(f"[CommercialPrior] Failed to load data: {e}")
            return
        
        self.n_recipes = len(recipes)
        ing_count = Counter()
        pair_count = defaultdict(int)
        note_count = defaultdict(lambda: Counter())
        cat_count = Counter()
        
        for recipe in recipes:
            ings = recipe.get('ingredients', [])
            ids = [i['id'] for i in ings]
            
            # 단일 빈도
            for i in ings:
                ing_count[i['id']] += 1
                note_count[i['id']][i.get('note', 'middle')] += 1
                cat_count[i.get('note', 'middle')] += 1
            
            # 공기출현 빈도
            for a, b in combinations(sorted(set(ids)), 2):
                pair_count[(a, b)] += 1
                self.cooccurrence[(a, b)] += 1
                self.cooccurrence[(b, a)] += 1
        
        # 단일 빈도 정규화
        self.ingredient_freq = {k: v / self.n_recipes for k, v in ing_count.items()}
        
        # Note type 분포
        for ing_id, counts in note_count.items():
            total = sum(counts.values())
            self.note_dist[ing_id] = {k: v / total for k, v in counts.items()}
        
        # Category 분포 (이상적 top:mid:base 비율)
        total_notes = sum(cat_count.values())
        self.category_dist = {k: v / total_notes for k, v in cat_count.items()}
        
        # PMI 계산
        for (a, b), count in pair_count.items():
            p_ab = count / self.n_recipes
            p_a = self.ingredient_freq.get(a, 1e-6)
            p_b = self.ingredient_freq.get(b, 1e-6)
            
            pmi = math.log2(max(p_ab / (p_a * p_b), 1e-10))
            self.pmi_matrix[(a, b)] = pmi
            self.pmi_matrix[(b, a)] = pmi
        
        print(f"[CommercialPrior] Built from {self.n_recipes} recipes, "
              f"{len(ing_count)} ingredients, {len(pair_count)} pairs")
    
    def plausibility_score(self, ingredient_ids):
        """레시피의 상업적 그럴듯함 점수 (0~1)
        
        높은 점수 = 성공한 향수들과 비슷한 원료 조합
        낮은 점수 = 이질적/엽기적 조합
        """
        if not ingredient_ids or len(ingredient_ids) < 2:
            return 0.5
        
        # 1) PMI 기반 조합 점수
        pmi_scores = []
        for a, b in combinations(sorted(set(ingredient_ids)), 2):
            pmi = self.pmi_matrix.get((a, b), -5.0)  # 미등장 쌍은 -5
            pmi_scores.append(pmi)
        
        if pmi_scores:
            avg_pmi = np.mean(pmi_scores)
            # PMI를 0~1로 정규화 (sigmoid-like)
            pmi_score = 1.0 / (1.0 + math.exp(-avg_pmi * 0.5))
        else:
            pmi_score = 0.5
        
        # 2) 원료 인기도 점수
        freq_scores = [self.ingredient_freq.get(i, 0.001) for i in ingredient_ids]
        avg_freq = np.mean(freq_scores)
        freq_score = min(1.0, avg_freq * 5)  # 20% 이상 = 1.0
        
        # 3) 종합 (PMI 70% + 인기도 30%)
        final = pmi_score * 0.7 + freq_score * 0.3
        return round(float(final), 4)
    
    def note_balance_score(self, ingredients_with_notes):
        """노트 밸런스 점수 (top:mid:base 비율 적절성)
        
        Args:
            ingredients_with_notes: [{'id': 'bergamot', 'note': 'top'}, ...]
        """
        note_counts = Counter(i.get('note', 'middle') for i in ingredients_with_notes)
        total = sum(note_counts.values())
        if total == 0:
            return 0.5
        
        actual = {k: note_counts.get(k, 0) / total for k in ['top', 'middle', 'base']}
        ideal = self.category_dist if self.category_dist else {'top': 0.25, 'middle': 0.40, 'base': 0.35}
        
        # KL Divergence (actual vs ideal)
        kl = 0
        for k in ['top', 'middle', 'base']:
            p = actual.get(k, 1e-6)
            q = ideal.get(k, 1e-6)
            if p > 0:
                kl += p * math.log(p / max(q, 1e-6))
        
        # KL → 0~1 점수 (KL=0이면 1.0, KL 크면 0)
        score = math.exp(-kl * 3.0)
        return round(float(score), 4)
    
    def get_top_cooccurrences(self, ingredient_id, top_k=10):
        """특정 원료와 가장 자주 함께 쓰이는 원료"""
        pairs = [(other, count) for (a, other), count 
                 in self.cooccurrence.items() if a == ingredient_id]
        pairs.sort(key=lambda x: -x[1])
        return pairs[:top_k]


# 싱글톤 인스턴스
_prior = None

def get_commercial_prior():
    """CommercialPrior 싱글톤"""
    global _prior
    if _prior is None:
        _prior = CommercialPrior()
    return _prior
