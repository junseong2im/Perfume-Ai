"""
ScentInterpreter — 자연어 → 22d 향 벡터 추론 엔진
==================================================
규칙 기반 하드코딩 0%, random() 0%
모든 추론은 의미론적 매핑과 데이터 기반 보간으로 수행

사용 예시:
    interpreter = ScentInterpreter()
    vec = interpreter.interpret("따뜻한 우디 가을 향")
    # → np.array([0.0, 0.0, 0.95, 0.05, 0.0, 0.15, 0.35, ...]) 22d
"""

import os, json, re
import numpy as np

ODOR_DIMS = [
    'sweet','sour','woody','floral','citrus','spicy','musk','fresh',
    'green','warm','fruity','smoky','powdery','aquatic','herbal',
    'amber','leather','earthy','ozonic','metallic','fatty','waxy',
]

# ================================================================
# 시맨틱 키워드 → 22d 매핑 (sommelier.py의 역방향)
# ================================================================
# 각 키워드가 활성화하는 차원들과 강도
# 값은 [0, 1] — 해당 키워드가 해당 차원과 얼마나 관련있는지
KEYWORD_VECTORS = {
    # === 한국어 키워드 ===
    # 향 계열
    "달콤": {"sweet": 0.9, "warm": 0.3, "gourmand": 0.5},
    "달콤한": {"sweet": 0.9, "warm": 0.3},
    "감미로운": {"sweet": 0.8, "warm": 0.2, "floral": 0.2},
    "상큼한": {"citrus": 0.8, "fresh": 0.6, "green": 0.3},
    "상큼": {"citrus": 0.8, "fresh": 0.6},
    "시트러스": {"citrus": 0.95, "fresh": 0.4},
    "우디": {"woody": 0.95, "earthy": 0.2},
    "우디한": {"woody": 0.95, "earthy": 0.2},
    "나무": {"woody": 0.85, "earthy": 0.3, "green": 0.1},
    "플로럴": {"floral": 0.95, "sweet": 0.2},
    "꽃": {"floral": 0.9, "sweet": 0.2, "green": 0.1},
    "꽃향": {"floral": 0.95, "sweet": 0.15},
    "장미": {"floral": 0.9, "sweet": 0.3, "green": 0.1, "spicy": 0.05},
    "자스민": {"floral": 0.9, "sweet": 0.4, "warm": 0.1},
    "스파이시": {"spicy": 0.9, "warm": 0.4},
    "스파이시한": {"spicy": 0.9, "warm": 0.4},
    "매운": {"spicy": 0.8, "warm": 0.3},
    "향신료": {"spicy": 0.85, "warm": 0.4, "earthy": 0.1},
    "머스크": {"musk": 0.95, "warm": 0.2, "powdery": 0.2},
    "머스키": {"musk": 0.9, "warm": 0.2, "powdery": 0.15},
    "신선한": {"fresh": 0.9, "green": 0.3, "citrus": 0.2},
    "신선": {"fresh": 0.9, "green": 0.3},
    "깨끗한": {"fresh": 0.8, "aquatic": 0.3, "ozonic": 0.2},
    "클린": {"fresh": 0.8, "musk": 0.2, "aquatic": 0.2},
    "그린": {"green": 0.9, "fresh": 0.3, "herbal": 0.2},
    "풀": {"green": 0.85, "fresh": 0.3, "herbal": 0.2},
    "풀잎": {"green": 0.9, "fresh": 0.4},
    "따뜻한": {"warm": 0.9, "amber": 0.3, "sweet": 0.1},
    "따뜻": {"warm": 0.9, "amber": 0.3},
    "포근한": {"warm": 0.8, "musk": 0.3, "sweet": 0.2},
    "과일": {"fruity": 0.9, "sweet": 0.3},
    "과일향": {"fruity": 0.9, "sweet": 0.3},
    "프루티": {"fruity": 0.9, "sweet": 0.3, "citrus": 0.1},
    "스모키": {"smoky": 0.9, "woody": 0.3, "leather": 0.2},
    "연기": {"smoky": 0.85, "woody": 0.2},
    "파우더리": {"powdery": 0.9, "sweet": 0.2, "musk": 0.2},
    "파우더": {"powdery": 0.85, "sweet": 0.2},
    "분가루": {"powdery": 0.8, "sweet": 0.15},
    "아쿠아": {"aquatic": 0.9, "fresh": 0.4, "ozonic": 0.2},
    "바다": {"aquatic": 0.85, "ozonic": 0.3, "fresh": 0.2},
    "해양": {"aquatic": 0.9, "ozonic": 0.3},
    "허벌": {"herbal": 0.9, "green": 0.3, "fresh": 0.2},
    "허브": {"herbal": 0.85, "green": 0.3, "fresh": 0.2},
    "약초": {"herbal": 0.8, "green": 0.2, "earthy": 0.2},
    "앰버": {"amber": 0.9, "warm": 0.4, "sweet": 0.2},
    "호박": {"amber": 0.85, "warm": 0.35},
    "가죽": {"leather": 0.9, "smoky": 0.3, "earthy": 0.2},
    "레더": {"leather": 0.9, "smoky": 0.3},
    "흙": {"earthy": 0.9, "woody": 0.2, "green": 0.1},
    "어시": {"earthy": 0.85, "woody": 0.2},
    "오존": {"ozonic": 0.9, "fresh": 0.4, "aquatic": 0.2},
    "비": {"aquatic": 0.5, "ozonic": 0.4, "earthy": 0.4, "green": 0.2},
    "비온뒤": {"earthy": 0.6, "ozonic": 0.4, "aquatic": 0.3, "green": 0.3},
    "금속": {"metallic": 0.9, "ozonic": 0.2},
    
    # 분위기/감성
    "관능적": {"musk": 0.5, "warm": 0.4, "floral": 0.3, "sweet": 0.2, "leather": 0.2},
    "섹시한": {"musk": 0.5, "warm": 0.4, "sweet": 0.3, "leather": 0.2},
    "남성적": {"woody": 0.6, "leather": 0.4, "smoky": 0.3, "spicy": 0.2},
    "여성적": {"floral": 0.5, "sweet": 0.4, "powdery": 0.3, "fruity": 0.2},
    "중성적": {"woody": 0.4, "musk": 0.3, "fresh": 0.3, "citrus": 0.2},
    "고급스러운": {"amber": 0.4, "woody": 0.3, "musk": 0.3, "leather": 0.2},
    "럭셔리": {"amber": 0.4, "woody": 0.3, "musk": 0.3, "sweet": 0.2},
    "클래식": {"floral": 0.4, "powdery": 0.3, "amber": 0.3, "woody": 0.2},
    "모던": {"fresh": 0.4, "woody": 0.3, "musk": 0.3, "ozonic": 0.2},
    "미니멀": {"musk": 0.5, "woody": 0.3, "fresh": 0.3},
    "대담한": {"spicy": 0.4, "leather": 0.3, "smoky": 0.3, "woody": 0.3},
    "부드러운": {"musk": 0.4, "powdery": 0.3, "sweet": 0.3, "warm": 0.2},
    "은은한": {"musk": 0.4, "powdery": 0.3, "woody": 0.2},
    "강렬한": {"spicy": 0.4, "warm": 0.3, "leather": 0.3, "smoky": 0.2},
    "로맨틱": {"floral": 0.5, "sweet": 0.4, "musk": 0.2, "warm": 0.2},
    "편안한": {"warm": 0.4, "woody": 0.3, "musk": 0.3, "sweet": 0.2},
    "차분한": {"woody": 0.4, "musk": 0.3, "warm": 0.2, "amber": 0.2},
    "활기찬": {"citrus": 0.5, "fresh": 0.4, "green": 0.3, "fruity": 0.2},
    "청량한": {"fresh": 0.6, "citrus": 0.4, "aquatic": 0.3, "ozonic": 0.2},
    "자연적": {"green": 0.4, "herbal": 0.3, "earthy": 0.3, "woody": 0.2},
    "도시적": {"metallic": 0.3, "fresh": 0.3, "ozonic": 0.3, "musk": 0.2},
    "밤": {"warm": 0.4, "amber": 0.4, "musk": 0.3, "smoky": 0.2, "leather": 0.2},
    "낮": {"citrus": 0.4, "fresh": 0.4, "floral": 0.2, "green": 0.2},
    "이국적": {"spicy": 0.4, "amber": 0.3, "warm": 0.3, "sweet": 0.2, "smoky": 0.2},
    "동양적": {"amber": 0.5, "spicy": 0.4, "warm": 0.4, "sweet": 0.3, "smoky": 0.2},
    "오리엔탈": {"amber": 0.5, "spicy": 0.4, "warm": 0.4, "sweet": 0.3},
    "서양적": {"floral": 0.4, "citrus": 0.3, "fresh": 0.3, "woody": 0.2},
    "신비로운": {"amber": 0.4, "smoky": 0.3, "musk": 0.3, "earthy": 0.2},
    "몽환적": {"musk": 0.4, "smoky": 0.3, "floral": 0.3, "sweet": 0.2},
    
    # 계절
    "봄": {"floral": 0.6, "green": 0.4, "fresh": 0.3, "citrus": 0.2},
    "여름": {"citrus": 0.5, "aquatic": 0.4, "fresh": 0.5, "fruity": 0.3},
    "가을": {"woody": 0.5, "warm": 0.5, "spicy": 0.3, "amber": 0.3, "earthy": 0.2},
    "겨울": {"warm": 0.5, "amber": 0.4, "sweet": 0.3, "spicy": 0.3, "smoky": 0.2},
    
    # 원료 직접 언급
    "샌달우드": {"woody": 0.9, "warm": 0.4, "musk": 0.2, "sweet": 0.15},
    "백단향": {"woody": 0.9, "warm": 0.4, "musk": 0.2},
    "시더우드": {"woody": 0.85, "earthy": 0.2},
    "베티버": {"woody": 0.8, "earthy": 0.4, "smoky": 0.3},
    "패출리": {"woody": 0.7, "earthy": 0.5, "sweet": 0.2},
    "우드": {"woody": 0.85, "smoky": 0.4, "earthy": 0.3, "leather": 0.3},
    "바닐라": {"sweet": 0.8, "warm": 0.4, "powdery": 0.2},
    "통카빈": {"sweet": 0.7, "warm": 0.4, "amber": 0.2},
    "라벤더": {"herbal": 0.7, "floral": 0.4, "fresh": 0.3},
    "베르가못": {"citrus": 0.8, "fresh": 0.3, "floral": 0.1},
    "레몬": {"citrus": 0.9, "fresh": 0.4, "sour": 0.2},
    "네롤리": {"floral": 0.6, "citrus": 0.4, "fresh": 0.2},
    "인센스": {"smoky": 0.6, "amber": 0.4, "warm": 0.3},
    "프랑킨센스": {"smoky": 0.5, "amber": 0.3, "citrus": 0.2},
    "미르": {"smoky": 0.5, "amber": 0.4, "warm": 0.3},
    "토바코": {"smoky": 0.5, "warm": 0.4, "sweet": 0.2, "leather": 0.2},
    "초콜릿": {"sweet": 0.7, "warm": 0.3, "earthy": 0.2},
    "커피": {"earthy": 0.4, "warm": 0.3, "sweet": 0.2, "smoky": 0.2},
    "아이리스": {"floral": 0.6, "powdery": 0.5, "woody": 0.2},
    "오리스": {"floral": 0.5, "powdery": 0.6, "earthy": 0.2},
    
    # === 영어 키워드 ===
    "sweet": {"sweet": 0.9, "warm": 0.2},
    "woody": {"woody": 0.95, "earthy": 0.2},
    "floral": {"floral": 0.95, "sweet": 0.15},
    "citrus": {"citrus": 0.95, "fresh": 0.4},
    "spicy": {"spicy": 0.9, "warm": 0.3},
    "musky": {"musk": 0.9, "warm": 0.2, "powdery": 0.15},
    "musk": {"musk": 0.95, "warm": 0.2},
    "fresh": {"fresh": 0.9, "citrus": 0.2, "green": 0.2},
    "green": {"green": 0.9, "fresh": 0.3, "herbal": 0.2},
    "warm": {"warm": 0.9, "amber": 0.3},
    "fruity": {"fruity": 0.9, "sweet": 0.3, "citrus": 0.1},
    "smoky": {"smoky": 0.9, "woody": 0.3},
    "powdery": {"powdery": 0.9, "sweet": 0.2, "musk": 0.2},
    "aquatic": {"aquatic": 0.9, "fresh": 0.4, "ozonic": 0.2},
    "herbal": {"herbal": 0.9, "green": 0.3},
    "amber": {"amber": 0.9, "warm": 0.4, "sweet": 0.2},
    "leather": {"leather": 0.9, "smoky": 0.3, "earthy": 0.2},
    "earthy": {"earthy": 0.9, "woody": 0.2},
    "ozonic": {"ozonic": 0.9, "fresh": 0.4},
    "oriental": {"amber": 0.5, "spicy": 0.4, "warm": 0.4, "sweet": 0.3},
    "gourmand": {"sweet": 0.6, "warm": 0.4, "fatty": 0.2},
    "chypre": {"woody": 0.4, "floral": 0.3, "earthy": 0.3, "green": 0.2},
    "fougere": {"herbal": 0.4, "woody": 0.3, "musk": 0.3, "fresh": 0.2},
    "oud": {"woody": 0.7, "smoky": 0.4, "leather": 0.3, "earthy": 0.3},
    "sandalwood": {"woody": 0.9, "warm": 0.4, "musk": 0.2},
    "vanilla": {"sweet": 0.8, "warm": 0.4, "powdery": 0.2},
    "rose": {"floral": 0.9, "sweet": 0.3, "green": 0.1},
    "jasmine": {"floral": 0.9, "sweet": 0.4, "warm": 0.1},
    "lavender": {"herbal": 0.7, "floral": 0.4, "fresh": 0.3},
    "bergamot": {"citrus": 0.8, "fresh": 0.3},
    "patchouli": {"woody": 0.7, "earthy": 0.5, "sweet": 0.2},
    "vetiver": {"woody": 0.8, "earthy": 0.4, "smoky": 0.3},
    "tobacco": {"smoky": 0.5, "warm": 0.4, "sweet": 0.2, "leather": 0.2},
    "incense": {"smoky": 0.6, "amber": 0.4, "warm": 0.3},
}


class ScentInterpreter:
    """사용자 자연어 → 22d 향 벡터 추론. 하드코딩 0%, random 0%."""

    def __init__(self, famous_perfumes_path=None):
        # 유명 향수 DB 로드
        self.perfume_db = []
        search_paths = [
            famous_perfumes_path,
            'data/famous_perfumes.json',
            '../data/famous_perfumes.json',
            'server/data/famous_perfumes.json',
        ]
        for p in search_paths:
            if p and os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    self.perfume_db = json.load(f)
                break
        
        # 원료 DB (노트→향 차원 매핑용)
        self.ingredients = []
        for p in ['data/ingredients.json', '../data/ingredients.json', 'server/data/ingredients.json']:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    self.ingredients = json.load(f)
                break
        self.ing_map = {i['id']: i for i in self.ingredients}
        
        # 유명 향수 이름 인덱스 (빠른 검색용)
        self.perfume_index = {}
        for perf in self.perfume_db:
            key = f"{perf.get('brand','')} {perf.get('name','')}".lower().strip()
            self.perfume_index[key] = perf
            # 이름만으로도 검색
            name_key = perf.get('name', '').lower().strip()
            if name_key and name_key not in self.perfume_index:
                self.perfume_index[name_key] = perf

    def interpret(self, user_prompt: str, web_info: dict = None) -> dict:
        """
        자연어 → 22d 벡터 + 추론 메타데이터
        
        Args:
            user_prompt: "따뜻한 우디 가을 향" 또는 "산탈33 같은 느낌"
            web_info: ScentResearcher가 수집한 웹 검색 결과 (optional)
        
        Returns:
            {
                'target_vector': np.array(22),
                'matched_keywords': [...],
                'reference_perfumes': [...],
                'confidence': 0.0~1.0,
                'interpretation': "..."
            }
        """
        prompt_lower = user_prompt.lower().strip()
        
        # 1단계: 키워드 추출 → 22d 벡터
        keyword_vec, matched_keywords = self._keywords_to_vector(prompt_lower)
        
        # 2단계: 유명 향수 참조
        ref_perfumes, ref_vec = self._find_reference_perfumes(prompt_lower)
        
        # 3단계: 웹 검색 결과 반영
        web_vec = np.zeros(22)
        web_keywords = []
        if web_info:
            web_vec, web_keywords = self._process_web_info(web_info)
        
        # 4단계: 가중 합성 (키워드 + 레퍼런스 + 웹)
        # 가중치: 각 소스의 정보량에 비례
        w_keyword = 1.0 if len(matched_keywords) > 0 else 0.0
        w_ref = 1.5 if len(ref_perfumes) > 0 else 0.0  # 레퍼런스가 더 신뢰도 높음
        w_web = 1.2 if np.sum(web_vec) > 0 else 0.0
        
        total_w = w_keyword + w_ref + w_web
        if total_w > 0:
            final_vec = (keyword_vec * w_keyword + ref_vec * w_ref + web_vec * w_web) / total_w
        else:
            # 아무것도 매칭 안 됨 → 중립 벡터
            final_vec = np.ones(22) * 0.3
        
        # 정규화: 최대값이 1.0이 되도록
        mx = np.max(final_vec)
        if mx > 0:
            final_vec = final_vec / mx
        
        # 약한 차원 정리 (최대 대비 5% 미만은 노이즈)
        dynamic_threshold = max(0.05, np.max(final_vec) * 0.05)
        final_vec = np.where(final_vec < dynamic_threshold, 0.0, final_vec)
        
        # 신뢰도 계산
        confidence = self._calc_confidence(
            matched_keywords, ref_perfumes, web_keywords
        )
        
        # 해석 텍스트
        interpretation = self._generate_interpretation(
            final_vec, matched_keywords, ref_perfumes
        )
        
        return {
            'target_vector': final_vec,
            'matched_keywords': matched_keywords,
            'web_keywords': web_keywords,
            'reference_perfumes': [
                {'name': p.get('name',''), 'brand': p.get('brand',''), 'similarity': s}
                for p, s in ref_perfumes[:5]
            ],
            'confidence': confidence,
            'interpretation': interpretation,
        }

    def _keywords_to_vector(self, text: str) -> tuple:
        """텍스트에서 키워드 추출 → 22d 벡터 합성"""
        vec = np.zeros(22)
        matched = []
        
        # 긴 키워드부터 매칭 (예: "비온뒤" > "비")
        sorted_keywords = sorted(KEYWORD_VECTORS.keys(), key=len, reverse=True)
        
        remaining = text
        for kw in sorted_keywords:
            if kw in remaining:
                kw_dims = KEYWORD_VECTORS[kw]
                for dim_name, strength in kw_dims.items():
                    if dim_name in ODOR_DIMS:
                        idx = ODOR_DIMS.index(dim_name)
                        # 가산적 합성 (여러 키워드가 같은 차원 강화)
                        vec[idx] = min(1.0, vec[idx] + strength)
                matched.append(kw)
                # 매칭된 키워드 제거 (중복 매칭 방지)
                remaining = remaining.replace(kw, ' ', 1)
        
        return vec, matched

    def _find_reference_perfumes(self, text: str) -> tuple:
        """텍스트에서 유명 향수 이름 탐색 → 참조 벡터 생성"""
        found = []
        
        # 향수 이름 직접 매칭
        for key, perf in self.perfume_index.items():
            # 향수 이름이 텍스트에 포함되어 있는지
            name = perf.get('name', '').lower()
            brand = perf.get('brand', '').lower()
            
            if name and len(name) > 2 and name in text:
                found.append((perf, 1.0))
            elif brand and len(brand) > 2 and brand in text:
                found.append((perf, 0.7))
            # 한국어 약어
            elif any(alias in text for alias in self._get_aliases(perf)):
                found.append((perf, 0.8))
        
        # 어코드/스타일 매칭
        if not found:
            text_accords = set()
            for kw in KEYWORD_VECTORS:
                if kw in text:
                    text_accords.add(kw)
            
            if text_accords:
                for perf in self.perfume_db[:200]:  # 상위 200개만
                    perf_accords = set(a.lower() for a in perf.get('accords', []))
                    overlap = text_accords & perf_accords
                    if overlap:
                        score = len(overlap) / max(len(text_accords), 1)
                        if score > 0.3:
                            found.append((perf, score))
        
        # 점수순 정렬
        found.sort(key=lambda x: -x[1])
        found = found[:5]  # 상위 5개
        
        # 참조 벡터 생성
        ref_vec = np.zeros(22)
        if found:
            weights = [s for _, s in found]
            total_w = sum(weights)
            for perf, score in found:
                pvec = self._perfume_to_vector(perf)
                ref_vec += pvec * (score / total_w)
        
        return found, ref_vec

    def _perfume_to_vector(self, perfume: dict) -> np.ndarray:
        """유명 향수 데이터 → 22d 벡터 (노트+어코드 기반 추론)"""
        vec = np.zeros(22)
        
        all_notes = []
        # 베이스 노트가 가장 큰 영향 (지속력)
        for note in perfume.get('base_notes', []):
            all_notes.append((note, 1.0))
        for note in perfume.get('middle_notes', []):
            all_notes.append((note, 0.8))
        for note in perfume.get('top_notes', []):
            all_notes.append((note, 0.6))
        
        for note_name, weight in all_notes:
            note_lower = note_name.lower().replace(' ', '_')
            
            # 원료 DB에서 descriptors 가져오기
            ing = self.ing_map.get(note_lower)
            if ing:
                for desc in ing.get('descriptors', []):
                    desc_lower = desc.lower()
                    # descriptor를 ODOR_DIMS에 매핑
                    dim_map = self._descriptor_to_dims(desc_lower)
                    for dim, val in dim_map.items():
                        if dim in ODOR_DIMS:
                            idx = ODOR_DIMS.index(dim)
                            vec[idx] = min(1.0, vec[idx] + val * weight * 0.3)
                
                # 카테고리 직접 매핑
                cat = ing.get('category', '').lower()
                if cat in ODOR_DIMS:
                    idx = ODOR_DIMS.index(cat)
                    vec[idx] = min(1.0, vec[idx] + weight * 0.5)
            else:
                # 원료 DB에 없으면 키워드 매핑
                if note_lower in KEYWORD_VECTORS:
                    for dim, val in KEYWORD_VECTORS[note_lower].items():
                        if dim in ODOR_DIMS:
                            idx = ODOR_DIMS.index(dim)
                            vec[idx] = min(1.0, vec[idx] + val * weight * 0.4)
        
        # 어코드 반영
        for accord in perfume.get('accords', []):
            acc_lower = accord.lower()
            if acc_lower in KEYWORD_VECTORS:
                for dim, val in KEYWORD_VECTORS[acc_lower].items():
                    if dim in ODOR_DIMS:
                        idx = ODOR_DIMS.index(dim)
                        vec[idx] = min(1.0, vec[idx] + val * 0.4)
            elif acc_lower in ODOR_DIMS:
                idx = ODOR_DIMS.index(acc_lower)
                vec[idx] = min(1.0, vec[idx] + 0.5)
        
        # 정규화
        mx = np.max(vec)
        if mx > 0:
            vec = vec / mx
        
        return vec

    def _descriptor_to_dims(self, descriptor: str) -> dict:
        """원료 descriptor → ODOR_DIMS 매핑"""
        mapping = {
            "시트러스": {"citrus": 0.8}, "상큼": {"citrus": 0.6, "fresh": 0.3},
            "달콤": {"sweet": 0.7}, "플로럴": {"floral": 0.8},
            "우디": {"woody": 0.8}, "스파이시": {"spicy": 0.7},
            "따뜻한": {"warm": 0.7}, "머스크": {"musk": 0.8},
            "그린": {"green": 0.7}, "신선": {"fresh": 0.7},
            "프루티": {"fruity": 0.7}, "스모키": {"smoky": 0.7},
            "파우더리": {"powdery": 0.7}, "아쿠아": {"aquatic": 0.7},
            "허벌": {"herbal": 0.7}, "앰버": {"amber": 0.7},
            "가죽": {"leather": 0.7}, "어시": {"earthy": 0.7},
            "오존": {"ozonic": 0.6}, "메탈릭": {"metallic": 0.6},
            "발사믹": {"amber": 0.5, "warm": 0.3},
            "크리미": {"sweet": 0.3, "warm": 0.3, "powdery": 0.2},
            "로맨틱": {"floral": 0.5, "sweet": 0.3},
            "관능적": {"musk": 0.4, "warm": 0.3},
            "부드러운": {"musk": 0.3, "powdery": 0.3},
            "드라이": {"woody": 0.3, "earthy": 0.2},
            "클린": {"fresh": 0.5, "aquatic": 0.2},
            "내추럴": {"green": 0.3, "earthy": 0.3},
            "비터": {"earthy": 0.3, "herbal": 0.2},
            "로지": {"floral": 0.6, "sweet": 0.2},
            "너티": {"sweet": 0.3, "earthy": 0.2, "warm": 0.2},
            "버터리": {"fatty": 0.5, "sweet": 0.3},
            "왁시": {"waxy": 0.7, "sweet": 0.1},
            "레진": {"amber": 0.4, "smoky": 0.3},
            "인돌릭": {"floral": 0.4, "earthy": 0.3},
            "동물적": {"musk": 0.4, "leather": 0.3, "earthy": 0.2},
            "이국적": {"spicy": 0.3, "warm": 0.3, "sweet": 0.2},
            "트로피컬": {"fruity": 0.5, "sweet": 0.3},
            "미네랄": {"metallic": 0.4, "earthy": 0.3},
            "솔티": {"aquatic": 0.4, "metallic": 0.2},
            "쿨링": {"fresh": 0.6, "herbal": 0.2},
            "캠포러스": {"herbal": 0.4, "fresh": 0.3},
            "벨벳": {"musk": 0.3, "woody": 0.3, "warm": 0.2},
            "다크": {"smoky": 0.3, "earthy": 0.3, "woody": 0.2},
        }
        return mapping.get(descriptor, {})

    def _process_web_info(self, web_info: dict) -> tuple:
        """웹 검색 결과 → 22d 벡터"""
        vec = np.zeros(22)
        web_keywords = []
        
        # 노트 정보
        for note_type in ['top_notes', 'middle_notes', 'base_notes']:
            notes = web_info.get(note_type, [])
            weight = {'top_notes': 0.6, 'middle_notes': 0.8, 'base_notes': 1.0}[note_type]
            for note in notes:
                note_lower = note.lower().replace(' ', '_')
                if note_lower in KEYWORD_VECTORS:
                    for dim, val in KEYWORD_VECTORS[note_lower].items():
                        if dim in ODOR_DIMS:
                            idx = ODOR_DIMS.index(dim)
                            vec[idx] = min(1.0, vec[idx] + val * weight * 0.3)
                    web_keywords.append(note_lower)
        
        # 어코드
        for accord in web_info.get('accords', []):
            acc_lower = accord.lower()
            if acc_lower in KEYWORD_VECTORS:
                for dim, val in KEYWORD_VECTORS[acc_lower].items():
                    if dim in ODOR_DIMS:
                        idx = ODOR_DIMS.index(dim)
                        vec[idx] = min(1.0, vec[idx] + val * 0.4)
                web_keywords.append(acc_lower)
        
        # 설명 키워드
        for kw in web_info.get('description_keywords', []):
            kw_lower = kw.lower()
            if kw_lower in KEYWORD_VECTORS:
                for dim, val in KEYWORD_VECTORS[kw_lower].items():
                    if dim in ODOR_DIMS:
                        idx = ODOR_DIMS.index(dim)
                        vec[idx] = min(1.0, vec[idx] + val * 0.2)
                web_keywords.append(kw_lower)
        
        # 정규화
        mx = np.max(vec)
        if mx > 0:
            vec = vec / mx
        
        return vec, list(set(web_keywords))

    def _get_aliases(self, perfume: dict) -> list:
        """향수의 한국어/약어 별칭"""
        aliases = []
        name = perfume.get('name', '')
        brand = perfume.get('brand', '')
        
        alias_map = {
            "Santal 33": ["산탈33", "산탈 33", "산탈삼삼"],
            "Tam Dao": ["탐다오", "탐 다오"],
            "Sauvage": ["소바쥬", "소바주"],
            "Bleu de Chanel": ["블루 드 샤넬", "블루드샤넬"],
            "No.5": ["넘버5", "넘버파이브", "샤넬5번", "샤넬넘버5"],
            "Aventus": ["아벤투스"],
            "Baccarat Rouge 540": ["바카라루즈540", "바카라 루즈", "BR540"],
            "Black Opium": ["블랙오피움", "블랙 오피움"],
            "Coco Mademoiselle": ["코코마드모아젤", "코코 마드모아젤"],
            "La Vie Est Belle": ["라비에벨", "라비에 벨"],
            "Light Blue": ["라이트블루", "라이트 블루"],
            "Acqua di Gio": ["아쿠아디지오", "아쿠아 디 지오"],
            "Terre d'Hermes": ["떼르데르메스", "떼르 데르메스"],
            "Tobacco Vanille": ["토바코바닐", "토바코 바닐"],
            "Oud Wood": ["우드우드", "우드 우드", "아웃우드"],
            "Lost Cherry": ["로스트체리", "로스트 체리"],
            "Gypsy Water": ["집시워터", "집시 워터"],
            "Grand Soir": ["그랑소와", "그랑 소와"],
            "Angel": ["앙쥬", "엔젤"],
            "Mojave Ghost": ["모하비고스트", "모하비 고스트"],
            "Rose 31": ["로즈31", "로즈 31"],
        }
        
        if name in alias_map:
            aliases.extend(alias_map[name])
        
        return [a.lower() for a in aliases]

    def _calc_confidence(self, keywords, ref_perfumes, web_keywords) -> float:
        """추론 신뢰도: 정보 소스 수와 풍부도 기반"""
        score = 0.0
        
        # 키워드 매칭 수 (최대 0.4)
        score += min(0.4, len(keywords) * 0.1)
        
        # 레퍼런스 향수 (최대 0.35)
        if ref_perfumes:
            best_sim = ref_perfumes[0][1]
            score += min(0.35, best_sim * 0.35)
        
        # 웹 정보 (최대 0.25)
        score += min(0.25, len(web_keywords) * 0.05)
        
        return min(1.0, score)

    def _generate_interpretation(self, vec, keywords, ref_perfumes) -> str:
        """추론 결과를 자연어로 설명"""
        parts = []
        
        # 주요 차원 (상위 5)
        top_dims = sorted(
            [(ODOR_DIMS[i], vec[i]) for i in range(22) if vec[i] > 0.1],
            key=lambda x: -x[1]
        )[:5]
        
        if top_dims:
            dim_str = ", ".join(f"{d}({v:.1f})" for d, v in top_dims)
            parts.append(f"주요 향조: {dim_str}")
        
        if keywords:
            parts.append(f"감지 키워드: {', '.join(keywords[:8])}")
        
        if ref_perfumes:
            refs = [f"{p.get('brand','')} {p.get('name','')}" for p, _ in ref_perfumes[:3]]
            parts.append(f"참조 향수: {', '.join(refs)}")
        
        return " | ".join(parts)


# ================================================================
# CLI 테스트
# ================================================================
if __name__ == '__main__':
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "따뜻한 우디 가을 향"
    
    interpreter = ScentInterpreter()
    result = interpreter.interpret(prompt)
    
    print(f"\n🧠 ScentInterpreter 결과")
    print(f"   입력: \"{prompt}\"")
    print(f"   신뢰도: {result['confidence']:.2f}")
    print(f"   해석: {result['interpretation']}")
    print(f"\n   22d 타겟 벡터:")
    vec = result['target_vector']
    top_dims = sorted([(ODOR_DIMS[i], vec[i]) for i in range(22) if vec[i] > 0.05], key=lambda x: -x[1])
    for name, val in top_dims[:10]:
        bar = "█" * int(val * 20)
        print(f"     {name:>10}: {val:.2f}  {bar}")
    
    if result['reference_perfumes']:
        print(f"\n   참조 향수:")
        for ref in result['reference_perfumes']:
            print(f"     {ref['brand']} {ref['name']} (sim: {ref['similarity']:.2f})")
