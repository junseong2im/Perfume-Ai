# sommelier.py — Engine 3: The Sommelier (향 언어 생성기)
# ==================================================================
# 숫자 벡터(20d) → 시적인 자연어 표현
# 규칙 기반 템플릿 엔진 (외부 API 없이 자체 동작)
# ==================================================================

import hashlib
import numpy as np

# 냄새 차원 정의 (odor_engine.py와 동일)
ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',  # V22: 독립 차원 추가
]

# ================================================================
# 단일 차원 → 한국어 표현 사전
# ================================================================

DIMENSION_DESCRIPTIONS = {
    'sweet': {
        'adjectives': ['달콤한', '감미로운', '꿀같은', '벨벳 같은'],
        'metaphors': ['설탕을 녹인 듯한', '꿀벌의 정원', '캐러멜이 녹아내리는'],
        'nouns': ['달콤함', '감미로움', '감미'],
    },
    'sour': {
        'adjectives': ['상쾌한', '톡 쏘는', '새콤한', '생동감 있는'],
        'metaphors': ['레몬을 한 입 베어문 듯한', '여름 과일의 첫 맛'],
        'nouns': ['상쾌함', '새콤함', '산미'],
    },
    'woody': {
        'adjectives': ['나무향의', '깊이 있는', '고요한', '숲속의'],
        'metaphors': ['고목의 나이테를 만지는', '깊은 숲의 산책로', '백단목의 기도'],
        'nouns': ['나무결', '숲의 기억', '우디함'],
    },
    'floral': {
        'adjectives': ['꽃향기의', '우아한', '여성스러운', '화사한'],
        'metaphors': ['봄 정원의 한가운데', '장미 꽃잎이 흩날리는', '새벽이슬에 젖은 꽃밭'],
        'nouns': ['꽃향', '플로럴', '꽃의 속삭임'],
    },
    'citrus': {
        'adjectives': ['시트러스의', '밝은', '경쾌한', '상큼한'],
        'metaphors': ['오렌지 껍질을 비틀 때', '지중해 레몬 밭', '아침 햇살에 반짝이는'],
        'nouns': ['시트러스', '상큼함', '밝은 활력'],
    },
    'spicy': {
        'adjectives': ['스파이시한', '따끔한', '대담한', '도발적인'],
        'metaphors': ['아라비안 바자르의 향신료', '계피 스틱이 따뜻한 와인에', '후추알이 부서지는'],
        'nouns': ['스파이시함', '향신료의 열정', '매운 온기'],
    },
    'musk': {
        'adjectives': ['관능적인', '피부에 밀착하는', '은밀한', '부드러운'],
        'metaphors': ['피부에 스며드는 속삭임', '실크 시트의 온기', '가까이 다가가야 느끼는'],
        'nouns': ['머스크', '사향', '피부의 기억'],
    },
    'fresh': {
        'adjectives': ['청량한', '깨끗한', '시원한', '투명한'],
        'metaphors': ['갓 세탁한 린넨', '산골 계곡의 물소리', '첫 눈이 내리기 직전의 공기'],
        'nouns': ['청량감', '깨끗함', '맑은 공기'],
    },
    'green': {
        'adjectives': ['풀잎의', '싱싱한', '자연스러운', '초록의'],
        'metaphors': ['갓 깎은 잔디', '이른 아침의 대나무 숲', '비 온 뒤 돋아나는 새싹'],
        'nouns': ['초록', '풀내음', '자연의 숨결'],
    },
    'warm': {
        'adjectives': ['따뜻한', '포근한', '아늑한', '편안한'],
        'metaphors': ['벽난로 앞의 겨울 저녁', '캐시미어 담요에 감싸인', '햇살이 머무는 나무 벤치'],
        'nouns': ['온기', '따뜻함', '포근함'],
    },
    'fruity': {
        'adjectives': ['과일향의', '즙이 많은', '달콤상큼한', '풍성한'],
        'metaphors': ['잘 익은 복숭아를 한 입', '여름 과수원의 오후', '열대 과일 칵테일'],
        'nouns': ['과일향', '과즙', '싱그러움'],
    },
    'smoky': {
        'adjectives': ['이끼낀', '그을린', '심오한', '불의'],
        'metaphors': ['오래된 위스키 통', '가을 밤 캠프파이어', '향나무가 타오르는 사원'],
        'nouns': ['연기', '스모키함', '그을음의 시'],
    },
    'powdery': {
        'adjectives': ['파우더리한', '부드러운', '솜털 같은', '몽환적인'],
        'metaphors': ['실크 파우더가 피부에 내려앉는', '아기의 뺨처럼 보송보송한'],
        'nouns': ['파우더', '보송함', '섬세함'],
    },
    'aquatic': {
        'adjectives': ['바다향의', '물결치는', '투명한', '소금기 있는'],
        'metaphors': ['파도가 발목을 적시는 순간', '수평선 너머의 새벽', '젖은 모래 위의 조개껍데기'],
        'nouns': ['바다향', '파도소리', '소금기'],
    },
    'herbal': {
        'adjectives': ['허브의', '약초향의', '건강한', '정갈한'],
        'metaphors': ['프로방스 라벤더 밭', '할머니의 약초 정원', '커밍 티를 우려내는'],
        'nouns': ['허브향', '약초', '치유의 향'],
    },
    'amber': {
        'adjectives': ['호박색의', '깊고 달콤한', '고대의', '신비로운'],
        'metaphors': ['오래된 도서관의 가죽 의자', '호박빛 노을이 지는 사막', '시간이 멈춘 성당'],
        'nouns': ['앰버', '호박', '고대의 따뜻함'],
    },
    'leather': {
        'adjectives': ['가죽의', '강인한', '세련된', '도시적인'],
        'metaphors': ['빈티지 가죽 자켓', '오래된 서재의 책등', '클래식 카의 시트'],
        'nouns': ['가죽', '레더', '거친 우아함'],
    },
    'earthy': {
        'adjectives': ['흙내음의', '원초적인', '대지의', '생명력 있는'],
        'metaphors': ['비 온 뒤 젖은 흙냄새', '숲속 낙엽이 쌓인 오솔길', '뿌리가 살아 숨쉬는'],
        'nouns': ['흙냄새', '대지', '페트리코르'],
    },
    'ozonic': {
        'adjectives': ['비 온 뒤의', '전기적인', '고요한', '투명한'],
        'metaphors': ['소나기가 지나간 직후의 공기', '번개가 갈라놓은 하늘', '높은 산 정상의 바람'],
        'nouns': ['오존', '비의 냄새', '하늘의 숨결'],
    },
    'metallic': {
        'adjectives': ['금속적인', '차가운', '날카로운', '미네랄의'],
        'metaphors': ['은빛 동전을 손에 쥔', '차가운 철 문고리', '새벽 안개 속 강철 다리'],
        'nouns': ['금속향', '미네랄', '차가운 광택'],
    },
    'fatty': {
        'adjectives': ['기름진', '풍부한', '버터 같은', '부드러운'],
        'metaphors': ['갓 구운 빵의 버터 향', '올리브 오일이 흐르는', '코코넛 크림의 부드러움'],
        'nouns': ['유지감', '기름향', '풍부한 질감'],
    },
    'waxy': {
        'adjectives': ['왁스향의', '매끈한', '촛불 같은', '은은한'],
        'metaphors': ['촛불이 타는 저녁 식탁', '밀랍으로 봉인된 편지', '백합의 꽃잎 표면'],
        'nouns': ['왁스', '밀랍', '매끄러운 질감'],
    },
}

# ================================================================
# 조합 패턴 → 시적 표현 (2~3차원 조합)
# ================================================================

COMBINATION_METAPHORS = {
    ('woody', 'earthy'): '비 온 뒤 젖은 흙 위의 고목',
    ('woody', 'smoky'): '가을 캠프파이어 옆 삼나무',
    ('woody', 'warm'): '오래된 서재의 편안한 오후',
    ('floral', 'fresh'): '이른 아침 이슬 맺힌 장미 정원',
    ('floral', 'sweet'): '한여름 꽃밭의 달콤한 바람',
    ('floral', 'green'): '봄비 맞은 풀밭 위의 들꽃',
    ('citrus', 'fresh'): '지중해 해변의 아침 레모네이드',
    ('citrus', 'aquatic'): '파도 위에 떨어진 오렌지 한 조각',
    ('citrus', 'green'): '라임을 짠 칵테일 위의 민트',
    ('musk', 'warm'): '실크 시트 위의 체온',
    ('musk', 'powdery'): '아이의 뺨에 닿는 부드러운 입술',
    ('musk', 'amber'): '호박빛 노을 아래 포옹',
    ('spicy', 'warm'): '겨울 벽난로 앞의 멀드 와인',
    ('spicy', 'woody'): '향나무로 지은 사원의 문',
    ('sweet', 'warm'): '카라멜이 녹아내리는 핫초코',
    ('sweet', 'fruity'): '여름 과수원의 복숭아 아이스크림',
    ('aquatic', 'ozonic'): '폭풍 전야의 바다',
    ('aquatic', 'fresh'): '산호초 위를 스치는 맑은 바닷물',
    ('smoky', 'leather'): '빈티지 위스키 바의 가죽 소파',
    ('smoky', 'amber'): '고대 사원에서 피어오르는 유향',
    ('green', 'herbal'): '프로방스 약초 밭을 걷는 아침',
    ('earthy', 'ozonic'): '소나기 직후 흙에서 올라오는 페트리코르',
    ('leather', 'woody'): '클래식 서재의 마호가니 책장',
    ('amber', 'warm'): '황금빛 일몰이 머무는 사막의 밤',
    ('powdery', 'floral'): '드레싱 테이블 위의 빈티지 파우더',
    ('fruity', 'floral'): '열대 과일과 자스민이 어우러진 정원',
}

# ================================================================
# 시간대별 분위기 표현
# ================================================================

TIME_ATMOSPHERE = {
    (0, 5): '첫 스프레이의 폭발적인',
    (5, 15): '처음 피어오르는',
    (15, 30): '서서히 드러나는',
    (30, 60): '안정되어 가는',
    (60, 120): '피부에 녹아드는',
    (120, 240): '은은하게 감싸는',
    (240, 480): '오랫동안 머무는',
}

NOTE_PHASE_NAMES = {
    'top': '탑 노트 (첫인상)',
    'middle': '미들 노트 (심장)',
    'base': '베이스 노트 (잔향)',
}


class Sommelier:
    """향수 소믈리에 — 숫자를 시로 번역"""
    
    def __init__(self):
        # 조합 SMARTS 미리 컴파일 (random 제거)
        self.combo_keys = list(COMBINATION_METAPHORS.keys())
        print("[Sommelier] Language engine initialized | Templates loaded")
    
    def _deterministic_pick(self, items, seed_val):
        """리스트에서 결정론적 선택 (random 대신 해시 기반)"""
        if not items:
            return ''
        idx = abs(hash(str(seed_val))) % len(items)
        return items[idx]
    
    def _get_top_dims(self, vec, threshold=0.3, top_k=5):
        """벡터에서 주요 차원 추출"""
        indices = np.argsort(vec)[::-1][:top_k]
        result = []
        for i in indices:
            if vec[i] >= threshold:
                result.append((ODOR_DIMENSIONS[i], float(vec[i])))
        return result
    
    def _intensity_word(self, value):
        """강도 → 부사"""
        if value > 0.8:
            return '강렬하게'
        elif value > 0.6:
            return '선명하게'
        elif value > 0.4:
            return '은은하게'
        elif value > 0.2:
            return '아주 살짝'
        else:
            return '미세하게'
    
    def describe_moment(self, odor_vector, time_min=0, note_phase='top'):
        """단일 시점의 냄새 → 자연어 표현
        
        Args:
            odor_vector: [20] 냄새 벡터
            time_min: 시간 (분)
            note_phase: 'top'/'middle'/'base'
        
        Returns:
            str: 시적 향 표현
        """
        vec = np.array(odor_vector)
        top_dims = self._get_top_dims(vec, threshold=0.25, top_k=5)
        
        if not top_dims:
            return "향이 거의 증발했습니다."
        
        # 시간대 분위기
        time_mood = ''
        for (t_start, t_end), mood in TIME_ATMOSPHERE.items():
            if t_start <= time_min < t_end:
                time_mood = mood
                break
        if not time_mood:
            time_mood = '잔향으로 남는'
        
        parts = []
        
        # 1) 조합 메타포 시도
        if len(top_dims) >= 2:
            d1, d2 = top_dims[0][0], top_dims[1][0]
            for key in [(d1, d2), (d2, d1)]:
                if key in COMBINATION_METAPHORS:
                    parts.append(COMBINATION_METAPHORS[key])
                    break
        
        # 2) 메인 차원 서술
        if not parts:
            main_dim = top_dims[0][0]
            desc = DIMENSION_DESCRIPTIONS.get(main_dim, {})
            metaphors = desc.get('metaphors', [])
            if metaphors:
                parts.append(self._deterministic_pick(metaphors, main_dim))
        
        # 3) 서브 차원 추가
        if len(top_dims) >= 2:
            sub_dim = top_dims[1][0]
            sub_desc = DIMENSION_DESCRIPTIONS.get(sub_dim, {})
            adjectives = sub_desc.get('adjectives', [])
            nouns = sub_desc.get('nouns', ['향'])
            if adjectives:
                intensity = self._intensity_word(top_dims[1][1])
                adj = self._deterministic_pick(adjectives, sub_dim)
                noun = self._deterministic_pick(nouns, sub_dim)
                parts.append(f"{intensity} {adj} {noun}이 어우러집니다")
        
        # 4) 세 번째 차원 (있다면)
        if len(top_dims) >= 3:
            third_dim = top_dims[2][0]
            third_desc = DIMENSION_DESCRIPTIONS.get(third_dim, {})
            nouns = third_desc.get('nouns', ['향'])
            parts.append(f"{self._deterministic_pick(nouns, third_dim)}의 뉘앙스가 감돕니다")
        
        # 조합
        phase_name = NOTE_PHASE_NAMES.get(note_phase, '')
        sentence = f"[{time_min}분] {time_mood} 향 — " + '. '.join(parts) + '.'
        
        return sentence
    
    def describe_evolution(self, odor_timeline):
        """시간별 냄새 타임라인 → 향 스토리
        
        Args:
            odor_timeline: OdorPredictor.predict_timeline의 출력
        
        Returns:
            List[str]: 각 시점의 표현 리스트
        """
        descriptions = []
        prev_dominant = None
        
        for snap in odor_timeline:
            t_min = snap['time_min']
            vec = np.array(snap['odor_vector'])
            dominant = snap.get('dominant', 'none')
            note = snap.get('note_balance', {})
            
            # 노트 페이즈 결정
            if note:
                phase = max(note, key=note.get)
            else:
                phase = 'top' if t_min < 30 else ('middle' if t_min < 180 else 'base')
            
            # 주요 전환점만 서술 (매 시점 다 쓰면 too verbose)
            is_transition = (dominant != prev_dominant) or t_min == 0
            
            if is_transition or t_min in [0, 30, 60, 120, 240, 480]:
                desc = self.describe_moment(vec, t_min, phase)
                
                # 전환 표현 추가
                if prev_dominant and dominant != prev_dominant and t_min > 0:
                    desc = f"🔄 향의 전환 — " + desc
                
                descriptions.append(desc)
            
            prev_dominant = dominant
        
        return descriptions
    
    def generate_story(self, recipe_name, odor_timeline, target_description=None):
        """전체 향 스토리 생성"""
        import re
        
        def _extract_time(desc):
            """서술문에서 시간(분) 추출 — 🔄 등 접두사 안전 처리"""
            m = re.search(r'\[(\d+)분\]', desc)
            return int(m.group(1)) if m else -1
        
        lines = []
        lines.append(f"═══ {recipe_name} ═══")
        lines.append("")
        
        if target_description:
            lines.append(f'🎯 컨셉: "{target_description}"')
            lines.append("")
        
        # 시간별 서술
        descriptions = self.describe_evolution(odor_timeline)
        
        # 3-Act 구조로 정리
        lines.append("▸ Act 1: 첫인상 (0~30분)")
        lines.append("─" * 40)
        for desc in descriptions:
            t = _extract_time(desc)
            if 0 <= t <= 30:
                lines.append(f"  {desc}")
        
        lines.append("")
        lines.append("▸ Act 2: 심장 (30분~3시간)")
        lines.append("─" * 40)
        for desc in descriptions:
            t = _extract_time(desc)
            if 30 < t <= 180:
                lines.append(f"  {desc}")
        
        lines.append("")
        lines.append("▸ Act 3: 잔향 (3시간~)")
        lines.append("─" * 40)
        for desc in descriptions:
            t = _extract_time(desc)
            if t > 180:
                lines.append(f"  {desc}")
        
        # 마무리
        lines.append("")
        if odor_timeline:
            last = odor_timeline[-1]
            intensity = last.get('intensity', 0)
            if intensity > 0.3:
                lines.append(f"✦ 마지막까지 {int(intensity*100)}%의 향이 피부에 머물러 있습니다.")
            else:
                lines.append("✦ 향이 서서히 피부에 녹아들어 은밀한 잔향으로 남습니다.")
        
        return '\n'.join(lines)
    
    def quick_describe(self, odor_vector):
        """빠른 한 줄 설명"""
        vec = np.array(odor_vector)
        top = self._get_top_dims(vec, threshold=0.3, top_k=3)
        if not top:
            return "거의 무취"
        
        parts = []
        for dim, val in top:
            desc = DIMENSION_DESCRIPTIONS.get(dim, {})
            adj = desc.get('adjectives', [''])[0]
            parts.append(adj)
        
        return ' + '.join(parts)


# ================================================================
# 전역 인스턴스
# ================================================================

_sommelier = None

def get_sommelier():
    global _sommelier
    if _sommelier is None:
        _sommelier = Sommelier()
    return _sommelier

def describe_moment(odor_vector, time_min=0, note_phase='top'):
    return get_sommelier().describe_moment(odor_vector, time_min, note_phase)

def describe_evolution(odor_timeline):
    return get_sommelier().describe_evolution(odor_timeline)

def generate_story(recipe_name, odor_timeline, target=None):
    return get_sommelier().generate_story(recipe_name, odor_timeline, target)

def quick_describe(odor_vector):
    return get_sommelier().quick_describe(odor_vector)
