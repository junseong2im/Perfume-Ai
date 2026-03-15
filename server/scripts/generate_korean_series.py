# -*- coding: utf-8 -*-
"""
🇰🇷 한국 지역 특산식물 기반 향수 레시피 시리즈
===============================================
V16 OdorGNN 완전체 파이프라인으로 지역별 레시피 생성
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, '.')

import numpy as np
from odor_engine import ODOR_DIMENSIONS, N_ODOR_DIM, PrincipalOdorMap
import recipe_engine

# ─── 어코드 → 실제 원료 조합 매핑 ───
# 어코드(Accord)는 여러 실제 원료를 혼합해서 만드는 추상적 향입니다.
# 레시피에 어코드가 포함되면 실제 구현 원료도 함께 표시합니다.
ACCORD_COMPOSITIONS = {
    # 자연 현상 어코드
    "jirisan_morning": {
        "name": "지리산 아침안개",
        "components": [
            {"name": "Hedione (메틸 디하이드로자스모네이트)", "pct": 40, "role": "투명한 공기감"},
            {"name": "Calone (워터멜론 케톤)", "pct": 25, "role": "물안개/수분감"},
            {"name": "Iso E Super", "pct": 20, "role": "나무 속 부유감"},
            {"name": "Dihydromyrcenol", "pct": 15, "role": "차가운 청량감"},
        ]
    },
    "bukhansan_granite": {
        "name": "북한산 화강암",
        "components": [
            {"name": "Ambrox (암브록스)", "pct": 35, "role": "미네랄 따뜻함"},
            {"name": "Iso E Super", "pct": 25, "role": "건조한 나무-돌 느낌"},
            {"name": "Vetiver (베티버)", "pct": 20, "role": "흙/뿌리 미네랄감"},
            {"name": "Javanol", "pct": 20, "role": "크리미한 샌달 베이스"},
        ]
    },
    "gyeongju_soil": {
        "name": "경주 흙",
        "components": [
            {"name": "Geosmin (게오스민, 0.01% 희석)", "pct": 10, "role": "젖은 흙 핵심 분자"},
            {"name": "Vetiver Haiti (아이티 베티버)", "pct": 35, "role": "뿌리/흙 깊이"},
            {"name": "Patchouli (패출리)", "pct": 30, "role": "어두운 흙 느낌"},
            {"name": "Oakmoss (오크모스 합성)", "pct": 25, "role": "이끼 낀 고대 토양"},
        ]
    },
    "bamboo_dew": {
        "name": "대나무 이슬",
        "components": [
            {"name": "Calone (칼론)", "pct": 30, "role": "이슬/수분감"},
            {"name": "Hedione (헤디온)", "pct": 30, "role": "투명한 신선함"},
            {"name": "Linalool (리날룰)", "pct": 25, "role": "녹색 플로럴"},
            {"name": "cis-3-Hexenol (풀잎 알코올)", "pct": 15, "role": "싱싱한 녹색"},
        ]
    },
    "autumn_foliage": {
        "name": "가을 단풍",
        "components": [
            {"name": "Iso E Super", "pct": 30, "role": "마른 나무 따뜻함"},
            {"name": "Cashmeran (캐시머란)", "pct": 25, "role": "따뜻한 스파이시 우드"},
            {"name": "Firmenich Leaf Accord", "pct": 20, "role": "마른 잎 느낌"},
            {"name": "Benzyl Benzoate (벤질벤조에이트)", "pct": 15, "role": "부드러운 달콤함"},
            {"name": "cis-3-Hexenyl Salicylate", "pct": 10, "role": "풀잎/녹색 뉘앙스"},
        ]
    },
    "spring_rain": {
        "name": "봄비 (페트리코)",
        "components": [
            {"name": "Geosmin (게오스민, 0.01% 희석)", "pct": 10, "role": "비 온 뒤 흙냄새 핵심"},
            {"name": "Calone (칼론)", "pct": 30, "role": "깨끗한 수분감"},
            {"name": "Hedione (헤디온)", "pct": 30, "role": "투명한 공기감"},
            {"name": "Iso E Super", "pct": 20, "role": "부유하는 우디"},
            {"name": "Ozone Accord (오존)", "pct": 10, "role": "비 직전 전기감"},
        ]
    },
    "snow_accord": {
        "name": "한국 눈",
        "components": [
            {"name": "Dihydromyrcenol (디하이드로미르세놀)", "pct": 35, "role": "차가운 청량감"},
            {"name": "Calone (칼론)", "pct": 25, "role": "투명한 수분"},
            {"name": "Hedione (헤디온)", "pct": 25, "role": "가벼운 공기감"},
            {"name": "Ethyl Maltol (에틸 말톨, 소량)", "pct": 15, "role": "부드러운 파우더리 눈 느낌"},
        ]
    },
    "ondol": {
        "name": "온돌 (따뜻한 바닥)",
        "components": [
            {"name": "Cashmeran (캐시머란)", "pct": 35, "role": "따뜻한 스파이시 우드"},
            {"name": "Cedarwood Virginia (시더우드)", "pct": 25, "role": "나무 바닥 느낌"},
            {"name": "Benzoin (벤조인 레진)", "pct": 20, "role": "따뜻한 바닐릭 수지"},
            {"name": "Ambrox (암브록스)", "pct": 20, "role": "포근한 앰버감"},
        ]
    },
    # 전통 문화 소재 어코드
    "hanji": {
        "name": "한지 (한국 종이)",
        "components": [
            {"name": "Papyrus (파피루스)", "pct": 35, "role": "종이/섬유 느낌"},
            {"name": "Iso E Super", "pct": 25, "role": "건조한 나무 느낌"},
            {"name": "Cashmeran (캐시머란)", "pct": 20, "role": "면직물 따뜻함"},
            {"name": "Hedione (헤디온)", "pct": 20, "role": "가벼운 투명감"},
        ]
    },
    "meok": {
        "name": "먹 (한국 먹물)",
        "components": [
            {"name": "Birch Tar (자작나무 타르)", "pct": 30, "role": "탄소/그을음 핵심"},
            {"name": "Guaiacwood (파나마 나무)", "pct": 30, "role": "스모키 우디"},
            {"name": "Black Pepper CO2 (흑후추)", "pct": 20, "role": "날카로운 스파이시"},
            {"name": "Musk Ketone (무스크 케톤)", "pct": 20, "role": "부드러운 마무리"},
        ]
    },
    "hanbok_silk": {
        "name": "한복 비단",
        "components": [
            {"name": "Heliotropin (헬리오트로핀)", "pct": 35, "role": "파우더리 달콤함"},
            {"name": "Orris Butter (아이리스 버터)", "pct": 25, "role": "실크 파우더리"},
            {"name": "Musk (화이트 무스크)", "pct": 25, "role": "부드러운 피부감"},
            {"name": "Benzyl Benzoate", "pct": 15, "role": "살짝 달콤한 마무리"},
        ]
    },
    "nurungji": {
        "name": "누룽지 (눌은 밥)",
        "components": [
            {"name": "Maltol (말톨)", "pct": 30, "role": "달콤한 구운 냄새"},
            {"name": "Tonka Bean (통카빈 앱솔루트)", "pct": 25, "role": "고소 달콤한 쿠마린"},
            {"name": "Cedarwood (시더우드)", "pct": 25, "role": "마른 나무/곡물 느낌"},
            {"name": "Vanillin (바닐린, 소량)", "pct": 20, "role": "따뜻한 달콤함"},
        ]
    },
    "makkoli": {
        "name": "막걸리",
        "components": [
            {"name": "Ethyl Acetate (에틸 아세테이트)", "pct": 25, "role": "발효 과일 산미"},
            {"name": "Maltol (말톨)", "pct": 25, "role": "달콤한 쌀 느낌"},
            {"name": "Lactic Acid Accord", "pct": 25, "role": "유산 발효감"},
            {"name": "Ethanol (에탄올 뉘앙스)", "pct": 25, "role": "알코올 톡 쏘는 느낌"},
        ]
    },
    "doenjang": {
        "name": "된장",
        "components": [
            {"name": "Costus Root (코스투스 뿌리)", "pct": 35, "role": "발효/동물적 깊이"},
            {"name": "Cumin (쿠민, 소량)", "pct": 20, "role": "발효 스파이시"},
            {"name": "Patchouli Dark (패출리 다크)", "pct": 25, "role": "어두운 흙 느낌"},
            {"name": "Castoreum Synthetic (합성 캐스토리움)", "pct": 20, "role": "동물적 따뜻함"},
        ]
    },
    "gama": {
        "name": "가마솥",
        "components": [
            {"name": "Birch Tar (자작나무 타르)", "pct": 30, "role": "그을린 나무/탄소"},
            {"name": "Cade Oil (카데 오일)", "pct": 30, "role": "훈연/스모키"},
            {"name": "Guaiacwood (과이악 우드)", "pct": 25, "role": "따뜻한 스모키 우디"},
            {"name": "Iron Accord (철 느낌)", "pct": 15, "role": "금속 무기질 뉘앙스"},
        ]
    },
    "hanok_wood": {
        "name": "한옥 나무",
        "components": [
            {"name": "Hinoki Oil (히노키 오일)", "pct": 30, "role": "일본편백 = 한옥 기둥"},
            {"name": "Cedarwood Atlas (시더우드)", "pct": 25, "role": "오래된 목재"},
            {"name": "Sandalwood (샌달우드)", "pct": 20, "role": "크리미한 나무"},
            {"name": "Iso E Super", "pct": 15, "role": "건조한 우디 아우라"},
            {"name": "Benzoin (벤조인)", "pct": 10, "role": "따뜻한 수지감"},
        ]
    },
    # 해양 어코드
    "jeju_sea_salt": {
        "name": "제주 바다 소금",
        "components": [
            {"name": "Calone (칼론)", "pct": 35, "role": "바다/수박 느낌"},
            {"name": "Ambrox (암브록스)", "pct": 25, "role": "소금기 앰버"},
            {"name": "Dihydromyrcenol", "pct": 25, "role": "시원한 바람"},
            {"name": "Helional (헬리오날)", "pct": 15, "role": "해변 공기"},
        ]
    },
    "haepung": {
        "name": "해풍 (바닷바람)",
        "components": [
            {"name": "Calone (칼론)", "pct": 30, "role": "바다 느낌"},
            {"name": "Dihydromyrcenol", "pct": 30, "role": "시원한 바람"},
            {"name": "Hedione (헤디온)", "pct": 25, "role": "투명 공기감"},
            {"name": "Ozone Accord", "pct": 15, "role": "바다 오존"},
        ]
    },
    "haejoRyu": {
        "name": "해조류",
        "components": [
            {"name": "Calone (칼론)", "pct": 30, "role": "해양 수분"},
            {"name": "Seaweed Absolute (해조류 앱솔루트)", "pct": 30, "role": "실제 해조 추출물"},
            {"name": "Dimethyl Sulfide (DMS, 극소량)", "pct": 10, "role": "바다/해초 핵심 분자"},
            {"name": "Helional (헬리오날)", "pct": 30, "role": "깨끗한 해변"},
        ]
    },
    "gaetbeol": {
        "name": "갯벌",
        "components": [
            {"name": "Geosmin (게오스민, 극소량)", "pct": 15, "role": "젖은 진흙"},
            {"name": "Seaweed Absolute (해초)", "pct": 25, "role": "바다 유기물"},
            {"name": "Vetiver (베티버)", "pct": 30, "role": "진흙/뿌리"},
            {"name": "Calone (칼론)", "pct": 20, "role": "바닷물 수분"},
            {"name": "Dimethyl Sulfide (DMS, 극소량)", "pct": 10, "role": "갯벌 특유 냄새"},
        ]
    },
    "jeju_lava": {
        "name": "제주 현무암",
        "components": [
            {"name": "Ambrox (암브록스)", "pct": 30, "role": "미네랄 따뜻함"},
            {"name": "Vetiver (베티버)", "pct": 25, "role": "화산 토양"},
            {"name": "Birch Tar (자작나무 타르)", "pct": 20, "role": "그을린 돌"},
            {"name": "Black Pepper CO2 (흑후추)", "pct": 15, "role": "화산 스파이시"},
            {"name": "Iso E Super", "pct": 10, "role": "건조 미네랄"},
        ]
    },
    "hallasan_moss": {
        "name": "한라산 이끼",
        "components": [
            {"name": "Oakmoss Synthetic (합성 오크모스)", "pct": 35, "role": "이끼 핵심"},
            {"name": "Vetiver (베티버)", "pct": 25, "role": "습한 흙"},
            {"name": "Patchouli (패출리)", "pct": 20, "role": "어두운 이끼"},
            {"name": "Evernyl (에버닐)", "pct": 20, "role": "IFRA 안전 오크모스 대체"},
        ]
    },
    # 전통 음료 어코드
    "ssanghwa": {
        "name": "쌍화탕",
        "components": [
            {"name": "Cinnamon Bark (계피 오일)", "pct": 25, "role": "따뜻한 스파이시"},
            {"name": "Angelica Root (당귀 뿌리 오일)", "pct": 25, "role": "약초 허벌"},
            {"name": "Ginger CO2 (생강)", "pct": 20, "role": "매운 따뜻함"},
            {"name": "Jujube (대추 추출)", "pct": 15, "role": "달콤한 과실"},
            {"name": "Licorice (감초)", "pct": 15, "role": "달콤한 뿌리"},
        ]
    },
    "sujeonggwa": {
        "name": "수정과",
        "components": [
            {"name": "Cinnamon Bark (계피 바크)", "pct": 35, "role": "수정과 핵심 — 계피"},
            {"name": "Ginger CO2 (생강)", "pct": 25, "role": "수정과 핵심 — 생강"},
            {"name": "Pine Nut (잣 향)", "pct": 15, "role": "고소한 잣 가니쉬"},
            {"name": "Dried Persimmon Accord", "pct": 25, "role": "곶감 달콤함"},
        ]
    },
    "sikhye": {
        "name": "식혜",
        "components": [
            {"name": "Maltol (말톨)", "pct": 35, "role": "달콤한 쌀엿 느낌"},
            {"name": "Ethyl Maltol", "pct": 25, "role": "쌀 발효 달콤함"},
            {"name": "Rice Bran Accord (쌀겨)", "pct": 25, "role": "쌀 곡물 느낌"},
            {"name": "Ginger (생강, 소량)", "pct": 15, "role": "살짝 매운 마무리"},
        ]
    },
    # 나무 관련 어코드
    "sonhyang": {
        "name": "솔향 (사찰)",
        "components": [
            {"name": "Pine Needle Oil (소나무 잎 오일)", "pct": 35, "role": "솔잎 핵심"},
            {"name": "Pine Resin (송진)", "pct": 25, "role": "소나무 수지"},
            {"name": "Frankincense (유향)", "pct": 25, "role": "사찰 향연"},
            {"name": "Cedarwood (시더우드)", "pct": 15, "role": "나무 보조"},
        ]
    },
    "jeolhyang": {
        "name": "절향 (사찰 향)",
        "components": [
            {"name": "Frankincense (유향)", "pct": 30, "role": "사찰 향 핵심"},
            {"name": "Sandalwood (샌달우드)", "pct": 25, "role": "명상적 우디"},
            {"name": "Benzoin Siam (시암 벤조인)", "pct": 20, "role": "따뜻한 수지"},
            {"name": "Agarwood (침향, 합성)", "pct": 15, "role": "깊은 영적 향"},
            {"name": "Camphor (장뇌, 소량)", "pct": 10, "role": "약간의 청량감"},
        ]
    },
}

# ─── 한국 지역별 향수 프로필 ───
KOREAN_RECIPES = [
    {
        "id": "mugunghwa",
        "name": "무궁화 — 나라꽃",
        "region": "대한민국",
        "description": "무궁화의 은은한 플로럴과 한국 산야의 따뜻한 흙과 나무 향기",
        "plants": "무궁화 (Rose of Sharon), 소나무, 참나무",
        "mood": "elegant",
        "season": "spring",
        "preferences": ["floral", "woody", "warm", "powdery", "green"],
        "intensity": 60,
        "target_boost": {
            "floral": 0.9, "warm": 0.6, "woody": 0.5, "powdery": 0.5,
            "green": 0.4, "sweet": 0.3, "herbal": 0.3,
        },
    },
    {
        "id": "jeju",
        "name": "제주 — 동백과 감귤",
        "region": "제주특별자치도",
        "description": "제주 동백꽃의 붉은 꽃잎과 감귤밭의 상큼한 바람, 바다 소금기",
        "plants": "동백꽃 (Camellia), 감귤 (Tangerine), 유채꽃",
        "mood": "relaxed",
        "season": "spring",
        "preferences": ["floral", "citrus", "aquatic", "fresh", "sweet"],
        "intensity": 55,
        "target_boost": {
            "floral": 0.8, "citrus": 0.8, "aquatic": 0.5, "fresh": 0.6,
            "sweet": 0.4, "green": 0.3, "fruity": 0.4,
        },
    },
    {
        "id": "gyeongju",
        "name": "경주 — 천년 매화",
        "region": "경상북도 경주",
        "description": "경주 불국사의 천년 매화와 소나무 숲, 고즈넉한 한옥의 나무 향기",
        "plants": "매화 (Plum Blossom), 소나무, 국화 (Chrysanthemum)",
        "mood": "calm",
        "season": "spring",
        "preferences": ["floral", "woody", "herbal", "powdery", "warm"],
        "intensity": 50,
        "target_boost": {
            "floral": 0.7, "woody": 0.7, "herbal": 0.4, "powdery": 0.4,
            "warm": 0.5, "sweet": 0.3, "earthy": 0.3,
        },
    },
    {
        "id": "boseong",
        "name": "보성 — 녹차밭 아침",
        "region": "전라남도 보성",
        "description": "보성 녹차밭의 이른 아침 안개와 초록 찻잎, 젖은 흙 냄새",
        "plants": "녹차 (Green Tea), 대나무, 매실 (Plum)",
        "mood": "calm",
        "season": "spring",
        "preferences": ["green", "herbal", "earthy", "fresh", "woody"],
        "intensity": 45,
        "target_boost": {
            "green": 0.9, "herbal": 0.6, "earthy": 0.5, "fresh": 0.5,
            "woody": 0.3, "sweet": 0.2, "aquatic": 0.3,
        },
    },
    {
        "id": "gangwon",
        "name": "강원 — 자작나무 숲",
        "region": "강원특별자치도 인제",
        "description": "인제 자작나무 숲의 하얀 나무껍질과 솔잎, 겨울 눈 덮인 산의 청량한 공기",
        "plants": "자작나무 (Birch), 소나무, 잣나무 (Korean Pine)",
        "mood": "calm",
        "season": "winter",
        "preferences": ["woody", "fresh", "green", "ozonic", "earthy"],
        "intensity": 50,
        "target_boost": {
            "woody": 0.8, "fresh": 0.7, "green": 0.5, "ozonic": 0.5,
            "earthy": 0.4, "herbal": 0.3, "smoky": 0.2,
        },
    },
    {
        "id": "buyeo",
        "name": "부여 — 궁남지 연꽃",
        "region": "충청남도 부여",
        "description": "부여 궁남지의 연꽃 향기와 물안개, 백제의 고요한 아침",
        "plants": "연꽃 (Lotus), 창포 (Iris), 버드나무 (Willow)",
        "mood": "calm",
        "season": "summer",
        "preferences": ["floral", "aquatic", "fresh", "green", "powdery"],
        "intensity": 45,
        "target_boost": {
            "floral": 0.8, "aquatic": 0.6, "fresh": 0.5, "green": 0.5,
            "powdery": 0.4, "sweet": 0.3, "warm": 0.2,
        },
    },
    {
        "id": "jirisan",
        "name": "지리산 — 철쭉 능선",
        "region": "전라남도·경상남도 지리산",
        "description": "지리산 철쭉 군락의 분홍빛 물결과 참나무 숲, 산안개 속 허브 향",
        "plants": "철쭉 (Azalea), 참나무 (Oak), 산초 (Sichuan Pepper)",
        "mood": "relaxed",
        "season": "spring",
        "preferences": ["floral", "woody", "herbal", "spicy", "green"],
        "intensity": 55,
        "target_boost": {
            "floral": 0.8, "woody": 0.6, "herbal": 0.5, "spicy": 0.4,
            "green": 0.5, "earthy": 0.3, "warm": 0.3,
        },
    },
    {
        "id": "damyang",
        "name": "담양 — 대나무 달빛",
        "region": "전라남도 담양",
        "description": "담양 죽녹원의 대나무 바스락거리는 소리와 달빛 아래 이슬, 대숲의 녹색 공기",
        "plants": "대나무 (Bamboo), 메밀꽃 (Buckwheat Flower), 자두 (Plum)",
        "mood": "calm",
        "season": "summer",
        "preferences": ["green", "fresh", "woody", "herbal", "aquatic"],
        "intensity": 45,
        "target_boost": {
            "green": 0.9, "fresh": 0.7, "woody": 0.5, "herbal": 0.4,
            "aquatic": 0.3, "earthy": 0.3, "ozonic": 0.3,
        },
    },
]

def generate_target_vector(profile):
    """프로필의 target_boost를 22d 냄새 벡터로 변환"""
    target = np.zeros(N_ODOR_DIM)
    for dim, val in profile["target_boost"].items():
        idx = ODOR_DIMENSIONS.index(dim)
        target[idx] = val
    target = target / (np.linalg.norm(target) + 1e-8)
    return target


def recipe_to_markdown(profile, recipe, target):
    """레시피를 .md 포맷으로 변환"""
    lines = []
    a = lines.append
    
    # Header
    a(f"# 🌸 {profile['name']}")
    a(f"### {profile['region']}")
    a(f"")
    a(f"> {profile['description']}")
    a(f"> ")
    a(f"> 대표 식물: {profile['plants']}")
    a(f">")
    a(f"> AI Perfumer V16 (CosSim 0.8092) 생성 | 2026-02-18")
    a(f"")
    a(f"---")
    a(f"")
    
    # Basic info
    a(f"## 기본 정보")
    a(f"")
    stats = recipe['stats']
    cost = recipe['cost']
    a(f"- **향수 이름:** {recipe['name_ko']} ({recipe['name']})")
    a(f"- **농도:** {recipe['concentration']} — {stats['total_concentrate_pct']}%")
    a(f"- **배치:** {recipe['batch_ml']}ml")
    a(f"- **원료:** {stats['total_ingredients']}개")
    a(f"- **지속:** 약 {stats['longevity_hours']}시간")
    a(f"- **비용:** {cost['total_formatted']}")
    a(f"")
    a(f"---")
    a(f"")
    
    # Target odor
    a(f"## 목표 냄새 프로파일")
    a(f"")
    top_dims = sorted(
        [(ODOR_DIMENSIONS[i], float(target[i])) for i in range(N_ODOR_DIM) if target[i] > 0.05],
        key=lambda x: x[1], reverse=True
    )
    for dim, val in top_dims[:6]:
        bar = '█' * int(val * 20)
        a(f"- **{dim}** {val:.3f} {bar}")
    a(f"")
    a(f"---")
    a(f"")
    
    # Pyramid
    a(f"## 향 피라미드")
    a(f"")
    pyramid = recipe['pyramid']
    a(f"**탑 노트** (0~30분)  ")
    a(f"🌿 {' · '.join(pyramid['top'])}")
    a(f"")
    a(f"**미들 노트** (30분~3시간)  ")
    a(f"🌸 {' · '.join(pyramid['middle'])}")
    a(f"")
    a(f"**베이스 노트** (3시간~)  ")
    a(f"🪵 {' · '.join(pyramid['base'])}")
    a(f"")
    a(f"---")
    a(f"")
    
    # Load origin data from DB
    import database as db_module
    origin_cache = {}
    try:
        conn = db_module.get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, country, region, terroir_notes FROM ingredients WHERE country IS NOT NULL")
        for row in cur.fetchall():
            origin_cache[row[0]] = {"country": row[1], "region": row[2], "terroir": row[3]}
    except:
        pass

    # Formula
    accord_used = []  # Track accords for later breakdown section
    origin_used = []  # Track ingredients with origin info
    for step_data in recipe['mixing_steps']:
        if step_data['note_type'] == 'solvent':
            a(f"### Step {step_data['step']}: 에탄올 희석")
            a(f"")
            sol = step_data['ingredients'][0]
            a(f"| 원료 | 비율 | 용량 | 무게 |")
            a(f"|------|------|------|------|")
            a(f"| 에탄올 95% | {sol['percentage']}% | {sol['ml']}ml | {sol['grams']}g |")
        else:
            note_label = {'base': '베이스 노트 (먼저 혼합)', 'middle': '미들 노트 (하트 추가)', 'top': '탑 노트 (마지막 투입)'}
            a(f"### Step {step_data['step']}: {note_label.get(step_data['note_type'], step_data['label'])}")
            a(f"")
            a(f"| 원료 | 카테고리 | 원산지 | 비율 | 용량 | 무게 |")
            a(f"|------|----------|--------|------|------|------|")
            for ing_item in step_data['ingredients']:
                ing_id = ing_item.get('id', '')
                is_accord = ing_id in ACCORD_COMPOSITIONS
                marker = " ⚗️" if is_accord else ""
                origin = origin_cache.get(ing_id, {})
                origin_str = f"{origin.get('country', '—')}" if origin else "—"
                a(f"| {ing_item['name_ko']} ({ing_item.get('name_en','')}){marker} | {ing_item.get('category','')} | {origin_str} | {ing_item['percentage']}% | {ing_item['ml']}ml | {ing_item['grams']}g |")
                if is_accord:
                    accord_used.append((ing_id, ing_item['name_ko'], float(ing_item['ml'])))
                if origin and origin.get('terroir'):
                    origin_used.append((ing_item['name_ko'], origin))
        a(f"")
    
    # Accord decomposition section
    if accord_used:
        a(f"---")
        a(f"")
        a(f"## ⚗️ 어코드 실제 원료 분해")
        a(f"")
        a(f"> 위 포뮬라에서 ⚗️ 표시된 원료는 **어코드(Accord)**입니다.")
        a(f"> 어코드는 여러 실제 원료를 혼합한 복합 향입니다.")
        a(f"> 아래는 각 어코드를 실제 원료로 구현하는 방법입니다.")
        a(f"")
        for accord_id, accord_ko, accord_ml in accord_used:
            acc = ACCORD_COMPOSITIONS[accord_id]
            a(f"### ⚗️ {accord_ko} → 실제 구현")
            a(f"")
            a(f"*{accord_ml}ml 기준 혼합 비율:*")
            a(f"")
            a(f"| 실제 원료 | 비율 | 용량 | 역할 |")
            a(f"|----------|------|------|------|")
            for comp in acc['components']:
                comp_ml = round(accord_ml * comp['pct'] / 100, 2)
                a(f"| {comp['name']} | {comp['pct']}% | {comp_ml}ml | {comp['role']} |")
            a(f"")
    
    # Origin / Terroir section
    if origin_used:
        a(f"---")
        a(f"")
        a(f"## 🌍 원산지 테루아르 정보")
        a(f"")
        a(f"> 같은 원료도 산지에 따라 향이 다릅니다. 아래는 이 레시피에 사용된 원료의 원산지 특성입니다.")
        a(f"")
        seen = set()
        for name_ko, origin in origin_used:
            if name_ko in seen:
                continue
            seen.add(name_ko)
            a(f"- **{name_ko}** — {origin['country']}, {origin['region']}")
            if origin.get('terroir'):
                a(f"  > {origin['terroir']}")
        a(f"")
    
    a(f"---")
    a(f"")
    
    # Mixing
    a(f"## 혼합 순서")
    a(f"")
    a(f"1. 깨끗한 유리 비커에 **베이스 원료**를 순서대로 넣고 천천히 저어줍니다.")
    a(f"2. **미들 원료**를 추가하고 부드럽게 혼합합니다.")
    a(f"3. **탑 원료**를 마지막에 추가합니다. (휘발성 높음)")
    a(f"4. **에탄올 95%**를 천천히 부어 희석합니다.")
    a(f"5. 차광 유리병에 밀봉 후 숙성합니다.")
    a(f"")
    a(f"---")
    a(f"")
    
    # Aging
    aging = recipe['aging']
    a(f"## 숙성")
    a(f"")
    a(f"- **최소:** {aging['min_days']}일 / **권장:** {aging['recommended_days']}일")
    a(f"- 직사광선 피해 서늘하고 어두운 곳 (15~22°C)")
    a(f"- 매일 한 번 가볍게 흔들어주기")
    a(f"")
    a(f"---")
    a(f"")
    
    # Cost
    a(f"## 비용")
    a(f"")
    a(f"- 향료: ₩{cost['ingredients_krw']:,}")
    a(f"- 에탄올: ₩{cost['alcohol_krw']:,}")
    a(f"- **합계: {cost['total_formatted']}**")
    a(f"")
    a(f"---")
    a(f"")
    
    # Harmony
    mh = recipe.get('molecular_harmony', {})
    a(f"## 분자 조화 분석")
    a(f"")
    a(f"- 조화도: {mh.get('harmony', 0):.3f} / 1.000")
    a(f"- 시너지: {mh.get('synergy_count', 0)}쌍 / 마스킹 위험: {mh.get('masking_count', 0)}쌍")
    a(f"- IFRA 경고: {'없음 ✅' if not recipe.get('ifra_warnings') else recipe['ifra_warnings']}")
    a(f"")
    
    if mh.get('synergy_bonuses'):
        for syn in mh['synergy_bonuses'][:3]:
            a(f"- 시너지: {syn['pair']} ({syn['bonus']})")
    a(f"")
    a(f"---")
    a(f"")
    a(f"*V16 OdorGNN(22d) + POM + AIRecipeEngine + MolecularHarmony*")
    
    return '\n'.join(lines)


# ─── 메인 실행 ───
print("=" * 60)
print("  🇰🇷 한국 지역 특산식물 향수 시리즈")
print("  V16 OdorGNN 완전체 파이프라인")
print("=" * 60)

output_dir = os.path.join('weights', 'korean_series')
os.makedirs(output_dir, exist_ok=True)

all_results = []

for i, profile in enumerate(KOREAN_RECIPES, 1):
    print(f"\n{'─'*60}")
    print(f"  [{i}/{len(KOREAN_RECIPES)}] {profile['name']} ({profile['region']})")
    print(f"  {profile['description']}")
    print(f"{'─'*60}")
    
    try:
        # 목표 벡터
        target = generate_target_vector(profile)
        
        # 레시피 생성
        recipe = recipe_engine.generate_recipe(
            mood=profile['mood'],
            season=profile['season'],
            preferences=profile['preferences'],
            intensity=profile['intensity'],
            complexity=12,
            batch_ml=100,
        )
        
        print(f"  ✅ {recipe['name_ko']} ({recipe['name']})")
        print(f"     {recipe['concentration']} | {recipe['stats']['total_ingredients']}개 원료 | {recipe['cost']['total_formatted']}")
        print(f"     조화도: {recipe.get('molecular_harmony',{}).get('harmony',0):.3f}")
        
        # .md 저장
        md_content = recipe_to_markdown(profile, recipe, target)
        md_path = os.path.join(output_dir, f"{profile['id']}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        all_results.append({
            'id': profile['id'],
            'name': profile['name'],
            'region': profile['region'],
            'recipe_name': f"{recipe['name_ko']} ({recipe['name']})",
            'concentration': recipe['concentration'],
            'ingredients': recipe['stats']['total_ingredients'],
            'cost': recipe['cost']['total_formatted'],
            'harmony': recipe.get('molecular_harmony',{}).get('harmony', 0),
        })
        
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        import traceback
        traceback.print_exc()

# ─── 총정리 index.md 생성 ───
print(f"\n{'='*60}")
print(f"  📋 전체 레시피 인덱스 생성")
print(f"{'='*60}")

index_lines = []
index_lines.append("# 🇰🇷 한국 지역 특산식물 향수 시리즈")
index_lines.append("")
index_lines.append("> V16 OdorGNN (CosSim 0.8092) 완전체 AI 파이프라인으로 생성")
index_lines.append("> 2026-02-18")
index_lines.append("")
index_lines.append("---")
index_lines.append("")
index_lines.append("| # | 지역 | 향수 이름 | 농도 | 원료 | 비용 | 조화도 |")
index_lines.append("|---|------|----------|------|------|------|--------|")

for i, r in enumerate(all_results, 1):
    index_lines.append(
        f"| {i} | {r['region']} | {r['recipe_name']} | {r['concentration'][:3]} | {r['ingredients']}개 | {r['cost']} | {r['harmony']:.3f} |"
    )

index_lines.append("")
index_lines.append("---")
index_lines.append("")
index_lines.append("## 레시피 파일")
index_lines.append("")
for r in all_results:
    index_lines.append(f"- [{r['name']}]({r['id']}.md)")
index_lines.append("")

index_path = os.path.join(output_dir, 'index.md')
with open(index_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(index_lines))

# JSON 전체 저장
json_path = os.path.join(output_dir, 'all_recipes.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n  ✅ {len(all_results)}개 레시피 생성 완료!")
print(f"  📁 저장: {output_dir}/")
print(f"     index.md (총정리)")
for r in all_results:
    print(f"     {r['id']}.md — {r['name']}")
print(f"\n완료! 🎉")
