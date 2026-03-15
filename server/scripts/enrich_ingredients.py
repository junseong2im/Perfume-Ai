"""
원료 DB 프로 필드 추가: CAS 번호, 대체 원료, 희석 정보, 기능/특징
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
ING_PATH = os.path.join(BASE, '..', '..', 'data', 'ingredients.json')

with open(ING_PATH, 'r', encoding='utf-8') as f:
    ings = json.load(f)

ing_map = {i['id']: i for i in ings}

# ================================================
# CAS 번호 매핑 (주요 원료)
# ================================================
CAS = {
    # 시트러스
    "bergamot":"8007-75-8","lemon":"8008-56-8","orange":"8008-57-9","grapefruit":"8016-20-4",
    "mandarin":"8008-31-9","lime":"8008-26-2","yuzu":"None","petitgrain":"8014-17-3",
    "blood_orange":"68514-75-0","tangerine":"8016-85-1","clementine":"68917-33-9",
    "bitter_orange":"68916-04-1","neroli_oil":"8016-38-4",
    "limonene":"5989-27-5","linalool":"78-70-6","citral":"5392-40-5",
    "dihydromyrcenol":"18479-58-8","lemon_verbena":"8024-12-2",
    # 플로럴
    "rose":"8007-01-0","jasmine":"8022-96-6","ylang_ylang":"8006-81-3",
    "tuberose":"8024-55-3","iris":"8002-73-1","peony":"None","neroli":"8016-38-4",
    "lily_of_valley":"None","magnolia":"None","freesia":"None","geranium":"8000-46-2",
    "violet":"8024-08-6","orange_blossom":"8016-38-4","gardenia":"None",
    "rose_absolute":"8007-01-0","jasmine_absolute":"8022-96-6",
    "osmanthus":"68917-05-5","cherry_blossom":"None",
    "rose_damascena":"8007-01-0","rose_centifolia":"8007-01-0",
    "jasmine_sambac":"91770-14-8","jasmine_grandiflorum":"84776-64-7",
    "taif_rose":"8007-01-0","turkish_rose":"8007-01-0",
    # 스파이시
    "pink_pepper":"68650-39-5","black_pepper":"8006-82-4","cardamom":"8000-66-6",
    "ginger":"8007-08-7","saffron":"8022-19-3","cinnamon":"8015-91-6",
    "clove":"8000-34-8","nutmeg":"8008-45-5","coriander":"8008-52-4",
    "star_anise":"8007-70-3","cumin":"8014-13-9","juniper":"8002-68-4",
    # 우디
    "sandalwood":"8006-87-9","cedarwood":"8000-27-9","vetiver":"8016-96-4",
    "patchouli":"8014-09-3","oud":"None","guaiac_wood":"8016-23-7",
    "birch":"8001-88-5","cypress":"8013-86-3","hinoki":"None",
    "rosewood":"8015-77-8","pine":"8002-09-3","fir":"8021-28-1",
    "atlas_cedar":"8023-85-6","virginia_cedar":"8000-27-9",
    "iso_e_super":"54464-57-2","cashmeran":"33704-61-9",
    # 머스크/앰버
    "musk":"None","white_musk":"None","galaxolide":"1222-05-5",
    "muscone":"541-91-3","habanolide":"34902-57-3",
    "ambroxan":"6790-58-5","cetalox":"3738-00-9",
    "ambergris":"None","amber":"None",
    "coumarin":"91-64-5","ethylene_brassylate":"105-95-3",
    "helvetolide":"141773-73-1","tonalide":"21145-77-7",
    # 발사믹/레진
    "benzoin":"9000-05-9","frankincense":"8016-36-2","myrrh":"8016-37-3",
    "labdanum":"8016-26-0","incense":"None","styrax":"8024-01-9",
    "tolu_balsam":"8024-29-1","peru_balsam":"8007-00-9",
    "olibanum":"8016-36-2",
    # 구르망
    "vanilla":"8024-06-4","tonka_bean":"8046-22-8","caramel":"None",
    "chocolate":"None","coffee":"None","honey":"None",
    "cocoa":"8002-31-1","praline":"None",
    # 합성 (아로마케미컬)
    "vanillin":"121-33-5","ethyl_vanillin":"121-32-4",
    "maltol":"118-71-8","ethyl_maltol":"4940-11-8",
    "hedione":"24851-98-7","phenylethyl_alcohol":"60-12-8",
    "geraniol":"106-24-1","citronellol":"106-22-9",
    "eugenol":"97-53-0","isoeugenol":"97-54-1",
    "indole":"120-72-9","skatole":"83-34-1",
    "benzyl_acetate":"140-11-4","benzaldehyde":"100-52-7",
    "cinnamaldehyde":"104-55-2","acetophenone":"98-86-2",
    "methyl_salicylate":"119-36-8","heliotropin":"120-57-0",
    "frambinone":"5471-51-2","furaneol":"3658-77-3",
    "calone":"28940-11-6",
    # 알데히드
    "aldehyde_c8":"124-13-0","aldehyde_c9":"124-19-6",
    "aldehyde_c10":"112-31-2","aldehyde_c11":"112-44-7","aldehyde_c12":"112-54-9",
    # 테르펜
    "alpha_pinene":"80-56-8","beta_pinene":"127-91-3",
    "menthol":"89-78-1","menthone":"14073-97-3",
    "carvone":"6485-40-1","thymol":"89-83-8",
    "borneol":"507-70-0","camphor":"76-22-2",
    "bisabolol":"515-69-5","nerolidol":"7212-44-4",
    # 락톤
    "gamma_decalactone":"706-14-9","gamma_undecalactone":"104-67-6",
    "gamma_nonalactone":"104-61-0","delta_decalactone":"705-86-2",
    "ambrettolide":"7779-50-2","exaltolide":"106-02-5",
    # 그린
    "cis3_hexenol":"928-96-1","cis3_hexenyl_acetate":"3681-71-8",
    "galbanum":"8023-88-9","violet_leaf":"8024-08-6",
    # 기타
    "leather":"None","castoreum":"8023-83-4","tobacco":"None",
    "geosmin":"19700-21-1","guaiacol":"90-05-1",
    # 어로마틱
    "lavender":"8000-28-0","rosemary":"8000-25-7","thyme":"8007-46-3",
    "basil":"8015-73-4","sage":"8022-56-8","eucalyptus":"8000-48-4",
    "chamomile":"8002-66-2","clary_sage":"8016-63-5",
    # 아쿠아
    "sea_salt":"None","marine":"None",
    # 프루티
    "blackcurrant":"68606-81-5","raspberry":"None","peach":"None",
    "green_apple":"None","pear":"None","fig":"None",
    # 새로운 합성
    "hedione_hc":"24851-98-7","javanol":"198404-98-7",
    "clearwood":"None","akigalawood":"None",
    "amber_xtreme":"None","vertofix":"32388-55-9",
    # 한국
    "pine_needle_korean":"8002-09-3","omija":"None",
}

# ================================================
# 대체 원료 매핑 (같은 향 계열에서 대체 가능한 원료 2~3개)
# ================================================
SUBS = {
    # 시트러스
    "bergamot": ["lemon","petitgrain","linalyl_acetate"],
    "lemon": ["citral","lime","lemon_verbena"],
    "orange": ["mandarin","tangerine","blood_orange"],
    "grapefruit": ["lime","lemon","pink_pepper"],
    "mandarin": ["tangerine","clementine","orange"],
    "lime": ["lemon","grapefruit","citral"],
    "yuzu": ["bergamot","lemon","grapefruit"],
    "petitgrain": ["neroli","bergamot","linalyl_acetate"],
    "blood_orange": ["orange","mandarin","grapefruit"],
    # 플로럴
    "rose": ["geranium","citronellol","phenylethyl_alcohol"],
    "jasmine": ["hedione","jasmine_absolute","methyl_jasmonate"],
    "ylang_ylang": ["ylang_extra","benzyl_acetate","jasmine"],
    "tuberose": ["tuberose_absolute","gardenia","jasmine"],
    "iris": ["orris","orris_butter","ionone_alpha"],
    "peony": ["rose","magnolia","lily_of_valley"],
    "neroli": ["neroli_oil","petitgrain","orange_blossom"],
    "lily_of_valley": ["hydroxycitronellal","lilial","muguet_aldehyde"],
    "magnolia": ["champaca","ylang_ylang","linalool"],
    "freesia": ["lily_of_valley","violet","peony"],
    "geranium": ["citronellol","geraniol","rose"],
    "violet": ["ionone_alpha","ionone_beta","orris"],
    "orange_blossom": ["neroli","linalool","petitgrain"],
    "gardenia": ["tuberose","jasmine","gardenia"],
    "osmanthus": ["champaca","davana","osmanthus"],
    "rose_absolute": ["rose_damascena","turkish_rose","geranium"],
    "jasmine_absolute": ["jasmine_sambac","jasmine_grandiflorum","hedione"],
    # 스파이시
    "pink_pepper": ["black_pepper","cardamom","ginger"],
    "black_pepper": ["pink_pepper","szechuan_pepper","ginger"],
    "cardamom": ["elemi","coriander","ginger"],
    "ginger": ["cardamom","galangal","pink_pepper"],
    "saffron": ["safranal","turmeric","osmanthus"],
    "cinnamon": ["cinnamaldehyde","clove","nutmeg"],
    "clove": ["eugenol","cinnamon","nutmeg"],
    "nutmeg": ["clove","cinnamon","cardamom"],
    # 우디
    "sandalwood": ["javanol","bacdanol","polysantol"],
    "cedarwood": ["atlas_cedar","virginia_cedar","iso_e_super"],
    "vetiver": ["vetiver_bourbon","vetiver_haiti","clearwood"],
    "patchouli": ["clearwood","akigalawood","patchouli_dark"],
    "oud": ["oud_synth","agarwood","guaiac_wood"],
    "guaiac_wood": ["guaiacol","vetiver","cedarwood"],
    "birch": ["birch_tar","guaiacol","leather"],
    "cypress": ["pine","juniper","cedarwood"],
    "hinoki": ["hinoki_japanese","cypress","cedarwood"],
    "iso_e_super": ["cashmeran","vertofix","cedarwood"],
    "pine": ["pine_needle_korean","fir","spruce"],
    "white_cedar": ["cedarwood","atlas_cedar","cypress"],
    # 머스크
    "musk": ["white_musk","galaxolide","ethylene_brassylate"],
    "white_musk": ["galaxolide","habanolide","helvetolide"],
    "galaxolide": ["habanolide","ethylene_brassylate","tonalide"],
    "muscone": ["cosmone","exaltolide","ambrette"],
    # 앰버
    "ambroxan": ["cetalox","ambrox_dl","amber_xtreme"],
    "amber": ["labdanum","ambroxan","benzoin"],
    "ambergris": ["ambroxan","cetalox","ambrox_dl"],
    "labdanum": ["amber_resinoid","benzoin","styrax"],
    # 베이스
    "vanilla": ["vanillin","ethyl_vanillin","madagascar_vanilla"],
    "tonka_bean": ["coumarin","vanilla","benzoin"],
    "benzoin": ["tolu_balsam","styrax","vanilla"],
    "frankincense": ["olibanum","elemi","somalian_frankincense"],
    "myrrh": ["opoponax","ethiopian_myrrh","frankincense"],
    # 합성
    "hedione": ["hedione_hc","methyl_dihydrojasmonate","linalool"],
    "vanillin": ["ethyl_vanillin","vanilla","benzoin"],
    "indole": ["jasmine_absolute","tuberose","skatole"],
    "coumarin": ["tonka_bean","hay","coumarin_lactone"],
    "calone": ["helional","adoxal","marine"],
    "iso_e_super": ["cashmeran","vertofix","georgywood"],
    # 가죽/스모키
    "leather": ["leather_accord","birch_tar","castoreum_synth"],
    "tobacco": ["immortelle","hay","vanilla"],
    # 구르망
    "caramel": ["maltol","ethyl_maltol","furaneol"],
    "chocolate": ["cocoa","vanilla","coffee"],
    "coffee": ["cocoa","chocolate","rum"],
    # 그린
    "galbanum": ["violet_leaf","cis3_hexenol","fig_leaf"],
    # 한국
    "pine_needle_korean": ["pine","fir","spruce"],
}

# ================================================
# 희석 정보 (강한 원료는 희석 필수)
# ================================================
DILUTION = {
    # format: (용매, 희석%) — 100이면 원액
    # 강한 원료
    "ambroxan": ("DPG", 10), "indole": ("DPG", 10), "skatole": ("DPG", 1),
    "vanillin": ("DPG", 10), "ethyl_vanillin": ("DPG", 10),
    "castoreum": ("DPG", 10), "civet": ("DPG", 5), "castoreum_synth": ("DPG", 10),
    "civet_synth": ("DPG", 5),
    "muscone": ("DPG", 10), "cosmone": ("DPG", 10),
    "saffron": ("DPG", 20), "safranal": ("DPG", 20),
    "oud": ("DPG", 10), "agarwood": ("DPG", 10),
    "oud_synth": ("DPG", 10),
    "para_cresol": ("DPG", 1),
    "phenylacetaldehyde": ("DPG", 10),
    "damascenone": ("DPG", 1),
    "geosmin": ("DPG", 1),
    # 머스크 (일부 IPM 기반)
    "galaxolide": ("IPM", 50), "tonalide": ("IPM", 50),
    "habanolide": ("IPM", 50),
    # 왁시 알데히드
    "aldehyde_c11": ("DPG", 10), "aldehyde_c12": ("DPG", 10),
    # 대부분 원액
}

# ================================================
# 기능/특징 (비고란)
# ================================================
FUNC = {
    "bergamot": "상쾌한 시작, 광독성 제거 추천",
    "lemon": "시트러스 부스팅, 휘발 빠름",
    "orange": "달콤한 시트러스 바디",
    "petitgrain": "시트러스-우디 브릿지",
    "linalool": "플로럴-우디 연결 키노트",
    "linalyl_acetate": "베르가못/라벤더의 뼈대",
    "hedione": "확산성(Sillage) 부여, 볼륨감",
    "hedione_hc": "헤디온 고농축, 래디언스 극대화",
    "iso_e_super": "우디 벨벳 질감, 스킨 센트",
    "cashmeran": "무스키 우디, 따뜻한 감싸기",
    "sandalwood": "크리미 우디 베이스",
    "javanol": "샌달우드 대체 (합성), 지속력 가",
    "cedarwood": "드라이 우디 구조",
    "vetiver": "어시 우디, 남성적 깊이",
    "patchouli": "어시 우디, 드라이다운 앵커",
    "ambroxan": "앰버그리스 합성, 지속력 강화",
    "cetalox": "앰브록산 대체, 클린 앰버",
    "galaxolide": "화이트 머스크, 베이스 안착",
    "habanolide": "클린 머스크, 피치 머스크",
    "vanillin": "달콤함 부여 (소량!)",
    "ethyl_vanillin": "바닐린의 3배 강도",
    "coumarin": "쿠마린 따뜻한 달콤함, 통카 대체",
    "indole": "자스민 핵심, 극소량 (동물적 주의)",
    "rose": "로즈 하트, 고급감",
    "jasmine": "자스민 하트, 관능미",
    "geranium": "로즈 대체, 가성비",
    "labdanum": "앰버 베이스, 레더리 뉘앙스",
    "benzoin": "발사믹 달콤한 베이스 고정",
    "frankincense": "인센스 스모키 레진",
    "vanilla": "구르망 베이스 앵커",
    "tonka_bean": "쿠마린 천연, 따뜻한 마무리",
    "leather": "가죽 어코드, 남성적",
    "cinnamon": "스파이시 따뜻함 (소량)",
    "calone": "아쿠아틱 오존, 마린 노트",
    "maltol": "솜사탕 달콤함, 과일 부스팅",
    "ethyl_maltol": "말톨의 10배 강도",
    "muscone": "애니멀 머스크 핵심",
    "gamma_decalactone": "피치 크리미 락톤",
    "frambinone": "라즈베리 키노트",
    "menthol": "쿨링 효과, 극소량",
    "guaiac_wood": "스모키 우디, 부드러운",
    "birch_tar": "가죽/스모키 원료",
    "pine": "침엽수 우디 그린",
    "pine_needle_korean": "한국 소나무 우디 그린",
    "clearwood": "패출리 대체 (클린 버전)",
    "akigalawood": "패출리 대체 (앰버릭)",
}

# ================================================
# 적용
# ================================================
updated = 0
for ing in ings:
    iid = ing['id']
    changed = False
    
    # CAS
    if iid in CAS and CAS[iid] != "None":
        ing['cas_number'] = CAS[iid]
        changed = True
    
    # 대체 원료
    if iid in SUBS:
        # 실제 존재하는 원료만
        valid_subs = [s for s in SUBS[iid] if s in ing_map and s != iid]
        if valid_subs:
            ing['substitutes'] = valid_subs[:3]
            changed = True
    
    # 희석
    if iid in DILUTION:
        solvent, pct = DILUTION[iid]
        ing['dilution_solvent'] = solvent
        ing['dilution_pct'] = pct
        changed = True
    else:
        ing['dilution_solvent'] = "-"
        ing['dilution_pct'] = 100
    
    # 기능
    if iid in FUNC:
        ing['function_note'] = FUNC[iid]
        changed = True
    
    if changed:
        updated += 1

print(f"업데이트: {updated}개 원료")
print(f"CAS 번호: {sum(1 for i in ings if i.get('cas_number'))}개")
print(f"대체 원료: {sum(1 for i in ings if i.get('substitutes'))}개")
print(f"희석 필요: {sum(1 for i in ings if i.get('dilution_pct',100) < 100)}개")
print(f"기능 설명: {sum(1 for i in ings if i.get('function_note'))}개")

with open(ING_PATH, 'w', encoding='utf-8') as f:
    json.dump(ings, f, ensure_ascii=False, indent=2)
print(f"✅ 저장 완료: {ING_PATH}")
