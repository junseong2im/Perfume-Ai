"""
향료 원료 DB 대규모 확장 스크립트
기존 82개 → 500+ 원료로 확장
중복 없이, SMILES 매핑 포함
"""
import json, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ING_PATH = os.path.join(BASE, '..', 'data', 'ingredients.json')
SMILES_PATH = os.path.join(BASE, '..', 'data', 'ingredient_smiles.json')

# 기존 로드
with open(ING_PATH, 'r', encoding='utf-8') as f:
    existing = json.load(f)
existing_ids = {ing['id'] for ing in existing}
print(f"기존 원료: {len(existing)}개")

# 기존 SMILES 로드
smiles_map = {}
if os.path.exists(SMILES_PATH):
    with open(SMILES_PATH, 'r', encoding='utf-8') as f:
        smiles_map = json.load(f)

def I(id, ko, en, cat, note, vol, inten, long, desc, typ_pct, max_pct, smiles=None):
    """원료 생성 헬퍼"""
    if id in existing_ids:
        return None
    existing_ids.add(id)
    if smiles:
        smiles_map[id] = smiles
    return {"id":id,"name_ko":ko,"name_en":en,"category":cat,"note_type":note,
            "volatility":vol,"intensity":inten,"longevity":long,
            "descriptors":desc,"typical_pct":typ_pct,"max_pct":max_pct}

new_ingredients = []

# ================================================================
# 시트러스 계열 (Citrus) - TOP
# ================================================================
for item in [
    I("blood_orange","블러드 오렌지","Blood Orange","citrus","top",9,6,2,["시트러스","달콤","베리"],5,12,"OC(=O)/C=C\\c1ccccc1"),
    I("tangerine","탄제린","Tangerine","citrus","top",9,5,2,["시트러스","달콤","과일"],5,12,"CC(=CCC/C(C)=C\\C)C"),
    I("kumquat","금귤","Kumquat","citrus","top",8.5,5,2,["시트러스","달콤","그린"],4,10,None),
    I("citron","시트론","Citron","citrus","top",9,7,2,["시트러스","상큼","허벌"],5,12,"CC(=CCCC(=CC=O)C)C"),
    I("bitter_orange","비터 오렌지","Bitter Orange","citrus","top",8.5,7,2.5,["시트러스","비터","아로마틱"],5,12,None),
    I("lemon_verbena","레몬 버베나","Lemon Verbena","citrus","top",8,6,2.5,["시트러스","허벌","그린"],4,10,"CC(=CCCC(=CC=O)C)C"),
    I("neroli_oil","네롤리 오일","Neroli Oil","citrus","top",7.5,6,3,["시트러스","플로럴","그린"],5,12,None),
    I("clementine","클레멘타인","Clementine","citrus","top",9,5,2,["시트러스","달콤","밝은"],5,12,None),
    # 아로마 화합물
    I("limonene","리모넨","Limonene","citrus","top",9.5,7,1.5,["시트러스","오렌지","신선"],4,10,"CC(=CCC1CC=C(C)C1)C" if "limonene" not in smiles_map else None),
    I("linalool","리날룰","Linalool","aromatic","top",8,5,3,["플로럴","시트러스","우디"],5,12,"CC(=CCC(O)(C)C=C)C"),
    I("citral","시트랄","Citral","citrus","top",9,7,2,["시트러스","레몬","샤프"],3,8,"CC(=CCCC(=CC=O)C)C"),
    I("dihydromyrcenol","디히드로미르세놀","Dihydromyrcenol","fresh","top",8.5,6,2.5,["시트러스","메탈릭","클린"],5,12,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 그린/허벌 계열 - TOP/MIDDLE
# ================================================================
for item in [
    I("galbanum","갈바넘","Galbanum","green","top",8,7,3,["그린","발사믹","샤프"],3,8,None),
    I("rosemary","로즈마리","Rosemary","aromatic","top",8,6,2.5,["허벌","캠포러스","그린"],4,10,"CC1(C)C2CC(=O)C1(C)CC2"),
    I("thyme","타임","Thyme","aromatic","top",7.5,7,3,["허벌","스파이시","그린"],3,8,"Cc1ccc(O)c(C(C)C)c1"),
    I("sage","세이지","Sage","aromatic","top",7.5,6,3,["허벌","캠포러스","그린"],3,8,None),
    I("clary_sage","클라리 세이지","Clary Sage","aromatic","middle",6,5,4,["허벌","머스크","앰버"],4,10,None),
    I("artemisia","쑥","Artemisia","herbal","top",7.5,6,3,["허벌","그린","아로마틱"],3,8,None),
    I("tarragon","타라곤","Tarragon","herbal","top",7.5,5,2.5,["허벌","아니스","그린"],3,7,None),
    I("eucalyptus","유칼립투스","Eucalyptus","aromatic","top",9,7,2,["캠포러스","쿨링","신선"],3,8,"CC1(C)C2CCC(C)(C2)O1"),
    I("fennel","펜넬","Fennel","herbal","top",7.5,5,2.5,["아니스","허벌","달콤"],3,8,None),
    I("marjoram","마조람","Marjoram","herbal","middle",6.5,5,3.5,["허벌","따뜻한","우디"],3,8,None),
    I("oregano","오레가노","Oregano","herbal","top",7.5,7,3,["허벌","스파이시","따뜻한"],2,5,None),
    I("dill","딜","Dill","herbal","top",8,5,2,["허벌","그린","시트러스"],3,7,None),
    I("chamomile","카모마일","Chamomile","herbal","middle",6,4,4,["허벌","플로럴","달콤"],4,10,None),
    I("mugwort","쑥","Mugwort","herbal","middle",6.5,5,4,["허벌","비터","그린"],3,7,None),
    I("violet_leaf","바이올렛 리프","Violet Leaf","green","top",7.5,6,3,["그린","아쿠아","크리스피"],3,8,None),
    I("fig_leaf","무화과 잎","Fig Leaf","green","top",7,5,3,["그린","우디","코코넛"],4,10,None),
    I("hay","건초","Hay","green","middle",5.5,4,4,["그린","쿠마린","따뜻한"],4,10,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 플로럴 계열 - MIDDLE
# ================================================================
for item in [
    I("gardenia","가드니아","Gardenia","floral","middle",5,7,5,["플로럴","크리미","달콤"],5,12,None),
    I("frangipani","프랜지파니","Frangipani","floral","middle",5,6,5,["플로럴","트로피컬","달콤"],4,10,None),
    I("mimosa","미모사","Mimosa","floral","middle",5.5,5,4.5,["플로럴","파우더리","그린"],4,10,None),
    I("lily","릴리","Lily","floral","middle",5.5,6,4.5,["플로럴","그린","왁시"],5,12,None),
    I("heliotrope","헬리오트로프","Heliotrope","floral","middle",4.5,5,5,["플로럴","파우더리","아몬드"],4,10,None),
    I("carnation","카네이션","Carnation","floral","middle",5.5,6,4.5,["플로럴","스파이시","클로브"],4,10,None),
    I("chrysanthemum","국화","Chrysanthemum","floral","middle",5.5,5,4,["플로럴","그린","허벌"],4,10,None),
    I("lotus","연꽃","Lotus","floral","middle",6,4,3.5,["플로럴","아쿠아","클린"],4,10,None),
    I("plum_blossom","매화","Plum Blossom","floral","middle",6,4,3.5,["플로럴","달콤","그린"],4,10,None),
    I("wisteria","등나무꽃","Wisteria","floral","middle",6,4,3.5,["플로럴","달콤","그린"],4,10,None),
    I("honeysuckle","허니서클","Honeysuckle","floral","middle",6,5,4,["플로럴","달콤","허니"],5,12,None),
    I("orchid","오키드","Orchid","floral","middle",5,5,5,["플로럴","트로피컬","크리미"],4,10,None),
    I("acacia","아카시아","Acacia","floral","middle",5.5,4,4,["플로럴","허니","파우더리"],4,10,None),
    I("linden_blossom","린덴 블라썸","Linden Blossom","floral","middle",5.5,5,4.5,["플로럴","허니","그린"],4,10,None),
    I("hyacinth","히아신스","Hyacinth","floral","middle",6,6,3.5,["플로럴","그린","달콤"],4,10,None),
    I("narcissus","나르시스","Narcissus","floral","middle",5,6,5,["플로럴","그린","동물적"],3,8,None),
    I("champaca","참파카","Champaca","floral","middle",4.5,7,5.5,["플로럴","달콤","이국적"],3,8,None),
    I("tiare","티아레","Tiare","floral","middle",5,6,5,["플로럴","크리미","트로피컬"],4,10,None),
    I("boronia","보로니아","Boronia","floral","middle",5,5,4.5,["플로럴","프루티","그린"],3,8,None),
    I("davana","다바나","Davana","floral","middle",5.5,5,4.5,["플로럴","프루티","우디"],3,8,None),
    I("tagetes","타게테스","Tagetes","floral","middle",6.5,6,3,["플로럴","프루티","허벌"],3,7,None),
    I("rose_absolute","로즈 앱솔루트","Rose Absolute","floral","middle",4.5,9,6,["플로럴","로맨틱","리치"],6,15,None),
    I("jasmine_absolute","자스민 앱솔루트","Jasmine Absolute","floral","middle",4.5,9,6,["플로럴","관능적","인돌릭"],5,12,None),
    I("tuberose_absolute","튜베로즈 앱솔루트","Tuberose Absolute","floral","middle",4,10,7,["플로럴","크리미","버터리"],3,8,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 스파이시 계열 - TOP/MIDDLE
# ================================================================
for item in [
    I("star_anise","스타 아니스","Star Anise","spicy","middle",6,7,4,["아니스","달콤","스파이시"],2,6,None),
    I("cumin","쿠민","Cumin","spicy","middle",5.5,7,4.5,["스파이시","어시","따뜻한"],2,5,None),
    I("coriander","코리앤더","Coriander","spicy","top",7,5,3,["스파이시","시트러스","그린"],3,8,None),
    I("juniper","주니퍼","Juniper","aromatic","top",7.5,6,3,["아로마틱","우디","그린"],4,10,None),
    I("elemi","엘레미","Elemi","spicy","top",7,5,3,["스파이시","시트러스","레진"],3,8,None),
    I("bay_leaf","월계수","Bay Leaf","herbal","middle",6.5,5,3.5,["허벌","스파이시","아로마틱"],3,7,None),
    I("anise","아니스","Anise","spicy","middle",6,6,4,["아니스","달콤","허벌"],2,6,None),
    I("caraway","캐러웨이","Caraway","spicy","middle",6,5,3.5,["스파이시","허벌","따뜻한"],2,6,None),
    I("galangal","갈랑갈","Galangal","spicy","middle",6,6,4,["스파이시","시트러스","따뜻한"],2,6,None),
    I("wasabi","와사비","Wasabi","spicy","top",9,8,1.5,["스파이시","그린","샤프"],1,3,None),
    I("szechuan_pepper","쓰촨 페퍼","Szechuan Pepper","spicy","top",7.5,7,2.5,["스파이시","시트러스","마비"],2,5,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 우디 계열 - BASE
# ================================================================
for item in [
    I("hinoki","히노키","Hinoki","woody","base",3.5,5,7,["우디","시트러스","클린"],5,12,None),
    I("agarwood","침향","Agarwood","woody","base",1.5,9,10,["우디","스모키","앰버"],3,8,None),
    I("rosewood","로즈우드","Rosewood","woody","base",4,5,6,["우디","플로럴","시트러스"],5,12,"CC(O)=CCC=C(C)C"),
    I("teak","티크","Teak","woody","base",2.5,5,7,["우디","가죽","드라이"],4,10,None),
    I("ebony","에보니","Ebony","woody","base",2,6,8,["우디","다크","스모키"],4,10,None),
    I("pine","소나무","Pine","woody","top",7.5,6,3,["우디","그린","신선"],5,12,"CC1=CCC2CC1C2(C)C"),
    I("fir","전나무","Fir","woody","top",7,5,3.5,["우디","그린","발사믹"],5,12,None),
    I("spruce","가문비나무","Spruce","woody","top",7,5,3.5,["우디","그린","신선"],5,12,None),
    I("bamboo_wood","대나무","Bamboo","woody","middle",5.5,4,4.5,["우디","그린","클린"],4,10,None),
    I("driftwood","드리프트우드","Driftwood","woody","base",2.5,4,7,["우디","아쿠아","미네랄"],4,10,None),
    I("cashmeran","캐시머란","Cashmeran","woody","base",2.5,5,8,["우디","머스크","스파이시"],5,12,"CC1(C)C(=O)CCC2(C)CCCC12"),
    I("iso_e_super","이소이슈퍼","Iso E Super","woody","base",2,4,9,["우디","앰버","벨벳"],8,20,None),
    I("okoumal","오쿠말","Okoumal","woody","base",2.5,4,8,["우디","앰버","크리미"],5,12,None),
    I("amyris","아미리스","Amyris","woody","base",2.5,4,7,["우디","크리미","발사믹"],5,12,None),
    I("gaiac","가이악","Gaiac","woody","base",2.5,5,7,["우디","스모키","로지"],5,12,None),
    I("atlas_cedar","아틀라스 시더","Atlas Cedar","woody","base",3,5,7,["우디","드라이","따뜻한"],6,15,"CC1CCC2C(C)(CCCC2(C)C)C1"),
    I("virginia_cedar","버지니아 시더","Virginia Cedar","woody","base",3,5,7,["우디","드라이","펜슬"],6,15,None),
    I("hemlock","헴록","Hemlock","woody","base",3.5,4,6,["우디","그린","발사믹"],4,10,None),
    I("massoia_bark","마소이아 바크","Massoia Bark","woody","base",3,6,7,["우디","코코넛","크리미"],3,8,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 머스크/앰버/파우더리 - BASE
# ================================================================
for item in [
    I("galaxolide","갈락솔라이드","Galaxolide","musk","base",1.5,5,9,["머스크","클린","플로럴"],6,15,None),
    I("muscone","무스콘","Muscone","musk","base",1,6,10,["머스크","파우더리","동물적"],3,8,"CCCCCCCCCCCC(=O)CC/C=C\\C"),
    I("habanolide","하바노라이드","Habanolide","musk","base",1.5,4,9,["머스크","크리미","피치"],5,12,None),
    I("ethylene_brassylate","에틸렌 브라실레이트","Ethylene Brassylate","musk","base",1,4,10,["머스크","파우더리","플로럴"],5,12,None),
    I("helvetolide","헬베톨라이드","Helvetolide","musk","base",1.5,4,9,["머스크","프루티","클린"],5,12,None),
    I("ambrette","앰브레트","Ambrette","musk","base",2,5,8,["머스크","플로럴","와인"],4,10,None),
    I("ambroxan","앰브록산","Ambroxan","amber","base",1.5,6,10,["앰버","우디","머스크"],5,12,"CC1(C)CCCC2(C)C1CCC1(C)OCCCC12"),
    I("cetalox","세탈록스","Cetalox","amber","base",1.5,5,10,["앰버","우디","클린"],5,12,None),
    I("coumarin","쿠마린","Coumarin","amber","base",3,5,7,["쿠마린","달콤","따뜻한"],4,10,"O=c1ccc2ccccc2o1"),
    I("heliotropin","헬리오트로핀","Heliotropin","powdery","base",3,5,6,["파우더리","달콤","플로럴"],4,10,"O=Cc1ccc2OCOc2c1"),
    I("orris","오리스","Orris","powdery","base",2.5,5,7,["파우더리","우디","바이올렛"],4,10,None),
    I("styrax","스티락스","Styrax","balsamic","base",2.5,6,7,["발사믹","가죽","앰버"],3,8,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 프루티 계열 - TOP/MIDDLE
# ================================================================
for item in [
    I("mango","망고","Mango","fruity","top",8,5,2.5,["프루티","트로피컬","달콤"],4,10,None),
    I("passion_fruit","패션프루트","Passion Fruit","fruity","top",8.5,6,2,["프루티","트로피컬","산뜻"],4,10,None),
    I("watermelon","수박","Watermelon","fruity","top",8.5,4,2,["프루티","아쿠아","신선"],4,10,None),
    I("papaya","파파야","Papaya","fruity","top",8,5,2,["프루티","트로피컬","크리미"],4,10,None),
    I("guava","구아바","Guava","fruity","top",8,5,2.5,["프루티","트로피컬","그린"],4,10,None),
    I("pomegranate","석류","Pomegranate","fruity","top",8,5,2.5,["프루티","달콤","탄닌"],4,10,None),
    I("cherry","체리","Cherry","fruity","top",8,6,2.5,["프루티","달콤","아몬드"],4,10,None),
    I("apricot","살구","Apricot","fruity","top",7.5,5,3,["프루티","달콤","벨벳"],4,10,None),
    I("melon","멜론","Melon","fruity","top",8.5,4,2,["프루티","그린","아쿠아"],4,10,None),
    I("banana","바나나","Banana","fruity","top",8,5,2,["프루티","크리미","달콤"],3,8,"CCCCCC(=O)OCC"),
    I("strawberry","딸기","Strawberry","fruity","top",8,5,2,["프루티","달콤","크리미"],4,10,None),
    I("blueberry","블루베리","Blueberry","fruity","top",7.5,5,2.5,["프루티","달콤","그린"],4,10,None),
    I("cranberry","크랜베리","Cranberry","fruity","top",8,5,2,["프루티","시트러스","타르트"],4,10,None),
    I("grape","포도","Grape","fruity","middle",6,4,3.5,["프루티","달콤","와인"],4,10,None),
    I("quince","모과","Quince","fruity","middle",6,4,4,["프루티","플로럴","따뜻한"],4,10,None),
    I("rhubarb","루바브","Rhubarb","fruity","top",7.5,5,2.5,["프루티","그린","타르트"],3,8,None),
    I("cassis","카시스","Cassis","fruity","top",8,6,2.5,["프루티","그린","베리"],4,10,None),
    I("pineapple","파인애플","Pineapple","fruity","top",8.5,6,2,["프루티","트로피컬","달콤"],4,10,None),
    I("kiwi","키위","Kiwi","fruity","top",8,4,2,["프루티","그린","시트러스"],4,10,None),
    I("persimmon","감","Persimmon","fruity","middle",5.5,4,4,["프루티","달콤","따뜻한"],4,10,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 구르망 계열 - MIDDLE/BASE
# ================================================================
for item in [
    I("brown_sugar","흑설탕","Brown Sugar","gourmand","base",3,5,6,["달콤","따뜻한","럼"],4,10,None),
    I("maple","메이플","Maple","gourmand","base",3,5,6,["달콤","따뜻한","우디"],4,10,None),
    I("marshmallow","마시멜로","Marshmallow","gourmand","base",3,4,5,["달콤","파우더리","크리미"],4,10,None),
    I("cotton_candy","솜사탕","Cotton Candy","gourmand","middle",5,5,3.5,["달콤","파우더리","프루티"],4,10,None),
    I("hazelnut","헤이즐넛","Hazelnut","gourmand","middle",5,5,4,["너티","달콤","크리미"],4,10,None),
    I("pistachio","피스타치오","Pistachio","gourmand","middle",5,4,4,["너티","달콤","그린"],4,10,None),
    I("rum","럼","Rum","gourmand","base",3,6,6,["알코올","따뜻한","달콤"],3,8,None),
    I("whiskey","위스키","Whiskey","gourmand","base",3,6,6,["알코올","스모키","오크"],3,8,None),
    I("cocoa","코코아","Cocoa","gourmand","base",3,6,6,["달콤","비터","파우더리"],4,10,None),
    I("tiramisu","티라미수","Tiramisu","gourmand","base",3,5,5,["달콤","커피","크리미"],3,8,None),
    I("dulce_de_leche","둘세 데 레체","Dulce de Leche","gourmand","base",3,5,6,["달콤","크리미","카라멜"],3,8,None),
    I("milk","밀크","Milk","gourmand","middle",5,3,4,["크리미","달콤","클린"],4,10,None),
    I("matcha","말차","Matcha","gourmand","middle",5.5,5,3.5,["그린","비터","파우더리"],4,10,None),
    I("popcorn","팝콘","Popcorn","gourmand","middle",5.5,5,3.5,["버터리","따뜻한","달콤"],3,8,None),
    I("almond_milk","아몬드 밀크","Almond Milk","gourmand","middle",5,4,4,["너티","크리미","달콤"],4,10,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 아쿠아/오존/프레시 - TOP/MIDDLE
# ================================================================
for item in [
    I("rain","레인","Rain","aquatic","top",8,4,2,["아쿠아","오존","클린"],4,10,None),
    I("ocean","오션","Ocean","aquatic","middle",5.5,4,4,["아쿠아","미네랄","솔티"],4,10,None),
    I("seaweed","해초","Seaweed","aquatic","middle",5.5,5,4,["아쿠아","그린","솔티"],3,8,None),
    I("water_lily","수련","Water Lily","aquatic","middle",6,4,3.5,["아쿠아","플로럴","클린"],4,10,None),
    I("dew","이슬","Dew","fresh","top",9,3,1.5,["신선","클린","그린"],3,8,None),
    I("ice","아이스","Ice","fresh","top",9,3,1.5,["쿨링","클린","미네랄"],3,8,None),
    I("snow","스노우","Snow","fresh","top",8.5,3,2,["클린","미네랄","오존"],3,8,None),
    I("calone","칼론","Calone","aquatic","top",8,5,3,["아쿠아","멜론","오존"],3,8,"O=C1CCC(=O)c2ccccc21"),
    I("hedione","헤디온","Hedione","fresh","middle",5.5,4,5,["자스민","시트러스","클린"],8,20,"CCOC(=O)CC1CCC(CC(=O)OC)C1"),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 레더/스모키/어시 - BASE
# ================================================================
for item in [
    I("birch_tar","자작나무 타르","Birch Tar","smoky","base",2,8,9,["스모키","가죽","타르"],2,5,None),
    I("guaiacol","과이아콜","Guaiacol","smoky","base",3,7,7,["스모키","우디","크리미"],2,5,"COc1ccccc1O"),
    I("cade","카드","Cade","smoky","base",2.5,7,8,["스모키","타르","우디"],2,6,None),
    I("smoke","스모크","Smoke","smoky","base",2.5,6,7,["스모키","따뜻한","드라이"],3,8,None),
    I("leather_accord","레더 어코드","Leather Accord","animalic","base",2,7,9,["가죽","스모키","동물적"],3,8,None),
    I("suede_accord","스웨이드 어코드","Suede Accord","animalic","base",2.5,5,7,["가죽","부드러운","파우더리"],4,10,None),
    I("peat","피트","Peat","earthy","base",2,6,8,["어시","스모키","따뜻한"],3,8,None),
    I("soil","흙","Soil","earthy","base",2.5,5,7,["어시","미네랄","젖은"],3,8,None),
    I("geosmin","지오스민","Geosmin","earthy","base",2.5,6,7,["어시","비 내리는","자연"],2,5,"OC1(C)CCCC2(C)CCCCC12"),
    I("papyrus","파피루스","Papyrus","woody","base",3,4,6,["우디","그린","드라이"],4,10,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 레지노스/발사믹 - BASE
# ================================================================
for item in [
    I("opoponax","오포포낙스","Opoponax","balsamic","base",2.5,6,7,["발사믹","달콤","따뜻한"],4,10,None),
    I("tolu_balsam","톨루 발삼","Tolu Balsam","balsamic","base",2.5,6,7,["발사믹","달콤","바닐라"],4,10,None),
    I("peru_balsam","페루 발삼","Peru Balsam","balsamic","base",2,7,8,["발사믹","달콤","시나몬"],3,8,None),
    I("copal","코팔","Copal","balsamic","base",3,5,6,["레진","인센스","시트러스"],3,8,None),
    I("mastic","매스틱","Mastic","balsamic","base",3,5,6,["레진","신선","그린"],3,8,None),
    I("dragon_blood","드래곤 블러드","Dragon's Blood","balsamic","base",2,7,8,["레진","달콤","우디"],3,8,None),
    I("camphor","캄포","Camphor","aromatic","top",8,7,2,["캠포러스","쿨링","메디컬"],2,5,"CC1(C)C2CC(=O)C1(C)CC2"),
    I("elemi_resin","엘레미 레진","Elemi Resin","balsamic","top",7,5,3,["레진","시트러스","그린"],3,8,None),
    I("amber_resinoid","앰버 레지노이드","Amber Resinoid","amber","base",2,7,9,["앰버","따뜻한","발사믹"],5,12,None),
    I("olibanum","올리바넘","Olibanum","balsamic","base",3,6,7,["인센스","레진","스모키"],4,10,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 합성 향료 (아로마케미컬) - 매우 중요
# ================================================================
for item in [
    I("ethyl_vanillin","에틸 바닐린","Ethyl Vanillin","synthetic","base",2.5,8,7,["바닐라","달콤","크리미"],5,12,"CCOc1cc(C=O)ccc1O"),
    I("vanillin","바닐린","Vanillin","synthetic","base",2.5,7,7,["바닐라","달콤","크리미"],5,12,"COc1cc(C=O)ccc1O"),
    I("maltol","말톨","Maltol","synthetic","middle",5,5,4,["달콤","카라멜","코튼캔디"],3,8,"CC1=C(O)C(=O)C=CO1"),
    I("ethyl_maltol","에틸 말톨","Ethyl Maltol","synthetic","base",3,6,6,["달콤","카라멜","솜사탕"],3,8,"CCC1=C(O)C(=O)C=CO1"),
    I("hydroxycitronellal","하이드록시시트로넬랄","Hydroxycitronellal","synthetic","middle",5.5,5,4,["릴리","플로럴","클린"],5,12,None),
    I("ionone_alpha","알파 이오논","Alpha Ionone","synthetic","middle",5,5,4.5,["바이올렛","우디","파우더리"],4,10,"CC(=CC(=O)C1=CCCCC1)C" if "ionone_alpha" not in smiles_map else None),
    I("ionone_beta","베타 이오논","Beta Ionone","synthetic","middle",5,5,4.5,["바이올렛","우디","플로럴"],4,10,"CC(=CC(=O)/C=C/C1C(C)(C)CC=C1)C" if "ionone_beta" not in smiles_map else None),
    I("lyral","리랄","Lyral","synthetic","middle",5.5,5,4,["릴리","뮤게","그린"],5,12,None),
    I("galaxolide_50","갈락솔라이드 50%","Galaxolide 50%","synthetic","base",1.5,4,9,["머스크","클린","달콤"],6,15,None),
    I("phenylethyl_alcohol","페닐에틸알코올","Phenylethyl Alcohol","synthetic","middle",5.5,5,4,["로즈","플로럴","허니"],5,12,"OCCC1=CC=CC=C1"),
    I("geraniol","제라니올","Geraniol","synthetic","middle",5.5,5,4,["로즈","시트러스","플로럴"],4,10,"CC(=CCC/C(C)=C/CO)C"),
    I("citronellol","시트로넬롤","Citronellol","synthetic","middle",5.5,4,4,["로즈","시트러스","그린"],5,12,"CC(CCO)CCC=C(C)C"),
    I("eugenol","유게놀","Eugenol","synthetic","middle",5.5,7,4.5,["클로브","스파이시","따뜻한"],3,7,"COc1cc(CC=C)ccc1O"),
    I("isoeugenol","이소유게놀","Isoeugenol","synthetic","middle",5,6,5,["클로브","스파이시","크리미"],3,7,"COc1cc(/C=C/C)ccc1O"),
    I("indole","인돌","Indole","synthetic","base",3,8,7,["플로럴","동물적","자스민"],1,3,"c1ccc2[nH]ccc2c1"),
    I("skatole","스카톨","Skatole","synthetic","base",2.5,9,8,["동물적","자스민","내추럴"],0.5,2,"Cc1c[nH]c2ccccc12"),
    I("musk_ketone","머스크 케톤","Musk Ketone","synthetic","base",1.5,5,9,["머스크","달콤","파우더리"],5,12,None),
    I("benzyl_acetate","벤질 아세테이트","Benzyl Acetate","synthetic","top",7,5,3,["자스민","프루티","달콤"],4,10,"CC(=O)OCc1ccccc1"),
    I("methyl_salicylate","메틸 살리실레이트","Methyl Salicylate","synthetic","top",7.5,6,2.5,["민티","메디컬","그린"],2,5,"COC(=O)c1ccccc1O"),
    I("benzaldehyde","벤즈알데히드","Benzaldehyde","synthetic","top",8,6,2,["아몬드","체리","비터"],3,7,"O=Cc1ccccc1"),
    I("acetophenone","아세토페논","Acetophenone","synthetic","middle",6,5,3.5,["체리","아몬드","달콤"],3,8,"CC(=O)c1ccccc1"),
    I("cinnamaldehyde","신남알데히드","Cinnamaldehyde","synthetic","middle",6,8,4,["시나몬","달콤","스파이시"],2,5,"O=C/C=C/c1ccccc1"),
    I("methyl_jasmonate","메틸 자스모네이트","Methyl Jasmonate","synthetic","middle",5,6,5,["자스민","그린","프루티"],3,8,None),
    I("hedione_hc","헤디온 HC","Hedione HC","synthetic","middle",5,5,5.5,["자스민","시트러스","라디언트"],6,15,None),
    I("norlimbanol","노르림바놀","Norlimbanol","synthetic","base",2,5,8,["우디","앰버","따뜻한"],5,12,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 동물적/동양/특수 - BASE
# ================================================================
for item in [
    I("oud_synth","우드 합성","Oud Synthetic","animalic","base",2,8,9,["우디","스모키","레더"],3,8,None),
    I("civet_synth","시벳 합성","Civet Synthetic","animalic","base",2,7,9,["동물적","머스크","따뜻한"],1,3,None),
    I("castoreum_synth","캐스토리움 합성","Castoreum Synthetic","animalic","base",2,7,9,["동물적","가죽","버치"],2,5,None),
    I("hyrax","하이락스","Hyrax","animalic","base",2,7,9,["동물적","앰버","따뜻한"],1,3,None),
    I("beeswax","밀랍","Beeswax","animalic","base",3,4,6,["왁시","허니","따뜻한"],4,10,None),
    I("santal_mysore","마이소르 샌달","Mysore Sandalwood","woody","base",2,7,9,["우디","크리미","밀키"],8,20,None),
    I("oudh_laos","라오스 우드","Laos Oud","woody","base",1.5,9,10,["우디","스위트","허니"],3,8,None),
    I("siam_benzoin","시암 벤조인","Siam Benzoin","balsamic","base",2.5,6,7,["발사믹","바닐라","달콤"],4,10,None),
    I("tincture_ambergris","앰버그리스 팅크처","Ambergris Tincture","amber","base",1,6,10,["앰버","바다","따뜻한"],3,8,None),
    I("musk_deer","사향","Musk Deer","musk","base",1,7,10,["머스크","동물적","파우더리"],2,5,None),
]:
    if item: new_ingredients.append(item)

# ================================================================
# 기타 천연 추출물 - MIDDLE/BASE
# ================================================================
for item in [
    I("spikenard","스파이크나드","Spikenard","woody","base",2.5,5,7,["우디","어시","발사믹"],3,8,None),
    I("costus","코스터스","Costus","woody","base",2,7,8,["동물적","우디","비올렛"],2,5,None),
    I("angelica","안젤리카","Angelica","herbal","middle",5.5,6,4.5,["허벌","머스크","어시"],3,8,None),
    I("immortelle","이모르텔","Immortelle","herbal","middle",5,6,5,["카레","허벌","달콤"],3,8,None),
    I("orris_butter","오리스 버터","Orris Butter","floral","base",2.5,5,8,["파우더리","바이올렛","우디"],3,8,None),
    I("rose_otto","로즈 오토","Rose Otto","floral","middle",4.5,8,5.5,["로즈","플로럴","시트러스"],5,12,None),
    I("neroli_absolute","네롤리 앱솔루트","Neroli Absolute","floral","middle",5,7,4.5,["플로럴","시트러스","앰버"],4,10,None),
    I("ylang_extra","일랑 엑스트라","Ylang Extra","floral","middle",4.5,8,5.5,["플로럴","이국적","크리미"],4,10,None),
    I("anis_stellatum","팔각","Star Anise Oil","spicy","middle",6,6,4,["아니스","달콤","리코리스"],2,6,None),
    I("black_tea","홍차","Black Tea","aromatic","middle",5,5,4,["아로마틱","스모키","따뜻한"],5,12,None),
    I("oolong_tea","우롱차","Oolong Tea","aromatic","middle",5.5,4,3.5,["아로마틱","플로럴","그린"],5,12,None),
    I("white_tea","백차","White Tea","aromatic","middle",6,3,3,["아로마틱","클린","그린"],5,12,None),
    I("rooibos","루이보스","Rooibos","aromatic","middle",5,4,4,["아로마틱","달콤","따뜻한"],4,10,None),
    I("mate","마테","Mate","aromatic","middle",5.5,5,3.5,["그린","스모키","허벌"],4,10,None),
    I("sake","사케","Sake","gourmand","middle",5,4,3.5,["달콤","라이스","클린"],3,8,None),
    I("rice","쌀","Rice","gourmand","middle",5,3,4,["클린","달콤","파우더리"],4,10,None),
    I("ginseng","인삼","Ginseng","herbal","base",3,5,6,["허벌","어시","비터"],3,8,None),
    I("turmeric","강황","Turmeric","spicy","middle",5.5,6,4,["스파이시","어시","따뜻한"],2,6,None),
    I("hemp","대마","Hemp","green","middle",5.5,4,4,["그린","어시","우디"],3,8,None),
    I("cannabis","칸나비스","Cannabis","green","middle",5.5,5,4,["그린","허벌","어시"],3,8,None),
]:
    if item: new_ingredients.append(item)

# 결과 합치기
all_ingredients = existing + new_ingredients
print(f"새로 추가: {len(new_ingredients)}개")
print(f"전체 합계: {len(all_ingredients)}개")

# ID 중복 확인
ids = [ing['id'] for ing in all_ingredients]
dupes = [x for x in ids if ids.count(x) > 1]
if dupes:
    print(f"⚠ 중복 ID 발견: {set(dupes)}")
else:
    print("✅ 중복 없음")

# 저장
with open(ING_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_ingredients, f, ensure_ascii=False, indent=2)
print(f"✅ {ING_PATH} 저장 완료")

# SMILES 저장
with open(SMILES_PATH, 'w', encoding='utf-8') as f:
    json.dump(smiles_map, f, ensure_ascii=False, indent=2)
print(f"✅ {SMILES_PATH} 저장 완료 ({len(smiles_map)}개 SMILES)")

# 카테고리별 통계
cat_count = {}
note_count = {'top':0,'middle':0,'base':0}
for ing in all_ingredients:
    c = ing.get('category','other')
    cat_count[c] = cat_count.get(c, 0) + 1
    nt = ing.get('note_type','middle')
    if nt in note_count: note_count[nt] += 1

print(f"\n📊 카테고리별 분포:")
for cat, cnt in sorted(cat_count.items(), key=lambda x:-x[1]):
    print(f"  {cat:>15}: {cnt}개")
print(f"\n📊 노트별 분포:")
for nt, cnt in note_count.items():
    print(f"  {nt:>10}: {cnt}개")
