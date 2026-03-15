"""
2차 확장 — 아로마케미컬 (합성 향료) + 천연 추출물 세분화
SMILES 정확도 최우선
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
ING_PATH = os.path.join(BASE, '..', '..', 'data', 'ingredients.json')
SMILES_PATH = os.path.join(BASE, '..', '..', 'data', 'ingredient_smiles.json')

with open(ING_PATH, 'r', encoding='utf-8') as f:
    existing = json.load(f)
existing_ids = {ing['id'] for ing in existing}

with open(SMILES_PATH, 'r', encoding='utf-8') as f:
    smiles_map = json.load(f)

count = 0
def I(id,ko,en,cat,note,vol,it,lon,desc,tp,mp,smi=None):
    global count
    if id in existing_ids: return None
    existing_ids.add(id)
    if smi: smiles_map[id] = smi
    count += 1
    return {"id":id,"name_ko":ko,"name_en":en,"category":cat,"note_type":note,
            "volatility":vol,"intensity":it,"longevity":lon,
            "descriptors":desc,"typical_pct":tp,"max_pct":mp}

new = []

# ============ 아로마케미컬: 알데히드 계열 ============
for x in [
    I("aldehyde_c6","알데히드 C6","Aldehyde C-6","synthetic","top",9,6,1.5,["그린","플로럴","리프"],2,5,"CCCCCC=O"),
    I("aldehyde_c7","알데히드 C7","Aldehyde C-7","synthetic","top",9,6,1.5,["그린","시트러스","오일리"],2,5,"CCCCCCC=O"),
    I("aldehyde_c8","알데히드 C8","Aldehyde C-8","synthetic","top",8.5,7,2,["시트러스","왁시","오렌지"],2,5,"CCCCCCCC=O"),
    I("aldehyde_c9","알데히드 C9","Aldehyde C-9","synthetic","top",8.5,7,2,["시트러스","왁시","로즈"],2,5,"CCCCCCCCC=O"),
    I("aldehyde_c10","알데히드 C10","Aldehyde C-10 (Decanal)","synthetic","top",8,7,2.5,["시트러스","왁시","오렌지필"],2,5,"CCCCCCCCCC=O"),
    I("aldehyde_c11","알데히드 C11","Aldehyde C-11 (Undecanal)","synthetic","top",7.5,7,3,["왁시","클린","메탈릭"],2,4,"CCCCCCCCCCC=O"),
    I("aldehyde_c12","알데히드 C12","Aldehyde C-12 (Dodecanal)","synthetic","top",7,6,3.5,["왁시","시트러스","앰버"],2,4,"CCCCCCCCCCCC=O"),
    I("aldehyde_c12_mna","알데히드 C12 MNA","Aldehyde C-12 MNA","synthetic","top",7,6,3.5,["메탈릭","왁시","앰버"],2,4,None),
]: 
    if x: new.append(x)

# ============ 아로마케미컬: 테르펜 계열 ============
for x in [
    I("alpha_pinene","알파 피넨","Alpha-Pinene","synthetic","top",9,6,1.5,["파인","그린","신선"],3,8,"CC1=CCC2CC1C2(C)C"),
    I("beta_pinene","베타 피넨","Beta-Pinene","synthetic","top",9,5,1.5,["파인","우디","그린"],3,8,"CC1(C)C2CCC(=C)C1C2"),
    I("myrcene","미르센","Myrcene","synthetic","top",9,5,1.5,["발사믹","허벌","스파이시"],3,8,"CC(=CCCC(=C)C=C)C"),
    I("ocimene","오시멘","Ocimene","synthetic","top",9,5,1.5,["허벌","플로럴","시트러스"],3,8,"CC(=CC=CC(=C)C)C"),
    I("terpinolene","테르피놀렌","Terpinolene","synthetic","top",8.5,5,2,["시트러스","파인","허벌"],3,8,"CC1=CCC(=C(C)C)CC1"),
    I("terpineol","테르피네올","Terpineol","synthetic","middle",6,5,4,["라일락","플로럴","클린"],4,10,"CC1=CCC(CC1)C(C)(C)O"),
    I("menthol","멘톨","Menthol","synthetic","top",9,8,2,["쿨링","민트","클린"],2,5,"CC(C)C1CCC(C)CC1O"),
    I("menthone","멘톤","Menthone","synthetic","top",8.5,7,2,["민트","쿨링","그린"],2,5,"CC(C)C1CCC(C)CC1=O"),
    I("carvone","카르본","Carvone","synthetic","top",8,6,2.5,["민트","캐러웨이","달콤"],3,7,"CC(=C)C1CC=C(C)C(=O)C1"),
    I("thymol","티몰","Thymol","synthetic","middle",6,7,4,["허벌","메디컬","따뜻한"],2,5,"Cc1ccc(O)c(C(C)C)c1"),
    I("borneol","보르네올","Borneol","synthetic","middle",5.5,5,4,["캠포러스","우디","허벌"],3,7,"CC1(C)C2CCC1(C)C(O)C2"),
    I("fenchone","펜콘","Fenchone","synthetic","top",8,6,2,["캠포러스","우디","그린"],2,5,None),
    I("nerolidol","네롤리돌","Nerolidol","synthetic","base",3,4,7,["플로럴","우디","그린"],4,10,"CC(=CCC/C(C)=C/CC(O)(C=C)C)C"),
    I("farnesol","파르네솔","Farnesol","synthetic","base",3,4,6,["플로럴","릴리","그린"],4,10,"CC(=CCC/C(C)=C/CC/C(C)=C/CO)C"),
    I("cedrol","세드롤","Cedrol","synthetic","base",2.5,5,7,["우디","시더","드라이"],5,12,None),
    I("santalol","산탈롤","Santalol","synthetic","base",2,6,8,["우디","크리미","밀키"],6,15,None),
    I("guaiazulene","과이아줄렌","Guaiazulene","synthetic","base",2.5,5,7,["우디","스모키","블루"],3,8,"CC1=CC2=C(CC1)C(C)=CC2(C)C"),
    I("patchoulol","패출롤","Patchoulol","synthetic","base",2,7,9,["우디","어시","캠포러스"],5,12,None),
    I("vetiverol","베티베롤","Vetiverol","synthetic","base",2,6,8,["우디","어시","스모키"],4,10,None),
    I("bisabolol","비사볼롤","Bisabolol","synthetic","middle",5,4,5,["플로럴","카모마일","클린"],4,10,"CC(=CCC(C)(O)C1CC=C(C)CC1)C"),
]: 
    if x: new.append(x)

# ============ 아로마케미컬: 에스테르/락톤 계열 ============
for x in [
    I("gamma_decalactone","감마 데카락톤","Gamma-Decalactone","synthetic","middle",5,6,5,["피치","크리미","코코넛"],3,8,"CCCCCCC1CCC(=O)O1"),
    I("gamma_undecalactone","감마 운데카락톤","Gamma-Undecalactone","synthetic","base",3,5,6,["피치","크리미","파우더리"],3,8,"CCCCCCCC1CCC(=O)O1"),
    I("gamma_nonalactone","감마 노나락톤","Gamma-Nonalactone","synthetic","middle",5,5,4.5,["코코넛","크리미","달콤"],3,8,"CCCCCC1CCC(=O)O1"),
    I("delta_decalactone","델타 데카락톤","Delta-Decalactone","synthetic","base",3,5,6,["크리미","밀키","버터"],3,8,"CCCCCC1CCCC(=O)O1"),
    I("coumarin_lactone","쿠마린 락톤","Coumarin Lactone","synthetic","base",3,5,7,["쿠마린","건초","달콤"],4,10,"O=c1ccc2ccccc2o1"),
    I("jasmine_lactone","자스민 락톤","Jasmine Lactone","synthetic","middle",5.5,5,4.5,["자스민","프루티","피치"],3,8,None),
    I("whiskey_lactone","위스키 락톤","Whiskey Lactone","synthetic","base",3,5,6,["우디","코코넛","바닐라"],3,8,None),
    I("massoia_lactone","마소이아 락톤","Massoia Lactone","synthetic","base",3,6,6,["코코넛","크리미","달콤"],3,8,None),
    I("ambrettolide","앰브레톨라이드","Ambrettolide","synthetic","base",1.5,5,9,["머스크","프루티","플로럴"],4,10,None),
    I("exaltolide","엑살토라이드","Exaltolide","synthetic","base",1.5,4,9,["머스크","파우더리","클린"],4,10,None),
    I("linalyl_acetate","리날릴 아세테이트","Linalyl Acetate","synthetic","top",8,5,2.5,["라벤더","시트러스","플로럴"],5,12,"CC(=CCC(OC(C)=O)(C)C=C)C"),
    I("geranyl_acetate","게라닐 아세테이트","Geranyl Acetate","synthetic","top",7.5,5,3,["로즈","플로럴","시트러스"],4,10,"CC(=CCOC(C)=O)/C=C/C=C(C)C"),
    I("neryl_acetate","네릴 아세테이트","Neryl Acetate","synthetic","top",7.5,5,3,["로즈","시트러스","허벌"],4,10,None),
    I("methyl_benzoate","메틸 벤조에이트","Methyl Benzoate","synthetic","top",8,5,2,["플로럴","허벌","일랑"],3,7,"COC(=O)c1ccccc1"),
    I("benzyl_benzoate","벤질 벤조에이트","Benzyl Benzoate","synthetic","base",2,4,7,["발사믹","달콤","아몬드"],5,12,"O=C(OCc1ccccc1)c1ccccc1"),
    I("ethyl_cinnamate","에틸 시나메이트","Ethyl Cinnamate","synthetic","middle",5.5,6,4.5,["발사믹","시나몬","프루티"],3,8,"CCOC(=O)/C=C/c1ccccc1"),
    I("terpinyl_acetate","테르피닐 아세테이트","Terpinyl Acetate","synthetic","top",7.5,5,2.5,["허벌","시트러스","라벤더"],4,10,None),
    I("methyl_anthranilate","메틸 안트라닐레이트","Methyl Anthranilate","synthetic","middle",5.5,7,5,["포도","플로럴","인돌릭"],2,5,"NC1=CC=CC=C1C(=O)OC"),
    I("dimethyl_anthranilate","디메틸 안트라닐레이트","Dimethyl Anthranilate","synthetic","middle",5.5,6,4.5,["포도","플로럴","만다린"],2,5,None),
]: 
    if x: new.append(x)

# ============ 아로마케미컬: 머스크/앰버/우디 특화 ============
for x in [
    I("tonalide","토나라이드","Tonalide","synthetic","base",1.5,5,9,["머스크","클린","달콤"],5,12,None),
    I("celestolide","셀레스톨라이드","Celestolide","synthetic","base",1.5,4,9,["머스크","클린","왁시"],5,12,None),
    I("phantolide","판톨라이드","Phantolide","synthetic","base",1.5,4,9,["머스크","클린","우디"],5,12,None),
    I("muscenone","무스세논","Muscenone","synthetic","base",1,5,10,["머스크","파우더리","클린"],4,10,None),
    I("musk_r1","머스크 R-1","Musk R-1","synthetic","base",1,5,10,["머스크","클린","파우더리"],4,10,None),
    I("nirvanolide","니르바놀라이드","Nirvanolide","synthetic","base",1.5,4,9,["머스크","클린","플로럴"],4,10,None),
    I("silvanone","실바논","Silvanone","synthetic","base",1.5,5,9,["머스크","우디","앰버"],4,10,None),
    I("cosmone","코스몬","Cosmone","synthetic","base",1,5,10,["머스크","동물적","크리미"],3,8,None),
    I("velvione","벨비온","Velvione","synthetic","base",1.5,4,9,["머스크","파우더리","과일"],4,10,None),
    I("romandolide","로만돌라이드","Romandolide","synthetic","base",1.5,5,9,["머스크","프루티","클린"],4,10,None),
    I("timberol","팀베롤","Timberol","synthetic","base",2.5,5,7,["우디","앰버","드라이"],5,12,None),
    I("javanol","자바놀","Javanol","synthetic","base",2,6,8,["우디","크리미","샌달"],6,15,None),
    I("bacdanol","박다놀","Bacdanol","synthetic","base",2,6,8,["우디","크리미","파우더리"],5,12,None),
    I("polysantol","폴리산톨","Polysantol","synthetic","base",2,5,8,["우디","크리미","밀키"],5,12,None),
    I("firsantol","퍼산톨","Firsantol","synthetic","base",2,5,8,["우디","크리미","카드보드"],5,12,None),
    I("ebanol_synthetic","에바놀","Ebanol","synthetic","base",2,6,8,["우디","크리미","샌달우드"],5,12,None),
    I("vertofix","버토픽스","Vertofix","synthetic","base",2.5,5,8,["우디","앰버","벨벳"],5,12,None),
    I("georgywood","조지우드","Georgywood","synthetic","base",2.5,5,7,["우디","시더","드라이"],5,12,None),
    I("clearwood","클리어우드","Clearwood","synthetic","base",2,6,8,["우디","패출리","클린"],5,12,None),
    I("akigalawood","아키갈라우드","Akigalawood","synthetic","base",2,6,8,["우디","패출리","앰버"],5,12,None),
    I("karanal","카라날","Karanal","synthetic","base",2.5,4,7,["앰버","우디","머스크"],5,12,None),
    I("ambrinol","앰브리놀","Ambrinol","synthetic","base",2,5,8,["앰버","우디","클린"],4,10,None),
    I("ambrox_dl","앰브록스 DL","Ambrox DL","synthetic","base",1.5,5,10,["앰버","우디","드라이"],5,12,None),
    I("amber_xtreme","앰버 엑스트림","Amber Xtreme","synthetic","base",1.5,6,10,["앰버","우디","강렬"],4,10,None),
    I("sclareolide","스클라레올라이드","Sclareolide","synthetic","base",2,5,8,["앰버","달콤","발사믹"],4,10,None),
    I("norlimbanol_2","노르림바놀 II","Norlimbanol II","synthetic","base",2,5,8,["우디","앰버","테리"],5,12,None),
]: 
    if x: new.append(x)

# ============ 아로마케미컬: 플로럴 특화 ============
for x in [
    I("phenoxanol","페녹사놀","Phenoxanol","synthetic","middle",5.5,5,4.5,["로즈","플로럴","그린"],4,10,None),
    I("rosoxide","로즈옥사이드","Rose Oxide","synthetic","middle",6,5,3.5,["로즈","그린","메탈릭"],3,8,None),
    I("damascone_alpha","알파 다마스콘","Alpha-Damascone","synthetic","middle",5,7,5,["로즈","프루티","따뜻한"],2,5,None),
    I("damascone_beta","베타 다마스콘","Beta-Damascone","synthetic","middle",5,7,5,["로즈","프루티","플럼"],2,5,None),
    I("damascenone","다마세논","Damascenone","synthetic","middle",4.5,8,5.5,["로즈","프루티","잼"],1,3,None),
    I("dihydro_jasmone","디히드로자스몬","Dihydrojasmone","synthetic","middle",5.5,5,4,["자스민","셀러리","그린"],3,8,"CCCCC1CCC(=O)C1=C"),
    I("jasmolactone","자스몰락톤","Jasmolactone","synthetic","middle",5,5,5,["자스민","크리미","락토닉"],3,8,None),
    I("heptine_carbonate","헵틴 카보네이트","Heptine Carbonate","synthetic","middle",5.5,6,4,["바이올렛","그린","리프"],3,8,None),
    I("lilial","릴리알","Lilial","synthetic","middle",5.5,5,4,["릴리","시클라멘","플로럴"],5,12,"CC(C)c1ccc(CC=O)cc1"),
    I("florhydral","플로르히드랄","Florhydral","synthetic","middle",6,5,3.5,["릴리","아쿠아","그린"],5,12,None),
    I("muguet_aldehyde","뮤게 알데히드","Muguet Aldehyde","synthetic","middle",6,5,3.5,["뮤게","릴리","그린"],5,12,None),
    I("cyclamen_aldehyde","시클라멘 알데히드","Cyclamen Aldehyde","synthetic","middle",5.5,5,4,["시클라멘","플로럴","그린"],4,10,None),
    I("helional","헬리오날","Helional","synthetic","middle",5.5,5,4.5,["아쿠아","그린","헤이"],4,10,"COCCC1CCCO1"),
    I("floralozone","플로랄로존","Floralozone","synthetic","middle",6,5,3.5,["오존","아쿠아","플로럴"],4,10,None),
    I("methyl_ionone","메틸 이오논","Methyl Ionone","synthetic","middle",5,5,5,["바이올렛","우디","오리스"],4,10,None),
    I("iso_e_gamma","이소이감마","Iso E Gamma","synthetic","base",2,4,8,["우디","앰버","벨벳"],6,15,None),
]: 
    if x: new.append(x)

# ============ 아로마케미컬: 프루티/구르망 ============
for x in [
    I("ethyl_butyrate","에틸 부티레이트","Ethyl Butyrate","synthetic","top",9,6,1.5,["파인애플","프루티","달콤"],3,7,"CCCC(=O)OCC"),
    I("ethyl_acetate","에틸 아세테이트","Ethyl Acetate","synthetic","top",9.5,5,1,["프루티","솔벤트","달콤"],3,7,"CC(=O)OCC"),
    I("isoamyl_acetate","이소아밀 아세테이트","Isoamyl Acetate","synthetic","top",9,6,1.5,["바나나","프루티","달콤"],3,7,"CC(C)CCOC(C)=O"),
    I("hexyl_acetate","헥실 아세테이트","Hexyl Acetate","synthetic","top",8,5,2,["그린","프루티","배"],3,8,"CCCCCCOC(C)=O"),
    I("allyl_caproate","알릴 카프로에이트","Allyl Caproate","synthetic","top",8.5,5,2,["파인애플","프루티","왁시"],3,7,None),
    I("fructone","프룩톤","Fructone","synthetic","top",8,5,2,["애플","프루티","그린"],3,8,None),
    I("nectaryl","넥타릴","Nectaryl","synthetic","middle",5.5,5,4,["프루티","와일드베리","크리미"],3,8,None),
    I("frambinone","프람비논","Frambinone","synthetic","middle",5.5,6,4,["라즈베리","프루티","달콤"],3,8,"CC(=O)c1ccc(O)cc1"),
    I("furaneol","퓨라네올","Furaneol","synthetic","middle",5,6,4,["카라멜","딸기","달콤"],3,8,"CC1=C(O)C(=O)CO1"),
    I("methyl_cinnamate","메틸 시나메이트","Methyl Cinnamate","synthetic","middle",5.5,6,4,["딸기","발사믹","달콤"],3,8,"COC(=O)/C=C/c1ccccc1"),
    I("cis3_hexenol","시스-3-헥세놀","cis-3-Hexenol","synthetic","top",8.5,6,2,["그린","풀잎","신선"],3,7,"CCC=CCCO"),
    I("cis3_hexenyl_acetate","시스-3-헥세닐 아세테이트","cis-3-Hexenyl Acetate","synthetic","top",8,5,2.5,["그린","프루티","바나나"],3,7,"CCC=CCCOC(C)=O"),
    I("trans2_hexenal","트랜스-2-헥세날","trans-2-Hexenal","synthetic","top",9,6,1.5,["그린","리프","애플"],2,5,"CCC/C=C/C=O"),
]: 
    if x: new.append(x)

# ============ 천연: 에센셜 오일 세분화 ============
for x in [
    I("rose_damascena","다마스크 로즈","Rose Damascena","floral","middle",4.5,8,5.5,["로즈","따뜻한","허니"],5,12,None),
    I("rose_centifolia","센티폴리아 로즈","Rose Centifolia","floral","middle",4.5,8,5.5,["로즈","왁시","그린"],5,12,None),
    I("rose_de_mai","로즈 드 메","Rose de Mai","floral","middle",4.5,9,6,["로즈","리치","달콤"],5,12,None),
    I("jasmine_sambac","삼박 자스민","Jasmine Sambac","floral","middle",4.5,9,6,["자스민","인돌릭","따뜻한"],4,10,None),
    I("jasmine_grandiflorum","그란디플로럼 자스민","Jasmine Grandiflorum","floral","middle",4.5,8,5.5,["자스민","그린","달콤"],5,12,None),
    I("lavender_maillette","라벤더 마이에트","Lavender Maillette","aromatic","top",7,6,3,["라벤더","허벌","달콤"],8,20,None),
    I("lavender_spike","스파이크 라벤더","Spike Lavender","aromatic","top",7.5,7,2.5,["라벤더","캠포러스","허벌"],5,12,None),
    I("lavandin","라반딘","Lavandin","aromatic","top",7.5,6,2.5,["라벤더","캠포러스","시트러스"],6,15,None),
    I("sandalwood_australian","호주 샌달우드","Australian Sandalwood","woody","base",2,5,7.5,["우디","허니","드라이"],6,15,None),
    I("sandalwood_new_caledonia","뉴칼레도니아 샌달우드","New Caledonia Sandalwood","woody","base",2,6,8,["우디","크리미","밀키"],6,15,None),
    I("cedarwood_texas","텍사스 시더우드","Texas Cedarwood","woody","base",3,5,7,["우디","드라이","스모키"],5,12,None),
    I("cedarwood_himalaya","히말라야 시더우드","Himalayan Cedarwood","woody","base",3,5,7,["우디","드라이","앰버"],5,12,None),
    I("vetiver_bourbon","버번 베티버","Bourbon Vetiver","woody","base",2,7,8,["우디","어시","스모키"],5,12,None),
    I("vetiver_haiti","아이티 베티버","Haitian Vetiver","woody","base",2,6,8,["우디","그린","클린"],5,12,None),
    I("patchouli_dark","다크 패출리","Dark Patchouli","woody","base",2,8,9,["우디","어시","초콜릿"],4,10,None),
    I("patchouli_light","라이트 패출리","Light Patchouli","woody","base",2.5,6,8,["우디","그린","클린"],5,12,None),
    I("oud_cambodian","캄보디안 우드","Cambodian Oud","woody","base",1.5,10,10,["우디","동물적","스위트"],2,5,None),
    I("oud_indian","인디안 우드","Indian Oud","woody","base",1.5,9,10,["우디","스모키","가죽"],2,5,None),
    I("oud_indonesian","인도네시안 우드","Indonesian Oud","woody","base",1.5,8,10,["우디","허벌","그린"],3,7,None),
    I("bergamot_calabrian","칼라브리안 베르가못","Calabrian Bergamot","citrus","top",9,7,2,["시트러스","달콤","그린"],5,12,None),
    I("lemon_sicilian","시칠리안 레몬","Sicilian Lemon","citrus","top",9.5,8,1.5,["시트러스","상큼","비터"],5,12,None),
    I("orange_sweet_brazil","브라질 오렌지","Brazilian Sweet Orange","citrus","top",9,6,2,["시트러스","달콤","과일"],5,12,None),
    I("lime_persian","페르시안 라임","Persian Lime","citrus","top",9,7,1.5,["시트러스","비터","그린"],4,10,None),
    I("geranium_bourbon","버번 제라늄","Bourbon Geranium","floral","middle",6,6,4,["플로럴","로즈","그린"],5,15,None),
    I("geranium_egyptian","이집트 제라늄","Egyptian Geranium","floral","middle",6,6,4,["플로럴","민트","로즈"],5,12,None),
    I("ylang_ylang_complete","일랑일랑 컴플리트","Ylang Ylang Complete","floral","middle",5,8,5,["플로럴","이국적","바나나"],5,15,None),
    I("neroli_tunisia","튀니지 네롤리","Tunisian Neroli","floral","middle",6,7,4,["플로럴","시트러스","허니"],5,12,None),
]: 
    if x: new.append(x)

# ============ 특수/희귀/한국 전통 ============
for x in [
    I("pine_needle_korean","한국 소나무","Korean Pine Needle","woody","top",7.5,5,3,["우디","그린","레진"],4,10,None),
    I("mugwort_korean","한국 쑥","Korean Mugwort","herbal","middle",6.5,5,4,["허벌","비터","그린"],3,7,None),
    I("yuzu_korean","한국 유자","Korean Yuzu","citrus","top",9,6,2,["시트러스","허벌","그린"],4,10,None),
    I("omija","오미자","Omija (Schisandra)","fruity","middle",5.5,5,4,["프루티","베리","허벌"],3,8,None),
    I("lotus_korean","한국 연꽃","Korean Lotus","floral","middle",5.5,4,4,["플로럴","아쿠아","클린"],4,10,None),
    I("chrysanthemum_korean","한국 국화","Korean Chrysanthemum","floral","middle",5.5,5,4,["플로럴","그린","허벌"],4,10,None),
    I("citron_korean","한국 유자 과피","Korean Citron Peel","citrus","top",9,6,2,["시트러스","허벌","비터"],4,10,None),
    I("plum_blossom_korean","한국 매화","Korean Plum Blossom","floral","top",6.5,4,3,["플로럴","달콤","그린"],4,10,None),
    I("bamboo_korean","한국 대나무","Korean Bamboo","green","middle",5.5,4,4,["그린","클린","아쿠아"],4,10,None),
    I("hinoki_japanese","일본 히노키","Japanese Hinoki","woody","base",3.5,5,7,["우디","시트러스","클린"],5,12,None),
    I("hokkaido_lavender","홋카이도 라벤더","Hokkaido Lavender","aromatic","top",7,6,3,["라벤더","달콤","허벌"],6,15,None),
    I("indian_jasmine","인도 자스민","Indian Jasmine","floral","middle",4.5,9,6,["자스민","인돌릭","리치"],5,12,None),
    I("taif_rose","타이프 로즈","Taif Rose","floral","middle",4.5,9,6,["로즈","스파이시","허니"],4,10,None),
    I("turkish_rose","터키 로즈","Turkish Rose","floral","middle",4.5,8,5.5,["로즈","달콤","잼"],5,12,None),
    I("grasse_jasmine","그라스 자스민","Grasse Jasmine","floral","middle",4.5,9,6,["자스민","프루티","앱솔루트"],4,10,None),
    I("grasse_rose","그라스 로즈","Grasse Rose","floral","middle",4.5,9,6,["로즈","허니","왁시"],5,12,None),
    I("persian_rose","페르시안 로즈","Persian Rose","floral","middle",4.5,8,5.5,["로즈","달콤","파우더리"],5,12,None),
    I("somalian_frankincense","소말리아 프랑킨센스","Somalian Frankincense","balsamic","base",3,6,7,["인센스","시트러스","레진"],4,10,None),
    I("ethiopian_myrrh","에티오피아 미르","Ethiopian Myrrh","balsamic","base",2.5,6,8,["발사믹","스모키","따뜻한"],4,10,None),
    I("madagascar_vanilla","마다가스카르 바닐라","Madagascar Vanilla","gourmand","base",2,7,8,["바닐라","달콤","크리미"],6,15,None),
    I("tahitian_vanilla","타히티안 바닐라","Tahitian Vanilla","gourmand","base",2,6,8,["바닐라","프루티","플로럴"],5,12,None),
    I("bourbon_vanilla","버번 바닐라","Bourbon Vanilla","gourmand","base",2,7,8,["바닐라","달콤","초콜릿"],6,15,None),
    I("sri_lankan_cinnamon","스리랑카 시나몬","Sri Lankan Cinnamon","spicy","middle",6,8,4,["시나몬","달콤","따뜻한"],2,5,None),
    I("indian_sandalwood","인도 샌달우드","Indian Sandalwood","woody","base",2,7,9,["우디","크리미","밀키"],8,20,None),
    I("australian_tea_tree","호주 티트리","Australian Tea Tree","herbal","top",8,6,2,["허벌","캠포러스","클린"],3,7,None),
    I("egyptian_jasmine","이집트 자스민","Egyptian Jasmine","floral","middle",4.5,8,5.5,["자스민","그린","달콤"],5,12,None),
    I("moroccan_rose","모로코 로즈","Moroccan Rose","floral","middle",5,7,5,["로즈","허벌","우디"],5,12,None),
    I("haitian_vetiver","아이티 베티버 앱솔루트","Haitian Vetiver Absolute","woody","base",2,7,9,["우디","스모키","어시"],4,10,None),
    I("burmese_oud","버마 우드","Burmese Oud","woody","base",1.5,9,10,["우디","허니","레더"],3,8,None),
    I("thai_oud","태국 우드","Thai Oud","woody","base",1.5,8,10,["우디","스위트","허벌"],3,8,None),
]: 
    if x: new.append(x)

# ============ 더 많은 합성 + 천연 ============
for x in [
    I("safranal","사프라날","Safranal","synthetic","top",7,7,3,["사프란","허벌","가죽"],2,5,"CC1=C(C=O)C(C)=CC1"),
    I("anethole","아네톨","Anethole","synthetic","middle",6,6,4,["아니스","달콤","허벌"],3,8,"COc1ccc(/C=C/C)cc1"),
    I("estragole","에스트라골","Estragole","synthetic","top",7.5,6,2.5,["아니스","허벌","타라곤"],2,5,"COc1ccc(CC=C)cc1"),
    I("cinnamyl_alcohol","시나밀 알코올","Cinnamyl Alcohol","synthetic","middle",5.5,6,4.5,["발사믹","히아신스","플로럴"],3,8,"OC/C=C/c1ccccc1"),
    I("cinnamyl_acetate","시나밀 아세테이트","Cinnamyl Acetate","synthetic","middle",5.5,6,4,["시나몬","플로럴","발사믹"],3,8,None),
    I("cinnamic_acid","시남산","Cinnamic Acid","synthetic","base",3,5,6,["발사믹","따뜻한","파우더리"],3,8,"OC(=O)/C=C/c1ccccc1"),
    I("para_cresol","파라 크레솔","para-Cresol","synthetic","base",3,7,7,["동물적","가죽","나르시스"],1,3,"Cc1ccc(O)cc1"),
    I("phenylacetic_acid","페닐아세트산","Phenylacetic Acid","synthetic","base",3,6,6,["허니","초콜릿","동물적"],2,5,"OC(=O)Cc1ccccc1"),
    I("phenylacetaldehyde","페닐아세트알데히드","Phenylacetaldehyde","synthetic","top",8,7,2,["허니","로즈","그린"],2,5,"O=CCc1ccccc1"),
    I("benzyl_alcohol","벤질 알코올","Benzyl Alcohol","synthetic","middle",6,4,3.5,["플로럴","바디","마일드"],4,10,"OCc1ccccc1"),
    I("anisaldehyde","아니스알데히드","Anisaldehyde","synthetic","middle",5.5,6,4,["아니스","달콤","호손"],3,8,"COc1ccc(C=O)cc1"),
    I("vanillic_acid","바닐릭산","Vanillic Acid","synthetic","base",3,4,6,["바닐라","마일드","파우더리"],3,8,None),
    I("methyl_dihydrojasmonate","메틸 디히드로자스모네이트","Methyl Dihydrojasmonate","synthetic","middle",5,5,5,["자스민","셀러리","클린"],5,12,None),
    I("hexenyl_salicylate","헥세닐 살리실레이트","Hexenyl Salicylate","synthetic","middle",6,4,4,["그린","플로럴","아쿠아"],4,10,None),
    I("methyl_octine_carbonate","메틸 옥틴 카보네이트","Methyl Octine Carbonate","synthetic","middle",5.5,6,4,["바이올렛","그린","리프"],3,8,None),
    I("triplal","트리플랄","Triplal","synthetic","top",7.5,6,3,["그린","플로럴","알데히드"],3,8,None),
    I("adoxal","아독살","Adoxal","synthetic","middle",5.5,5,4.5,["마린","오존","멜론"],4,10,None),
    I("oxyphenylon","옥시페닐론","Oxyphenylon","synthetic","base",3,5,6,["앰버","발사믹","카시스"],3,8,None),
    I("syringa_aldehyde","시린가 알데히드","Syringaldehyde","synthetic","middle",5.5,5,4,["바닐라","스파이시","우디"],3,8,"COc1cc(C=O)cc(OC)c1O"),
    I("veratraldehyde","베라트르알데히드","Veratraldehyde","synthetic","middle",5.5,5,4,["바닐라","우디","플로럴"],3,8,"COc1ccc(C=O)cc1OC"),
    I("piperonal","피페로날","Piperonal","synthetic","middle",5,5,5,["헬리오트로프","바닐라","체리"],4,10,"O=Cc1ccc2OCOc2c1"),
    I("aubepine","오베핀","Aubepine","synthetic","middle",5,5,4.5,["호손","아몬드","아니스"],3,8,None),
]: 
    if x: new.append(x)

# 병합
all_ings = existing + new
print(f"2차 확장: +{count}개")
print(f"전체: {len(all_ings)}개")

ids = [i['id'] for i in all_ings]
dupes = set(x for x in ids if ids.count(x)>1)
if dupes:
    print(f"⚠ 중복: {dupes}")
    # 중복 제거
    seen = set()
    deduped = []
    for i in all_ings:
        if i['id'] not in seen:
            seen.add(i['id'])
            deduped.append(i)
    all_ings = deduped
    print(f"중복 제거 후: {len(all_ings)}개")
else:
    print("✅ 중복 없음")

with open(ING_PATH,'w',encoding='utf-8') as f:
    json.dump(all_ings,f,ensure_ascii=False,indent=2)

with open(SMILES_PATH,'w',encoding='utf-8') as f:
    json.dump(smiles_map,f,ensure_ascii=False,indent=2)

# 통계
cats={}; notes={'top':0,'middle':0,'base':0}
for i in all_ings:
    c=i.get('category','other'); cats[c]=cats.get(c,0)+1
    n=i.get('note_type','middle')
    if n in notes: notes[n]+=1

print(f"\n📊 카테고리:")
for c,n in sorted(cats.items(),key=lambda x:-x[1]):
    print(f"  {c:>15}: {n}")
print(f"\n📊 노트: top={notes['top']} mid={notes['middle']} base={notes['base']}")
print(f"✅ SMILES: {len(smiles_map)}개")
