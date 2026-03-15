# -*- coding: utf-8 -*-
"""
🔬 Pro-Grade 원료 DB 업그레이드
=================================
1. 스키마 업그레이드 (CAS, IFRA, allergens, odor_descriptors, price_per_kg, volatility, odor_strength)
2. 중복 삭제
3. 핵심 원료 프로데이터 보강
"""
import sys,os,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
sys.path.insert(0,'.')
import database

def upgrade_schema():
    """스키마 업그레이드"""
    conn = database.get_conn()
    cur = conn.cursor()
    cols = [
        ("cas_number", "VARCHAR(30)"),
        ("ifra_limit", "DECIMAL"),         # IFRA 최대 사용 % (피부)
        ("allergens", "TEXT"),              # 쉼표구분 알레르겐
        ("odor_descriptors", "TEXT"),       # 다중 향 디스크립터 (쉼표구분)
        ("odor_strength", "INTEGER"),       # 향 강도 1-10
        ("price_usd_kg", "DECIMAL"),        # USD/kg
        ("volatility", "VARCHAR(20)"),      # top/middle/base/fixative
    ]
    for col,dtype in cols:
        try:
            cur.execute(f"ALTER TABLE ingredients ADD COLUMN {col} {dtype}")
            print(f"  ✅ {col} 추가")
        except Exception as e:
            if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
                print(f"  ⏭ {col} — 이미 존재")
                conn.rollback()
            else:
                print(f"  ❌ {col}: {e}")
                conn.rollback()

def remove_duplicates():
    """중복 원료 삭제 (country-specific 버전을 살리고 generic 제거)"""
    conn = database.get_conn()
    cur = conn.cursor()
    
    # 정확한 중복 (같은 name_en)
    dupes_to_remove = [
        # generic → country-specific 으로 통합
        "jasmine_absolute",     # jasmine_abs 유지 (이집트산)
        "rose_absolute",        # rose_abs 유지 (불가리아산)
        "br_tonka_abs",         # tonka_abs 유지
        "cn_osmanthus_abs",     # osmanthus_abs 유지
        "cn_lapsang_souchong",  # lapsang 유지
        "fr_hay_abs",           # hay_absolute 유지
        # 같은 의미 중복
        "beeswax_absolute",     # beeswax 유지
        "rose_otto",            # rose_abs에 통합
        "lavender_bulgarian",   # bg_lavender_bg에 통합
        "clove_indonesian",     # id_clove_maluku에 통합
        "vetiver_java",         # id_vetiver_java에 통합
        "jasmine_egyptian",     # eg_jasmine_nile에 통합
        "rose_moroccan",        # ma_rose_dades에 통합
        "rose_turkish",         # tr_rose_isparta에 통합
        "neroli_moroccan",      # ma_neroli_fez에 통합
        "sandalwood_australian",# au_sandalwood_west에 통합
        "vetiver_haiti",        # ht_vetiver_haiti에 통합
        "geranium_bourbon",     # fr_geranium_bourbon에 통합
        "cardamom_guatemala",   # 단일유지, 스킵
        "tuberose_india",       # in_tuberose_mysore에 통합
        "clove_bud",            # id_clove_bud_oil에 통합
        "yuzu_japanese",        # jp_yuzu_kochi에 통합
        "orange_spanish",       # es_orange_seville에 통합
        "cedarwood_virginia",   # us_cedar_virginia에 통합
        "blood_orange",         # it_blood_orange_oil에 통합
        "magnolia_chinese",     # cn_magnolia_denudata에 통합
        "chamomile_blue",       # ma_chamomile_blue에 통합
        "pepper_sichuan",       # cn_sichuan_pepper_oil에 통합
    ]
    
    removed = 0
    for did in dupes_to_remove:
        cur.execute("SELECT id FROM ingredients WHERE id=%s",(did,))
        if cur.fetchone():
            cur.execute("DELETE FROM ingredients WHERE id=%s",(did,))
            removed += 1
    
    print(f"  🗑 중복 삭제: {removed}개")
    return removed


# ═══════════════════════════════════════
# 프로 데이터: CAS, IFRA, 가격, 향 디스크립터
# ═══════════════════════════════════════
# 형식: (id, cas, ifra_limit%, allergens, odor_descriptors, strength, usd_kg, volatility)
PRO_DATA = [
    # ─── 핵심 천연 플로럴 ───
    ("rose_abs","8007-01-0",2.5,"citronellol,geraniol,linalool,eugenol","rose,honey,spicy,rich,deep",9,6500,"middle"),
    ("jasmine_abs","8022-96-6",None,"benzyl_acetate,benzyl_benzoate,linalool","jasmine,indolic,narcotic,sweet,animalic",10,8000,"middle"),
    ("fr_rose_centifolia","8007-01-0",2.5,"citronellol,geraniol,linalool","rose,honey,butter,sweet,waxy",9,7200,"middle"),
    ("fr_rose_de_mai","90106-38-0",2.5,"citronellol,geraniol","rose,butter,honey,green,waxy",10,12000,"middle"),
    ("bg_rose_kazanlak","8007-01-0",2.5,"citronellol,geraniol,linalool","rose,spicy,honey,deep,damask",10,8500,"middle"),
    ("bg_rose_absolute","8007-01-0",2.5,"citronellol,geraniol","rose,honey,spicy,dark,rich",10,9000,"middle"),
    ("ir_damask_rose","8007-01-0",2.5,"citronellol,geraniol","rose,rosewater,fresh,spicy",9,7000,"middle"),
    ("tr_rose_isparta","8007-01-0",2.5,"citronellol,geraniol","rose,sweet,spicy,honey",9,5500,"middle"),
    ("ma_rose_dades","8007-01-0",2.5,"citronellol,geraniol","rose,herbal,dry,spicy",8,4500,"middle"),
    ("ylang_ylang","8006-81-3",None,"linalool,benzyl_acetate,geraniol","ylang,jasmine,banana,creamy,narcotic",9,350,"middle"),
    ("mg_ylang_extra","8006-81-3",None,"linalool,benzyl_acetate","ylang,creamy,jasmine,banana,exotic",10,500,"middle"),
    ("neroli","8016-38-4",None,"linalool,linalyl_acetate,geraniol","orange_blossom,fresh,floral,green,citrus",8,4500,"top"),
    ("fr_tuberose_grasse","8024-55-1",None,"benzyl_benzoate,benzyl_salicylate","tuberose,creamy,narcotic,sweet,white_floral",10,5000,"middle"),
    ("iris","89957-98-2",3.1,"None","iris,powdery,violet,earthy,buttery",9,45000,"base"),
    ("it_orris_concrete","89957-98-2",3.1,"None","orris,powdery,violet,earthy,luxurious",10,70000,"base"),
    ("it_orris_absolute","89957-98-2",3.1,"None","orris,powdery,violet,creamy,precious",10,100000,"base"),
    
    # ─── 핵심 시트러스 ───
    ("bergamot","8007-75-8",None,"linalool,linalyl_acetate,limonene","bergamot,citrus,floral,fresh,green",7,180,"top"),
    ("it_bergamot_calabria","8007-75-8",None,"linalool,linalyl_acetate","bergamot,earl_grey,fresh,sparkling,citrus",8,250,"top"),
    ("lemon","8008-56-8",None,"limonene,citral","lemon,fresh,citrus,bright,clean",7,35,"top"),
    ("mandarin","8008-31-9",None,"limonene","mandarin,sweet,citrus,sparkling,juicy",6,85,"top"),
    ("yuzu","None",None,"limonene,linalool","yuzu,citrus,tart,green,bergamot-like",7,450,"top"),
    ("jp_yuzu_kochi","None",None,"limonene,linalool","yuzu,tart,green,bright,japanese",7,500,"top"),
    ("grapefruit","8016-20-4",None,"limonene,nootkatone","grapefruit,bitter,fresh,citrus,sparkling",7,65,"top"),
    ("lime","8008-26-2",None,"limonene,citral","lime,citrus,green,sharp,fresh",7,45,"top"),

    # ─── 핵심 우디 ───
    ("sandalwood","8006-87-9",None,"None","sandalwood,creamy,milky,woody,soft",8,2500,"base"),
    ("in_sandalwood_mysore","8006-87-9",None,"None","sandalwood,creamy,milky,buttery,precious",10,3500,"base"),
    ("au_sandalwood_west","8024-35-9",None,"None","sandalwood,dry,woody,lighter,sustainable",7,800,"base"),
    ("cedarwood","8000-27-9",None,"None","cedar,pencil,dry,woody,clean",5,25,"base"),
    ("us_cedar_virginia","8000-27-9",None,"None","cedar,pencil,dry,woody,sharp",6,28,"base"),
    ("ma_cedarwood_atlas","92201-55-3",None,"None","cedar,warm,balsamic,dry,creamy",6,45,"base"),
    ("vetiver","8016-96-4",None,"None","vetiver,earthy,smoky,woody,rooty",8,250,"base"),
    ("ht_vetiver_haiti","8016-96-4",None,"None","vetiver,clean,green,chocolate,earthy",9,350,"base"),
    ("patchouli","8014-09-3",None,"None","patchouli,dark,earthy,sweet,camphor",8,85,"base"),
    ("id_patchouli_sulawesi","8014-09-3",None,"None","patchouli,dark,earthy,deep,chocolate",9,95,"base"),
    ("oud","None",None,"None","oud,smoky,animalic,leather,sweet,woody",10,35000,"base"),
    ("natural_oud","None",None,"None","oud,smoky,barnyard,woody,complex",10,50000,"base"),
    ("hinoki","None",None,"None","hinoki,lemony,woody,herbal,calming",7,350,"base"),
    ("jp_hinoki_wood","None",None,"None","hinoki,temple,woody,citrus,zen",8,400,"base"),
    
    # ─── 핵심 합성 ───
    ("iso_e_super","68155-67-9",None,"None","woody,amber,velvet,dry,modern",5,35,"base"),
    ("ambroxan","6790-58-5",None,"None","ambergris,mineral,skin,woody,salty",7,120,"base"),
    ("hedione","24851-98-7",None,"None","jasmine,airy,transparent,radiant,green",5,45,"middle"),
    ("galaxolide","1222-05-5",None,"None","musk,clean,sweet,powdery,skin",4,35,"base"),
    ("cashmeran","33704-61-9",None,"None","musky,spicy,woody,warm,cashmeran",6,85,"base"),
    ("javanol","198404-98-7",None,"None","sandalwood,creamy,milky,woody",7,250,"base"),
    ("calone","28940-11-6",None,"None","marine,watermelon,ozone,fresh,sea",8,180,"top"),
    ("linalool","78-70-6",None,"linalool","linalool,floral,woody,lavender,fresh",5,25,"top"),
    ("coumarin","91-64-5",0.5,"coumarin","coumarin,hay,tonka,almond,sweet",6,30,"base"),
    ("vanillin","121-33-5",None,"None","vanilla,sweet,creamy,warm,gourmand",7,15,"base"),
    ("ethyl_vanillin","121-32-4",None,"None","vanilla,sweet,creamy,strong,gourmand",8,25,"base"),
    ("muscone","541-91-3",None,"None","musk,animalic,powdery,skin,warm",8,4500,"base"),
    ("dihydromyrcenol","18479-58-8",None,"None","citrus,metallic,cool,fresh,ozonic",6,20,"top"),
    ("ethyl_maltol","4940-11-8",None,"None","cotton_candy,caramel,sweet,fruity",7,25,"middle"),
    ("indole","120-72-9",None,"None","jasmine,animalic,mothball,floral,narcotic",8,65,"middle"),
    
    # ─── 핵심 스파이스 ───
    ("cinnamon","8015-91-6",0.5,"cinnamaldehyde,eugenol","cinnamon,sweet,spicy,warm,balsamic",8,45,"middle"),
    ("lk_cinnamon_ceylon","8015-91-6",0.5,"cinnamaldehyde,eugenol","cinnamon,sweet,mild,complex,warm",9,120,"middle"),
    ("cardamom","8000-66-6",None,"linalool,linalyl_acetate","cardamom,spicy,sweet,eucalyptus,warm",7,250,"top"),
    ("in_cardamom_kerala","8000-66-6",None,"linalool","cardamom,spicy,eucalyptus,camphor,fresh",8,280,"top"),
    ("saffron","8022-12-0",None,"None","saffron,leather,honey,hay,metallic",9,8000,"middle"),
    ("in_saffron_kashmir","8022-12-0",None,"None","saffron,leather,honey,golden,precious",10,12000,"middle"),
    ("ir_saffron_khorasan","8022-12-0",None,"None","saffron,leather,honey,dried_hay",9,6000,"middle"),
    ("black_pepper","8006-82-4",None,"None","pepper,spicy,warm,woody,dry",6,65,"top"),
    ("vn_pepper_phuquoc","8006-82-4",None,"None","pepper,sharp,warm,complex,aromatic",8,120,"top"),
    ("ginger","8007-08-7",None,"None","ginger,fresh,spicy,citrus,warm",7,80,"top"),
    ("nutmeg","8008-45-5",None,"None","nutmeg,warm,spicy,sweet,woody",7,45,"middle"),
    
    # ─── 핵심 레진/앰버 ───
    ("frankincense","8016-36-2",None,"None","frankincense,lemon,pine,resin,sacred",7,120,"base"),
    ("myrrh","8016-37-3",None,"None","myrrh,bitter,sweet,balsamic,resin",7,180,"base"),
    ("benzoin","9000-72-0",None,"None","benzoin,vanilla,cinnamon,balsamic,sweet",6,85,"base"),
    ("la_benzoin_siam","9000-72-0",None,"None","benzoin,vanilla,chocolate,balsamic,warm",7,120,"base"),
    ("labdanum","8016-26-0",None,"None","labdanum,amber,leather,animalic,warm",8,280,"base"),
    ("vanilla","8024-06-4",None,"None","vanilla,sweet,creamy,warm,balsamic",8,450,"base"),
    ("mg_vanilla_bourbon","8024-06-4",None,"None","vanilla,creamy,caramel,rich,warm",9,600,"base"),
    ("tonka_bean","8046-22-8",None,"coumarin","tonka,almond,caramel,hay,sweet",8,120,"base"),
    ("amber","None",None,"None","amber,warm,resinous,powdery,sweet",6,350,"base"),

    # ─── 핵심 허벌 ───
    ("lavender","8000-28-0",None,"linalool,linalyl_acetate,geraniol","lavender,floral,herbal,fresh,clean",7,85,"top"),
    ("fr_lavender_fine","8000-28-0",None,"linalool,linalyl_acetate","lavender,fine,floral,soft,AOC",8,180,"top"),
    ("rosemary","8000-25-7",None,"None","rosemary,camphor,herbal,green,fresh",7,25,"top"),
    ("es_rosemary_spain","8000-25-7",None,"None","rosemary,camphor,intense,herbal,green",8,30,"top"),
    ("thyme","8007-46-3",None,"thymol","thyme,herbal,green,medicinal,warm",7,35,"top"),
    ("basil","8015-73-4",None,"linalool,eugenol,methyl_chavicol","basil,herbal,sweet,anise,green",7,45,"top"),
    ("sage","8022-56-8",None,"None","sage,herbal,camphor,woody,dry",6,55,"middle"),
    ("clary_sage","8016-63-5",None,"linalool","clary_sage,herbal,musky,tea-like,sweet",6,120,"middle"),

    # ─── 핵심 프루티 ───
    ("fig","None",None,"None","fig,green,milky,coconut,lactonic",7,450,"middle"),
    ("peach","None",None,"None","peach,juicy,sweet,fuzzy,lactonic",6,280,"middle"),
    ("blackcurrant","68606-81-5",None,"None","blackcurrant,fruity,green,catty,sulfurous",8,650,"top"),
    ("raspberry","None",None,"None","raspberry,sweet,berry,fruity,jammy",6,350,"middle"),
    ("apple","None",None,"None","apple,green,fresh,crisp,sweet",5,200,"top"),
    ("coconut","8001-31-8",None,"None","coconut,tropical,creamy,sweet,lactonic",6,35,"middle"),

    # ─── 핵심 해양/아쿠아틱 ───
    ("ambergris","None",None,"None","ambergris,mineral,salty,skin,oceanic",9,120000,"base"),
    ("seaweed","None",None,"None","seaweed,marine,iodine,salty,ozonic",6,280,"middle"),

    # ─── 핵심 차 ───
    ("jp_matcha_uji","None",None,"None","matcha,green,umami,sweet,ceremonial",7,800,"middle"),
    ("cn_pu_erh","None",None,"None","pu-erh,earthy,woody,mushroom,aged",7,350,"base"),
    ("jp_hojicha","None",None,"None","hojicha,roasted,smoky,caramel,warm",6,200,"middle"),

    # ─── 핵심 구르망 ───
    ("coffee","None",None,"None","coffee,roasted,chocolate,bitter,rich",8,180,"middle"),
    ("et_coffee_yirga","None",None,"None","coffee,fruity,floral,wine,complex",9,350,"middle"),
    ("chocolate","None",None,"None","chocolate,cocoa,sweet,bitter,rich",7,120,"middle"),
    ("maple","None",None,"None","maple,sweet,woody,caramel,warm",6,280,"middle"),
    ("honey","None",None,"None","honey,sweet,warm,waxy,floral",7,350,"middle"),

    # ─── 핵심 한국 원료 ───
    ("mugunghwa","None",None,"None","mugunghwa,soft_floral,powdery,gentle,korean",6,400,"middle"),
    ("green_tea","None",None,"None","green_tea,fresh,grassy,clean,bright",6,120,"top"),
    ("ginseng","None",None,"None","ginseng,earthy,herbal,spicy,sweet",7,350,"base"),
    ("yuzu_peel","None",None,"limonene","yuzu_peel,citrus,tart,green,bright",7,380,"top"),
    ("bamboo","None",None,"None","bamboo,green,fresh,aquatic,woody",5,150,"middle"),
    ("lotus_korean","None",None,"None","lotus,aquatic,floral,powdery,serene",7,450,"middle"),
    ("maehwa","None",None,"None","plum_blossom,clean,floral,green,spring",7,500,"middle"),
    ("ssuk","None",None,"None","mugwort,herbal,green,warm,earthy",6,80,"middle"),
    ("omija","None",None,"None","five_flavors,sour,sweet,bitter,spicy,salty",8,200,"middle"),
    ("gamgyul","None",None,"limonene","tangerine,sweet,juicy,citrus,jeju",6,120,"top"),
    ("beotkkot","None",None,"None","cherry_blossom,powdery,sweet,green,spring",6,600,"middle"),
    ("ondol","None",None,"None","ondol,warm_wood,cozy,earthy,comforting",5,None,"base"),
    ("hanji","None",None,"None","hanji,paper,dry,warm,cellulose",4,None,"base"),

    # ─── 핵심 동물성 (합성) ───
    ("civet","68916-26-7",None,"None","civet,animalic,warm,musky,fecal",9,850,"base"),
    ("castoreum","8023-83-4",None,"None","castoreum,leather,birch,animalic,warm",8,1200,"base"),
    ("hyraceum","None",None,"None","hyraceum,animalic,amber,tobacco,musky",8,2800,"base"),

    # ─── 핵심 그린 ───
    ("galbanum","8023-91-4",None,"None","galbanum,green,metallic,resinous,bitter",8,380,"top"),
    ("ir_galbanum_ir","8023-91-4",None,"None","galbanum,intense,green,metallic,knife-sharp",9,420,"top"),
    ("violet_leaf","8024-08-6",None,"None","violet_leaf,green,creamy,earthy,leafy",7,550,"middle"),
    ("oakmoss","9000-50-4",0.1,"atranol,chloroatranol,evernic_acid","oakmoss,mossy,green,earthy,damp",8,450,"base"),
]

def enrich_pro_data():
    """프로 데이터 보강"""
    conn = database.get_conn()
    cur = conn.cursor()
    updated = 0
    for item in PRO_DATA:
        ing_id = item[0]
        cas = item[1]
        ifra = item[2]
        allergens = item[3] if item[3] != "None" else None
        descriptors = item[4]
        strength = item[5]
        price = item[6]
        vol = item[7]
        
        cur.execute("SELECT id FROM ingredients WHERE id=%s",(ing_id,))
        if not cur.fetchone():
            continue
        
        cur.execute("""
            UPDATE ingredients SET
                cas_number=%s, ifra_limit=%s, allergens=%s,
                odor_descriptors=%s, odor_strength=%s, price_usd_kg=%s,
                volatility=%s
            WHERE id=%s
        """, (cas, ifra, allergens, descriptors, strength, price, vol, ing_id))
        updated += 1
    return updated


def main():
    print("="*60)
    print("  🔬 Pro-Grade 원료 DB 업그레이드")
    print("="*60)
    
    # 1. 스키마
    print("\n[1/4] 스키마 업그레이드...")
    upgrade_schema()
    
    # 2. 중복삭제
    print("\n[2/4] 중복 원료 삭제...")
    removed = remove_duplicates()
    
    # 3. 프로데이터
    print("\n[3/4] 프로 데이터 보강...")
    updated = enrich_pro_data()
    print(f"  📊 {updated}개 원료에 CAS/IFRA/가격/디스크립터 추가")
    
    # 4. 통계
    print("\n[4/4] 최종 통계...")
    conn = database.get_conn()
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM ingredients")
    total = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM ingredients WHERE cas_number IS NOT NULL")
    cas_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM ingredients WHERE ifra_limit IS NOT NULL")
    ifra_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM ingredients WHERE price_usd_kg IS NOT NULL")
    price_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM ingredients WHERE odor_descriptors IS NOT NULL")
    desc_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM ingredients WHERE volatility IS NOT NULL")
    vol_count = cur.fetchone()[0]
    
    print(f"\n  총 원료: {total}개")
    print(f"  CAS 번호: {cas_count}개 ({cas_count/total*100:.0f}%)")
    print(f"  IFRA 제한: {ifra_count}개")
    print(f"  가격 정보: {price_count}개 ({price_count/total*100:.0f}%)")
    print(f"  향 디스크립터: {desc_count}개 ({desc_count/total*100:.0f}%)")
    print(f"  노트 분류: {vol_count}개 ({vol_count/total*100:.0f}%)")
    
    # 가격 범위 (가장 비싼/저렴한)
    cur.execute("SELECT name_ko, price_usd_kg FROM ingredients WHERE price_usd_kg IS NOT NULL ORDER BY price_usd_kg DESC LIMIT 10")
    print(f"\n  💎 가장 비싼 원료 TOP 10:")
    for r in cur.fetchall():
        print(f"     ${r[1]:>10,.0f}/kg — {r[0]}")
    
    cur.execute("SELECT name_ko, price_usd_kg FROM ingredients WHERE price_usd_kg IS NOT NULL ORDER BY price_usd_kg ASC LIMIT 5")
    print(f"\n  💰 가장 저렴한 원료:")
    for r in cur.fetchall():
        print(f"     ${r[1]:>10,.0f}/kg — {r[0]}")
    
    print(f"\n{'='*60}")
    print(f"  ✅ Pro-Grade 업그레이드 완료!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
