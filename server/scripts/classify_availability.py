# -*- coding: utf-8 -*-
"""
🏷 원료 실용성 분류 (Availability Tier)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tier 1: readily_available — 대형 공급업체에서 바로 구매 가능, $500/kg 이하
Tier 2: specialty         — 전문 공급업체, $500-5000/kg 또는 특수 소싱
Tier 3: rare              — 매우 제한적, $5000+/kg 또는 계절 한정  
Tier 4: restricted        — IFRA 규제/금지, 사용 제한 있음
Tier 5: conceptual        — 어코드/가상 원료, 단일 원료로 존재하지 않음
"""
import sys,os,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
sys.path.insert(0,'.')
import database

conn = database.get_conn()
cur = conn.cursor()

# 1) Add column
try:
    cur.execute("ALTER TABLE ingredients ADD COLUMN availability VARCHAR(30)")
    print("✅ availability 컬럼 추가")
except:
    conn.rollback()
    print("⏭ availability 컬럼 이미 존재")

# ═══════════════════════════════════════
# 분류 규칙 (여러 규칙을 순차 적용)
# ═══════════════════════════════════════

# === TIER 5: conceptual (어코드/가상) ===
CONCEPTUAL = [
    # 한국 어코드/가상
    "ondol","hanji","gaetbeol","haepung","spring_rain","snow_accord","jeju_lava",
    "jirisan_morning","bamboo_charcoal","ssanghwa","barley_tea",
    "gyeongju_soil",
    # 자연현상 어코드
    "fr_champagne_accord","us_blueberry_accord","us_bourbon_accord",
    "eg_kyphi_accord","uz_cotton","bluebells","freesia","sweet_pea",
    "la_sticky_rice","kw_amber_kuwaiti","ma_amber_accord","ae_bakhoor",
    "ma_oud_moroccan",
]

# === TIER 4: restricted (IFRA 규제) ===
RESTRICTED_IDS = [
    "oakmoss",        # IFRA 0.1% - 매우 엄격
    "musk_xylene",    # EU 규제
    "musk_ketone",    # EU 규제
    "lyral",          # EU 금지 (2022)
    "lilial",         # EU 금지 (2022)
    "coumarin",       # IFRA 0.5%
    "methyl_eugenol", # IFRA 제한
    "skatole",        # 극소량만
    "cresol",         # 극소량만
    "civet_synthetic",# 대체품만 사용
]

# === TIER 3: rare ($5000+/kg 또는 극소량) ===
RARE_IDS = [
    "it_orris_absolute","it_orris_concrete","iris_butter","orris_root",
    "ambergris_tincture","natural_oud","oud_oil","oud",
    "in_agarwood_assam","cn_agarwood_cn","my_gaharu_malay",
    "fr_rose_de_mai","fr_boronia_abs","au_boronia_mega",
    "damascenone","muscenone","muscone",
    "lily","hyacinth","honeysuckle","champaca_gold",
    "hi_sandalwood_hawaii","fj_sandalwood_fiji",
    "in_saffron_kashmir","es_saffron_mancha","saffron_abs",
    "eg_lotus_blue","in_mogra","pk_jasmine_multan",
    "hi_tuberose_hawaii","mx_tuberose_mexican","fr_tuberose_grasse",
    "co2_rose","ambroxan",
    "paradisone","habanolide","nirvanolide","romandolide",
    "in_sandalwood_mysore","in_costus_root",
    "castoreum","hyraceum",
    "co_coffee_colombian",
    "bg_rose_concrete",
    "fr_jonquil","fr_narcissus_abs",
]

# === TIER 2: specialty ($500-5000/kg 또는 특수 소싱) ===
# 가격 기반 + 수동 지정
SPECIALTY_IDS = [
    # 비싼 천연 원료
    "rose_abs","jasmine_abs","neroli_oil",
    "bg_rose_kazanlak","ir_damask_rose","ro_rose_romanian",
    "tr_rose_isparta","tr_rose_abs","ma_rose_dades","ma_rose_abs",
    "tuberose_abs","osmanthus_abs",
    "in_sandalwood_mysore","au_sandalwood_west","sandalwood_oil",
    "fr_mimosa_absolute","fr_cassie",
    "jp_kinmokusei","jp_sakura_abs","jp_wisteria_jp",
    "mg_ylang_extra",
    "fr_labdanum_abs","cy_labdanum_cyprus",
    "in_champaca_abs_in","in_kewda",
    "fr_clary_sage_oil","fr_angelica_root",
    "black_currant_bud","violet_leaf_abs",
    "hr_immortelle_croatia","at_edelweiss",
    "co2_vanilla","co2_coffee","co2_frankincense",
    "pe_cacao_criollo","ec_cacao_arriba",
    "fr_cognac_oil","fr_genet_abs","fr_broom_absolute",
    "fr_truffle_black","it_truffle_white",
    "tn_neroli_tunisia","ke_geranium_kenya",
    "pf_tiare_tahiti","hi_plumeria",
    "np_jatamansi","np_rhododendron",
    "in_attar_mitti","in_attar_shamama","in_attar_gulab","in_attar_hina",
    "in_davana_in","in_nagarmotha_in",
    "so_olibanum_extra","om_frankincense_royal",
    "jp_matcha_uji","jp_gyokuro",
    "carnation","frangipani","gardenia","peony",
    "ma_saffron_taliouine","ir_saffron_khorasan",
    "water_lily","heliotrope","wisteria",
    "de_chamomile_german","co2_chamomile",
    "bg_melissa_bg",
    "civetone","castoreum_synth",
    "za_buchu","za_cape_chamomile",
    "pe_palo_santo",
    "au_blue_cypress","au_buddawood",
    "kh_kampot_pepper","kh_kravan",
    "vn_pepper_phuquoc",
    "javanol","polysantol","ebanol","bacdanol",
    "akigalawood","clearwood","norlimbanol",
    "hindinol","firsantol",
    "hedione_hc","helvetolide","ambrettolide",
    "safranal_synth","safraleine",
    "nectaryl","raspberry_ketone",
    "ambrocenide","karanal",
    "suederal",
    "mg_vanilla_bourbon","mg_vanilla_abs",
    "mx_vanilla_mexican","fr_vanilla_tahiti",
    "lk_cinnamon_ceylon",
    "se_birch_tar","ru_birch_russian",
    "ma_blue_tansy_oil",
    "jp_kuromoji","jp_sandalwood_jp",
    "fr_vetiver_bourbon","ht_vetiver_haiti",
    "br_rosewood_oil","br_priprioca",
    "gy_rosewood_guyana",
    "ta_ge_tes","tagetes",
    "eg_geranium_nile",
    "et_coffee_yirga",
    "fr_fig_leaf_abs",
]

def classify():
    # 1) Default: readily_available
    cur.execute("UPDATE ingredients SET availability='readily_available' WHERE availability IS NULL")
    default_count = cur.rowcount
    
    # 2) Auto-classify by price
    cur.execute("UPDATE ingredients SET availability='specialty' WHERE price_usd_kg >= 500 AND price_usd_kg < 5000 AND availability='readily_available'")
    price_spec = cur.rowcount
    
    cur.execute("UPDATE ingredients SET availability='rare' WHERE price_usd_kg >= 5000 AND availability IN ('readily_available','specialty')")
    price_rare = cur.rowcount
    
    # 3) Manual overrides
    for iid in CONCEPTUAL:
        cur.execute("UPDATE ingredients SET availability='conceptual' WHERE id=%s",(iid,))
    
    for iid in RESTRICTED_IDS:
        cur.execute("UPDATE ingredients SET availability='restricted' WHERE id=%s",(iid,))
    
    for iid in RARE_IDS:
        cur.execute("UPDATE ingredients SET availability='rare' WHERE id=%s AND availability NOT IN ('restricted','conceptual')",(iid,))
    
    for iid in SPECIALTY_IDS:
        cur.execute("UPDATE ingredients SET availability='specialty' WHERE id=%s AND availability NOT IN ('restricted','conceptual','rare')",(iid,))
    
    print(f"  기본(readily_available): {default_count}개 → 가격 기반 재분류")

classify()

# ═══ 최종 통계 ═══
print(f"\n{'='*60}")
print(f"  🏷 원료 실용성 분류 결과")
print(f"{'='*60}")

tiers = [
    ("readily_available", "🟢 바로 구매 가능", "대형 공급업체, <$500/kg"),
    ("specialty", "🟡 전문 소싱 필요", "$500-5000/kg, 전문 공급업체"),
    ("rare", "🔴 희귀/고가", ">$5000/kg, 극소량"),
    ("restricted", "⛔ IFRA 규제", "사용 제한/금지"),
    ("conceptual", "💭 가상/어코드", "단일 원료 아님"),
]

total_all = 0
for tier_id, label, desc in tiers:
    cur.execute("SELECT COUNT(*) FROM ingredients WHERE availability=%s",(tier_id,))
    cnt = cur.fetchone()[0]
    total_all += cnt
    pct = cnt / 1136 * 100
    print(f"\n  {label} — {cnt}개 ({pct:.1f}%)")
    print(f"     {desc}")
    
    # 대표 원료 5개
    cur.execute("SELECT name_ko, price_usd_kg FROM ingredients WHERE availability=%s ORDER BY RANDOM() LIMIT 5",(tier_id,))
    for r in cur.fetchall():
        p = f"${r[1]:,.0f}/kg" if r[1] else "—"
        print(f"     • {r[0]} ({p})")

# 실용 vs 비실용 요약
cur.execute("SELECT COUNT(*) FROM ingredients WHERE availability IN ('readily_available','specialty')")
usable = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM ingredients WHERE availability IN ('rare','restricted','conceptual')")
unusable = cur.fetchone()[0]

print(f"\n{'='*60}")
print(f"  📊 요약")
print(f"{'='*60}")
print(f"  ✅ 실용 가능:   {usable}개 ({usable/total_all*100:.1f}%)")
print(f"     → 바로 사용:  ", end="")
cur.execute("SELECT COUNT(*) FROM ingredients WHERE availability='readily_available'")
print(f"{cur.fetchone()[0]}개")
print(f"     → 전문 소싱:  ", end="")
cur.execute("SELECT COUNT(*) FROM ingredients WHERE availability='specialty'")
print(f"{cur.fetchone()[0]}개")
print(f"  ❌ 실용 불가:   {unusable}개 ({unusable/total_all*100:.1f}%)")
print(f"     → 희귀/고가:  ", end="")
cur.execute("SELECT COUNT(*) FROM ingredients WHERE availability='rare'")
print(f"{cur.fetchone()[0]}개")
print(f"     → IFRA 규제:  ", end="")
cur.execute("SELECT COUNT(*) FROM ingredients WHERE availability='restricted'")
print(f"{cur.fetchone()[0]}개")
print(f"     → 가상/어코드: ", end="")
cur.execute("SELECT COUNT(*) FROM ingredients WHERE availability='conceptual'")
print(f"{cur.fetchone()[0]}개")
