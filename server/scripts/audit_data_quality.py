# -*- coding: utf-8 -*-
"""🔍 원료 DB 데이터 품질 감사 (Data Quality Audit)"""
import sys,os,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
sys.path.insert(0,'.')
import database

conn = database.get_conn()
cur = conn.cursor()

print("="*70)
print("  🔍 원료 데이터베이스 품질 감사 (Data Quality Audit)")
print("="*70)

# ══════════════════════════════════════
# 1. 기본 통계
# ══════════════════════════════════════
cur.execute("SELECT COUNT(*) FROM ingredients")
total = cur.fetchone()[0]
print(f"\n📊 총 원료: {total}개\n")

# ══════════════════════════════════════
# 2. 필드 완성도
# ══════════════════════════════════════
fields = [
    ("id", "id IS NOT NULL"),
    ("name_ko", "name_ko IS NOT NULL AND name_ko != ''"),
    ("name_en", "name_en IS NOT NULL AND name_en != ''"),
    ("category", "category IS NOT NULL AND category != ''"),
    ("country", "country IS NOT NULL AND country != ''"),
    ("region", "region IS NOT NULL AND region != ''"),
    ("terroir_notes", "terroir_notes IS NOT NULL AND terroir_notes != ''"),
    ("note_type", "note_type IS NOT NULL AND note_type != ''"),
    ("cas_number", "cas_number IS NOT NULL AND cas_number != '' AND cas_number != 'None'"),
    ("ifra_limit", "ifra_limit IS NOT NULL"),
    ("allergens", "allergens IS NOT NULL AND allergens != '' AND allergens != 'None'"),
    ("odor_descriptors", "odor_descriptors IS NOT NULL AND odor_descriptors != ''"),
    ("odor_strength", "odor_strength IS NOT NULL"),
    ("price_usd_kg", "price_usd_kg IS NOT NULL"),
]

print("┌─────────────────────┬────────┬────────┐")
print("│ 필드                │ 완성   │ 비율   │")
print("├─────────────────────┼────────┼────────┤")
for name, cond in fields:
    cur.execute(f"SELECT COUNT(*) FROM ingredients WHERE {cond}")
    cnt = cur.fetchone()[0]
    pct = cnt/total*100
    bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
    grade = "✅" if pct >= 80 else "⚠️" if pct >= 30 else "❌"
    print(f"│ {name:19s} │ {cnt:4d}개 │ {pct:5.1f}% {grade}│")
print("└─────────────────────┴────────┴────────┘")

# ══════════════════════════════════════
# 3. 중복 검사
# ══════════════════════════════════════
print("\n🔄 중복 검사:")
# Exact name_en duplicates
cur.execute("""
    SELECT LOWER(name_en), COUNT(*), array_agg(id) 
    FROM ingredients WHERE name_en IS NOT NULL 
    GROUP BY LOWER(name_en) HAVING COUNT(*) > 1
""")
dupes = cur.fetchall()
if dupes:
    print(f"  ⚠️ name_en 중복: {len(dupes)}건")
    for d in dupes[:10]:
        print(f"     {d[0]:40s} → {d[2]}")
else:
    print("  ✅ name_en 중복 없음")

# name_ko duplicates
cur.execute("""
    SELECT name_ko, COUNT(*), array_agg(id) 
    FROM ingredients WHERE name_ko IS NOT NULL 
    GROUP BY name_ko HAVING COUNT(*) > 1
""")
dupes_ko = cur.fetchall()
if dupes_ko:
    print(f"  ⚠️ name_ko 중복: {len(dupes_ko)}건")
    for d in dupes_ko[:10]:
        print(f"     {d[0]:40s} → {d[2]}")
else:
    print("  ✅ name_ko 중복 없음")

# ══════════════════════════════════════
# 4. 카테고리 분포 & 이상치
# ══════════════════════════════════════
print("\n📂 카테고리 분포:")
cur.execute("SELECT category, COUNT(*) FROM ingredients GROUP BY category ORDER BY COUNT(*) DESC")
for r in cur.fetchall():
    print(f"  {r[0] or '(NULL)':20s} {r[1]:4d}개")

# ══════════════════════════════════════
# 5. CAS 번호 품질
# ══════════════════════════════════════
print("\n🧪 CAS 번호 품질:")
cur.execute("SELECT COUNT(*) FROM ingredients WHERE cas_number = 'None'")
fake_cas = cur.fetchone()[0]
if fake_cas > 0:
    print(f"  ⚠️ CAS='None' (문자열): {fake_cas}개 — 이건 NULL이어야 함")
    cur.execute("SELECT id, name_en FROM ingredients WHERE cas_number = 'None' LIMIT 5")
    for r in cur.fetchall():
        print(f"     {r[0]:30s} {r[1]}")

cur.execute("SELECT COUNT(*) FROM ingredients WHERE cas_number IS NOT NULL AND cas_number != 'None'")
real_cas = cur.fetchone()[0]
print(f"  ✅ 유효 CAS: {real_cas}개")

# CAS format check (should be XXXXX-XX-X)
cur.execute("SELECT id, cas_number FROM ingredients WHERE cas_number IS NOT NULL AND cas_number != 'None' AND cas_number !~ '^[0-9]+-[0-9]+-[0-9]+$'")
bad_cas = cur.fetchall()
if bad_cas:
    print(f"  ⚠️ CAS 형식 오류: {len(bad_cas)}개")
    for r in bad_cas[:5]:
        print(f"     {r[0]:30s} CAS={r[1]}")
else:
    print("  ✅ CAS 형식 모두 정상 (XXXXX-XX-X)")

# ══════════════════════════════════════
# 6. 가격 이상치
# ══════════════════════════════════════
print("\n💰 가격 데이터 검증:")
cur.execute("SELECT COUNT(*) FROM ingredients WHERE price_usd_kg IS NOT NULL AND price_usd_kg <= 0")
neg_price = cur.fetchone()[0]
if neg_price > 0:
    print(f"  ❌ 가격 0 이하: {neg_price}개")
else:
    print("  ✅ 음수/0 가격 없음")

cur.execute("SELECT AVG(price_usd_kg), STDDEV(price_usd_kg) FROM ingredients WHERE price_usd_kg IS NOT NULL")
avg_p, std_p = cur.fetchone()
print(f"  평균: ${avg_p:,.0f}/kg, 표준편차: ${std_p:,.0f}")

# Outliers (> mean + 3*std)
threshold = avg_p + 3 * std_p
cur.execute("SELECT name_ko, price_usd_kg FROM ingredients WHERE price_usd_kg > %s ORDER BY price_usd_kg DESC", (threshold,))
outliers = cur.fetchall()
if outliers:
    print(f"  ⚠️ 가격 이상치 (3σ 초과): {len(outliers)}개")
    for r in outliers:
        print(f"     ${r[1]:>10,.0f}/kg — {r[0]}")
    print(f"     (단, 오리스/앰버그리스 등은 실제로 비싸므로 정상)")

# ══════════════════════════════════════
# 7. IFRA 제한 vs 카테고리 교차검증
# ══════════════════════════════════════
print("\n🛡 IFRA 규제 분석:")
cur.execute("SELECT name_ko, ifra_limit, category FROM ingredients WHERE ifra_limit IS NOT NULL ORDER BY ifra_limit")
ifra_items = cur.fetchall()
for r in ifra_items:
    level = "🔴 위험" if r[1] < 0.5 else "🟡 주의" if r[1] < 3 else "🟢 안전"
    print(f"  {level} {r[0]:30s} — 최대 {r[1]}% ({r[2]})")

# ══════════════════════════════════════
# 8. odor_descriptors 분석
# ══════════════════════════════════════
print("\n🌸 향 디스크립터 분석:")
cur.execute("SELECT odor_descriptors FROM ingredients WHERE odor_descriptors IS NOT NULL")
all_desc = cur.fetchall()
tag_count = {}
for row in all_desc:
    for tag in row[0].split(","):
        tag = tag.strip()
        tag_count[tag] = tag_count.get(tag, 0) + 1

print(f"  총 고유 태그: {len(tag_count)}개")
print(f"  상위 20 태그:")
for tag, cnt in sorted(tag_count.items(), key=lambda x: -x[1])[:20]:
    print(f"    {tag:25s} {cnt}회")

# ══════════════════════════════════════
# 9. 종합 품질 점수
# ══════════════════════════════════════
print("\n" + "="*70)
print("  📋 종합 데이터 품질 리포트")
print("="*70)

scores = {}
# 기본 필드
for name, cond in [("name_ko","name_ko IS NOT NULL"),("name_en","name_en IS NOT NULL"),
    ("category","category IS NOT NULL"),("country","country IS NOT NULL")]:
    cur.execute(f"SELECT COUNT(*) FROM ingredients WHERE {cond}")
    scores[name] = cur.fetchone()[0] / total * 100

# Pro 필드
for name, cond in [("CAS","cas_number IS NOT NULL AND cas_number != 'None'"),
    ("가격","price_usd_kg IS NOT NULL"),("향디스크립터","odor_descriptors IS NOT NULL"),
    ("노트분류","note_type IS NOT NULL AND note_type != ''")]:
    cur.execute(f"SELECT COUNT(*) FROM ingredients WHERE {cond}")
    scores[name] = cur.fetchone()[0] / total * 100

basic_avg = (scores["name_ko"]+scores["name_en"]+scores["category"]+scores["country"])/4
pro_avg = (scores["CAS"]+scores["가격"]+scores["향디스크립터"]+scores["노트분류"])/4
overall = basic_avg * 0.6 + pro_avg * 0.4

print(f"\n  기본 데이터 품질: {basic_avg:.1f}% {'✅' if basic_avg>90 else '⚠️'}")
print(f"    name_ko: {scores['name_ko']:.1f}% | name_en: {scores['name_en']:.1f}%")
print(f"    category: {scores['category']:.1f}% | country: {scores['country']:.1f}%")
print(f"\n  프로 데이터 품질: {pro_avg:.1f}% {'✅' if pro_avg>50 else '⚠️' if pro_avg>20 else '❌'}")
print(f"    CAS: {scores['CAS']:.1f}% | 가격: {scores['가격']:.1f}%")
print(f"    향디스크립터: {scores['향디스크립터']:.1f}% | 노트분류: {scores['노트분류']:.1f}%")
print(f"\n  ★ 종합 품질 점수: {overall:.1f}/100")
print(f"    {'🏆 우수' if overall>80 else '✅ 양호' if overall>60 else '⚠️ 보통' if overall>40 else '❌ 미흡'}")

print(f"\n  📌 개선 권고사항:")
if scores['CAS'] < 50:
    print(f"     1. CAS 번호 보강 필요 ({scores['CAS']:.0f}% → 목표 50%+)")
if scores['가격'] < 50:
    print(f"     2. 가격 데이터 보강 필요 ({scores['가격']:.0f}% → 목표 50%+)")
if scores['향디스크립터'] < 50:
    print(f"     3. 향 디스크립터 보강 필요 ({scores['향디스크립터']:.0f}% → 목표 50%+)")
if len(dupes) > 0:
    print(f"     4. name_en 중복 {len(dupes)}건 해결 필요")
if fake_cas > 0:
    print(f"     5. CAS='None' 문자열 → NULL 변환 필요 ({fake_cas}개)")
