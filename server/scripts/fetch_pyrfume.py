# fetch_pyrfume.py — Pyrfume Leffingwell 데이터 수집 + PostgreSQL 적재
# ====================================================================
import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("psycopg2 미설치. 설치 중...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'psycopg2-binary'])
    import psycopg2
    import psycopg2.extras

# ===== 설정 =====
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'perfumer',
    'user': 'perfumer',
    'password': 'perfumer123'
}

PYRFUME_BASE = 'https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/leffingwell'
DATA_DIR = Path(__file__).parent.parent.parent / 'data'


def download_csv(url):
    """URL에서 CSV 다운로드 → 리스트 반환"""
    print(f"  다운로드: {url}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    text = resp.read().decode('utf-8')
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    print(f"  → {len(rows)}개 행")
    return rows


def connect_db(retries=5):
    for i in range(retries):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except psycopg2.OperationalError:
            print(f"  DB 연결 대기 ({i+1}/{retries})...")
            time.sleep(3)
    raise Exception("PostgreSQL 연결 실패")


def load_pyrfume_molecules(conn):
    """Pyrfume Leffingwell molecules.csv → DB"""
    print("\n[1/4] Pyrfume 분자 데이터 다운로드...")
    molecules = download_csv(f"{PYRFUME_BASE}/molecules.csv")

    cur = conn.cursor()
    inserted = 0
    for mol in molecules:
        cid = mol.get('CID', '')
        if not cid:
            continue
        try:
            cid_int = int(float(cid))
        except (ValueError, OverflowError):
            cid_int = hash(cid) % (10**15)

        smiles = mol.get('IsomericSMILES', '')
        name = mol.get('name', '') or mol.get('IUPACName', '') or f'CID_{cid}'
        mw = float(mol.get('MolecularWeight', 0) or 0)
        iupac = mol.get('IUPACName', '')

        cur.execute("""
            INSERT INTO molecules (cid, name, iupac_name, smiles, molecular_weight, source)
            VALUES (%s, %s, %s, %s, %s, 'pyrfume')
            ON CONFLICT (cid) DO NOTHING
            RETURNING id
        """, (cid_int, name[:500], iupac[:1000] if iupac else None, smiles, mw))

        if cur.fetchone():
            inserted += 1

    conn.commit()
    print(f"  → {inserted}개 분자 삽입됨")
    return inserted


def load_pyrfume_behaviors(conn):
    """Pyrfume Leffingwell behavior.csv → molecule_odors 매핑"""
    print("\n[2/4] Pyrfume 냄새 행동 데이터 다운로드...")
    behaviors = download_csv(f"{PYRFUME_BASE}/behavior.csv")

    cur = conn.cursor()

    # descriptor name → id 매핑
    cur.execute("SELECT id, name FROM odor_descriptors")
    desc_map = {row[1]: row[0] for row in cur.fetchall()}

    if not behaviors:
        print("  behavior 데이터 없음")
        return

    # 첫 행에서 descriptor 컬럼 추출
    all_cols = list(behaviors[0].keys())
    desc_cols = [c for c in all_cols if c != 'Stimulus' and c in desc_map]
    print(f"  매칭 descriptor: {len(desc_cols)}개")

    mapped = 0
    for row in behaviors:
        cid_str = row.get('Stimulus', '')
        if not cid_str:
            continue
        try:
            cid_int = int(float(cid_str))
        except (ValueError, OverflowError):
            cid_int = hash(cid_str) % (10**15)

        # molecule_id 찾기
        cur.execute("SELECT id FROM molecules WHERE cid = %s", (cid_int,))
        mol_row = cur.fetchone()
        if not mol_row:
            continue
        mol_id = mol_row[0]

        # 각 descriptor에 대해 1인 것만 삽입
        for col in desc_cols:
            val = row.get(col, '0')
            if val == '1':
                cur.execute("""
                    INSERT INTO molecule_odors (molecule_id, descriptor_id, strength, source)
                    VALUES (%s, %s, 1.0, 'pyrfume')
                    ON CONFLICT (molecule_id, descriptor_id) DO NOTHING
                """, (mol_id, desc_map[col]))
                mapped += 1

    conn.commit()
    print(f"  → {mapped}개 분자-냄새 매핑 삽입됨")


def load_existing_ingredients(conn):
    """기존 data/ingredients.json → DB 이전"""
    print("\n[3/4] 기존 향료 데이터 이전...")
    path = DATA_DIR / 'ingredients.json'
    if not path.exists():
        print("  ingredients.json 없음, 건너뜀")
        return

    with open(path, 'r', encoding='utf-8') as f:
        ingredients = json.load(f)

    cur = conn.cursor()
    inserted = 0
    for ing in ingredients:
        cur.execute("""
            INSERT INTO ingredients (id, name_ko, category, note_type,
                volatility, intensity, longevity, typical_pct, max_pct,
                descriptors, moods, seasons, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'custom')
            ON CONFLICT (id) DO NOTHING
        """, (
            ing['id'], ing.get('name_ko', ''), ing.get('category', ''),
            ing.get('note_type', ''), ing.get('volatility', 5),
            ing.get('intensity', 5), ing.get('longevity', 5),
            ing.get('typical_pct', 5), ing.get('max_pct', 15),
            ing.get('descriptors', []), ing.get('moods', []),
            ing.get('seasons', [])
        ))
        inserted += 1

    conn.commit()
    print(f"  → {inserted}개 향료 삽입됨")


def load_existing_molecules(conn):
    """기존 data/molecules.json → DB 이전 (기존 분자 보존)"""
    print("\n[4/4] 기존 분자 데이터 이전...")
    path = DATA_DIR / 'molecules.json'
    if not path.exists():
        print("  molecules.json 없음, 건너뜀")
        return

    with open(path, 'r', encoding='utf-8') as f:
        mols = json.load(f)

    cur = conn.cursor()
    inserted = 0
    for mol in mols:
        # CID가 없는 기존 분자 → 이름 해시를 CID로
        name = mol.get('name', mol.get('id', ''))
        smiles = mol.get('smiles', '')
        cid = hash(name) % (10**12) + 900000000000  # 충돌 방지

        cur.execute("""
            INSERT INTO molecules (cid, name, name_ko, smiles, molecular_weight,
                logp, hbd, hba, rotatable_bonds, rings, aromatic_rings,
                functional_groups, odor_strength, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'custom')
            ON CONFLICT (cid) DO UPDATE SET name_ko = EXCLUDED.name_ko
            RETURNING id
        """, (
            cid, name, mol.get('name', ''), smiles,
            mol.get('mw', 0), mol.get('logP', 0),
            mol.get('hbd', 0), mol.get('hba', 0),
            mol.get('rotatable', 0), mol.get('rings', 0),
            mol.get('aromatic_rings', 0),
            mol.get('functional_groups', []),
            mol.get('odor_strength', 5)
        ))

        result = cur.fetchone()
        if result:
            mol_id = result[0]
            # 냄새 레이블도 넣기
            for label in mol.get('odor_labels', []):
                cur.execute("SELECT id FROM odor_descriptors WHERE name = %s", (label,))
                desc_row = cur.fetchone()
                if desc_row:
                    cur.execute("""
                        INSERT INTO molecule_odors (molecule_id, descriptor_id, strength, source)
                        VALUES (%s, %s, 1.0, 'custom')
                        ON CONFLICT (molecule_id, descriptor_id) DO NOTHING
                    """, (mol_id, desc_row[0]))
            inserted += 1

    conn.commit()
    print(f"  → {inserted}개 분자 삽입/업데이트됨")


def print_stats(conn):
    """최종 통계"""
    cur = conn.cursor()
    stats = {}
    for table in ['molecules', 'odor_descriptors', 'molecule_odors', 'ingredients']:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cur.fetchone()[0]

    print("\n" + "=" * 50)
    print("📊 데이터베이스 통계")
    print("=" * 50)
    print(f"  분자 (molecules):        {stats['molecules']:,}개")
    print(f"  냄새 descriptor:         {stats['odor_descriptors']}개")
    print(f"  분자-냄새 매핑:          {stats['molecule_odors']:,}개")
    print(f"  향료 원료 (ingredients): {stats['ingredients']}개")
    print("=" * 50)


if __name__ == '__main__':
    print("🧪 AI Perfumer 데이터베이스 구축 시작")
    print("=" * 50)

    conn = connect_db()
    try:
        load_pyrfume_molecules(conn)
        load_pyrfume_behaviors(conn)
        load_existing_ingredients(conn)
        load_existing_molecules(conn)
        print_stats(conn)
        print("\n✅ 완료!")
    finally:
        conn.close()
