# database.py — PostgreSQL 연결 + 쿼리
import psycopg2
import psycopg2.extras

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'perfumer',
    'user': 'perfumer',
    'password': 'perfumer123'
}

_conn = None


def get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(**DB_CONFIG)
        _conn.autocommit = True
    return _conn


def get_all_molecules(limit=None):
    """전체 분자 목록 (냄새 레이블 포함)"""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    sql = """
        SELECT m.id, m.cid, m.name, m.name_ko, m.smiles, m.molecular_weight as mw,
               m.logp as "logP", m.hbd, m.hba, m.rotatable_bonds as rotatable,
               m.rings, m.aromatic_rings, m.functional_groups, m.odor_strength,
               m.source,
               COALESCE(array_agg(DISTINCT od.name) FILTER (WHERE od.name IS NOT NULL), '{}') as odor_labels
        FROM molecules m
        LEFT JOIN molecule_odors mo ON m.id = mo.molecule_id
        LEFT JOIN odor_descriptors od ON mo.descriptor_id = od.id
        GROUP BY m.id
        ORDER BY m.id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    return [dict(r) for r in cur.fetchall()]


def get_molecule_by_id(mol_id):
    """ID로 분자 조회"""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT m.*, 
               COALESCE(array_agg(DISTINCT od.name) FILTER (WHERE od.name IS NOT NULL), '{}') as odor_labels
        FROM molecules m
        LEFT JOIN molecule_odors mo ON m.id = mo.molecule_id
        LEFT JOIN odor_descriptors od ON mo.descriptor_id = od.id
        WHERE m.id = %s
        GROUP BY m.id
    """, (mol_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def get_molecule_by_name(name):
    """이름으로 분자 조회"""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT m.*, 
               COALESCE(array_agg(DISTINCT od.name) FILTER (WHERE od.name IS NOT NULL), '{}') as odor_labels
        FROM molecules m
        LEFT JOIN molecule_odors mo ON m.id = mo.molecule_id
        LEFT JOIN odor_descriptors od ON mo.descriptor_id = od.id
        WHERE m.name ILIKE %s OR m.name_ko ILIKE %s
        GROUP BY m.id LIMIT 1
    """, (f'%{name}%', f'%{name}%'))
    row = cur.fetchone()
    return dict(row) if row else None


def search_molecules(query, limit=20):
    """분자 검색 (이름/SMILES)"""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT m.id, m.name, m.name_ko, m.smiles, m.molecular_weight as mw,
               COALESCE(array_agg(DISTINCT od.name) FILTER (WHERE od.name IS NOT NULL), '{}') as odor_labels
        FROM molecules m
        LEFT JOIN molecule_odors mo ON m.id = mo.molecule_id
        LEFT JOIN odor_descriptors od ON mo.descriptor_id = od.id
        WHERE m.name ILIKE %s OR m.smiles ILIKE %s OR m.name_ko ILIKE %s
        GROUP BY m.id
        LIMIT %s
    """, (f'%{query}%', f'%{query}%', f'%{query}%', limit))
    return [dict(r) for r in cur.fetchall()]


def get_molecules_by_odor(odor_name, limit=50):
    """특정 냄새 descriptor로 분자 조회"""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT m.id, m.name, m.smiles, m.molecular_weight as mw,
               COALESCE(array_agg(DISTINCT od2.name) FILTER (WHERE od2.name IS NOT NULL), '{}') as odor_labels
        FROM molecules m
        JOIN molecule_odors mo ON m.id = mo.molecule_id
        JOIN odor_descriptors od ON mo.descriptor_id = od.id AND od.name = %s
        LEFT JOIN molecule_odors mo2 ON m.id = mo2.molecule_id
        LEFT JOIN odor_descriptors od2 ON mo2.descriptor_id = od2.id
        GROUP BY m.id
        LIMIT %s
    """, (odor_name, limit))
    return [dict(r) for r in cur.fetchall()]


def get_all_descriptors():
    """전체 냄새 descriptor 목록"""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT od.id, od.name, od.category, od.name_ko,
               COUNT(mo.id) as molecule_count
        FROM odor_descriptors od
        LEFT JOIN molecule_odors mo ON od.id = mo.descriptor_id
        GROUP BY od.id
        ORDER BY molecule_count DESC
    """)
    return [dict(r) for r in cur.fetchall()]


def get_all_ingredients():
    """전체 향료 목록"""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM ingredients ORDER BY category, id")
    return [dict(r) for r in cur.fetchall()]


def get_db_stats():
    """DB 통계"""
    conn = get_conn()
    cur = conn.cursor()
    stats = {}
    for table in ['molecules', 'odor_descriptors', 'molecule_odors', 'ingredients']:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cur.fetchone()[0]
    return stats
