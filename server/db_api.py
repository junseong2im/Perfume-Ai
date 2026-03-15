"""
향 AI v3 — FastAPI CRUD 서버
=============================
PostgreSQL 향 DB에 대한 REST API

Endpoints:
    GET  /api/molecules               전체 분자 조회 (페이지네이션)
    GET  /api/molecules/{smiles}      단일 분자 조회
    GET  /api/molecules/{smiles}/odor  향 벡터 조회
    GET  /api/recipes                  레시피 목록
    GET  /api/recipes/{id}            레시피 상세
    GET  /api/safety/{smiles}         안전 정보
    GET  /api/export/training         학습 데이터 CSV export
    GET  /api/stats                   DB 통계
"""

import os
import io
import csv
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
import psycopg2
from psycopg2.extras import RealDictCursor

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://fragrance_admin:fragrance_ai_2024@localhost:5433/fragrance"
)


def get_conn():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)


app = FastAPI(
    title="향 AI Database API",
    description="Fragrance AI v3 — 분자/향/레시피/안전 데이터 API",
    version="3.0",
)


# ================================================================
# Molecules
# ================================================================

@app.get("/api/molecules")
def list_molecules(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    search: Optional[str] = None,
    source: Optional[str] = None,
):
    """분자 목록 (페이지네이션)"""
    offset = (page - 1) * size
    conn = get_conn()
    cur = conn.cursor()

    where = "WHERE 1=1"
    params = []
    if search:
        where += " AND (name ILIKE %s OR smiles ILIKE %s)"
        params.extend([f"%{search}%", f"%{search}%"])
    if source:
        where += " AND %s = ANY(sources)"
        params.append(source)

    cur.execute(f"SELECT COUNT(*) as total FROM molecules {where}", params)
    total = cur.fetchone()['total']

    cur.execute(f"""
        SELECT smiles, name, mw, logp, tpsa, has_chiral, sources
        FROM molecules {where}
        ORDER BY smiles
        LIMIT %s OFFSET %s
    """, params + [size, offset])
    rows = cur.fetchall()
    conn.close()

    return {
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,
        "data": rows,
    }


@app.get("/api/molecules/{smiles}")
def get_molecule(smiles: str):
    """단일 분자 상세"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM molecules WHERE smiles = %s", (smiles,))
    mol = cur.fetchone()
    if not mol:
        conn.close()
        raise HTTPException(404, f"Molecule not found: {smiles}")

    cur.execute("SELECT * FROM odor_labels WHERE smiles = %s", (smiles,))
    odor = cur.fetchall()

    cur.execute("SELECT * FROM note_positions WHERE smiles = %s", (smiles,))
    notes = cur.fetchall()

    cur.execute("SELECT * FROM hedonic_scores WHERE smiles = %s", (smiles,))
    hedonic = cur.fetchall()

    conn.close()
    return {"molecule": mol, "odor_labels": odor, "notes": notes, "hedonic": hedonic}


@app.get("/api/molecules/{smiles}/odor")
def get_odor(smiles: str):
    """22d 향 벡터"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT sweet, sour, woody, floral, citrus, spicy, musk, fresh,
               green, warm, fruity, smoky, powdery, aquatic, herbal,
               amber, leather, earthy, ozonic, metallic, fatty, waxy,
               confidence, source
        FROM odor_labels WHERE smiles = %s
    """, (smiles,))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        raise HTTPException(404, f"No odor data for: {smiles}")
    return rows


# ================================================================
# Recipes
# ================================================================

@app.get("/api/recipes")
def list_recipes(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    mood: Optional[str] = None,
    season: Optional[str] = None,
    style: Optional[str] = None,
):
    """레시피 목록"""
    conn = get_conn()
    cur = conn.cursor()

    where = "WHERE 1=1"
    params = []
    if mood:
        where += " AND mood ILIKE %s"
        params.append(f"%{mood}%")
    if season:
        where += " AND season ILIKE %s"
        params.append(f"%{season}%")
    if style:
        where += " AND style ILIKE %s"
        params.append(f"%{style}%")

    cur.execute(f"SELECT COUNT(*) as total FROM recipes {where}", params)
    total = cur.fetchone()['total']

    cur.execute(f"""
        SELECT id, name, style, mood, season, source
        FROM recipes {where}
        ORDER BY id
        LIMIT %s OFFSET %s
    """, params + [size, (page - 1) * size])
    rows = cur.fetchall()
    conn.close()

    return {"total": total, "page": page, "data": rows}


@app.get("/api/recipes/{recipe_id}")
def get_recipe(recipe_id: int):
    """레시피 상세 (원료 포함)"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM recipes WHERE id = %s", (recipe_id,))
    recipe = cur.fetchone()
    if not recipe:
        conn.close()
        raise HTTPException(404, f"Recipe not found: {recipe_id}")

    cur.execute("""
        SELECT ingredient, ratio, note_type, smiles
        FROM recipe_items WHERE recipe_id = %s ORDER BY ratio DESC
    """, (recipe_id,))
    items = cur.fetchall()
    conn.close()

    return {"recipe": recipe, "ingredients": items}


# ================================================================
# Safety
# ================================================================

@app.get("/api/safety/{smiles}")
def get_safety(smiles: str):
    """분자 안전 정보"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM safety_limits WHERE smiles = %s", (smiles,))
    limits = cur.fetchall()

    # Also check category-level limits
    cur.execute("SELECT * FROM safety_limits WHERE smiles IS NULL ORDER BY ifra_cat")
    cat_limits = cur.fetchall()
    conn.close()

    return {"molecule_limits": limits, "category_limits": cat_limits}


# ================================================================
# Export & Stats
# ================================================================

@app.get("/api/export/training")
def export_training(format: str = Query("csv", regex="^(csv|json)$")):
    """학습 데이터 export (training_export 뷰)"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM training_export LIMIT 100000")
    rows = cur.fetchall()
    conn.close()

    if format == "json":
        return rows

    # CSV streaming
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=training_data.csv"},
    )


@app.get("/api/stats")
def get_stats():
    """DB 통계"""
    conn = get_conn()
    cur = conn.cursor()

    stats = {}
    for table in ['molecules', 'odor_labels', 'note_positions',
                   'receptor_binds', 'hedonic_scores', 'recipes',
                   'recipe_items', 'safety_limits']:
        try:
            cur.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = cur.fetchone()['count']
        except:
            stats[table] = 0

    # Source distribution
    try:
        cur.execute("""
            SELECT unnest(sources) as source, COUNT(*) as count
            FROM molecules GROUP BY source ORDER BY count DESC
        """)
        stats['sources'] = cur.fetchall()
    except:
        stats['sources'] = []

    conn.close()
    return stats


@app.get("/")
def root():
    return {
        "name": "향 AI Database API",
        "version": "3.0",
        "endpoints": [
            "/api/molecules", "/api/recipes",
            "/api/safety/{smiles}", "/api/export/training",
            "/api/stats",
        ]
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
