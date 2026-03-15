"""
데이터 import 스크립트 — JSON/CSV → PostgreSQL
==============================================
Docker DB가 실행된 후 데이터를 일괄 import합니다.

사용법:
    python scripts/import_data.py                     # 전체 import
    python scripts/import_data.py --molecules-only    # 분자만
    python scripts/import_data.py --recipes-only      # 레시피만
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://fragrance_admin:fragrance_ai_2024@localhost:5433/fragrance"
)

ODOR_DIMS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]


def get_conn():
    return psycopg2.connect(DB_URL)


# ================================================================
# 1. 분자 기본 정보 import
# ================================================================
def import_molecules(conn):
    """unified_molecules.json + rdkit_properties.json → molecules 테이블"""
    print("\n=== Importing molecules ===")

    # Load unified molecules
    unified_path = PROCESSED_DIR / "unified_molecules.json"
    if not unified_path.exists():
        print("  [SKIP] No unified_molecules.json — run collect_data.py --unify first")
        return

    with open(unified_path, encoding='utf-8') as f:
        molecules = json.load(f)

    # Load RDKit properties
    props = {}
    props_path = PROCESSED_DIR / "rdkit_properties.json"
    if props_path.exists():
        with open(props_path) as f:
            props = json.load(f)

    # Insert
    cur = conn.cursor()
    inserted = 0
    skipped = 0

    for mol in molecules:
        smi = mol['smiles']
        p = props.get(smi, {})
        sources = mol.get('sources', [])

        try:
            cur.execute("""
                INSERT INTO molecules (smiles, canonical, name, mw, logp, tpsa,
                    hbd, hba, rotatable, rings, aromatic_rings, heavy_atoms,
                    fsp3, has_chiral, n_chiral, sources)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (smiles) DO UPDATE SET
                    name = COALESCE(EXCLUDED.name, molecules.name),
                    mw = COALESCE(EXCLUDED.mw, molecules.mw),
                    logp = COALESCE(EXCLUDED.logp, molecules.logp),
                    sources = EXCLUDED.sources
            """, (
                smi, smi, mol.get('name', ''),
                p.get('mw'), p.get('logp'), p.get('tpsa'),
                p.get('hbd'), p.get('hba'), p.get('rotatable'),
                p.get('rings'), p.get('aromatic_rings'), p.get('heavy_atoms'),
                p.get('fsp3'), p.get('has_chiral', False), p.get('n_chiral', 0),
                sources,
            ))
            inserted += 1
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"    Error: {e}")

    conn.commit()
    print(f"  Molecules: {inserted} inserted, {skipped} skipped")


# ================================================================
# 2. 향 벡터 라벨 import
# ================================================================
def import_odor_labels(conn):
    """unified_training_data.csv → odor_labels 테이블"""
    print("\n=== Importing odor labels ===")

    csv_path = DATA_DIR / "unified_training_data.csv"
    if not csv_path.exists():
        print("  [SKIP] No unified_training_data.csv")
        return

    cur = conn.cursor()
    inserted = 0

    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get('smiles', '')
            if not smi:
                continue

            values = []
            valid = True
            for dim in ODOR_DIMS:
                v = row.get(dim, '')
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    valid = False
                    break
            if not valid:
                continue

            confidence = float(row.get('confidence', row.get('weight', 1.0)))
            source = row.get('source', 'local')

            try:
                cur.execute(f"""
                    INSERT INTO odor_labels (smiles, source, {', '.join(ODOR_DIMS)}, confidence)
                    VALUES (%s, %s, {', '.join(['%s'] * 22)}, %s)
                    ON CONFLICT (smiles, source) DO UPDATE SET
                        confidence = EXCLUDED.confidence
                """, [smi, source] + values + [confidence])
                inserted += 1
            except Exception:
                pass

    conn.commit()
    print(f"  Odor labels: {inserted} inserted")


# ================================================================
# 3. 레시피 import
# ================================================================
def import_recipes(conn):
    """recipe_data.json + fragrantica_recipes.json → recipes/recipe_items"""
    print("\n=== Importing recipes ===")
    cur = conn.cursor()
    total_recipes = 0
    total_items = 0

    for recipe_file in ['recipe_data.json', 'fragrantica_recipes.json',
                        'recipe_training_data.json', 'famous_perfumes.json']:
        path = DATA_DIR / recipe_file
        if not path.exists():
            continue

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                recipes = list(data.values()) if isinstance(list(data.values())[0], dict) else [data]
            elif isinstance(data, list):
                recipes = data
            else:
                continue

            for recipe in recipes:
                name = recipe.get('name', recipe.get('recipe_name', 'Unknown'))
                style = recipe.get('style', recipe.get('category', ''))
                mood = recipe.get('mood', '')
                season = recipe.get('season', '')

                try:
                    cur.execute("""
                        INSERT INTO recipes (name, style, mood, season, source)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (name, style, mood, season, recipe_file))
                    recipe_id = cur.fetchone()[0]
                    total_recipes += 1

                    # Insert ingredients
                    ingredients = recipe.get('ingredients', recipe.get('formula', []))
                    if isinstance(ingredients, dict):
                        ingredients = [{'name': k, 'ratio': v} for k, v in ingredients.items()]

                    for ing in ingredients:
                        if isinstance(ing, dict):
                            ing_name = ing.get('name', ing.get('ingredient', ''))
                            ratio = float(ing.get('ratio', ing.get('percentage', ing.get('pct', 0))))
                        elif isinstance(ing, (list, tuple)):
                            ing_name = str(ing[0])
                            ratio = float(ing[1]) if len(ing) > 1 else 0
                        else:
                            continue

                        if ing_name:
                            cur.execute("""
                                INSERT INTO recipe_items (recipe_id, ingredient, ratio)
                                VALUES (%s, %s, %s)
                            """, (recipe_id, ing_name, ratio))
                            total_items += 1

                except Exception:
                    pass

            print(f"  {recipe_file}: processed")

        except Exception as e:
            print(f"  {recipe_file} error: {e}")

    conn.commit()
    print(f"  Recipes: {total_recipes} recipes, {total_items} items")


# ================================================================
# 4. IFRA 안전 데이터 import
# ================================================================
def import_safety(conn):
    """ifra_limits.json → safety_limits"""
    print("\n=== Importing safety limits ===")
    cur = conn.cursor()

    ifra_path = PROCESSED_DIR / "ifra_limits.json"
    if not ifra_path.exists():
        print("  [SKIP] No ifra_limits.json")
        return

    with open(ifra_path) as f:
        ifra = json.load(f)

    inserted = 0
    for cat, max_pct in ifra.get('categories', {}).items():
        try:
            cur.execute("""
                INSERT INTO safety_limits (ifra_cat, max_pct, source)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (cat, max_pct, 'ifra'))
            inserted += 1
        except:
            pass

    conn.commit()
    print(f"  Safety limits: {inserted} inserted")


# ================================================================
# 5. PubChem 물리속성 보강
# ================================================================
def import_pubchem_props(conn):
    """pubchem_properties.json → molecules 테이블 업데이트"""
    print("\n=== Importing PubChem properties ===")
    cur = conn.cursor()

    for props_file in ['pubchem_properties.json', 'rdkit_properties.json']:
        path = PROCESSED_DIR / props_file
        if not path.exists():
            continue

        with open(path) as f:
            props = json.load(f)

        updated = 0
        for smi, p in props.items():
            try:
                cur.execute("""
                    UPDATE molecules SET
                        mw = COALESCE(mw, %s),
                        logp = COALESCE(logp, %s),
                        tpsa = COALESCE(tpsa, %s),
                        hbd = COALESCE(hbd, %s),
                        hba = COALESCE(hba, %s),
                        rotatable = COALESCE(rotatable, %s)
                    WHERE smiles = %s
                """, (
                    p.get('mw'), p.get('logp'), p.get('tpsa'),
                    p.get('hbd'), p.get('hba'), p.get('rotatable'),
                    smi,
                ))
                if cur.rowcount > 0:
                    updated += 1
            except:
                pass

        conn.commit()
        print(f"  {props_file}: {updated} molecules updated")


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="데이터 import to PostgreSQL")
    parser.add_argument('--molecules-only', action='store_true')
    parser.add_argument('--recipes-only', action='store_true')
    parser.add_argument('--db-url', default=None, help='Override DATABASE_URL')
    args = parser.parse_args()

    global DB_URL
    if args.db_url:
        DB_URL = args.db_url

    print("=" * 60)
    print("  향 AI v3 — Data Import")
    print(f"  DB: {DB_URL.split('@')[1] if '@' in DB_URL else DB_URL}")
    print("=" * 60)

    try:
        conn = get_conn()
        print("  Connected ✓")
    except Exception as e:
        print(f"  Connection failed: {e}")
        print(f"  Make sure Docker DB is running: docker-compose up -d")
        sys.exit(1)

    try:
        if args.molecules_only:
            import_molecules(conn)
        elif args.recipes_only:
            import_recipes(conn)
        else:
            import_molecules(conn)
            import_odor_labels(conn)
            import_recipes(conn)
            import_safety(conn)
            import_pubchem_props(conn)
    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("  Import 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
