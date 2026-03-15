"""
Full Data Audit: Find ALL data subsets being used when full data is available
"""
import json, os, csv, glob

BASE = os.path.join(os.path.dirname(__file__), '..')

def audit():
    print("=" * 60)
    print("  FULL DATA AUDIT")
    print("=" * 60)

    # 1. IFRA
    print("\n--- 1. IFRA ---")
    ifra_files = {
        'ifra_51st_official.json': 'data/pom_upgrade/ifra_51st_official.json',
        'ifra_51st_cat4.json': 'data/pom_upgrade/ifra_51st_cat4.json',
        'ifra_51st_full.json': 'data/pom_upgrade/ifra_51st_full.json',
        'ifra_limits.json': 'data/processed/ifra_limits.json',
    }
    for name, path in ifra_files.items():
        fp = os.path.join(BASE, path)
        if os.path.exists(fp):
            d = json.load(open(fp, 'r', encoding='utf-8'))
            size = len(d)
            print(f"  {name}: {size} entries")
        else:
            print(f"  {name}: NOT FOUND")

    # 2. Pyrfume datasets
    print("\n--- 2. Pyrfume Datasets ---")
    pyrfume_dir = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all')
    if os.path.exists(pyrfume_dir):
        for subdir in sorted(os.listdir(pyrfume_dir)):
            subpath = os.path.join(pyrfume_dir, subdir)
            if os.path.isdir(subpath):
                files = os.listdir(subpath)
                mol_count = 0
                beh_count = 0
                for f in files:
                    fp = os.path.join(subpath, f)
                    if f == 'molecules.csv':
                        with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                            mol_count = sum(1 for _ in fh) - 1
                    elif f.startswith('behavior') and f.endswith('.csv'):
                        with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                            beh_count = sum(1 for _ in fh) - 1
                print(f"  {subdir}: {mol_count} molecules, {beh_count} behavior rows, files={files}")
    else:
        print(f"  pyrfume_all dir NOT FOUND")

    # Also check raw pyrfume
    raw_pyr = os.path.join(BASE, 'data', 'raw', 'pyrfume')
    if os.path.exists(raw_pyr):
        print(f"  raw/pyrfume:")
        for f in sorted(os.listdir(raw_pyr)):
            fp = os.path.join(raw_pyr, f)
            size = os.path.getsize(fp)
            if f.endswith('.csv'):
                with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                    rows = sum(1 for _ in fh) - 1
                print(f"    {f}: {rows} rows ({size//1024}KB)")
            else:
                print(f"    {f}: {size//1024}KB")

    # 3. Abraham ODT
    print("\n--- 3. Abraham ODT ---")
    abr_dir = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all', 'abraham_2012')
    if os.path.exists(abr_dir):
        for f in os.listdir(abr_dir):
            fp = os.path.join(abr_dir, f)
            if f.endswith('.csv'):
                with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                    rows = sum(1 for _ in fh) - 1
                print(f"  {f}: {rows} rows")

    # 4. Predicted ODT / QSPR
    print("\n--- 4. ODT / QSPR ---")
    odt_files = ['data/pom_upgrade/predicted_odt.json', 'data/pom_upgrade/dream_odt.json']
    for path in odt_files:
        fp = os.path.join(BASE, path)
        if os.path.exists(fp):
            d = json.load(open(fp, 'r', encoding='utf-8'))
            if isinstance(d, dict):
                odt_map = d.get('odt_map', d)
                print(f"  {os.path.basename(path)}: {len(odt_map)} entries")
            else:
                print(f"  {os.path.basename(path)}: {len(d)} entries")

    # 5. POM Embeddings
    print("\n--- 5. POM Embeddings ---")
    emb_files = ['data/pom_5050_db.json', 'data/pom_upgrade/pom_master_138d.json']
    for path in emb_files:
        fp = os.path.join(BASE, path)
        if os.path.exists(fp):
            d = json.load(open(fp, 'r', encoding='utf-8'))
            if isinstance(d, dict):
                if 'embeddings' in d:
                    print(f"  {os.path.basename(path)}: {len(d['embeddings'])} embeddings")
                elif 'smiles_to_idx' in d:
                    print(f"  {os.path.basename(path)}: {len(d['smiles_to_idx'])} molecules")
                else:
                    print(f"  {os.path.basename(path)}: {len(d)} entries, keys={list(d.keys())[:5]}")

    # 6. Curated CSV
    print("\n--- 6. Curated CSV ---")
    for pattern in ['data/curated_GS_LF_merged*.csv', 'data/*.csv']:
        for fp in glob.glob(os.path.join(BASE, pattern)):
            with open(fp, 'r', encoding='utf-8-sig', errors='replace') as f:
                rows = sum(1 for _ in f) - 1
            name = os.path.basename(fp)
            print(f"  {name}: {rows} rows")

    # 7. Ingredients
    print("\n--- 7. Ingredients ---")
    for path in ['data/ingredients.json', '../data/ingredients.json']:
        fp = os.path.join(BASE, path)
        if os.path.exists(fp):
            d = json.load(open(fp, 'r', encoding='utf-8'))
            print(f"  {path}: {len(d)} entries")
            # Check sources
            sources = {}
            for x in d:
                s = x.get('source', 'original')
                sources[s] = sources.get(s, 0) + 1
            print(f"    sources: {sources}")
            # IFRA coverage
            ifra_tagged = sum(1 for x in d if x.get('ifra_cas'))
            print(f"    IFRA tagged: {ifra_tagged}")

    # 8. Accords / compatibility
    print("\n--- 8. Accords & Compatibility ---")
    for path in ['../data/accords.json', '../data/compatibility.json']:
        fp = os.path.join(BASE, path)
        if os.path.exists(fp):
            d = json.load(open(fp, 'r', encoding='utf-8'))
            print(f"  {os.path.basename(path)}: {len(d)} entries")

    # 9. GoodScents data
    print("\n--- 9. GoodScents ---")
    gs_patterns = ['data/goodscents*.csv', 'data/*GS*.csv', 'data/*gs*.csv']
    for pattern in gs_patterns:
        for fp in glob.glob(os.path.join(BASE, pattern)):
            name = os.path.basename(fp)
            with open(fp, 'r', encoding='utf-8-sig', errors='replace') as f:
                rows = sum(1 for _ in f) - 1
            print(f"  {name}: {rows} rows")

    # 10. DREAM data
    print("\n--- 10. DREAM ---")
    dream_patterns = ['data/DREAM*.csv', 'data/pom_upgrade/dream*.json']
    for pattern in dream_patterns:
        for fp in glob.glob(os.path.join(BASE, pattern)):
            name = os.path.basename(fp)
            if fp.endswith('.csv'):
                with open(fp, 'r', encoding='utf-8-sig', errors='replace') as f:
                    rows = sum(1 for _ in f) - 1
                print(f"  {name}: {rows} rows")
            elif fp.endswith('.json'):
                d = json.load(open(fp, 'r', encoding='utf-8'))
                print(f"  {name}: {len(d)} entries")

if __name__ == '__main__':
    audit()
