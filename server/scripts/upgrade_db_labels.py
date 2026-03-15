"""Analyze current DB state and upgrade odor labels with real GoodScents+Leffingwell data"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import database as db
from rdkit import Chem

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "curated_GS_LF_merged_4983.csv")

def analyze_current_db():
    """Analyze current DB state"""
    print("=" * 60)
    print("  CURRENT DB STATE")
    print("=" * 60)
    
    stats = db.get_db_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
    
    descs = db.get_all_descriptors()
    print(f"\n  Top 20 descriptors ({len(descs)} total):")
    for d in descs[:20]:
        print(f"    {d['name']:25s} molecules: {d['molecule_count']:5d}")
    
    # Check how many molecules have vs don't have labels
    mols = db.get_all_molecules(limit=10000)
    has_labels = sum(1 for m in mols if m.get('odor_labels') and m['odor_labels'] != ['{}'] and len(m['odor_labels']) > 0 and m['odor_labels'][0] != '')
    no_labels = len(mols) - has_labels
    print(f"\n  Molecules with labels: {has_labels}")
    print(f"  Molecules without labels: {no_labels}")
    
    return mols


def analyze_matching(mols):
    """Check how many DB molecules match the GS+LF dataset"""
    print("\n" + "=" * 60)
    print("  MATCHING ANALYSIS")
    print("=" * 60)
    
    # Load GS+LF dataset
    with open(DATA_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    odor_cols = header[1:]
    
    # Build canonical SMILES → labels map from dataset
    gs_labels = {}
    for row in rows:
        smiles = row[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            active = [col for col, val in zip(odor_cols, row[1:]) if val == "1"]
            gs_labels[canonical] = active
    
    print(f"  GS+LF dataset: {len(gs_labels)} canonical SMILES")
    
    # Match against DB
    matched = 0
    unmatched = 0
    unmatched_examples = []
    
    for m in mols:
        smiles = m.get("smiles", "")
        if not smiles:
            unmatched += 1
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            if canonical in gs_labels:
                matched += 1
            else:
                unmatched += 1
                if len(unmatched_examples) < 10:
                    unmatched_examples.append(m.get("name", "?"))
        else:
            unmatched += 1
    
    print(f"  DB molecules matched: {matched}/{len(mols)} ({matched/len(mols)*100:.1f}%)")
    print(f"  DB molecules unmatched: {unmatched}")
    if unmatched_examples:
        print(f"  Unmatched examples: {unmatched_examples}")
    
    return gs_labels


def upgrade_labels(mols, gs_labels):
    """Replace DB odor labels with real GS+LF data"""
    print("\n" + "=" * 60)
    print("  UPGRADING DB LABELS")
    print("=" * 60)
    
    conn = db.get_conn()
    cur = conn.cursor()
    
    # Step 1: Get all current descriptors
    cur.execute("SELECT id, name FROM odor_descriptors")
    existing_descs = {name.lower(): id for id, name in cur.fetchall()}
    print(f"  Existing descriptors: {len(existing_descs)}")
    
    # Step 2: Collect all unique labels from GS+LF
    all_gs_labels = set()
    for labels in gs_labels.values():
        all_gs_labels.update(l.lower() for l in labels)
    print(f"  GS+LF unique labels: {len(all_gs_labels)}")
    
    # Step 3: Add missing descriptors to DB
    new_descs = all_gs_labels - set(existing_descs.keys())
    if new_descs:
        print(f"  Adding {len(new_descs)} new descriptors: {sorted(new_descs)[:20]}...")
        for desc_name in sorted(new_descs):
            cur.execute(
                "INSERT INTO odor_descriptors (name, category) VALUES (%s, %s) RETURNING id",
                (desc_name, 'real_gs_lf')
            )
            existing_descs[desc_name] = cur.fetchone()[0]
    
    # Step 4: Clear ALL existing molecule_odors (remove synthetic SMARTS labels)
    cur.execute("SELECT COUNT(*) FROM molecule_odors")
    old_count = cur.fetchone()[0]
    cur.execute("DELETE FROM molecule_odors")
    print(f"  Cleared {old_count:,} old molecule-odor associations")
    
    # Step 5: Insert real labels for matched molecules
    inserted = 0
    matched = 0
    for m in mols:
        smiles = m.get("smiles", "")
        if not smiles:
            continue
        
        mol_obj = Chem.MolFromSmiles(smiles)
        if not mol_obj:
            continue
        
        canonical = Chem.MolToSmiles(mol_obj)
        if canonical not in gs_labels:
            continue
        
        mol_id = m["id"]
        labels = gs_labels[canonical]
        matched += 1
        
        for label in labels:
            label_lower = label.lower()
            if label_lower in existing_descs:
                desc_id = existing_descs[label_lower]
                cur.execute(
                    "INSERT INTO molecule_odors (molecule_id, descriptor_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (mol_id, desc_id)
                )
                inserted += 1
    
    conn.commit()
    
    print(f"  Matched molecules: {matched}")
    print(f"  Inserted associations: {inserted:,}")
    
    # Step 6: Verify
    cur.execute("SELECT COUNT(*) FROM molecule_odors")
    new_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT molecule_id) FROM molecule_odors")
    mol_with_labels = cur.fetchone()[0]
    
    print(f"\n  === RESULT ===")
    print(f"  molecule_odors: {old_count:,} → {new_count:,}")
    print(f"  Molecules with labels: {mol_with_labels}")
    
    # Top descriptors after upgrade
    descs = db.get_all_descriptors()
    print(f"\n  Top 20 descriptors (after upgrade):")
    for d in descs[:20]:
        print(f"    {d['name']:25s} molecules: {d['molecule_count']:5d}")


if __name__ == "__main__":
    mols = analyze_current_db()
    gs_labels = analyze_matching(mols)
    
    print("\n" + "~" * 60)
    confirm = input("  Proceed with upgrade? (yes/no): ").strip().lower()
    if confirm == "yes":
        upgrade_labels(mols, gs_labels)
        print("\n  DONE! DB labels upgraded to real GoodScents+Leffingwell data.")
    else:
        print("  Aborted.")
