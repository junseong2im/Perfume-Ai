# -*- coding: utf-8 -*-
"""
V12 Phase 1+2: Data Expansion & Label Quality Improvement
==========================================================
1. Download pyrfume-data datasets from GitHub (Leffingwell, GoodScents, DREAM, etc.)
2. Match to unlabeled DB molecules by CID or canonical SMILES
3. Insert new odor labels into DB
4. Add well-known rare-class molecules (aquatic, ozonic, musk)
5. Fix unmapped labels (caraway, turpentine)
6. Report data quality metrics

Usage:
    python scripts/fetch_pyrfume_labels.py
"""
import sys, os, io, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
import numpy as np
import requests
from collections import Counter, defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')

import database as db
import psycopg2
import psycopg2.extras

DB_CONFIG = db.DB_CONFIG

# ================================================================
# PYRFUME-DATA GITHUB RAW URLS
# ================================================================
PYRFUME_BASE = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main"

DATASETS = {
    'leffingwell': {
        'molecules': f"{PYRFUME_BASE}/leffingwell/molecules.csv",
        'behavior':  f"{PYRFUME_BASE}/leffingwell/behavior.csv",
    },
    'goodscents': {
        'molecules': f"{PYRFUME_BASE}/goodscents/molecules.csv",
        'behavior':  f"{PYRFUME_BASE}/goodscents/behavior.csv",
    },
    'arctander': {
        'molecules': f"{PYRFUME_BASE}/arctander/molecules.csv",
        'behavior':  f"{PYRFUME_BASE}/arctander/behavior.csv",
    },
    'sigma_ff': {
        'molecules': f"{PYRFUME_BASE}/sigma_2014/molecules.csv",
        'behavior':  f"{PYRFUME_BASE}/sigma_2014/behavior.csv",
    },
    'ifra_2019': {
        'molecules': f"{PYRFUME_BASE}/ifra_2019/molecules.csv",
        'behavior':  f"{PYRFUME_BASE}/ifra_2019/behavior.csv",
    },
}

# ================================================================
# RARE CLASS MOLECULES (well-known reference compounds)
# ================================================================
RARE_CLASS_MOLECULES = [
    # AQUATIC — completely missing (0%)
    {"smiles": "O=C1Cc2ccccc2CC1=O", "name": "Calone (7-methyl-2H-1,5-benzodioxepin-3(4H)-one)",
     "labels": ["aquatic", "marine", "fresh", "ozonic", "melon"]},
    {"smiles": "CC(C)=CCCC(C)=CCOC", "name": "Dihydromyrcenol",
     "labels": ["fresh", "citrus", "aquatic"]},
    {"smiles": "O=C(CC1CC=CC1)c1ccccc1", "name": "Hedione (methyl dihydrojasmonate)",
     "labels": ["floral", "jasmine", "aquatic", "fresh"]},
    {"smiles": "CCCCCCCC/C=C\\CCCCCCCC(=O)OC", "name": "Methyl cis-9-octadecenoate",
     "labels": ["aquatic", "waxy", "oily"]},
    {"smiles": "O=C1CCCCCCCCC/C=C\\1", "name": "Cyclododecanone",
     "labels": ["aquatic", "woody", "musk"]},
    
    # OZONIC — very rare (0.7%)
    {"smiles": "O=CC=CC(=O)O", "name": "trans-2-Pentenal",
     "labels": ["ozonic", "green", "fruity"]},
    {"smiles": "CC(=O)/C=C/C", "name": "3-Penten-2-one",
     "labels": ["ozonic", "sweet", "fruity"]},
    {"smiles": "CCCCC=O", "name": "Pentanal (valeraldehyde)",
     "labels": ["ozonic", "green", "aldehydic"]},
    
    # MUSK — rare (4.9%)
    {"smiles": "CC1CCCCCCCCCCC(=O)C(C)C1", "name": "Muscone",
     "labels": ["musk", "animal", "powdery"]},
    {"smiles": "CC1(C)CC(=O)c2cc(C(C)(C)C)ccc2O1", "name": "Tonalide",
     "labels": ["musk", "sweet", "powdery"]},
    {"smiles": "CC12CCC(CC1)C(C)(C)OC2=O", "name": "Ambroxide",
     "labels": ["amber", "musk", "woody"]},
    {"smiles": "O=C1CCCCCCCCCCCCC1", "name": "Cyclopentadecanone (Exaltone)",
     "labels": ["musk", "animal", "sweet"]},
    {"smiles": "O=C1CCCCCCCCCCCCCC1", "name": "Cyclohexadecanone",
     "labels": ["musk", "powdery", "sweet"]},
    
    # POWDERY — rare (5.9%)
    {"smiles": "O=Cc1ccc(OC)c(OC)c1", "name": "Veratraldehyde",
     "labels": ["powdery", "sweet", "vanilla"]},
    {"smiles": "COc1ccc(C=O)cc1OC", "name": "3,4-Dimethoxybenzaldehyde",
     "labels": ["powdery", "vanilla", "sweet"]},
]


def download_csv(url, name=""):
    """Download CSV from URL with retries."""
    for attempt in range(3):
        try:
            print(f"    Downloading {name or url}...", end=" ", flush=True)
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                print(f"OK ({len(df)} rows)")
                return df
            elif resp.status_code == 404:
                print(f"404 Not Found")
                return None
            else:
                print(f"HTTP {resp.status_code}")
        except Exception as e:
            print(f"Error: {e}")
            if attempt < 2:
                time.sleep(2)
    return None


def canonicalize(smi):
    """Convert SMILES to canonical form."""
    if not smi or pd.isna(smi):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except:
        return None


def extract_labels_from_behavior(behavior_df, mol_df):
    """
    Extract odor labels from pyrfume behavior data.
    Different datasets have different formats — handle them all.
    """
    labels_by_cid = defaultdict(set)
    labels_by_smi = defaultdict(set)
    
    if behavior_df is None or mol_df is None:
        return labels_by_cid, labels_by_smi
    
    # Build CID→SMILES mapping from molecules
    cid_to_smi = {}
    if 'CID' in mol_df.columns:
        smi_col = None
        for c in ['IsomericSMILES', 'SMILES', 'smiles', 'nonStereoSMILES', 'CanonicalSMILES']:
            if c in mol_df.columns:
                smi_col = c
                break
        if smi_col:
            for _, row in mol_df.iterrows():
                cid = row.get('CID')
                smi = canonicalize(row.get(smi_col, ''))
                if cid and smi:
                    cid_to_smi[int(cid)] = smi
    
    # Check behavior format
    bcols = list(behavior_df.columns)
    
    # Format 1: Wide format (columns = descriptors, values = intensity)
    # e.g., leffingwell: CID, descriptor1, descriptor2, ...
    if 'CID' in bcols:
        id_col = 'CID'
    elif 'Stimulus' in bcols:
        id_col = 'Stimulus'
    else:
        id_col = bcols[0]
    
    descriptor_cols = [c for c in bcols if c not in [id_col, 'CID', 'Stimulus', 'Subject', 
                       'IsomericSMILES', 'SMILES', 'smiles', 'name', 'Name',
                       'MolecularWeight', 'IUPACName', 'CanonicalSMILES']]
    
    # Check if it's a behavior matrix (wide) or long format
    if len(descriptor_cols) > 5:
        # Wide format: each column is a descriptor, value indicates presence/intensity
        for _, row in behavior_df.iterrows():
            mol_id = row.get(id_col)
            smi = cid_to_smi.get(int(mol_id), None) if mol_id and not pd.isna(mol_id) else None
            
            for desc in descriptor_cols:
                val = row.get(desc)
                if pd.notna(val) and val != 0 and val != '' and val != 'nan':
                    # Normalize descriptor name
                    label = str(desc).lower().strip().replace('_', ' ')
                    if smi:
                        labels_by_smi[smi].add(label)
                    if mol_id and not pd.isna(mol_id):
                        labels_by_cid[int(mol_id)].add(label)
    else:
        # Long format: each row is (molecule, descriptor, value)
        desc_col = None
        for c in ['Descriptor', 'descriptor', 'Label', 'label', 'Words', 'words']:
            if c in bcols:
                desc_col = c
                break
        if desc_col:
            for _, row in behavior_df.iterrows():
                mol_id = row.get(id_col)
                label = str(row.get(desc_col, '')).lower().strip()
                if label and label != 'nan':
                    smi = cid_to_smi.get(int(mol_id), None) if mol_id and not pd.isna(mol_id) else None
                    if smi:
                        labels_by_smi[smi].add(label)
                    if mol_id and not pd.isna(mol_id):
                        labels_by_cid[int(mol_id)].add(label)
    
    return labels_by_cid, labels_by_smi


def get_db_state():
    """Get current DB state for matching."""
    mols = db.get_all_molecules(limit=50000)
    
    # Build canonical SMILES → molecule mapping
    smi_to_mol = {}
    cid_to_mol = {}
    labeled_smiles = set()
    
    for m in mols:
        smi = m.get('smiles', '')
        can = canonicalize(smi)
        if can:
            smi_to_mol[can] = m
        
        cid = m.get('cid')
        if cid:
            cid_to_mol[int(cid)] = m
        
        if m.get('odor_labels') and len(m['odor_labels']) > 0 and m['odor_labels'][0] != '':
            if can:
                labeled_smiles.add(can)
    
    return mols, smi_to_mol, cid_to_mol, labeled_smiles


def insert_labels(conn, mol_id, labels):
    """Insert odor labels for a molecule into DB."""
    cur = conn.cursor()
    inserted = 0
    for label in labels:
        label = label.lower().strip()
        if not label or label == 'nan' or len(label) < 2:
            continue
        try:
            # Ensure descriptor exists
            cur.execute("""
                INSERT INTO odor_descriptors (name) VALUES (%s)
                ON CONFLICT (name) DO NOTHING
            """, (label,))
            
            # Get descriptor ID
            cur.execute("SELECT id FROM odor_descriptors WHERE name = %s", (label,))
            desc_row = cur.fetchone()
            if desc_row:
                # Insert molecule-odor link
                cur.execute("""
                    INSERT INTO molecule_odors (molecule_id, descriptor_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (mol_id, desc_row[0]))
                inserted += 1
        except Exception as e:
            pass
    return inserted


def insert_molecule_with_labels(conn, smiles, name, labels, source='pyrfume_expand'):
    """Insert a new molecule with its labels."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    
    can_smi = Chem.MolToSmiles(mol)
    mw = round(Descriptors.ExactMolWt(mol), 2)
    
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO molecules (name, smiles, molecular_weight, source)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (smiles) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
        """, (name, can_smi, mw, source))
        mol_id = cur.fetchone()[0]
        insert_labels(conn, mol_id, labels)
        return True
    except Exception as e:
        return False


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print("=" * 65)
    print("  V12 Phase 1+2: Data Expansion & Label Quality Improvement")
    print("=" * 65)
    
    # 1) Get current DB state
    print("\n📊 Current DB state...")
    mols, smi_to_mol, cid_to_mol, labeled_smiles = get_db_state()
    print(f"  Total molecules: {len(mols)}")
    print(f"  Labeled: {len(labeled_smiles)}")
    print(f"  Unlabeled: {len(mols) - len(labeled_smiles)}")
    
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    
    total_new_labels = 0
    total_new_molecules = 0
    
    # 2) Download pyrfume datasets
    print(f"\n{'='*65}")
    print(f"  📥 Downloading pyrfume-data datasets from GitHub")
    print(f"{'='*65}")
    
    for ds_name, urls in DATASETS.items():
        print(f"\n  [{ds_name}]")
        mol_df = download_csv(urls['molecules'], f"{ds_name}/molecules")
        beh_df = download_csv(urls['behavior'], f"{ds_name}/behavior")
        
        if mol_df is None or beh_df is None:
            print(f"    ⚠️ Skipping {ds_name} (data not available)")
            continue
        
        print(f"    Molecules: {len(mol_df)} | Behavior: {len(beh_df)}")
        print(f"    Mol columns: {list(mol_df.columns)[:5]}")
        print(f"    Behavior columns: {list(beh_df.columns)[:5]}")
        
        # Extract labels
        labels_by_cid, labels_by_smi = extract_labels_from_behavior(beh_df, mol_df)
        
        print(f"    Extracted: {len(labels_by_smi)} molecules with labels")
        
        # Match to DB molecules
        matched = 0
        new_labels_added = 0
        
        for can_smi, labels in labels_by_smi.items():
            if can_smi in smi_to_mol:
                mol_entry = smi_to_mol[can_smi]
                existing_labels = set(l.lower() for l in mol_entry.get('odor_labels', []) if l)
                new_labels = labels - existing_labels
                
                if new_labels:
                    n = insert_labels(conn, mol_entry['id'], new_labels)
                    new_labels_added += n
                    matched += 1
        
        if matched:
            print(f"    ✅ Matched {matched} DB molecules, added {new_labels_added} new labels")
            total_new_labels += new_labels_added
        
        # Also check for new molecules (in pyrfume but not in our DB)
        smi_col = None
        for c in ['IsomericSMILES', 'SMILES', 'smiles', 'nonStereoSMILES', 'CanonicalSMILES']:
            if c in mol_df.columns:
                smi_col = c
                break
        
        if smi_col:
            new_mols = 0
            for _, row in mol_df.iterrows():
                smi = canonicalize(row.get(smi_col, ''))
                if smi and smi not in smi_to_mol:
                    # Check if this molecule has labels
                    mol_labels = labels_by_smi.get(smi, set())
                    if mol_labels and len(mol_labels) >= 1:
                        name = str(row.get('Name', row.get('name', row.get('IUPACName', 'unknown'))))
                        if insert_molecule_with_labels(conn, smi, name, mol_labels, f'pyrfume_{ds_name}'):
                            smi_to_mol[smi] = {'smiles': smi}  # Track as known
                            new_mols += 1
                            total_new_molecules += 1
            if new_mols:
                print(f"    ✅ Added {new_mols} NEW molecules with labels")
    
    # 3) Add rare class molecules
    print(f"\n{'='*65}")
    print(f"  🧪 Adding rare class reference molecules")
    print(f"{'='*65}")
    
    rare_added = 0
    for entry in RARE_CLASS_MOLECULES:
        smi = canonicalize(entry['smiles'])
        if smi and smi not in smi_to_mol:
            if insert_molecule_with_labels(conn, entry['smiles'], entry['name'], 
                                            entry['labels'], 'rare_class_ref'):
                rare_added += 1
                smi_to_mol[smi] = {'smiles': smi}
                total_new_molecules += 1
        elif smi and smi in smi_to_mol:
            # Molecule exists — add labels if missing
            mol_entry = smi_to_mol[smi]
            n = insert_labels(conn, mol_entry['id'], entry['labels'])
            if n > 0:
                rare_added += 1
                total_new_labels += n
    
    print(f"  ✅ Rare class molecules processed: {rare_added}")
    
    # 4) Fix unmapped labels
    print(f"\n{'='*65}")
    print(f"  🔧 Fixing unmapped labels")
    print(f"{'='*65}")
    
    # Add caraway and turpentine as descriptors if they don't exist
    cur = conn.cursor()
    for label in ['caraway', 'turpentine']:
        cur.execute("INSERT INTO odor_descriptors (name) VALUES (%s) ON CONFLICT DO NOTHING", (label,))
    print("  ✅ Ensured caraway and turpentine descriptors exist")
    
    # 5) Final report
    print(f"\n{'='*65}")
    print(f"  📊 V12 Data Expansion Report")
    print(f"{'='*65}")
    
    # Re-check DB state
    mols2, _, _, labeled_smiles2 = get_db_state()
    
    print(f"  Before: {len(mols)} molecules, {len(labeled_smiles)} labeled")
    print(f"  After:  {len(mols2)} molecules, {len(labeled_smiles2)} labeled")
    print(f"  New molecules added: {total_new_molecules}")
    print(f"  New labels added:    {total_new_labels}")
    print(f"  New labeled molecules: {len(labeled_smiles2) - len(labeled_smiles)}")
    
    # Check class distribution
    all_labels = Counter()
    for m in mols2:
        for l in m.get('odor_labels', []):
            if l:
                all_labels[l.lower()] += 1
    
    print(f"\n  Top 22d-relevant label counts:")
    dims = ['sweet','sour','woody','floral','citrus','spicy','musk','fresh','green','warm',
            'fruity','smoky','powdery','aquatic','herbal','amber','leather','earthy','ozonic','metallic',
            'fatty','waxy']
    from scripts.label_mapping import HARD_MAPPING
    for dim in dims:
        mapped_labels = [l for l, d in HARD_MAPPING.items() if d == dim]
        count = sum(all_labels.get(l, 0) for l in mapped_labels + [dim])
        print(f"    {dim:>10s}: {count:4d}")
    
    conn.close()
    print(f"\n✅ V12 Phase 1+2 complete!")
