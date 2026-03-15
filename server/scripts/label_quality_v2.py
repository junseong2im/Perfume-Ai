# -*- coding: utf-8 -*-
"""
Label Quality Enhancement — Phase 4
=====================================
Propagates labels from labeled molecules to unlabeled ones
using Morgan fingerprint (Tanimoto) similarity.

Also performs multi-source consensus and label cleaning.

Usage:
    python scripts/label_quality_v2.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from collections import Counter, defaultdict
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

import database as db

t_start = time.time()
print("=" * 70)
print("  ⚗️ Phase 4: Label Quality Enhancement")
print("=" * 70)

conn = db.get_conn()
cur = conn.cursor()

# ===================================================================
# STEP 1: Load all molecules + their labels
# ===================================================================
print("\n📊 Loading molecules...")
molecules = db.get_all_molecules()
print(f"  Total: {len(molecules)}")

labeled = [m for m in molecules if m.get('odor_labels') and m['odor_labels'] != ['{}'] and m['odor_labels'] != ['odorless']]
unlabeled = [m for m in molecules if not m.get('odor_labels') or m['odor_labels'] == ['{}'] or m['odor_labels'] == []]

print(f"  Labeled: {len(labeled)}")
print(f"  Unlabeled: {len(unlabeled)}")

# ===================================================================
# STEP 2: Build fingerprints for labeled molecules
# ===================================================================
print("\n🔬 Building fingerprints for labeled molecules...")

labeled_fps = []
labeled_info = []  # (mol_id, labels, smiles)
n_ok = n_fail = 0

for mol in labeled:
    smi = mol.get('smiles', '')
    if not smi: n_fail += 1; continue
    
    rdmol = Chem.MolFromSmiles(smi)
    if rdmol is None: n_fail += 1; continue
    
    fp = AllChem.GetMorganFingerprintAsBitVect(rdmol, 2, nBits=2048)
    labeled_fps.append(fp)
    labeled_info.append((mol['id'], mol['odor_labels'], smi))
    n_ok += 1

print(f"  Valid fingerprints: {n_ok} (failed: {n_fail})")

# ===================================================================
# STEP 3: Label Propagation via Tanimoto Similarity
# ===================================================================
print("\n🔄 Propagating labels to unlabeled molecules...")
print(f"  Processing {len(unlabeled)} unlabeled molecules...")
print(f"  Similarity threshold: 0.85 (Tanimoto)")

TANIMOTO_THRESHOLD = 0.85
MIN_NEIGHBORS = 2  # Need at least 2 similar labeled molecules
BATCH_SIZE = 5000

n_propagated = 0
n_labels_added = 0
n_processed = 0

for batch_start in range(0, len(unlabeled), BATCH_SIZE):
    batch = unlabeled[batch_start:batch_start+BATCH_SIZE]
    
    for mol in batch:
        smi = mol.get('smiles', '')
        if not smi: continue
        
        rdmol = Chem.MolFromSmiles(smi)
        if rdmol is None: continue
        
        fp = AllChem.GetMorganFingerprintAsBitVect(rdmol, 2, nBits=2048)
        
        # Find similar labeled molecules
        sims = DataStructs.BulkTanimotoSimilarity(fp, labeled_fps)
        
        # Collect labels from similar molecules
        neighbor_labels = Counter()
        n_similar = 0
        
        for i, sim in enumerate(sims):
            if sim >= TANIMOTO_THRESHOLD:
                for label in labeled_info[i][1]:
                    if label and label != '{}':
                        neighbor_labels[label] += 1
                n_similar += 1
        
        # Propagate if enough agreement
        if n_similar >= MIN_NEIGHBORS:
            # Only use labels that appear in majority of neighbors
            threshold = max(2, n_similar * 0.5)
            consensus_labels = [l for l, c in neighbor_labels.items() if c >= threshold]
            
            if consensus_labels:
                mol_id = mol['id']
                for label in consensus_labels:
                    label_clean = label.strip().lower()
                    if not label_clean or label_clean == '{}': continue
                    
                    # Get or create descriptor
                    cur.execute("SELECT id FROM odor_descriptors WHERE name = %s", (label_clean,))
                    row = cur.fetchone()
                    if row:
                        desc_id = row[0]
                    else:
                        cur.execute("INSERT INTO odor_descriptors (name) VALUES (%s) RETURNING id", (label_clean,))
                        desc_id = cur.fetchone()[0]
                    
                    try:
                        cur.execute("""
                            INSERT INTO molecule_odors (molecule_id, descriptor_id, source, strength)
                            VALUES (%s, %s, 'propagated', 0.7)
                            ON CONFLICT (molecule_id, descriptor_id) DO NOTHING
                        """, (mol_id, desc_id))
                        if cur.rowcount > 0: n_labels_added += 1
                    except:
                        conn.rollback()
                
                n_propagated += 1
        
        n_processed += 1
    
    elapsed = time.time() - t_start
    print(f"    [{batch_start+len(batch):>6d}/{len(unlabeled)}] "
          f"Propagated: {n_propagated}, Labels: +{n_labels_added} "
          f"({elapsed:.0f}s)")

print(f"\n  ✅ Label propagation complete:")
print(f"     Molecules labeled: {n_propagated}")
print(f"     Labels added: {n_labels_added}")

# ===================================================================
# STEP 4: Label Cleaning
# ===================================================================
print("\n🧹 Cleaning labels...")

# Remove very rare descriptors that appear in <3 molecules
cur.execute("""
    SELECT od.id, od.name, COUNT(mo.id) as cnt
    FROM odor_descriptors od
    LEFT JOIN molecule_odors mo ON od.id = mo.descriptor_id
    GROUP BY od.id
    HAVING COUNT(mo.id) < 2
""")
rare_descs = cur.fetchall()
print(f"  Rare descriptors (<2 molecules): {len(rare_descs)}")

# Clean up long/invalid descriptor names
cur.execute("SELECT id, name FROM odor_descriptors WHERE LENGTH(name) > 25")
long_descs = cur.fetchall()
print(f"  Long descriptor names (>25 chars): {len(long_descs)}")
for desc_id, name in long_descs:
    # Try to simplify
    simplified = name.split(',')[0].split('(')[0].strip()[:25]
    if simplified != name:
        cur.execute("UPDATE odor_descriptors SET name = %s WHERE id = %s", (simplified, desc_id))

# ===================================================================
# FINAL SUMMARY
# ===================================================================
cur.execute("SELECT COUNT(DISTINCT m.id) FROM molecules m JOIN molecule_odors mo ON m.id=mo.molecule_id")
labeled_after = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM molecule_odors")
links_after = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM odor_descriptors")
descs_after = cur.fetchone()[0]

elapsed = time.time() - t_start

print(f"\n{'='*70}")
print(f"  Phase 4 Complete")
print(f"{'='*70}")
print(f"  Labeled molecules: {len(labeled)} → {labeled_after}")
print(f"  Molecule-odor links: → {links_after}")
print(f"  Descriptors: → {descs_after}")
print(f"  Labels propagated: +{n_labels_added}")
print(f"  Time: {elapsed:.1f}s")
print(f"{'='*70}")
