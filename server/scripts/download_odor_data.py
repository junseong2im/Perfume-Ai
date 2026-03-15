"""Download and analyze the OpenPOM curated GoodScents + Leffingwell odor dataset"""
import urllib.request, os, csv, sys

URL = "https://raw.githubusercontent.com/ARY2260/openpom/main/openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
DST = os.path.join(os.path.dirname(__file__), "..", "data", "curated_GS_LF_merged_4983.csv")
os.makedirs(os.path.dirname(DST), exist_ok=True)

print("[1] Downloading curated_GS_LF_merged_4983.csv ...")
urllib.request.urlretrieve(URL, DST)
size = os.path.getsize(DST)
print(f"    OK: {size:,} bytes -> {DST}")

print("\n[2] Analyzing dataset ...")
with open(DST, encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

n_cols = len(header)
n_rows = len(rows)
odor_cols = header[1:]  # first column is SMILES
print(f"    Molecules: {n_rows}")
print(f"    Odor descriptors: {len(odor_cols)}")
print(f"    First 20 descriptors: {odor_cols[:20]}")

# Count label distribution
label_counts = {}
for row in rows:
    for col, val in zip(odor_cols, row[1:]):
        if val == "1":
            label_counts[col] = label_counts.get(col, 0) + 1

# Sort by frequency
sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
print(f"\n[3] Top 30 most common odor labels:")
for label, count in sorted_labels[:30]:
    pct = count / n_rows * 100
    print(f"    {label:25s} {count:5d} ({pct:.1f}%)")

print(f"\n[4] Label statistics:")
labels_per_mol = [sum(1 for v in row[1:] if v == "1") for row in rows]
avg_labels = sum(labels_per_mol) / len(labels_per_mol)
print(f"    Avg labels per molecule: {avg_labels:.1f}")
print(f"    Min labels: {min(labels_per_mol)}")
print(f"    Max labels: {max(labels_per_mol)}")

# Check SMILES coverage with our DB
print(f"\n[5] Checking overlap with our DB ...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import database as db
    all_mols = db.get_all_molecules(limit=10000)
    db_smiles = set()
    for mol in all_mols:
        s = mol.get("smiles", "")
        if s:
            db_smiles.add(s)
    
    dataset_smiles = set(row[0] for row in rows)
    overlap = db_smiles & dataset_smiles
    print(f"    Our DB: {len(db_smiles)} molecules")
    print(f"    Dataset: {len(dataset_smiles)} molecules")
    print(f"    Overlap: {len(overlap)} molecules ({len(overlap)/len(dataset_smiles)*100:.1f}%)")
    
    # Try canonical matching
    from rdkit import Chem
    db_canonical = set()
    for s in db_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            db_canonical.add(Chem.MolToSmiles(mol))
    
    dataset_canonical = set()
    for s in dataset_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            dataset_canonical.add(Chem.MolToSmiles(mol))
    
    canonical_overlap = db_canonical & dataset_canonical
    print(f"    Canonical overlap: {len(canonical_overlap)} molecules ({len(canonical_overlap)/len(dataset_canonical)*100:.1f}%)")
except Exception as e:
    print(f"    DB check failed: {e}")

print("\nDone!")
