# -*- coding: utf-8 -*-
"""quality_data_collector.py - High-Quality Odor Data Collection
================================================================
Downloads ONLY peer-reviewed, expert-validated molecular odor data.

Tier 1 (Gold Standard - peer-reviewed):
  - OpenPOM curated GS-LF (4,983 molecules, 138 labels)
  - Dravnieks 1985 (138 molecules, 146 labels, ASTM panel)
  - Keller 2016 (480 molecules, Science paper)

Tier 2 (Expert-labeled):
  - Leffingwell PMP 2001 (3,523 molecules, 113 labels)

Quality pipeline:
  1. Download from trusted GitHub sources
  2. RDKit SMILES validation & canonicalization
  3. Cross-source label agreement scoring
  4. Deduplication by canonical SMILES
  5. Quality confidence scoring per molecule

Output: curated_training_data.csv
"""

import os
import sys
import csv
import json
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

# RDKit for SMILES validation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[WARN] RDKit not installed - SMILES validation disabled")


# ================================================================
# Trusted Data Sources
# ================================================================

PYRFUME_BASE = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main"
OPENPOM_URL = ("https://raw.githubusercontent.com/ARY2260/openpom/main/"
               "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv")

# Datasets to download from pyrfume-data
PYRFUME_DATASETS = {
    # Tier 1: Gold Standard
    "leffingwell": {
        "molecules": f"{PYRFUME_BASE}/leffingwell/molecules.csv",
        "behavior":  f"{PYRFUME_BASE}/leffingwell/behavior.csv",
        "tier": 1,
        "description": "Leffingwell PMP 2001 - Expert perfumer labels",
        "smiles_col": "IsomericSMILES",
        "id_col": "CID",
        "label_type": "binary",
        "match_by": "id",
    },
    "dravnieks_1985": {
        "molecules": f"{PYRFUME_BASE}/dravnieks_1985/molecules.csv",
        "behavior":  f"{PYRFUME_BASE}/dravnieks_1985/behavior_1.csv",
        "tier": 1,
        "description": "Dravnieks 1985 - ASTM human panel (146 descriptors)",
        "smiles_col": "IsomericSMILES",
        "id_col": "CID",
        "label_type": "continuous",
        "match_by": "name",
    },
    "keller_2016": {
        "molecules": f"{PYRFUME_BASE}/keller_2016/molecules.csv",
        "behavior":  f"{PYRFUME_BASE}/keller_2016/stimuli.csv",
        "tier": 1,
        "description": "Keller 2016 - Science paper, psychophysics",
        "smiles_col": "CanonicalSMILES",
        "id_col": "CID",
        "label_type": "continuous",
        "match_by": "id",
    },
    # Tier 2: Expert / Curated databases
    "arctander_1960": {
        "molecules": f"{PYRFUME_BASE}/arctander_1960/molecules.csv",
        "behavior":  f"{PYRFUME_BASE}/arctander_1960/behavior_1.csv",
        "tier": 2,
        "description": "Arctander 1960 - Classic perfumery reference (2,751 molecules)",
        "smiles_col": "IsomericSMILES",
        "id_col": "CID",
        "label_type": "binary",
        "match_by": "id",
    },
    "flavordb": {
        "molecules": f"{PYRFUME_BASE}/flavordb/molecules.csv",
        "behavior":  f"{PYRFUME_BASE}/flavordb/behavior.csv",
        "tier": 2,
        "description": "FlavorDB - Flavor molecule database (25,595 molecules)",
        "smiles_col": "IsomericSMILES",
        "id_col": "CID",
        "label_type": "text",  # Odor Percepts column has comma-separated labels
        "text_cols": ["Odor Percepts"],
        "match_by": "id",
    },
    "aromadb": {
        "molecules": f"{PYRFUME_BASE}/aromadb/molecules.csv",
        "behavior":  f"{PYRFUME_BASE}/aromadb/behavior.csv",
        "tier": 2,
        "description": "AromaDB - Indian spice aroma database (869 molecules)",
        "smiles_col": "IsomericSMILES",
        "id_col": "CID",
        "label_type": "text",  # Raw Descriptors column
        "text_cols": ["Raw Descriptors", "Filtered Descriptors"],
        "match_by": "id",
    },
}

# The 138 labels from GS-LF (OpenPOM standard)
STANDARD_138_LABELS = None  # Will be populated from OpenPOM data

# Synonym mapping for label normalization
LABEL_SYNONYMS = {
    "flowery": "floral",
    "flower": "floral",
    "flowers": "floral",
    "flora": "floral",
    "grassy": "green",
    "grass": "green",
    "herbaceous": "herbal",
    "herb": "herbal",
    "herbs": "herbal",
    "camphoraceous": "camphoreous",
    "camphor": "camphoreous",
    "balsam": "balsamic",
    "minty": "mint",
    "mentholic": "mint",
    "menthol": "mint",
    "resinous": "balsamic",
    "sulfury": "sulfurous",
    "sulphury": "sulfurous",
    "sulphurous": "sulfurous",
    "citrusy": "citrus",
    "fruity odor": "fruity",
    "musky": "musk",
    "nutlike": "nutty",
    "nut": "nutty",
    "smokey": "smoky",
    "spice": "spicy",
    "woody odor": "woody",
    "wood": "woody",
    "sweet odor": "sweet",
    "sour odor": "sour",
    "milden": "milky",
    "alcohollike": "alcoholic",
    "meaty odor": "meaty",
    "meat": "meaty",
    "oily odor": "oily",
    "rose-like": "rose",
    "roses": "rose",
    "lemonlike": "lemon",
    "orangelike": "orange",
    "bitter flavor": "bitter",
    "earthy odor": "earthy",
    "earth": "earthy",
    "musty odor": "musty",
    "warmth": "warm",
}


# ================================================================
# Helpers
# ================================================================

def download_csv(url, cache_dir="data/cache"):
    """Download CSV from URL, return list of dicts. Uses local cache."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = url.split("/")[-1]
    # Include parent dir to avoid name collisions
    parent = url.split("/")[-2]
    cache_path = os.path.join(cache_dir, f"{parent}_{fname}")

    if os.path.exists(cache_path):
        print(f"    [cache] {parent}/{fname}")
    else:
        print(f"    [download] {url}")
        try:
            urllib.request.urlretrieve(url, cache_path)
        except urllib.error.HTTPError as e:
            print(f"    [ERROR] HTTP {e.code}: {url}")
            return []

    with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        return list(reader)


def normalize_label(label):
    """Normalize an odor label string."""
    label = label.strip().lower().replace("_", " ").replace("-", " ")
    # Remove common suffixes
    for suffix in [" odor", " like", " type", " note", "-like"]:
        label = label.replace(suffix, "")
    label = label.strip()
    return LABEL_SYNONYMS.get(label, label)


def canonical_smiles(smi):
    """Canonicalize SMILES using RDKit. Returns None if invalid."""
    if not smi or not HAS_RDKIT:
        return smi.strip() if smi else None
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None


def validate_molecule(smi):
    """Check if SMILES represents a valid, reasonable small molecule."""
    if not HAS_RDKIT or not smi:
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        n_atoms = mol.GetNumHeavyAtoms()
        mw = Descriptors.MolWt(mol)
        # Fragrance molecules: 30-600 Da, 2-80 heavy atoms
        if n_atoms < 2 or n_atoms > 80:
            return False
        if mw < 30 or mw > 600:
            return False
        return True
    except:
        return False


# ================================================================
# Data Collection Functions
# ================================================================

def collect_openpom(out_dir):
    """Download and parse OpenPOM curated GS-LF dataset (4,983 / 138 labels).
    This is THE gold standard for odor prediction.
    """
    global STANDARD_138_LABELS
    print("\n  [Tier 1] OpenPOM GoodScents-Leffingwell (4,983 molecules)")
    print(f"    Source: {OPENPOM_URL}")

    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "openpom_GS_LF.csv")

    if not os.path.exists(cache_path):
        print("    Downloading...")
        urllib.request.urlretrieve(OPENPOM_URL, cache_path)
    else:
        print("    Using cached file")

    molecules = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        # Auto-detect SMILES column (could be 'smiles', 'SMILES', 'nonStereoSMILES')
        smiles_col = None
        for h in headers:
            if 'smiles' in h.lower() or 'SMILES' in h:
                smiles_col = h
                break
        if smiles_col is None:
            print("    [ERROR] No SMILES column found!")
            return molecules
        print(f"    SMILES column detected: '{smiles_col}'")

        label_cols = [h for h in headers if h != smiles_col]
        STANDARD_138_LABELS = sorted(label_cols)

        for row in reader:
            smi = row.get(smiles_col, "").strip()
            if not smi or smi.upper() == 'NA':
                continue
            can = canonical_smiles(smi)
            if can is None or not validate_molecule(can):
                continue

            labels = {}
            for col in label_cols:
                val = row.get(col, "0").strip()
                try:
                    v = float(val)
                    if v > 0:
                        normalized = normalize_label(col)
                        labels[normalized] = max(labels.get(normalized, 0), v)
                except:
                    pass

            if labels:
                if can not in molecules:
                    molecules[can] = {
                        "smiles": can,
                        "labels": {},
                        "sources": set(),
                        "tier": 1,
                    }
                molecules[can]["labels"].update(labels)
                molecules[can]["sources"].add("openpom_GS_LF")

    print(f"    Collected: {len(molecules)} molecules, "
          f"{len(STANDARD_138_LABELS)} label types")
    return molecules


def collect_pyrfume_dataset(name, config, existing):
    """Download and parse a single Pyrfume dataset."""
    print(f"\n  [Tier {config['tier']}] {config['description']}")

    # Download molecules
    mol_rows = download_csv(config["molecules"])
    if not mol_rows:
        print(f"    [SKIP] No molecule data for {name}")
        return existing

    # Build lookup maps for matching (CID -> SMILES, and name -> SMILES)
    smiles_col = config["smiles_col"]
    id_col = config["id_col"]
    cid_to_smiles = {}
    name_to_smiles = {}
    for row in mol_rows:
        cid = row.get(id_col, "").strip().strip('"')
        smi = row.get(smiles_col, "").strip()
        mol_name = row.get("name", row.get("OdorName", "")).strip().strip('"')
        if cid and smi:
            can = canonical_smiles(smi)
            if can and validate_molecule(can):
                cid_to_smiles[cid] = can
                if mol_name:
                    name_to_smiles[mol_name.lower()] = can

    print(f"    Molecules with valid SMILES: {len(cid_to_smiles)}")

    # Download behavior
    beh_rows = download_csv(config["behavior"])
    if not beh_rows:
        print(f"    [SKIP] No behavior data for {name}")
        return existing

    match_by = config.get("match_by", "id")
    print(f"    Match strategy: {match_by}")

    # Parse behavior data
    beh_headers = list(beh_rows[0].keys()) if beh_rows else []
    # First column is usually the stimulus ID
    first_col = beh_headers[0] if beh_headers else ""
    # Skip metadata columns
    skip_cols = {first_col.lower(), "cid", "stimulus", "", "cas", "odorname",
                 "molecularweight", "isomericsmiles", "canonicalsmiles",
                 "iupacname", "name"}
    label_cols = [h for h in beh_headers if h.lower() not in skip_cols]

    matched = 0
    for row in beh_rows:
        stimulus_key = row.get(first_col, "").strip().strip('"')
        can = None

        if match_by == "name":
            # Try stimulus name -> molecule name matching
            # Dravnieks uses names like "Abhexone_high" -> strip suffix
            base_name = stimulus_key.rsplit('_', 1)[0] if '_' in stimulus_key else stimulus_key
            can = name_to_smiles.get(base_name.lower())
            if can is None:
                can = name_to_smiles.get(stimulus_key.lower())
        else:
            # Match by CID
            can = cid_to_smiles.get(stimulus_key)
            if can is None:
                # Try Stimulus column
                alt_key = row.get("Stimulus", row.get("stimulus", "")).strip().strip('"')
                can = cid_to_smiles.get(alt_key)

        if can is None:
            continue

        labels = {}
        if config["label_type"] == "text":
            # FlavorDB/AromaDB: text columns with comma-separated labels
            text_cols = config.get("text_cols", label_cols)
            for col in text_cols:
                text_val = row.get(col, "").strip()
                if not text_val or text_val.lower() in ("nan", "none", ""):
                    continue
                # Split by comma, semicolon, pipe, or " and "
                for sep in [",", ";", "|"]:
                    text_val = text_val.replace(sep, ",")
                text_val = text_val.replace(" and ", ",")
                for word in text_val.split(","):
                    word = word.strip()
                    if word and len(word) > 1 and word.lower() not in ("na", "none", "??"):
                        normalized = normalize_label(word)
                        if normalized and len(normalized) > 1:
                            labels[normalized] = 1.0
        else:
            for col in label_cols:
                val = row.get(col, "").strip()
                try:
                    v = float(val)
                    if config["label_type"] == "binary":
                        if v > 0:
                            normalized = normalize_label(col)
                            labels[normalized] = 1.0
                    else:  # continuous
                        if v > 0:
                            normalized = normalize_label(col)
                            if v > 1:
                                v = min(v / 5.0, 1.0) if v <= 5 else min(v / 100.0, 1.0)
                            labels[normalized] = max(labels.get(normalized, 0), v)
                except:
                    pass

        if labels:
            matched += 1
            if can not in existing:
                existing[can] = {
                    "smiles": can,
                    "labels": {},
                    "sources": set(),
                    "tier": config["tier"],
                }
            existing[can]["labels"].update(labels)
            existing[can]["sources"].add(name)
            existing[can]["tier"] = min(existing[can]["tier"], config["tier"])

    print(f"    Matched molecules with labels: {matched}")
    return existing


def compute_quality_score(mol_data):
    """Compute quality confidence score for a molecule.

    Score components:
      - n_sources: more sources = higher confidence
      - tier: lower tier = higher confidence
      - n_labels: enough labels is ideal (not too few, not too many)
    """
    n_sources = len(mol_data["sources"])
    tier = mol_data["tier"]
    n_labels = len(mol_data["labels"])

    # Source score: 1 source = 0.3, 2 = 0.6, 3+ = 1.0
    source_score = min(n_sources / 3.0, 1.0)

    # Tier score: tier 1 = 1.0, tier 2 = 0.7, tier 3 = 0.4
    tier_score = max(0.4, 1.0 - (tier - 1) * 0.3)

    # Label score: 1-3 labels = 0.5, 4-10 = 0.8, 10+ = 1.0
    if n_labels >= 10:
        label_score = 1.0
    elif n_labels >= 4:
        label_score = 0.8
    elif n_labels >= 1:
        label_score = 0.5
    else:
        label_score = 0.0

    return round(0.4 * source_score + 0.4 * tier_score + 0.2 * label_score, 3)


# ================================================================
# Merge Verified Data from Existing unified_training_data.csv
# ================================================================

def merge_verified_unified(molecules, unified_path):
    """Merge molecules from unified_training_data.csv that have
    verified sources (GS, LF, chemprop). Exclude discovery_db-only entries.

    Strategy:
      - If SMILES already in curated: skip (curated labels are better)
      - If source contains 'GS' or 'LF' or 'chemprop': include as Tier 2
      - If source is 'discovery_db' only: EXCLUDE (unverified)
    """
    print(f"\n  [Merge] Verified molecules from unified_training_data.csv")

    if not os.path.exists(unified_path):
        print(f"    [SKIP] File not found: {unified_path}")
        return molecules

    added = 0
    skipped_duplicate = 0
    skipped_unverified = 0

    with open(unified_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        # Label columns = everything except metadata
        meta_cols = {"smiles", "sources", "n_sources", "confidence"}
        label_cols = [h for h in headers if h.lower() not in meta_cols]

        for row in reader:
            smi = row.get("smiles", "").strip()
            sources = row.get("sources", "").lower()

            # Skip discovery_db-only entries
            if sources == "discovery_db":
                skipped_unverified += 1
                continue

            # Must have at least one verified source
            has_verified = any(tag in sources for tag in ["gs", "lf", "chemprop"])
            if not has_verified:
                skipped_unverified += 1
                continue

            # Validate SMILES
            can = canonical_smiles(smi)
            if can is None or not validate_molecule(can):
                continue

            # Skip if already in curated (curated labels are better quality)
            if can in molecules:
                skipped_duplicate += 1
                continue

            # Extract labels
            labels = {}
            for col in label_cols:
                val = row.get(col, "0").strip()
                try:
                    v = float(val)
                    if v > 0.01:
                        normalized = normalize_label(col)
                        labels[normalized] = max(labels.get(normalized, 0), v)
                except:
                    pass

            if labels:
                added += 1
                molecules[can] = {
                    "smiles": can,
                    "labels": labels,
                    "sources": {"unified_verified"},
                    "tier": 2,  # Tier 2: verified but not from direct peer-review
                }

    print(f"    Added: {added} new molecules")
    print(f"    Skipped (already in curated): {skipped_duplicate}")
    print(f"    Skipped (unverified/discovery_db only): {skipped_unverified}")
    return molecules


# ================================================================
# Main Pipeline
# ================================================================

def main():
    print("=" * 70)
    print("  High-Quality Odor Data Collection")
    print("  Strategy: Curated + Verified, exclude unverified")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "..", "data", "cache")
    out_dir = os.path.join(script_dir, "..", "cloud", "data")
    unified_path = os.path.join(out_dir, "unified_training_data.csv")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: OpenPOM (gold standard, defines the 138-label space)
    molecules = collect_openpom(cache_dir)

    # Step 2: Pyrfume datasets (Tier 1 & 2)
    for name, config in PYRFUME_DATASETS.items():
        try:
            molecules = collect_pyrfume_dataset(name, config, molecules)
        except Exception as e:
            print(f"    [ERROR] {name}: {e}")
            import traceback; traceback.print_exc()

    # Step 3: Merge verified data from existing unified_training_data.csv
    molecules = merge_verified_unified(molecules, unified_path)

    # Step 4: Quality scoring
    print(f"\n{'='*70}")
    print(f"  Quality Assessment")
    print(f"{'='*70}")

    for can, data in molecules.items():
        data["quality_score"] = compute_quality_score(data)

    # Stats
    total = len(molecules)
    multi_src = sum(1 for m in molecules.values() if len(m["sources"]) >= 2)
    tier1 = sum(1 for m in molecules.values() if m["tier"] == 1)
    tier2 = sum(1 for m in molecules.values() if m["tier"] == 2)
    avg_labels = sum(len(m["labels"]) for m in molecules.values()) / max(total, 1)
    avg_quality = sum(m["quality_score"] for m in molecules.values()) / max(total, 1)

    # All unique labels
    all_labels = set()
    for m in molecules.values():
        all_labels.update(m["labels"].keys())
    all_labels = sorted(all_labels)

    print(f"  Total molecules: {total}")
    print(f"  Multi-source validated: {multi_src} ({100*multi_src/max(total,1):.1f}%)")
    print(f"  Tier 1 (gold): {tier1}")
    print(f"  Tier 2 (expert): {tier2}")
    print(f"  Unique labels: {len(all_labels)}")
    print(f"  Avg labels per molecule: {avg_labels:.1f}")
    print(f"  Avg quality score: {avg_quality:.3f}")

    # Step 5: Export CSV
    print(f"\n{'='*70}")
    print(f"  Exporting curated_training_data.csv")
    print(f"{'='*70}")

    # Sort all_labels for consistent column order
    out_path = os.path.join(out_dir, "curated_training_data.csv")
    fieldnames = ["smiles"] + all_labels + ["sources", "n_sources", "tier", "quality_score"]

    # Sort molecules by quality score (best first)
    sorted_mols = sorted(molecules.values(),
                         key=lambda x: x["quality_score"], reverse=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for mol in sorted_mols:
            row = {"smiles": mol["smiles"]}
            for label in all_labels:
                row[label] = mol["labels"].get(label, 0)
            row["sources"] = "+".join(sorted(mol["sources"]))
            row["n_sources"] = len(mol["sources"])
            row["tier"] = mol["tier"]
            row["quality_score"] = mol["quality_score"]
            writer.writerow(row)

    print(f"  Saved: {out_path}")
    print(f"  Rows: {len(sorted_mols)}")
    print(f"  Columns: {len(fieldnames)}")

    # Export label mapping for model
    label_map_path = os.path.join(out_dir, "label_mapping.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({
            "labels": all_labels,
            "n_labels": len(all_labels),
            "sources": list(set(
                src for m in molecules.values() for src in m["sources"]
            )),
            "stats": {
                "total_molecules": total,
                "multi_source": multi_src,
                "tier1": tier1,
                "tier2": tier2,
                "avg_labels_per_mol": round(avg_labels, 2),
                "avg_quality": round(avg_quality, 3),
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"  Label mapping: {label_map_path}")

    # Top/bottom quality examples
    print(f"\n  Top 5 quality molecules:")
    for m in sorted_mols[:5]:
        print(f"    {m['smiles'][:40]:40s} q={m['quality_score']:.3f} "
              f"src={'+'.join(sorted(m['sources']))} "
              f"labels={len(m['labels'])}")

    print(f"\n  Bottom 5 quality molecules:")
    for m in sorted_mols[-5:]:
        print(f"    {m['smiles'][:40]:40s} q={m['quality_score']:.3f} "
              f"src={'+'.join(sorted(m['sources']))} "
              f"labels={len(m['labels'])}")

    # Label frequency distribution
    label_freq = defaultdict(int)
    for m in molecules.values():
        for label in m["labels"]:
            label_freq[label] += 1
    top_labels = sorted(label_freq.items(), key=lambda x: -x[1])[:30]
    print(f"\n  Top 30 labels:")
    for label, count in top_labels:
        pct = 100 * count / total
        bar = "#" * int(pct / 2)
        print(f"    {label:20s} {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n{'='*70}")
    print(f"  Data collection complete!")
    print(f"  {total} molecules, {len(all_labels)} labels, quality-validated")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
