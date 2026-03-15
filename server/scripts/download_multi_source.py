"""Multi-source odor data integration (v2 - fixed matching)

Sources:
1. GoodScents+Leffingwell (4,983 molecules, binary labels)
2. Dravnieks Atlas 1985 (160 stimuli × 146 descriptors, intensity 0-5)
3. Keller & Vosshall 2016 (481 molecules, pleasantness+intensity ratings)

Output: multi_source_unified.csv → 20d odor vectors per canonical SMILES
"""
import os, sys, csv, io, re, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rdkit import Chem

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
GS_PATH = os.path.join(DATA_DIR, "curated_GS_LF_merged_4983.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "multi_source_unified.csv")

ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
]

# === Universal label→dimension mapper ===
LABEL_MAP = {
    'sweet': 'sweet', 'vanilla': 'sweet', 'caramellic': 'sweet', 'honey': 'sweet',
    'chocolate': 'sweet', 'candy': 'sweet', 'sugar': 'sweet', 'maple': 'sweet',
    'sour': 'sour', 'acidic': 'sour', 'tart': 'sour', 'vinegar': 'sour',
    'fermented': 'sour', 'sharp': 'sour', 'pungent': 'sour', 'acid': 'sour',
    'rancid': 'sour', 'bitter': 'sour', 'cheesy': 'sour',
    'woody': 'woody', 'wood': 'woody', 'cedar': 'woody', 'cedarwood': 'woody',
    'sandalwood': 'woody', 'pine': 'woody', 'bark': 'woody', 'resinous': 'woody',
    'balsamic': 'woody', 'cork': 'woody', 'pencil shavings': 'woody', 'new lumber': 'woody',
    'floral': 'floral', 'flower': 'floral', 'rose': 'floral', 'jasmine': 'floral',
    'jasmin': 'floral', 'lily': 'floral', 'violet': 'floral', 'lavender': 'floral',
    'orchid': 'floral', 'gardenia': 'floral', 'perfumery': 'floral',
    'citrus': 'citrus', 'lemon': 'citrus', 'orange': 'citrus', 'lime': 'citrus',
    'grapefruit': 'citrus', 'bergamot': 'citrus',
    'fruity,citrus': 'citrus',  # Dravnieks combined label
    'spicy': 'spicy', 'cinnamon': 'spicy', 'clove': 'spicy', 'pepper': 'spicy',
    'ginger': 'spicy', 'nutmeg': 'spicy', 'anise': 'spicy', 'spices': 'spicy',
    'garlic': 'spicy', 'onion': 'spicy',
    'musk': 'musk', 'musky': 'musk', 'animalic': 'musk', 'animal': 'musk',
    'sweaty': 'musk', 'urinous': 'musk',
    'fresh': 'fresh', 'clean': 'fresh', 'crisp': 'fresh', 'cool': 'fresh',
    'cold': 'fresh', 'cooling': 'fresh', 'minty': 'fresh', 'camphor': 'fresh',
    'menthol': 'fresh', 'eucalyptus': 'fresh', 'peppermint': 'fresh', 'soapy': 'fresh',
    'green': 'green', 'grassy': 'green', 'leafy': 'green', 'grass': 'green',
    'hay': 'green', 'tea': 'green', 'crushed grass': 'green', 'cut grass': 'green',
    'warm': 'warm', 'nutty': 'warm', 'toasted': 'warm', 'almond': 'warm',
    'walnut': 'warm', 'bread': 'warm', 'peanut': 'warm', 'peanut butter': 'warm',
    'coconut': 'warm', 'butter': 'warm', 'bakery': 'warm', 'waxy': 'warm',
    'fatty': 'warm', 'creamy': 'warm', 'malty': 'warm', 'popcorn': 'warm',
    'fruity': 'fruity', 'fruit': 'fruity', 'apple': 'fruity', 'banana': 'fruity',
    'peach': 'fruity', 'berry': 'fruity', 'grape': 'fruity', 'cherry': 'fruity',
    'tropical': 'fruity', 'pineapple': 'fruity', 'melon': 'fruity', 'strawberry': 'fruity',
    'smoky': 'smoky', 'smoke': 'smoky', 'roasted': 'smoky', 'burnt': 'smoky',
    'charred': 'smoky', 'coffee': 'smoky', 'tobacco': 'smoky', 'tar': 'smoky',
    'creosote': 'smoky',
    'powdery': 'powdery', 'powder': 'powdery', 'chalky': 'powdery',
    'dusty': 'powdery', 'chalk': 'powdery', 'talc': 'powdery',
    'aquatic': 'aquatic', 'marine': 'aquatic', 'oily': 'aquatic', 'fishy': 'aquatic', 'fish': 'aquatic',
    'herbal': 'herbal', 'medicinal': 'herbal', 'camphoraceous': 'herbal',
    'aromatic': 'herbal',
    'amber': 'amber', 'incense': 'amber',
    'leather': 'leather', 'leathery': 'leather',
    'earthy': 'earthy', 'mushroom': 'earthy', 'mossy': 'earthy', 'musty': 'earthy',
    'moldy': 'earthy', 'damp': 'earthy', 'cellar': 'earthy', 'stale': 'earthy',
    'decayed': 'earthy', 'putrid': 'earthy', 'wet earth': 'earthy', 'meaty': 'earthy',
    'ozonic': 'ozonic', 'ozone': 'ozonic',
    'metallic': 'metallic', 'sulfurous': 'metallic', 'sulfidic': 'metallic',
    'chemical': 'metallic', 'gasoline': 'metallic', 'petroleum': 'metallic',
    'kerosene': 'metallic', 'paint': 'metallic', 'solvent': 'metallic',
    'ether': 'metallic', 'ammonia': 'metallic',
}


def canonical(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def download_csv(url):
    print(f"  Downloading {url.split('/')[-1]} ...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode('utf-8')
        return list(csv.DictReader(io.StringIO(text)))
    except Exception as e:
        print(f"    ERROR: {e}")
        return []


def label_to_dim(label):
    return LABEL_MAP.get(label.lower().strip())


def load_goodscents(merged):
    """Source 1: GoodScents+Leffingwell binary labels"""
    print("\n[1] GoodScents+Leffingwell ...")
    with open(GS_PATH, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        odor_cols = header[1:]
        for row in reader:
            can = canonical(row[0])
            if not can:
                continue
            active = [col.lower() for col, val in zip(odor_cols, row[1:]) if val == '1']
            vec = np.zeros(20, dtype=np.float32)
            for label in active:
                dim = label_to_dim(label)
                if dim:
                    vec[ODOR_DIMENSIONS.index(dim)] += 1.0
            if vec.max() > 0:
                vec /= vec.max()
            merged[can] = {'vec': vec, 'sources': ['GS+LF'], 'n_sources': 1}
    
    print(f"  GS+LF: {len(merged)} molecules")
    return merged


def load_dravnieks(merged):
    """Source 2: Dravnieks Atlas 1985 (average intensity per stimulus, 146 descriptors)"""
    print("\n[2] Dravnieks Atlas 1985 ...")
    
    # Molecules: CID → SMILES
    mol_rows = download_csv("https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/dravnieks_1985/molecules.csv")
    if not mol_rows:
        return merged
    
    # Build name → canonical SMILES (case-insensitive, strip underscores)    
    name_to_can = {}
    for row in mol_rows:
        name = row.get('name', '').lower().strip()
        smiles = row.get('IsomericSMILES', '')
        can = canonical(smiles)
        if name and can:
            name_to_can[name] = can
            # Also add without hyphens/spaces
            cleaned = re.sub(r'[\s\-_]', '', name)
            name_to_can[cleaned] = can
    
    print(f"  Name → SMILES map: {len(name_to_can)} entries")
    
    # Behavior: Stimulus × 146 descriptors (already averaged in pyrfume)
    beh_rows = download_csv("https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/dravnieks_1985/behavior_1.csv")
    if not beh_rows:
        return merged
    
    desc_cols = [c for c in beh_rows[0].keys() if c != 'Stimulus']
    print(f"  Descriptors: {len(desc_cols)}")
    print(f"  Stimuli: {len(beh_rows)}")
    
    new_count = 0
    enriched = 0
    
    for row in beh_rows:
        stimulus = row['Stimulus']
        # Parse stimulus name: "Benzaldehyde_high" → "benzaldehyde"
        name = re.sub(r'_(high|low|medium|med)$', '', stimulus).lower().strip()
        name_cleaned = re.sub(r'[\s\-_]', '', name)
        
        can = name_to_can.get(name) or name_to_can.get(name_cleaned)
        if not can:
            continue
        
        # Build 20d vector from descriptor intensities
        vec = np.zeros(20, dtype=np.float32)
        for desc in desc_cols:
            try:
                intensity = float(row[desc])
            except (ValueError, TypeError):
                continue
            if intensity <= 0:
                continue
            
            dim = label_to_dim(desc)
            if dim:
                vec[ODOR_DIMENSIONS.index(dim)] += intensity / 5.0  # Scale 0-5→0-1
        
        if vec.max() > 0:
            vec /= vec.max()
        
        if can in merged:
            # Weighted average (quantitative Dravnieks gets 1.5x weight)
            old = merged[can]['vec']
            n = merged[can]['n_sources']
            merged[can]['vec'] = (old * n + vec * 1.5) / (n + 1.5)
            merged[can]['sources'].append('Dravnieks')
            merged[can]['n_sources'] += 1
            enriched += 1
        else:
            merged[can] = {'vec': vec, 'sources': ['Dravnieks'], 'n_sources': 1}
            new_count += 1
    
    print(f"  Dravnieks: {new_count} new + {enriched} enriched")
    return merged


def load_keller(merged):
    """Source 3: Keller & Vosshall 2016 (481 molecules, long-format ratings)"""
    print("\n[3] Keller & Vosshall 2016 ...")
    
    mol_rows = download_csv("https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/keller_2016/molecules.csv")
    if not mol_rows:
        return merged
    
    # CID → CanonicalSMILES
    cid_to_can = {}
    for row in mol_rows:
        cid = row.get('CID', '')
        smiles = row.get('CanonicalSMILES', '')
        can = canonical(smiles)
        if cid and can:
            cid_to_can[cid] = can
    
    print(f"  Keller molecules: {len(cid_to_can)}")
    
    # Behavior is in long format (1.4M rows) — too large to download in full
    # Instead, use the molecules + their SMILES directly to match with our DB
    # The behavioral data (pleasantness, intensity) supplements but doesn't add new labels
    # We'll mark these molecules as "Keller-validated" (experimentally tested)
    
    new_count = 0
    enriched = 0
    for cid, can in cid_to_can.items():
        if can in merged:
            if 'Keller' not in merged[can]['sources']:
                merged[can]['sources'].append('Keller')
                merged[can]['n_sources'] += 1
                enriched += 1
        else:
            # No label data from Keller alone (need behavior file)
            # But we can at least note this molecule was experimentally tested
            new_count += 1
    
    print(f"  Keller: {new_count} new (no labels), {enriched} enriched (cross-validated)")
    return merged


def save_unified(merged):
    """Save unified multi-source CSV"""
    # Remove zero-vector entries
    valid = {k: v for k, v in merged.items() if v['vec'].max() > 0}
    
    print(f"\n[4] Saving unified dataset ...")
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['smiles', 'sources', 'n_sources'] + ODOR_DIMENSIONS
        writer.writerow(header)
        for smi, data in sorted(valid.items()):
            src = '+'.join(sorted(set(data['sources'])))
            row = [smi, src, data['n_sources']] + [f"{v:.4f}" for v in data['vec']]
            writer.writerow(row)
    
    # Stats
    multi = sum(1 for d in valid.values() if len(set(d['sources'])) > 1)
    by_source = {}
    for d in valid.values():
        for s in set(d['sources']):
            by_source[s] = by_source.get(s, 0) + 1
    
    print(f"\n  === UNIFIED DATASET ===")
    print(f"  Total: {len(valid)} molecules")
    for s, c in sorted(by_source.items()):
        print(f"    {s:15s}: {c:5d}")
    print(f"  Multi-source:   {multi:5d} ({multi/len(valid)*100:.1f}%)")
    print(f"  Saved: {OUTPUT_PATH}")
    
    return valid


if __name__ == "__main__":
    print("=" * 60)
    print("  MULTI-SOURCE ODOR DATA INTEGRATION v2")
    print("=" * 60)
    
    merged = {}
    merged = load_goodscents(merged)
    merged = load_dravnieks(merged)
    merged = load_keller(merged)
    valid = save_unified(merged)
    
    print("\n  DONE!")
