"""
Priority 1: Patch DREAM 165 molecules with Abraham REAL measured ODTs
=====================================================================
Abraham 2012 dataset has CID-indexed measured ODTs (Log 1/ODT).
DREAM dataset also uses CIDs. Direct CID matching + SMILES fallback.

Note: Abraham uses Log(1/ODT) where POSITIVE = strong smell
      Our system uses log10(ODT) where NEGATIVE = strong smell  
      Conversion: our_odt = -abraham_value
"""
import sys, os, csv, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE = os.path.join(os.path.dirname(__file__), '..')
ABR_DIR = os.path.join(BASE, 'data', 'pom_data', 'pyrfume_all', 'abraham_2012')
DREAM_CACHE = os.path.join(BASE, 'data', 'pom_upgrade', 'cid_smiles_cache.json')
ODT_FILE = os.path.join(BASE, 'data', 'pom_upgrade', 'dream_odt.json')

def main():
    print("="*60)
    print("  Priority 1: Abraham Real ODT Override")
    print("="*60)
    
    # 1. Load Abraham molecules (CID -> SMILES)
    abr_mol_path = os.path.join(ABR_DIR, 'molecules.csv')
    abr_beh_path = os.path.join(ABR_DIR, 'behavior.csv')
    
    abr_cid_smiles = {}
    with open(abr_mol_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        for row in csv.DictReader(f):
            cid = row.get('CID', '').strip()
            smi = row.get('IsomericSMILES', '').strip()
            if cid and smi:
                abr_cid_smiles[cid] = smi
    print(f"\n  Abraham molecules: {len(abr_cid_smiles)}")
    
    # 2. Load Abraham behavior (CID -> Log(1/ODT))
    abr_odt = {}  # CID -> log(1/ODT)
    abr_smiles_odt = {}  # SMILES -> log(1/ODT) for fallback
    with open(abr_beh_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        for row in csv.DictReader(f):
            cid = row.get('Stimulus', '').strip()
            odt_str = row.get('Log (1/ODT)', '').strip()
            if cid and odt_str:
                try:
                    val = float(odt_str)
                    abr_odt[cid] = val
                    smi = abr_cid_smiles.get(cid, '')
                    if smi:
                        abr_smiles_odt[smi] = val
                except:
                    pass
    print(f"  Abraham ODTs: {len(abr_odt)} CIDs, {len(abr_smiles_odt)} SMILES")
    
    # 3. Load DREAM CID -> SMILES
    with open(DREAM_CACHE, 'r') as f:
        dream_cids = json.load(f)
    dream_non_empty = {k: v for k, v in dream_cids.items() if v}
    print(f"  DREAM molecules: {len(dream_non_empty)}")
    
    # 4. Load existing ODT cache
    with open(ODT_FILE, 'r') as f:
        cache = json.load(f)
    old_odts = cache.get('odt', {})
    pred_cache = cache.get('pred_138d', {})
    
    # 5. Match: CID direct -> SMILES fallback -> QSPR estimate
    new_odts = {}
    match_cid = 0
    match_smi = 0
    qspr_kept = 0
    
    for cid, smiles in dream_non_empty.items():
        # Try 1: Direct CID match (Abraham CID == DREAM CID, both are PubChem CIDs)
        if cid in abr_odt:
            # Convert: Abraham Log(1/ODT) -> our log(ODT)
            # Log(1/ODT) = -log(ODT)
            new_odts[cid] = -abr_odt[cid]
            match_cid += 1
        # Try 2: SMILES match
        elif smiles in abr_smiles_odt:
            new_odts[cid] = -abr_smiles_odt[smiles]
            match_smi += 1
        # Try 3: Canonical SMILES normalization match
        else:
            matched = False
            try:
                from rdkit import Chem
                dream_mol = Chem.MolFromSmiles(smiles)
                if dream_mol:
                    dream_canonical = Chem.MolToSmiles(dream_mol)
                    for abr_smi, abr_val in abr_smiles_odt.items():
                        abr_mol = Chem.MolFromSmiles(abr_smi)
                        if abr_mol:
                            abr_canonical = Chem.MolToSmiles(abr_mol)
                            if dream_canonical == abr_canonical:
                                new_odts[cid] = -abr_val
                                match_smi += 1
                                matched = True
                                break
            except:
                pass
            
            if not matched:
                # Keep old QSPR estimate
                new_odts[cid] = old_odts.get(cid, -3.0)
                qspr_kept += 1
    
    total_real = match_cid + match_smi
    print(f"\n--- Matching Results ---")
    print(f"  CID direct match:   {match_cid}")
    print(f"  SMILES match:       {match_smi}")
    print(f"  QSPR kept:          {qspr_kept}")
    print(f"  TOTAL REAL ODT:     {total_real}/{len(dream_non_empty)} ({total_real/len(dream_non_empty)*100:.1f}%)")
    
    # 6. Compare old vs new
    real_vals = [v for k, v in new_odts.items() if k in dream_non_empty and 
                 (dream_non_empty[k] in abr_smiles_odt or k in abr_odt)]
    if real_vals:
        vals = np.array(real_vals)
        print(f"\n--- Real ODT Distribution ---")
        print(f"  Range: [{vals.min():.2f}, {vals.max():.2f}]")
        print(f"  Mean: {vals.mean():.2f}, Std: {vals.std():.2f}")
    
    # 7. Save patched ODTs
    cache['odt'] = new_odts
    with open(ODT_FILE, 'w') as f:
        json.dump(cache, f)
    
    print(f"\n[DONE] Patched ODTs saved to {ODT_FILE}")
    
    # Show before/after for first 10 matched
    print(f"\n--- Sample: Before (QSPR) vs After (Abraham) ---")
    count = 0
    for cid in list(dream_non_empty.keys())[:50]:
        if cid in abr_odt or dream_non_empty[cid] in abr_smiles_odt:
            old = old_odts.get(cid, -3.0)
            new = new_odts[cid]
            diff = abs(new - old)
            marker = " ***" if diff > 1.0 else ""
            print(f"  CID={cid:>8s}: QSPR={old:+.2f} -> Abraham={new:+.2f} (diff={diff:.2f}){marker}")
            count += 1
            if count >= 10:
                break

if __name__ == '__main__':
    main()
