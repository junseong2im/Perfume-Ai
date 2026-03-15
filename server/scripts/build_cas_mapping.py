"""
Build CAS -> CID mapping for GoodScents via PubChem PUG REST API
Individual GET requests with checkpoint resume support

API: GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{CAS}/cids/JSON
Rate limit: ~5 requests/sec (per PubChem guidelines)
"""
import csv
import json
import os
import time
import sys
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = os.path.join('data', 'pom_data', 'pyrfume_all', 'goodscents')
OUTPUT_PATH = os.path.join(DATA_DIR, 'cas_to_cid.json')

def query_pubchem_cas(cas):
    """Query PubChem for a single CAS number -> CID"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(cas)}/cids/JSON"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'OpenPOM-Training/1.0')
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        cids = data.get('IdentifierList', {}).get('CID', [])
        if cids and cids[0] != 0:
            return cas, str(cids[0])
    except Exception:
        pass
    return cas, None


def main():
    # 1. Collect CAS numbers
    print("=== Step 1: Collecting CAS numbers ===")
    cas_numbers = []
    with open(os.path.join(DATA_DIR, 'behavior.csv'), 'r', encoding='utf-8', errors='replace') as f:
        seen = set()
        for row in csv.DictReader(f):
            cas = row.get('Stimulus', '').strip()
            if cas and '-' in cas and cas not in seen:
                cas_numbers.append(cas)
                seen.add(cas)
    print(f"  Unique CAS numbers: {len(cas_numbers)}")
    
    # 2. Load existing checkpoint
    cas_to_cid = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r') as f:
            cas_to_cid = json.load(f)
        print(f"  Loaded checkpoint: {len(cas_to_cid)} already mapped")
    
    remaining = [c for c in cas_numbers if c not in cas_to_cid]
    print(f"  Remaining to query: {len(remaining)}")
    
    if not remaining:
        print("  All CAS numbers already mapped!")
    else:
        # 3. Query PubChem with ThreadPool (5 concurrent, respecting rate limit)
        print(f"\n=== Step 2: Querying PubChem ({len(remaining)} requests) ===")
        success = 0
        failed = 0
        start_time = time.time()
        
        # Use 3 threads to balance speed vs rate limit
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}
            for cas in remaining:
                fut = pool.submit(query_pubchem_cas, cas)
                futures[fut] = cas
            
            for i, fut in enumerate(as_completed(futures), 1):
                cas, cid = fut.result()
                if cid:
                    cas_to_cid[cas] = cid
                    success += 1
                else:
                    failed += 1
                
                # Progress every 100
                if i % 100 == 0 or i == len(remaining):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(remaining) - i) / rate if rate > 0 else 0
                    print(f"  [{i}/{len(remaining)}] "
                          f"OK={success} FAIL={failed} "
                          f"({rate:.1f} req/s, ETA={eta:.0f}s)")
                    
                    # Checkpoint save every 500
                    if i % 500 == 0:
                        with open(OUTPUT_PATH, 'w') as f:
                            json.dump(cas_to_cid, f)
        
        # Final save
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(cas_to_cid, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed:.1f}s")
        print(f"  Mapped: {success}, Failed: {failed}")
    
    # 4. Match with molecules.csv
    print(f"\n=== Step 3: CID -> SMILES matching ===")
    cid_to_smiles = {}
    with open(os.path.join(DATA_DIR, 'molecules.csv'), 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cid = row.get('CID', '')
            smi = row.get('IsomericSMILES', '')
            if cid and smi:
                cid_to_smiles[str(cid)] = smi
    
    matched = sum(1 for cid in cas_to_cid.values() if cid in cid_to_smiles)
    
    print(f"  Total CAS->CID mapped:    {len(cas_to_cid)}/{len(cas_numbers)}")
    print(f"  CID found in molecules:   {matched}/{len(cas_to_cid)}")
    print(f"  Mapping saved to:         {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
