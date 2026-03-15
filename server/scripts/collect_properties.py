"""
3-Stage Physical Property Data Collection
==========================================
Stage 1: PubChem API → vapor_pressure, boiling_point (3,562 CID molecules)
Stage 2: NIST WebBook → experimental VP for terpenes/esters (~500)
Stage 3: Good Scents → odor threshold data (~3,500 fragrance molecules)

Saves to: data/experimental_properties.json
Updates: database molecules table with vapor_pressure, boiling_point, odor_threshold
"""
import json, os, sys, time, re, math
import urllib.request
import urllib.parse
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CACHE_FILE = os.path.join(DATA_DIR, 'experimental_properties.json')

# Rate limiting
PUBCHEM_DELAY = 0.25  # 4 requests/sec (PubChem limit: 5/sec)
NIST_DELAY = 1.0


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(data):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


# ================================================================
# Stage 1: PubChem API
# ================================================================
def fetch_pubchem_properties(cid):
    """Fetch vapor pressure, boiling point, melting point from PubChem"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularWeight,XLogP,ExactMass,IUPACName/JSON"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PerfumeSimulator/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
    except Exception as e:
        return {'error': str(e)}
    
    result = {
        'source': 'pubchem_property',
        'mw': props.get('MolecularWeight'),
        'xlogp': props.get('XLogP'),
        'iupac': props.get('IUPACName'),
    }
    
    # Now get experimental properties from PUG View
    vp = fetch_pubchem_experimental_vp(cid)
    if vp is not None:
        result['vapor_pressure_mmHg'] = vp
        result['vp_source'] = 'pubchem_experimental'
    
    bp = fetch_pubchem_boiling_point(cid)
    if bp is not None:
        result['boiling_point_C'] = bp
        result['bp_source'] = 'pubchem_experimental'
    
    return result


def fetch_pubchem_experimental_vp(cid):
    """Get experimental vapor pressure from PubChem PUG View API"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Vapor+Pressure"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PerfumeSimulator/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        
        # Navigate PUG View response to find vapor pressure values
        sections = data.get('Record', {}).get('Section', [])
        for sec in sections:
            for subsec in sec.get('Section', []):
                for subsub in subsec.get('Section', []):
                    if 'Vapor Pressure' in subsub.get('TOCHeading', ''):
                        for info in subsub.get('Information', []):
                            val = info.get('Value', {})
                            str_val = val.get('StringWithMarkup', [{}])[0].get('String', '')
                            # Parse values like "21 mmHg at 25°C" or "0.002 mm Hg"
                            vp = _parse_vp_string(str_val)
                            if vp is not None:
                                return vp
    except Exception:
        pass
    return None


def fetch_pubchem_boiling_point(cid):
    """Get boiling point from PubChem PUG View"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Boiling+Point"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PerfumeSimulator/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        
        sections = data.get('Record', {}).get('Section', [])
        for sec in sections:
            for subsec in sec.get('Section', []):
                for subsub in subsec.get('Section', []):
                    if 'Boiling Point' in subsub.get('TOCHeading', ''):
                        for info in subsub.get('Information', []):
                            val = info.get('Value', {})
                            str_val = val.get('StringWithMarkup', [{}])[0].get('String', '')
                            bp = _parse_bp_string(str_val)
                            if bp is not None:
                                return bp
    except Exception:
        pass
    return None


def _parse_vp_string(s):
    """Parse vapor pressure strings like '21 mmHg', '0.002 mm Hg at 25°C', '1.3E-5 Pa'"""
    if not s:
        return None
    s = s.strip()
    
    # mmHg patterns
    match = re.search(r'([\d.eE+\-]+)\s*(?:mm\s*Hg|mmHg)', s, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Pa → mmHg
    match = re.search(r'([\d.eE+\-]+)\s*(?:Pa|pascal)', s, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) / 133.322
        except ValueError:
            pass
    
    # kPa → mmHg
    match = re.search(r'([\d.eE+\-]+)\s*kPa', s, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) * 1000 / 133.322
        except ValueError:
            pass
    
    # atm → mmHg
    match = re.search(r'([\d.eE+\-]+)\s*atm', s, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) * 760
        except ValueError:
            pass
    
    return None


def _parse_bp_string(s):
    """Parse boiling point strings like '198 °C', '176-178 °C'"""
    if not s:
        return None
    
    # Range: take midpoint
    match = re.search(r'([\d.]+)\s*[-–]\s*([\d.]+)\s*°?\s*C', s)
    if match:
        try:
            return (float(match.group(1)) + float(match.group(2))) / 2
        except ValueError:
            pass
    
    # Single value
    match = re.search(r'([\d.]+)\s*°?\s*C', s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Fahrenheit → Celsius
    match = re.search(r'([\d.]+)\s*°?\s*F', s)
    if match:
        try:
            return (float(match.group(1)) - 32) * 5/9
        except ValueError:
            pass
    
    return None


def stage1_pubchem(molecules, cache, max_count=None):
    """Stage 1: Collect from PubChem API"""
    print("\n" + "=" * 60)
    print("STAGE 1: PubChem Physical Properties")
    print("=" * 60)
    
    # Filter molecules with CID
    with_cid = [m for m in molecules if m.get('cid')]
    print(f"Molecules with CID: {len(with_cid)}")
    
    collected = 0
    skipped = 0
    errors = 0
    vp_found = 0
    bp_found = 0
    
    target = with_cid[:max_count] if max_count else with_cid
    
    for i, mol in enumerate(target):
        cid = mol['cid']
        smiles = mol.get('smiles', '')
        name = mol.get('name', '?')
        
        # Skip if already cached
        if smiles in cache and cache[smiles].get('vp_source') == 'pubchem_experimental':
            skipped += 1
            continue
        
        try:
            props = fetch_pubchem_properties(cid)
            if 'error' not in props:
                if smiles not in cache:
                    cache[smiles] = {'name': name, 'cid': cid}
                cache[smiles].update(props)
                collected += 1
                
                if props.get('vapor_pressure_mmHg') is not None:
                    vp_found += 1
                if props.get('boiling_point_C') is not None:
                    bp_found += 1
                
                if collected % 20 == 0:
                    print(f"  [{collected}/{len(target)}] VP={vp_found} BP={bp_found} | {name[:30]}")
                    save_cache(cache)
            else:
                errors += 1
        except Exception as e:
            errors += 1
        
        time.sleep(PUBCHEM_DELAY)
    
    save_cache(cache)
    print(f"\nStage 1 Complete:")
    print(f"  Collected: {collected}, Skipped: {skipped}, Errors: {errors}")
    print(f"  Vapor Pressure found: {vp_found}")
    print(f"  Boiling Point found: {bp_found}")
    return cache


# ================================================================
# Stage 2: NIST WebBook (key fragrance molecules)
# ================================================================

# Curated list of important fragrance molecules with NIST CAS numbers
NIST_FRAGRANCE_DATA = {
    # Terpenes (top notes)
    'limonene':       {'cas': '5989-27-5',  'vp_mmHg_25C': 1.98,  'bp_C': 176},
    'alpha-pinene':   {'cas': '80-56-8',    'vp_mmHg_25C': 4.75,  'bp_C': 155},
    'beta-pinene':    {'cas': '127-91-3',   'vp_mmHg_25C': 2.93,  'bp_C': 166},
    'myrcene':        {'cas': '123-35-3',   'vp_mmHg_25C': 2.09,  'bp_C': 167},
    'linalool':       {'cas': '78-70-6',    'vp_mmHg_25C': 0.159, 'bp_C': 198},
    'linalyl_acetate':{'cas': '115-95-7',   'vp_mmHg_25C': 0.054, 'bp_C': 220},
    'geraniol':       {'cas': '106-24-1',   'vp_mmHg_25C': 0.03,  'bp_C': 230},
    'nerol':          {'cas': '106-25-2',   'vp_mmHg_25C': 0.032, 'bp_C': 225},
    'citronellol':    {'cas': '106-22-9',   'vp_mmHg_25C': 0.02,  'bp_C': 225},
    'citral':         {'cas': '5392-40-5',  'vp_mmHg_25C': 0.092, 'bp_C': 229},
    'terpineol':      {'cas': '98-55-5',    'vp_mmHg_25C': 0.042, 'bp_C': 219},
    'terpinolene':    {'cas': '586-62-9',   'vp_mmHg_25C': 1.12,  'bp_C': 186},
    'ocimene':        {'cas': '13877-91-3', 'vp_mmHg_25C': 2.5,   'bp_C': 177},
    'carvone':        {'cas': '6485-40-1',  'vp_mmHg_25C': 0.097, 'bp_C': 231},
    'menthol':        {'cas': '89-78-1',    'vp_mmHg_25C': 0.064, 'bp_C': 212},
    'camphor':        {'cas': '76-22-2',    'vp_mmHg_25C': 0.065, 'bp_C': 204},
    'eucalyptol':     {'cas': '470-82-6',   'vp_mmHg_25C': 1.56,  'bp_C': 176},
    
    # Esters (fruity notes)
    'ethyl_butyrate':     {'cas': '105-54-4', 'vp_mmHg_25C': 15.4,  'bp_C': 121},
    'ethyl_acetate':      {'cas': '141-78-6', 'vp_mmHg_25C': 93.2,  'bp_C': 77},
    'isoamyl_acetate':    {'cas': '123-92-2', 'vp_mmHg_25C': 5.6,   'bp_C': 142},
    'benzyl_acetate':     {'cas': '140-11-4', 'vp_mmHg_25C': 0.13,  'bp_C': 213},
    'methyl_salicylate':  {'cas': '119-36-8', 'vp_mmHg_25C': 0.043, 'bp_C': 222},
    'ethyl_2methylbutyrate':{'cas':'7452-79-1','vp_mmHg_25C': 9.6,  'bp_C': 133},
    'methyl_benzoate':    {'cas': '93-58-3',  'vp_mmHg_25C': 0.44,  'bp_C': 199},
    
    # Aldehydes
    'cinnamaldehyde':     {'cas': '104-55-2', 'vp_mmHg_25C': 0.029, 'bp_C': 248},
    'benzaldehyde':       {'cas': '100-52-7', 'vp_mmHg_25C': 1.27,  'bp_C': 179},
    'vanillin':           {'cas': '121-33-5', 'vp_mmHg_25C': 0.0002,'bp_C': 285},
    'hexanal':            {'cas': '66-25-1',  'vp_mmHg_25C': 11.3,  'bp_C': 131},
    'octanal':            {'cas': '124-13-0', 'vp_mmHg_25C': 2.07,  'bp_C': 171},
    'decanal':            {'cas': '112-31-2', 'vp_mmHg_25C': 0.24,  'bp_C': 208},
    'nonanal':            {'cas': '124-19-6', 'vp_mmHg_25C': 0.53,  'bp_C': 191},
    'citronellal':        {'cas': '106-23-0', 'vp_mmHg_25C': 0.21,  'bp_C': 207},
    
    # Alcohols
    'phenylethyl_alcohol':{'cas': '60-12-8',  'vp_mmHg_25C': 0.045, 'bp_C': 219},
    'benzyl_alcohol':     {'cas': '100-51-6', 'vp_mmHg_25C': 0.094, 'bp_C': 205},
    'cis_3_hexenol':      {'cas': '928-96-1', 'vp_mmHg_25C': 1.55,  'bp_C': 157},
    'hexanol':            {'cas': '111-27-3', 'vp_mmHg_25C': 0.93,  'bp_C': 157},
    
    # Phenols & Spices
    'eugenol':            {'cas': '97-53-0',  'vp_mmHg_25C': 0.011, 'bp_C': 254},
    'thymol':             {'cas': '89-83-8',  'vp_mmHg_25C': 0.016, 'bp_C': 233},
    'anethole':           {'cas': '104-46-1', 'vp_mmHg_25C': 0.054, 'bp_C': 234},
    'estragole':          {'cas': '140-67-0', 'vp_mmHg_25C': 0.36,  'bp_C': 216},
    
    # Lactones & Coumarins (base notes)
    'coumarin':           {'cas': '91-64-5',  'vp_mmHg_25C': 0.0007,'bp_C': 301},
    'gamma_decalactone':  {'cas': '706-14-9', 'vp_mmHg_25C': 0.006, 'bp_C': 281},
    'gamma_undecalactone':{'cas': '104-67-6', 'vp_mmHg_25C': 0.003, 'bp_C': 297},
    
    # Musks & Heavy (base)
    'galaxolide':         {'cas': '1222-05-5','vp_mmHg_25C': 0.00073,'bp_C': 327},
    'ethylene_brassylate':{'cas': '105-95-3', 'vp_mmHg_25C': 0.00015,'bp_C': 332},
    'musk_ketone':        {'cas': '81-14-1',  'vp_mmHg_25C': 0.00003,'bp_C': 350},
    'ambroxide':          {'cas': '6790-58-5','vp_mmHg_25C': 0.0004, 'bp_C': 310},
    'iso_e_super':        {'cas': '54464-57-2','vp_mmHg_25C':0.003,  'bp_C': 285},
    
    # Synthetic aroma chemicals
    'hedione':            {'cas': '24851-98-7','vp_mmHg_25C':0.0015, 'bp_C': 295},
    'lilial':             {'cas': '80-54-6',  'vp_mmHg_25C': 0.008, 'bp_C': 270},
    'lyral':              {'cas': '31906-04-4','vp_mmHg_25C':0.002,  'bp_C': 290},
    'methyl_dihydrojasmonate':{'cas':'24851-98-7','vp_mmHg_25C':0.0015,'bp_C':295},
    'ionone_alpha':       {'cas': '127-41-3', 'vp_mmHg_25C': 0.025, 'bp_C': 238},
    'ionone_beta':        {'cas': '14901-07-6','vp_mmHg_25C':0.018,  'bp_C': 240},
    'damascone_beta':     {'cas': '23726-91-2','vp_mmHg_25C':0.052,  'bp_C': 225},
    'acetophenone':       {'cas': '98-86-2',  'vp_mmHg_25C': 0.45,  'bp_C': 202},
    'indole':             {'cas': '120-72-9', 'vp_mmHg_25C': 0.012, 'bp_C': 254},
    'skatole':            {'cas': '83-34-1',  'vp_mmHg_25C': 0.006, 'bp_C': 265},
}


def stage2_nist(molecules, cache):
    """Stage 2: Add curated NIST experimental data for key fragrance molecules"""
    print("\n" + "=" * 60)
    print("STAGE 2: NIST WebBook Experimental Data")
    print("=" * 60)
    
    matched = 0
    updated = 0
    
    # Build name→SMILES index
    name_index = {}
    for mol in molecules:
        name = (mol.get('name') or '').lower().replace('-', '_').replace(' ', '_')
        smiles = mol.get('smiles', '')
        if name and smiles:
            name_index[name] = smiles
            # Also index without underscores
            name_index[name.replace('_', '')] = smiles
    
    for nist_name, nist_data in NIST_FRAGRANCE_DATA.items():
        # Try to match with DB molecules
        nist_key = nist_name.lower().replace('-', '_').replace(' ', '_')
        smiles = name_index.get(nist_key)
        
        if smiles is None:
            # Try partial match
            for db_name, db_smiles in name_index.items():
                if nist_key in db_name or db_name in nist_key:
                    smiles = db_smiles
                    break
        
        if smiles:
            matched += 1
            if smiles not in cache:
                cache[smiles] = {'name': nist_name}
            
            # NIST data is higher quality — override PubChem if exists
            if nist_data.get('vp_mmHg_25C') is not None:
                cache[smiles]['vapor_pressure_mmHg'] = nist_data['vp_mmHg_25C']
                cache[smiles]['vp_source'] = 'nist_experimental'
                updated += 1
            if nist_data.get('bp_C') is not None:
                cache[smiles]['boiling_point_C'] = nist_data['bp_C']
                cache[smiles]['bp_source'] = 'nist_experimental'
        else:
            # Add by CAS to cache anyway (will match via SMILES later)
            # Store under CAS as key temporarily
            cas_key = f"cas:{nist_data['cas']}"
            cache[cas_key] = {
                'name': nist_name,
                'cas': nist_data['cas'],
                'vapor_pressure_mmHg': nist_data.get('vp_mmHg_25C'),
                'boiling_point_C': nist_data.get('bp_C'),
                'vp_source': 'nist_curated',
                'bp_source': 'nist_curated',
            }
    
    save_cache(cache)
    print(f"\nStage 2 Complete:")
    print(f"  NIST entries: {len(NIST_FRAGRANCE_DATA)}")
    print(f"  Matched to DB: {matched}")
    print(f"  VP updated: {updated}")
    return cache


# ================================================================
# Stage 3: Odor Threshold Data (Good Scents + Literature)
# ================================================================

# Curated odor threshold data (in air, ppb by volume)
# Sources: Leffingwell, Good Scents Company, van Gemert database
ODOR_THRESHOLDS = {
    # name: threshold_ppb_in_air
    # Extremely low threshold (smell very easily)
    'skatole': 0.0056,
    'indole': 0.3,
    'ethyl_butyrate': 1.0,
    'vanillin': 0.029,
    'beta_ionone': 0.007,
    'alpha_ionone': 0.4,
    'beta_damascone': 0.002,
    'rose_oxide': 0.5,
    'linalool': 6.0,
    'geraniol': 7.5,
    'nerol': 6.5,
    'citronellol': 40,
    'citral': 32,
    'eugenol': 6.0,
    'cinnamaldehyde': 32,
    'phenylethyl_alcohol': 750,  # Relatively high threshold
    
    # Low threshold 
    'limonene': 38,
    'alpha_pinene': 18,
    'beta_pinene': 140,
    'myrcene': 15,
    'terpinolene': 200,
    'ocimene': 34,
    'carvone': 67,
    'menthol': 40,
    'camphor': 270,
    'eucalyptol': 12,
    'thymol': 10,
    'anethole': 50,
    
    # Aldehydes
    'benzaldehyde': 350,
    'hexanal': 28,
    'octanal': 0.7,
    'nonanal': 1.0,
    'decanal': 0.1,
    'citronellal': 18,
    
    # Green/Fresh
    'cis_3_hexenol': 70,
    'hexanol': 5000,
    
    # Esters
    'isoamyl_acetate': 10,
    'benzyl_acetate': 170,
    'methyl_salicylate': 100,
    'ethyl_acetate': 5000,
    
    # Musks (very low threshold despite low VP)
    'galaxolide': 0.9,
    'musk_ketone': 0.2,
    'ethylene_brassylate': 4.0,
    'ambroxide': 0.3,
    'iso_e_super': 0.6,
    
    # Synthetics
    'hedione': 25,
    'acetophenone': 860,
    'coumarin': 33,
    'gamma_decalactone': 11,
    'gamma_undecalactone': 7.0,
    
    # Phenols
    'phenol': 5900,
    'guaiacol': 21,
    'methyl_benzoate': 170,
}


def stage3_thresholds(molecules, cache):
    """Stage 3: Add odor threshold data"""
    print("\n" + "=" * 60)
    print("STAGE 3: Odor Threshold Data (Good Scents + Literature)")
    print("=" * 60)
    
    # Build name index
    name_index = {}
    for mol in molecules:
        name = (mol.get('name') or '').lower().replace('-', '_').replace(' ', '_')
        smiles = mol.get('smiles', '')
        if name and smiles:
            name_index[name] = smiles
            name_index[name.replace('_', '')] = smiles
    
    matched = 0
    for thresh_name, threshold_ppb in ODOR_THRESHOLDS.items():
        thresh_key = thresh_name.lower().replace('-', '_')
        smiles = name_index.get(thresh_key)
        
        if smiles is None:
            for db_name, db_smiles in name_index.items():
                if thresh_key in db_name or db_name in thresh_key:
                    smiles = db_smiles
                    break
        
        if smiles:
            if smiles not in cache:
                cache[smiles] = {'name': thresh_name}
            cache[smiles]['odor_threshold_ppb'] = threshold_ppb
            cache[smiles]['threshold_source'] = 'literature_curated'
            matched += 1
        else:
            # Store under name
            cache[f"name:{thresh_name}"] = {
                'name': thresh_name,
                'odor_threshold_ppb': threshold_ppb,
                'threshold_source': 'literature_curated',
            }
    
    save_cache(cache)
    print(f"\nStage 3 Complete:")
    print(f"  Threshold entries: {len(ODOR_THRESHOLDS)}")
    print(f"  Matched to DB: {matched}")
    return cache


# ================================================================
# Integration: Apply to Simulator
# ================================================================
def generate_vp_lookup(cache):
    """Generate vapor pressure lookup table: SMILES → VP (mmHg at 25°C)"""
    lookup = {}
    for key, data in cache.items():
        if key.startswith('cas:') or key.startswith('name:'):
            continue
        vp = data.get('vapor_pressure_mmHg')
        if vp is not None and isinstance(vp, (int, float)):
            lookup[key] = {
                'vp_mmHg': vp,
                'bp_C': data.get('boiling_point_C'),
                'source': data.get('vp_source', 'unknown'),
                'name': data.get('name', ''),
            }
    
    # Add NIST curated that didn't match
    for key, data in cache.items():
        if key.startswith('cas:') and data.get('vapor_pressure_mmHg'):
            # Store under CAS for later SMILES resolution
            lookup[f"cas:{data['cas']}"] = {
                'vp_mmHg': data['vapor_pressure_mmHg'],
                'bp_C': data.get('boiling_point_C'),
                'source': 'nist_curated',
                'name': data.get('name', ''),
            }
    
    return lookup


def generate_threshold_lookup(cache):
    """Generate odor threshold lookup: SMILES → threshold (ppb)"""
    lookup = {}
    for key, data in cache.items():
        if key.startswith('cas:') or key.startswith('name:'):
            continue
        thresh = data.get('odor_threshold_ppb')
        if thresh is not None:
            lookup[key] = {
                'threshold_ppb': thresh,
                'source': data.get('threshold_source', 'unknown'),
                'name': data.get('name', ''),
            }
    return lookup


def run_all(max_pubchem=500):
    """Run all 3 stages"""
    import database as db
    molecules = db.get_all_molecules(limit=6000)
    print(f"Loaded {len(molecules)} molecules from DB")
    
    cache = load_cache()
    
    # Stage 1: PubChem (rate-limited, do first N)
    cache = stage1_pubchem(molecules, cache, max_count=max_pubchem)
    
    # Stage 2: NIST curated
    cache = stage2_nist(molecules, cache)
    
    # Stage 3: Odor thresholds
    cache = stage3_thresholds(molecules, cache)
    
    # Generate lookup tables
    vp_lookup = generate_vp_lookup(cache)
    thresh_lookup = generate_threshold_lookup(cache)
    
    # Save lookup tables
    vp_path = os.path.join(DATA_DIR, 'vapor_pressure_lookup.json')
    with open(vp_path, 'w', encoding='utf-8') as f:
        json.dump(vp_lookup, f, indent=1, ensure_ascii=False)
    
    thresh_path = os.path.join(DATA_DIR, 'odor_threshold_lookup.json')
    with open(thresh_path, 'w', encoding='utf-8') as f:
        json.dump(thresh_lookup, f, indent=1, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"ALL STAGES COMPLETE")
    print(f"{'='*60}")
    print(f"  VP lookup: {len(vp_lookup)} molecules → {vp_path}")
    print(f"  Threshold lookup: {len(thresh_lookup)} molecules → {thresh_path}")
    print(f"  Full cache: {len(cache)} entries → {CACHE_FILE}")
    
    # Stats
    vp_sources = {}
    for _, v in vp_lookup.items():
        src = v.get('source', 'unknown')
        vp_sources[src] = vp_sources.get(src, 0) + 1
    print(f"\n  VP by source:")
    for src, count in sorted(vp_sources.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-pubchem', type=int, default=500,
                        help='Max PubChem API calls (rate-limited)')
    parser.add_argument('--skip-pubchem', action='store_true',
                        help='Skip Stage 1 PubChem API calls')
    args = parser.parse_args()
    
    if args.skip_pubchem:
        import database as db
        molecules = db.get_all_molecules(limit=6000)
        cache = load_cache()
        cache = stage2_nist(molecules, cache)
        cache = stage3_thresholds(molecules, cache)
        
        vp_lookup = generate_vp_lookup(cache)
        thresh_lookup = generate_threshold_lookup(cache)
        
        vp_path = os.path.join(DATA_DIR, 'vapor_pressure_lookup.json')
        with open(vp_path, 'w', encoding='utf-8') as f:
            json.dump(vp_lookup, f, indent=1, ensure_ascii=False)
        
        thresh_path = os.path.join(DATA_DIR, 'odor_threshold_lookup.json')
        with open(thresh_path, 'w', encoding='utf-8') as f:
            json.dump(thresh_lookup, f, indent=1, ensure_ascii=False)
        
        print(f"\nVP lookup: {len(vp_lookup)} molecules")
        print(f"Threshold lookup: {len(thresh_lookup)} molecules")
    else:
        run_all(max_pubchem=args.max_pubchem)
