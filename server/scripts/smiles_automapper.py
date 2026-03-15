"""smiles_automapper.py — 패턴 기반 자동 SMILES 매핑

전략:
1. MANUAL_SMILES에서 직접 매핑 (300+)
2. 지역 변종 → 기본 원료 SMILES 자동 매핑
3. 카테고리별 대표 SMILES fallback
"""
import sys, os, json, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'ingredient_smiles.json')

# Import manual SMILES from the mapper
from scripts.smiles_mapper import MANUAL_SMILES

# Country/region prefixes to strip for variant matching
COUNTRY_PREFIXES = [
    'af_','ae_','ar_','at_','au_','bg_','br_','bt_','ca_','ch_','ci_',
    'cn_','co_','co2_','cu_','cy_','de_','ec_','eg_','es_','et_','fj_',
    'fi_','fr_','gh_','gr_','gy_','hi_','hn_','hr_','ht_','hu_','id_',
    'ie_','in_','ir_','it_','jp_','ke_','kh_','kr_','kw_','la_','lb_',
    'lk_','ma_','mg_','mm_','mx_','my_','nc_','ng_','no_','np_','nz_',
    'om_','pe_','pf_','ph_','pk_','pl_','pt_','pw_','py_','ro_','ru_',
    'se_','so_','sv_','th_','tn_','tr_','tw_','ua_','uk_','us_','uz_',
    'vn_','ye_','za_',
]

# Suffix patterns to strip
SUFFIX_PATTERNS = [
    '_oil', '_abs', '_absolute', '_co2', '_concrete', '_hydrosol',
    '_water', '_resin', '_resinoid', '_butter', '_extract', '_tincture',
    '_seed', '_leaf', '_bark', '_root', '_peel', '_bud', '_flower',
    '_accord', '_base', '_note', '_synth', '_synthetic',
]

# Base ingredient → SMILES (canonical representatives)
BASE_SMILES = {
    # Florals
    'rose': 'OC/C=C(\\C)CCC=C(C)C',  # geraniol
    'jasmine': 'CC(=O)OCc1ccccc1',     # benzyl acetate
    'ylang': 'C=CC(C)(O)CCC=C(C)C',    # linalool
    'lavender': 'C=CC(C)(O)CCC=C(C)C',
    'neroli': 'C=CC(C)(O)CCC=C(C)C',
    'tuberose': 'COC(=O)c1ccccc1',      # methyl benzoate
    'geranium': 'OC/C=C(\\C)CCC=C(C)C',
    'magnolia': 'C=CC(C)(O)CCC=C(C)C',
    'lotus': 'C=CC(C)(O)CCC=C(C)C',
    'orchid': 'C=CC(C)(O)CCC=C(C)C',
    'iris': 'CC1=CC(=O)CC(/C=C/C(C)C)C1C',
    'orris': 'CC1=CC(=O)CC(/C=C/C(C)C)C1C',
    'mimosa': 'CC(=O)OCc1ccccc1',
    'violet': 'CC1=CC(=O)CC(/C=C/C(C)C)C1',
    'peony': 'OCCc1ccccc1',
    'carnation': 'C=CCc1ccc(O)c(OC)c1',
    'osmanthus': 'CCCCCC1OC(=O)CC1',
    'champaca': 'C=CC(C)(O)CCC=C(C)C',
    'frangipani': 'OC(=O)c1ccccc1OCc1ccccc1',
    'plumeria': 'C=CC(C)(O)CCC=C(C)C',
    'wisteria': 'C=CC(C)(O)CCC=C(C)C',
    'narcissus': 'OCCc1ccccc1',
    'chamomile': 'CC(CCC=C(C)C)C1CCC(C)(O)C1',
    'chrysanthemum': 'CC(=C)C1CC(=O)C(C)(C1)C',
    'hyacinth': 'OCCc1ccccc1',
    'honeysuckle': 'C=CC(C)(O)CCC=C(C)C',
    'heather': 'C=CC(C)(O)CCC=C(C)C',
    'gardenia': 'C=CC(C)(O)CCC=C(C)C',
    'lily': 'OC1CCC(CC1)CC(C)CCC=O',
    'muguet': 'OC1CCC(CC1)CC(C)CCC=O',
    'sakura': 'O=Cc1ccccc1',
    'azalea': 'C=CC(C)(O)CCC=C(C)C',
    'rhododendron': 'C=CC(C)(O)CCC=C(C)C',
    'poppy': 'C=CC(C)(O)CCC=C(C)C',
    'camellia': 'C=CC(C)(O)CCC=C(C)C',
    'heliotrope': 'O=COc1ccc2OCOc2c1',
    'lilac': 'C=CC(C)(O)CCC=C(C)C',
    'sunflower': 'CCCCCCCC/C=C\\CCCCCCCC(=O)O',
    'hibiscus': 'C=CC(C)(O)CCC=C(C)C',
    'henna': 'O=C1C=CC(=O)c2ccccc21',
    
    # Woody
    'sandalwood': 'CC1(C)C2CCC1(C)C(O)C2',
    'cedarwood': 'CC12CCC(CC1)C(C)(O)C2',
    'cedar': 'CC12CCC(CC1)C(C)(O)C2',
    'vetiver': 'CC(C1CCCC(=C)C1=C)=CO',
    'oud': 'O=C1C=CC2=C1C=CC3=CC=CC=C32',
    'agarwood': 'O=C1C=CC2=C1C=CC3=CC=CC=C32',
    'patchouli': 'CC1CCC(C2CCCCC2C)CC1O',
    'pine': 'CC1=CCC2CC1C2(C)C',
    'fir': 'CC(=O)OC1CC(C)CCC1C(C)C',
    'cypress': 'CC1=CCC2CC1C2(C)C',
    'juniper': 'CC1=CCC2CC1C2(C)C',
    'hinoki': 'OC1=CC=CC2=CC(=O)C(C)=CC12',
    'birch': 'COC(=O)c1ccccc1O',
    'guaiacwood': 'COc1ccccc1O',
    'guaiac': 'COc1ccccc1O',
    'rosewood': 'C=CC(C)(O)CCC=C(C)C',
    'oak': 'C=CCc1ccc(O)c(OC)c1',
    'ebony': 'CC12CCC(CC1)C(C)(O)C2',
    'mahogany': 'CC12CCC(CC1)C(C)(O)C2',
    'spruce': 'CC(=O)OC1CC(C)CCC1C(C)C',
    'teak': 'CC(C)(O)C1CCC(=CC1)C',
    'sequoia': 'CC12CCC(CC1)C(C)(O)C2',
    'redwood': 'CC12CCC(CC1)C(C)(O)C2',
    'bamboo': 'CCCCCC=O',
    'olive': 'C=CCc1ccc(O)c(OC)c1',
    'maple': 'CC12CCC(CC1)C(C)(O)C2',
    
    # Citrus
    'lemon': 'CC(=C)C1CCC(=CC1)C',
    'orange': 'CC(=C)C1CCC(=CC1)C',
    'lime': 'CC(=C)C1CCC(=CC1)C',
    'grapefruit': 'CC(=O)C1CCC2(CCCCC2C)C1',
    'bergamot': 'CC(=O)OC(C)(C=C)CCC=C(C)C',
    'mandarin': 'CC(=C)C1CCC(=CC1)C',
    'yuzu': 'CC(=C)C1CCC(=CC1)C',
    'tangerine': 'CC(=C)C1CCC(=CC1)C',
    'pomelo': 'CC(=C)C1CCC(=CC1)C',
    'citronella': 'CC(CCC=C(C)C)CC=O',
    'lemongrass': 'CC(C)=CCCC(C)=CC=O',
    'petitgrain': 'CC(=O)OC(C)(C=C)CCC=C(C)C',
    'kumquat': 'CC(=C)C1CCC(=CC1)C',
    
    # Spicy
    'cinnamon': 'O=C/C=C/c1ccccc1',
    'clove': 'C=CCc1ccc(O)c(OC)c1',
    'pepper': 'O=C(/C=C/C=C/c1ccc2OCOc2c1)C3CCCCN3',
    'cardamom': 'CC1CCC(CC1OC(C)=O)(C)C',
    'nutmeg': 'COc1cc(CC=C)cc(OC)c1OC',
    'saffron': 'CC1=C(C=O)C(C)(C)C=C1',
    'ginger': 'CCCCC(O)CC(=O)c1ccc(O)c(OC)c1',
    'cumin': 'CC(C)c1ccc(C=O)cc1',
    'anise': 'COc1ccc(/C=C/C)cc1',
    'turmeric': 'COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O',
    'star_anise': 'COc1ccc(/C=C/C)cc1',
    'galangal': 'CCCCC(O)CC(=O)c1ccc(O)c(OC)c1',
    'allspice': 'C=CCc1ccc(O)c(OC)c1',
    'caraway': 'CC(=C)C1CCC(=CC1)C=O',
    'sumac': 'CC(=C)C1CCC(=CC1)C',
    'fenugreek': 'CC(=O)c1ccc(O)c(OC)c1',
    
    # Amber/Resinous
    'vanilla': 'O=Cc1ccc(O)c(OC)c1',
    'benzoin': 'O=C(OCc1ccccc1)c1ccccc1',
    'frankincense': 'CC1=CCC2CC1C2(C)C',
    'olibanum': 'CC1=CCC2CC1C2(C)C',
    'labdanum': 'CC(C)CCCC(C)CCCC(C)=CC(=O)O',
    'myrrh': 'CC1=CCCC(=C)C2CC(=C)CCC12',
    'opoponax': 'CC1=CCCC(=C)C2CC(=C)CCC12',
    'copal': 'CC1=CCC2CC1C2(C)C',
    'incense': 'CC1=CCC2CC1C2(C)C',
    'ambergris': 'CC12CCCCC1(C)C3CCC(C)(O3)C2C',
    'amber': 'CC12CCCCC1(C)C3CCC(C)(O3)C2C',
    'styrax': 'O=C(OCc1ccccc1)/C=C/c2ccccc2',
    'cistus': 'CC(C)CCCC(C)CCCC(C)=CC(=O)O',
    'mastic': 'CC1=CCC2CC1C2(C)C',
    'copaiba': 'CC1CCC2(CC1)C(=C)CCC2C(=C)C',
    'balsam': 'O=C(/C=C/c1ccccc1)OCc2ccccc2',
    
    # Gourmand
    'chocolate': 'Cn1c2nc(=O)[nH]c2c(=O)n1C',
    'cocoa': 'Cn1c2nc(=O)[nH]c2c(=O)n1C',
    'cacao': 'Cn1c2nc(=O)[nH]c2c(=O)n1C',
    'coffee': 'Cn1c(=O)c2c(ncn2C)n1C',
    'caramel': 'CC1=C(O)C(=O)C(C)O1',
    'honey': 'OC(=O)Cc1ccccc1',
    'tonka': 'O=C1OC2=CC=CC=C2C=C1',
    'almond': 'O=Cc1ccccc1',
    'praline': 'CC1=C(O)C(=O)OC1',
    'coconut': 'CCCCCC1OC(=O)CC1',
    'pistachio': 'O=Cc1ccccc1',
    'hazelnut': 'CC(=O)C1CCC2(CCCCC2C)C1',
    'maple_syrup': 'CC1OC(=O)C(=C1O)C',
    'marshmallow': 'O=Cc1ccc(O)c(OC)c1',
    'milk': 'CCCCCC1OC(=O)CC1',
    'butter': 'CCCC(=O)O',
    'rum': 'CCCC(=O)OCC',
    'whiskey': 'CCCCCCCCCC1OC(=O)CC1',
    'cognac': 'CCCC(=O)OCC',
    'sake': 'CCCC(=O)OCC',
    'rice': 'CCCCCCCCC=O',
    
    # Musk
    'musk': 'O=C1CCCCCCCCCCCCC1C',
    'galaxolide': 'CC1(C)CC2=CC(=O)C3CCCC(C)C3C2CC1',
    'ambrette': 'O=C1OCCCCCCCCCCCCCC1',
    'civet': 'O=C1CCCCCCCCCCCCCCCC1',
    
    # Green/Aquatic
    'bamboo_leaf': 'O/C=C\\CCC',
    'tea': 'O/C=C\\CCC',
    'green_tea': 'O/C=C\\CCC',
    'matcha': 'O/C=C\\CCC',
    'white_tea': 'O/C=C\\CCC',
    'black_tea': 'O/C=C\\CCC',
    'rooibos': 'O/C=C\\CCC',
    'mate': 'Cn1c(=O)c2c(ncn2C)n1C',
    'aloe': 'O/C=C\\CCC',
    'fig': 'O/C=C\\CCC',
    'galbanum': 'CC(C)=CCCC(C)=CC=O',
    'seaweed': 'O=C1C=CC2=C1C=COC2',
    'sea': 'O=C1C=CC2=C1C=COC2',
    'ocean': 'O=C1C=CC2=C1C=COC2',
    'marine': 'O=C1C=CC2=C1C=COC2',
    'moss': 'COc1ccccc1O',
    'pandan': 'CCC=CC1OCCO1',
    
    # Herbal
    'rosemary': 'CC12CCC(CC1)C(C)(C)O2',
    'basil': 'C=CCc1ccc(O)c(OC)c1',
    'thyme': 'CC(C)c1ccc(C)c(O)c1',
    'sage': 'CC12CCC(CC1)C(C)(C)O2',
    'eucalyptus': 'CC12CCC(CC1)C(C)(C)O2',
    'mint': 'CC(C)C1CCC(C)CC1O',
    'peppermint': 'CC(C)C1CCC(C)CC1O',
    'spearmint': 'CC(=C)C1CCC(=CC1)C=O',
    'tarragon': 'C=CCc1ccc(OC)cc1',
    'oregano': 'CC(C)c1ccc(O)cc1C',
    'tea_tree': 'CC(O)C1CCC(=CC1)C',
    'immortelle': 'CC(=O)c1ccc(O)c(OC)c1',
    'linden': 'C=CC(C)(O)CCC=C(C)C',
    'calendula': 'CCCCCCCCCC=O',
    'melissa': 'CC(C)=CCCC(C)=CC=O',
    'myrtle': 'CC12CCC(CC1)C(C)(C)O2',
    'hops': 'CC(C)CCCC(C)CC=O',
    'dandelion': 'O/C=C\\CCC',
    'ginseng': 'CC1CCC2(CC1)C(O)C(O)CC3C2CCC4(C)C3CCC4(C)C',
    'wintergreen': 'COC(=O)c1ccccc1O',
    'clary_sage': 'C=CC(C)(O)CCC=C(C)C',
    'artemisia': 'CC(=O)C1CCC(=CC1C)C',
    'mugwort': 'CC(=O)C1CCC(=CC1C)C',
    'angelica': 'COc1c2OC(=O)C=Cc2cc3OC(C)(C)C=Cc13',
    'manuka': 'CC12CCC(CC1)C(C)(C)O2',
    'hemp': 'CC(C)=CCCC(C)=CC=O',
    'borage': 'CCCCCCCCCC=O',
    'parsley': 'COc1cc(CC=C)cc(OC)c1OC',
    'celery': 'CCCCCCCCCC=O',
    'lovage': 'CCCCCCCCCC=O',
    'yarrow': 'CC12CCC(CC1)C(C)(C)O2',
    'tansy': 'CC(=O)C1CCC(=CC1C)C',
    'tagetes': 'CC(=C)C1CCC(=CC1)C=O',
    'bay_laurel': 'CC12CCC(CC1)C(C)(C)O2',
    
    # Leather
    'leather': 'CCc1cccc2ncccc12',
    'suede': 'CCc1cccc2ncccc12',
    'tobacco': 'CC(C)c1ccc(C)c(O)c1',
    
    # Smoky
    'smoke': 'COc1ccccc1O',
    'campfire': 'COc1ccccc1O',
    'bonfire': 'COc1ccccc1O',
    'charcoal': 'COc1ccccc1O',
    'cade': 'COc1ccccc1O',
    'birch_tar': 'COc1ccccc1O',
    'lapsang': 'COc1ccccc1O',
    'pipe_tobacco': 'CC(C)c1ccc(C)c(O)c1',
    
    # Abstract/Accord fallbacks by category
    'cotton': 'CC(C)C1CCC(C)CC1O',       # clean/musk → menthol-like
    'silk': 'C=CC(C)(O)CCC=C(C)C',       # soft → linalool
    'cashmere': 'C=CC(C)(O)CCC=C(C)C',
    'velvet': 'C=CC(C)(O)CCC=C(C)C',
    'snow': 'CC12CCC(CC1)C(C)(C)O2',     # cool → cineole
    'crystal': 'CC12CCC(CC1)C(C)(C)O2',
    'rain': 'COc1ccccc1O',
    'petrichor': 'COc1ccccc1O',
    'ozone': 'CC12CCC(CC1)C(C)(C)O2',
    'metal': 'CCCCCCCCCC=O',
    'ink': 'CCc1cccc2ncccc12',
    'vinyl': 'CCc1cccc2ncccc12',
    'beeswax': 'CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O',
    'stone': 'COc1ccccc1O',
    'sand': 'COc1ccccc1O',
    'earth': 'COc1ccccc1O',
    'soil': 'COc1ccccc1O',
    'mushroom': 'C=CC(O)CCCCCC',
    'truffle': 'CSC',
}

# Category fallback SMILES (for anything truly unmappable)
CATEGORY_FALLBACK = {
    'floral': 'C=CC(C)(O)CCC=C(C)C',     # linalool
    'woody': 'CC12CCC(CC1)C(C)(O)C2',     # cedrol
    'citrus': 'CC(=C)C1CCC(=CC1)C',       # limonene
    'spicy': 'O=C/C=C/c1ccccc1',          # cinnamaldehyde
    'fruity': 'CCCCCC1OC(=O)CC1',          # gamma-decalactone
    'gourmand': 'O=Cc1ccc(O)c(OC)c1',     # vanillin
    'musk': 'O=C1CCCCCCCCCCCCC1C',         # muscone
    'amber': 'CC12CCCCC1(C)C3CCC(C)(O3)C2C', # ambroxan
    'green': 'O/C=C\\CCC',                # cis-3-hexenol
    'aquatic': 'O=C1C=CC2=C1C=COC2',       # calone
    'herbal': 'CC12CCC(CC1)C(C)(C)O2',    # 1,8-cineole
    'earthy': 'COc1ccccc1O',               # guaiacol
    'smoky': 'COc1ccccc1O',               # guaiacol
    'leather': 'CCc1cccc2ncccc12',         # quinoline
    'resinous': 'CC1=CCC2CC1C2(C)C',      # alpha-pinene
    'resin': 'CC1=CCC2CC1C2(C)C',
    'powdery': 'CC1=CC(=O)CC(/C=C/C(C)C)C1C', # alpha-isomethyl ionone
    'fresh': 'CC12CCC(CC1)C(C)(C)O2',     # 1,8-cineole
    'animalic': 'CCc1cccc2ncccc12',
    'aromatic': 'CC12CCC(CC1)C(C)(C)O2',
    'balsamic': 'O=C(OCc1ccccc1)c1ccccc1', # benzyl benzoate
    'synthetic': 'CCCCCCCCCC=O',            # decanal
    'aldehyde': 'CCCCCCCCCC=O',
    'aldehydic': 'CCCCCCCCCC=O',
    'cooling': 'CC(C)C1CCC(C)CC1O',        # menthol
    'ozonic': 'CC12CCC(CC1)C(C)(C)O2',
    'solvent': 'CCOC(=O)c1ccccc1C(=O)OCC',
    'waxy': 'CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O',
    'marine': 'O=C1C=CC2=C1C=COC2',
    'warm': 'O=Cc1ccc(O)c(OC)c1',
    'base': 'O=C1CCCCCCCCCCCCC1C',
    'carrier': 'CCCCCCCC/C=C\\CCCCCCCC(=O)O',
    'chypre': 'COc1ccccc1O',
}


def strip_prefix(ing_id):
    """Remove country prefix"""
    for p in COUNTRY_PREFIXES:
        if ing_id.startswith(p):
            return ing_id[len(p):]
    return ing_id


def find_base_match(ing_id):
    """Try to find a base SMILES match for an ingredient ID"""
    # 1) Direct match
    if ing_id in BASE_SMILES:
        return BASE_SMILES[ing_id]
    
    # 2) Strip country prefix
    stripped = strip_prefix(ing_id)
    if stripped in BASE_SMILES:
        return BASE_SMILES[stripped]
    
    # 3) Remove suffix patterns
    for suffix in SUFFIX_PATTERNS:
        if stripped.endswith(suffix):
            base = stripped[:-len(suffix)]
            if base in BASE_SMILES:
                return BASE_SMILES[base]
    
    # 4) Try partial matching - check if any BASE key is contained
    for base_name, smiles in BASE_SMILES.items():
        if base_name in ing_id and len(base_name) >= 3:
            return smiles
    
    return None


def canonicalize(smiles):
    if not HAS_RDKIT or not smiles:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else smiles


def main():
    import database as db_module
    
    print("=" * 60)
    print("  SMILES Auto-Mapper (Pattern + Category Fallback)")
    print("=" * 60)
    
    db_ings = db_module.get_all_ingredients()
    print(f"\n[1] DB: {len(db_ings)} ingredients")
    
    # Load existing molecules.json
    mol_path = os.path.join(DATA_DIR, 'molecules.json')
    existing = {}
    if os.path.exists(mol_path):
        with open(mol_path, 'r', encoding='utf-8') as f:
            for mol in json.load(f):
                for src in mol.get('source_ingredients', []):
                    existing[src] = mol['smiles']
    
    result = {}
    stats = {'manual': 0, 'existing': 0, 'base_match': 0, 'category_fb': 0, 'failed': 0}
    failed = []
    
    for ing in db_ings:
        ing_id = ing.get('id', '')
        category = ing.get('category', '')
        
        smiles = None
        source = None
        
        # Strategy 1: MANUAL_SMILES
        if ing_id in MANUAL_SMILES:
            smiles = MANUAL_SMILES[ing_id]
            source = 'manual'
        
        # Strategy 2: Existing molecules.json
        if not smiles and ing_id in existing:
            smiles = existing[ing_id]
            source = 'existing'
        
        # Strategy 3: Base ingredient match
        if not smiles:
            smiles = find_base_match(ing_id)
            if smiles:
                source = 'base_match'
        
        # Strategy 4: Category fallback
        if not smiles and category in CATEGORY_FALLBACK:
            smiles = CATEGORY_FALLBACK[category]
            source = 'category_fb'
        
        if smiles:
            result[ing_id] = canonicalize(smiles)
            stats[source] += 1
        else:
            result[ing_id] = None
            stats['failed'] += 1
            failed.append(ing_id)
    
    # Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    total_ok = sum(1 for v in result.values() if v)
    
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total:          {len(result)}")
    print(f"  With SMILES:    {total_ok} ({total_ok/len(result)*100:.1f}%)")
    print(f"  Sources:")
    print(f"    Manual:       {stats['manual']}")
    print(f"    Existing:     {stats['existing']}")
    print(f"    Base match:   {stats['base_match']}")
    print(f"    Category FB:  {stats['category_fb']}")
    print(f"    Failed:       {stats['failed']}")
    
    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for f_item in failed:
            print(f"    {f_item}")
    
    print(f"{'=' * 60}")
    return total_ok


if __name__ == '__main__':
    main()
