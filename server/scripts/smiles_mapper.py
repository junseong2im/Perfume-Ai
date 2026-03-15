"""smiles_mapper.py — 다중 전략 SMILES 매핑

전략 (우선순위):
1. molecules.json 기존 매핑 (48개)
2. CAS 번호 → PubChem (107개 CAS)
3. name_en → PubChem 이름 검색
4. name_en 단어 기반 PubChem 검색 (괄호/부가 제거)
5. 수동 별칭 테이블 (100+ 향수 원료)
6. 천연물 → 주요 성분 분자로 대체

핵심: 천연 추출물(rose, jasmine 등)은 단일 SMILES가 없으므로
해당 원료의 "주요 향기 성분" 분자로 대체합니다.
"""
import sys, os, json, time, urllib.request, urllib.parse, urllib.error
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'ingredient_smiles.json')

# ================================================================
# 천연 원료 → 주요 향기 분자 (SMILES) 수동 매핑
# 천연 추출물은 수십~수백개 분자의 혼합물이므로
# 향기 프로필을 대표하는 주요 성분의 SMILES를 사용
# ================================================================
MANUAL_SMILES = {
    # === FLORALS ===
    'rose': 'OCC=C(C)CCC=C(C)C',          # geraniol
    'damascus_rose': 'OCC=C(C)CCC=C(C)C',
    'tr_rose': 'OCC=C(C)CCC=C(C)C',
    'bg_rose': 'OCC=C(C)CCC=C(C)C',
    'ma_rose': 'OCC=C(C)CCC=C(C)C',
    'in_rose': 'OCC=C(C)CCC=C(C)C',
    'fr_rose_de_mai': 'OCC=C(C)CCC=C(C)C',
    'eg_rose': 'OCC=C(C)CCC=C(C)C',
    'jasmine': 'CC(=O)OCc1ccccc1',         # benzyl acetate
    'in_jasmine': 'CC(=O)OCc1ccccc1',
    'eg_jasmine': 'CC(=O)OCc1ccccc1',
    'in_mogra': 'CC(=O)OCc1ccccc1',
    'jasmine_sambac': 'CC(=O)OCc1ccccc1',
    'jasmine_grandiflorum': 'CC(=O)OCc1ccccc1',
    'ylang_ylang': 'C=CC(C)(O)CCC=C(C)C',  # linalool
    'comoros_ylang': 'C=CC(C)(O)CCC=C(C)C',
    'md_ylang_extra': 'C=CC(C)(O)CCC=C(C)C',
    'tuberose': 'COC(=O)c1ccccc1',          # methyl benzoate
    'iris': 'CC1=CC(=O)CC(/C=C/C(C)C)C1C',  # alpha-isomethyl ionone
    'orris': 'CC1=CC(=O)CC(/C=C/C(C)C)C1C',
    'ir_orris': 'CC1=CC(=O)CC(/C=C/C(C)C)C1C',
    'peony': 'OCCc1ccccc1',                 # phenylethyl alcohol (PEA)
    'lily_of_valley': 'OC1CCC(CC1)CC(C)CCC=O', # hydroxycitronellal approx
    'magnolia': 'C=CC(C)(O)CCC=C(C)C',
    'freesia': 'C=CC(C)(O)CCC=C(C)C',
    'geranium': 'OCC=C(C)CCC=C(C)C',        # geraniol
    'eg_geranium': 'OCC=C(C)CCC=C(C)C',
    'violet': 'CC1=CC(=O)CC(/C=C/C(C)C)C1',  # ionone
    'neroli': 'C=CC(C)(O)CCC=C(C)C',
    'orange_blossom': 'C=CC(C)(O)CCC=C(C)C',
    'cherry_blossom': 'O=Cc1ccccc1',          # benzaldehyde
    'beotkkot': 'O=Cc1ccccc1',
    'osmanthus': 'CCCCCC1OC(=O)CC1',          # gamma-decalactone
    'sweet_pea': 'C=CC(C)(O)CCC=C(C)C',
    'frangipani': 'OC(=O)c1ccccc1OCC1=CC=CC=C1', # benzyl salicylate
    'champaca': 'C=CC(C)(O)CCC=C(C)C',
    'champaca_gold': 'C=CC(C)(O)CCC=C(C)C',
    'pikake': 'CC(=O)OCc1ccccc1',
    'kewda': 'OCCc1ccccc1',
    'lotus': 'C=CC(C)(O)CCC=C(C)C',
    'plumeria': 'C=CC(C)(O)CCC=C(C)C',
    'gardenia': 'C=CC(C)(O)CCC=C(C)C',
    'carnation': 'C=CCc1ccc(O)c(OC)c1',     # eugenol
    'hyacinth': 'OCCc1ccccc1',
    'mimosa': 'CC(=O)OCc1ccccc1',
    'narcissus': 'OCCc1ccccc1',
    'tiare': 'C=CC(C)(O)CCC=C(C)C',
    'wisteria': 'C=CC(C)(O)CCC=C(C)C',
    'heliotrope': 'O=COc1ccc2OCOc2c1',       # piperonal
    'muguet': 'OC1CCC(CC1)CC(C)CCC=O',
    'honeysuckle': 'C=CC(C)(O)CCC=C(C)C',
    'acacia': 'CC(=O)OCc1ccccc1',
    'chrysanthemum': 'CC(=C)C1CC(=O)C(C)(C1)C', # chrysanthenone
    'wonangnamu_flower': 'C=CC(C)(O)CCC=C(C)C',
    'rhododendron': 'C=CC(C)(O)CCC=C(C)C',
    'au_boronia_mega': 'CC(=O)OCc1ccccc1',

    # === WOODY ===
    'sandalwood': 'CC1(C)C2CCC1(C)C(O)C2',   # alpha-santalol approx
    'in_mysore_sandalwood': 'CC1(C)C2CCC1(C)C(O)C2',
    'au_sandalwood': 'CC1(C)C2CCC1(C)C(O)C2',
    'new_caledonia_sandalwood': 'CC1(C)C2CCC1(C)C(O)C2',
    'cedarwood': 'CC12CCC(CC1)C(C)(O)C2',     # cedrol approx
    'atlas_cedarwood': 'CC12CCC(CC1)C(C)(O)C2',
    'virginia_cedarwood': 'CC12CCC(CC1)C(C)(O)C2',
    'himalayan_cedarwood': 'CC12CCC(CC1)C(C)(O)C2',
    'vetiver': 'CC(C1CCCC(=C)C1=C)=CO',       # vetiverol approx
    'haiti_vetiver': 'CC(C1CCCC(=C)C1=C)=CO',
    'java_vetiver': 'CC(C1CCCC(=C)C1=C)=CO',
    'oud': 'O=C1C=CC2=C1C=CC3=CC=CC=C32',  # 2-(2-phenylethyl)chromone
    'cambodia_oud': 'O=C1C=CC2=C1C=CC3=CC=CC=C32',
    'laos_oud': 'O=C1C=CC2=C1C=CC3=CC=CC=C32',
    'india_oud': 'O=C1C=CC2=C1C=CC3=CC=CC=C32',
    'patchouli': 'CC1CCC(C2CCCCC2C)CC1O',     # patchoulol approx
    'in_patchouli': 'CC1CCC(C2CCCCC2C)CC1O',
    'id_patchouli': 'CC1CCC(C2CCCCC2C)CC1O',
    'birch': 'COC(=O)c1ccccc1O',              # methyl salicylate
    'guaiac': 'COc1ccccc1O',                   # guaiacol
    'cypress': 'CC1=CCC2CC1C2(C)C',            # alpha-pinene
    'hinoki': 'OC1=CC=CC2=CC(=O)C(C)=CC12',   # hinokitiol approx
    'bamboo': 'CCCCCC=O',                       # hexanal
    'driftwood': 'OC/C=C\\CCC',                # cis-3-hexenol
    'agarwood': 'O=C1C=CC2=C1C=CC3=CC=CC=C32',
    'teak': 'CC(C)(O)C1CCC(=CC1)C',           # alpha-terpineol
    'oak': 'C=CCc1ccc(O)c(OC)c1',             # eugenol
    'rosewood': 'C=CC(C)(O)CCC=C(C)C',        # linalool
    'bois_de_rose': 'C=CC(C)(O)CCC=C(C)C',
    'pine': 'CC1=CCC2CC1C2(C)C',
    'fir': 'CC(=O)OC1CC(C)CCC1C(C)C',         # bornyl acetate approx
    'juniper_berry': 'CC1=CCC2CC1C2(C)C',
    'elemi': 'C=CC(=C)CCC=C(C)C',             # elemicin approx
    
    # === CITRUS ===
    'bergamot': 'CC(=O)OC(C)CC=C(C)CCC=C(C)C',  # linalyl acetate
    'it_bergamot': 'CC(=O)OC(C)CC=C(C)CCC=C(C)C',
    'lemon': 'CC(=C)C1CCC(=CC1)C',             # limonene
    'it_lemon': 'CC(=C)C1CCC(=CC1)C',
    'ar_lemon': 'CC(=C)C1CCC(=CC1)C',
    'orange': 'CC(=C)C1CCC(=CC1)C',
    'br_orange': 'CC(=C)C1CCC(=CC1)C',
    'it_blood_orange': 'CC(=C)C1CCC(=CC1)C',
    'blood_orange': 'CC(=C)C1CCC(=CC1)C',
    'grapefruit': 'CC(=O)C1CCC2(CCCCC2C)C1',   # nootkatone approx
    'lime': 'CC(=C)C1CCC(=CC1)C',
    'mx_lime': 'CC(=C)C1CCC(=CC1)C',
    'mandarin': 'CC(=C)C1CCC(=CC1)C',
    'it_mandarin': 'CC(=C)C1CCC(=CC1)C',
    'yuzu': 'CC(=C)C1CCC(=CC1)C',
    'jp_yuzu': 'CC(=C)C1CCC(=CC1)C',
    'tangerine': 'CC(=C)C1CCC(=CC1)C',
    'pomelo': 'CC(=C)C1CCC(=CC1)C',
    'citron': 'CC(C)=CC=O',                    # citral (neral)
    'petitgrain': 'CC(=O)OC(C)CC=C(C)CCC=C(C)C',
    'kumquat': 'CC(=C)C1CCC(=CC1)C',
    'calamansi': 'CC(=C)C1CCC(=CC1)C',
    'sudachi': 'CC(=C)C1CCC(=CC1)C',
    'kabosu': 'CC(=C)C1CCC(=CC1)C',
    
    # === SPICY ===
    'cinnamon': 'O=C/C=C/c1ccccc1',            # cinnamaldehyde
    'cn_cinnamon': 'O=C/C=C/c1ccccc1',
    'sri_lanka_cinnamon': 'O=C/C=C/c1ccccc1',
    'clove': 'C=CCc1ccc(O)c(OC)c1',            # eugenol
    'id_clove': 'C=CCc1ccc(O)c(OC)c1',
    'mg_clove': 'C=CCc1ccc(O)c(OC)c1',
    'black_pepper': 'O=C(/C=C/C=C/c1ccc2OCOc2c1)C3CCCCN3',  # piperine
    'vn_black_pepper': 'O=C(/C=C/C=C/c1ccc2OCOc2c1)C3CCCCN3',
    'in_black_pepper': 'O=C(/C=C/C=C/c1ccc2OCOc2c1)C3CCCCN3',
    'pink_pepper': 'CC1=CCC2CC1C2(C)C',
    'cardamom': 'CC1CCC(CC1OC(C)=O)(C)C',
    'gt_cardamom': 'CC1CCC(CC1OC(C)=O)(C)C',
    'nutmeg': 'COc1cc(CC=C)cc(OC)c1OC',        # myristicin
    'id_nutmeg': 'COc1cc(CC=C)cc(OC)c1OC',
    'saffron': 'CC1=C(C=O)C(C)(C)C=C1',        # safranal
    'ir_saffron': 'CC1=C(C=O)C(C)(C)C=C1',
    'star_anise': 'COc1ccc(/C=C/C)cc1',         # anethole
    'ginger': 'CCCCC(O)CC(=O)c1ccc(O)c(OC)c1', # gingerol
    'cumin': 'CC(C)c1ccc(C=O)cc1',              # cuminaldehyde
    'turmeric': 'COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O', # curcumin
    'wasabi': 'C=CCCS(=O)C',
    'szechuan_pepper': 'CC(C)=CC=CC=CC(=O)NCC1=CC2=CC=CC=C2O1', #hydroxy-alpha-sanshool
    'coriander': 'C=CC(C)(O)CCC=C(C)C',        # linalool
    'fennel': 'COc1ccc(/C=C/C)cc1',             # anethole

    # === AMBER/BALSAMIC ===
    'vanilla': 'O=Cc1ccc(O)c(OC)c1',           # vanillin
    'mg_vanilla': 'O=Cc1ccc(O)c(OC)c1',
    'ug_vanilla': 'O=Cc1ccc(O)c(OC)c1',
    'tahiti_vanilla': 'O=Cc1ccc(O)c(OC)c1',
    'mx_vanilla': 'O=Cc1ccc(O)c(OC)c1',
    'in_vanilla': 'O=Cc1ccc(O)c(OC)c1',
    'benzoin': 'O=C(OCc1ccccc1)c2ccccc2',      # benzyl benzoate
    'sm_benzoin': 'O=C(OCc1ccccc1)c2ccccc2',
    'th_benzoin': 'O=C(OCc1ccccc1)c2ccccc2',
    'frankincense': 'CC1=CCC2CC1C2(C)C',
    'om_frankincense': 'CC1=CCC2CC1C2(C)C',
    'labdanum': 'CC(C)CCCC(C)CCCC(C)=CC(=O)O', # labdanolic acid approx
    'myrrh': 'CC1=CCCC(=C)C2CC(=C)CCC12',      # curzerene approx
    'so_myrrh': 'CC1=CCCC(=C)C2CC(=C)CCC12',
    'copal': 'CC1=CCC2CC1C2(C)C',
    'ambergris': 'CC12CCCCC1(C)C3CCC(C)(O3)C2C', # ambroxan
    'incense': 'CC1=CCC2CC1C2(C)C',
    'styrax': 'O=C(OCc1ccccc1)/C=C/c2ccccc2',  # cinnamyl benzoate
    'opoponax': 'CC1=CCCC(=C)C2CC(=C)CCC12',
    'elecampane': 'CC1(C)C2CCC3(CC2O)OC31',     # alantolactone approx
    'peru_balsam': 'O=C(/C=C/c1ccccc1)OCc2ccccc2', # benzyl cinnamate
    'tolu_balsam': 'O=C(OCc1ccccc1)c2ccccc2',
    'copaiba': 'CC1CCC2(CC1)C(=C)CCC2C(=C)C',

    # === GOURMAND ===
    'caramel': 'CC1=C(O)C(=O)C(C)O1',          # furaneol
    'chocolate': 'Cn1c2nc(=O)[nH]c2c(=O)n1C',  # theobromine
    'coffee': 'Cn1c(=O)c2c(ncn2C)n1C',          # caffeine
    'honey': 'OC(=O)Cc1ccccc1',                 # phenylacetic acid
    'praline': 'CC1=C(O)C(=O)OC1',              # maltol
    'tonka_bean': 'O=C1OC2=CC=CC=C2C=C1',       # coumarin
    'maple': 'CC1OC(=O)C(=C1O)C',               # sotolon
    'cocoa': 'Cn1c2nc(=O)[nH]c2c(=O)n1C',
    'almond': 'O=Cc1ccccc1',                     # benzaldehyde

    # === MUSK ===
    'white_musk': 'CC1(C)CC2=CC(=O)C3CCCC(C)C3C2CC1',  # galaxolide approx
    'musk': 'O=C1CCCCCCCCCCCCC1C',               # muscone approx
    'ambrette': 'O=C1OCCCCCCCCCCCCCC1',          # ambrettolide approx

    # === GREEN/AQUATIC ===
    'green_tea': 'O/C=C\\CCC',                   # cis-3-hexenol
    'fig': 'OC/C=C\\CCC',
    'sea_salt': 'O=C1C=CC2=C1C=COC2',
    'seaweed': 'O=C1C=CC2=C1C=COC2',
    'marine_note': 'O=C1C=CC2=C1C=COC2',         # calone
    'moss': 'COc1ccccc1O',
    'oakmoss': 'COc1ccccc1O',

    # === HERBAL ===
    'lavender': 'C=CC(C)(O)CCC=C(C)C',
    'fr_lavender': 'C=CC(C)(O)CCC=C(C)C',
    'bg_lavender': 'C=CC(C)(O)CCC=C(C)C',
    'uk_lavender': 'C=CC(C)(O)CCC=C(C)C',
    'rosemary': 'CC12CCC(CC1)C(C)(C)O2',        # 1,8-cineole
    'basil': 'C=CCc1ccc(O)c(OC)c1',
    'thyme': 'CC(C)c1ccc(C)c(O)c1',              # thymol
    'sage': 'CC12CCC(CC1)C(C)(C)O2',
    'chamomile': 'CC(CCC=C(C)C)C1CCC(C)(O)C1',  # bisabolol approx
    'eucalyptus': 'CC12CCC(CC1)C(C)(C)O2',
    'tea_tree': 'CC(O)C1CCC(=CC1)C',             # terpinen-4-ol
    'mint': 'CC(C)C1CCC(C)CC1O',                  # menthol
    'peppermint': 'CC(C)C1CCC(C)CC1O',
    'spearmint': 'CC(=C)C1CCC(=CC1)C=O',          # carvone
    'tarragon': 'C=CCc1ccc(OC)cc1',               # estragole
    'oregano': 'CC(C)c1ccc(O)cc1C',               # carvacrol
    'dill': 'CC(=C)C1CCC(=CC1)C=O',
    'mugwort': 'CC(=O)C1CCC(=CC1C)C',
    'wormwood': 'CC(C)C12CC3CC(=O)C1(C)C3O2',    # thujone approx
    'lemongrass': 'CC(C)=CCCC(C)=CC=O',           # citral
    'citronella': 'CC(CCC=C(C)C)CC=O',             # citronellal
    'palmarosa': 'OCC=C(C)CCC=C(C)C',
    'marjoram': 'CC(C)C1CCC(=CC1)C',
    'bay_leaf': 'CC12CCC(CC1)C(C)(C)O2',
    'hyssop': 'CC1=CCC2CC1C2(C)C',
    'verbena': 'CC(C)=CCCC(C)=CC=O',
    
    # === LEATHER/ANIMALIC ===
    'leather': 'CCc1cccc2ncccc12',                 # quinoline
    'suede': 'CCc1cccc2ncccc12',
    'castoreum': 'O=Cc1ccc(O)c(OC)c1',

    # === SYNTHETIC / AROMACHEM ===
    'iso_e_super': 'CC1(C)CCC2(CC1)C(=O)CCC2=CC', # iso e super
    'hedione': 'CCCC1CC(CC(=O)OC)C1',             # methyl dihydrojasmonate
    'hedione_hc': 'CCCC1CC(CC(=O)OC)C1',
    'galaxolide': 'CC1(C)CC2=CC(=O)C3CCCC(C)C3C2CC1',
    'cashmeran': 'CC1(C)CCC2=C1C(=O)CC(C)C2',
    'ambroxan': 'CC12CCCCC1(C)C3CCC(C)(O3)C2C',
    'javanol': 'CC(C)CC1CCC(CO)CC1',
    'habanolide': 'O=C1OCCCCCCCCCCC/C=C\\1',
    'ethylene_brassylate': 'O=C1OCCCCCCCCCCCOC1=O',
    'muscenone': 'O=CC/C=C\\CCCCCCCCCC',
    'velvione': 'O=C1CCCCCCCCCCCCC1',
    'exaltolide': 'O=C1OCCCCCCCCCCCCCC1',
    'calone': 'O=C1C=CC2=C1C=COC2',
    'linalool': 'C=CC(C)(O)CCC=C(C)C',
    'linalyl_acetate': 'CC(=O)OC(C)(C=C)CCC=C(C)C',
    'citronellol': 'OCC(C)CCC=C(C)C',
    'geraniol': 'OCC=C(C)CCC=C(C)C',
    'nerol': 'OC/C=C(\\C)CCC=C(C)C',
    'eugenol': 'C=CCc1ccc(O)c(OC)c1',
    'phenylethyl_alcohol': 'OCCc1ccccc1',
    'benzyl_acetate': 'CC(=O)OCc1ccccc1',
    'methyl_anthranilate': 'NC(=O)c1ccccc1OC',
    'indole': 'c1ccc2[nH]ccc2c1',
    'coumarin': 'O=C1OC2=CC=CC=C2C=C1',
    'vanillin': 'O=Cc1ccc(O)c(OC)c1',
    'heliotropin': 'O=COc1ccc2OCOc2c1',
    'anisaldehyde': 'O=Cc1ccc(OC)cc1',
    'methyl_salicylate': 'COC(=O)c1ccccc1O',
    'benzyl_salicylate': 'OC(=O)c1ccccc1OCc1ccccc1',
    'benzyl_benzoate': 'O=C(OCc1ccccc1)c1ccccc1',
    'dipropylene_glycol': 'CC(O)COC(C)CO',
    'diethyl_phthalate': 'CCOC(=O)c1ccccc1C(=O)OCC',
    'ionone_alpha': 'CC1=CC(=O)CC(/C=C/C(C)=C)C1',
    'ionone_beta': 'CC(/C=C/C1=C(C)C=CC(=O)C1)=C',
    'beta_ionone': 'CC(/C=C/C1=C(C)C=CC(=O)C1)=C',
    'alpha_isone': 'CC1=CC(=O)CC(/C=C/C(C)C)C1C',
    'safranal_synth': 'CC1=C(C=O)C(C)(C)C=C1',
    'damascone_alpha': 'CC(/C=C/C1=C(C)CCCC1(C)C)=O',
    'damascone_beta': 'CC(/C=C/C1C(=O)CCC1(C)C)=C',
    'methyl_dihydrojasmonate': 'CCCC1CC(CC(=O)OC)C1',
    'gamma_decalactone': 'CCCCCC1OC(=O)CC1',
    'delta_decalactone': 'CCCCCCC1OC(=O)CC1',
    'gamma_undecalactone': 'CCCCCCC1OC(=O)CC1',
    'gamma_nonalactone': 'CCCCC1OC(=O)CC1',
    'musk_ketone': 'CC1=CC(=CC(=C1[N+](=O)[O-])C)C(C)=O',
    'musk_xylene': 'CC1=CC(=C(C(=C1[N+](=O)[O-])C)[N+](=O)[O-])C(C)(C)C',
    'ethyl_maltol': 'CCC1=C(O)C(=O)OC=C1',
    'maltol': 'CC1=C(O)C(=O)OC=C1',
    'furaneol': 'CC1=C(O)C(=O)C(C)O1',
    'karanal': 'CC1(C)C2CCC(C)(C2)C1CC=O',
    'norlimbanol': 'CC(C)C1CCC(CC1)CC=O',
    'aldehyde_c10': 'CCCCCCCCCC=O',              # decanal
    'aldehyde_c11': 'CCCCCCCCCCC=O',             # undecanal 
    'aldehyde_c12': 'CCCCCCCCCCCC=O',            # dodecanal
    'aldehyde_c9': 'CCCCCCCCC=O',                # nonanal
    'aldehyde_c8': 'CCCCCCCC=O',                 # octanal
    'aldehyde_c14': 'CCCCCCCCCCCC(=O)C',         # methylundecanal actually C14
    'limonene': 'CC(=C)C1CCC(=CC1)C',
    'myrcene': 'CC(=C)CCC=C(C)C=C',
    'terpinolene': 'CC1=CCC(=C(C)C)CC1',
    'ocimene': 'CC(=C)C=CC=C(C)C',
    'pinene_alpha': 'CC1=CCC2CC1C2(C)C',
    'pinene_beta': 'CC1(C)C2CCC(=C)C1C2',
    'camphene': 'CC1(C)C2CCC1(C)C=C2',
    'cineole': 'CC12CCC(CC1)C(C)(C)O2',
    'menthol': 'CC(C)C1CCC(C)CC1O',
    'camphor': 'CC1(C)C2CCC1(C)C(=O)C2',
    'thymol': 'CC(C)c1ccc(C)c(O)c1',
    'carvacrol': 'CC(C)c1ccc(O)cc1C',
    'nerolidol': 'CC(=CCC/C(=C\\CC=C(C)C)CO)C',
    'farnesol': 'CC(=CCCC(=CCCC(=CCO)C)C)C',
    'squalene': 'CC(=CCC/C(=C\\CCC(=CCCC=C(C)CCC=C(C)C)C)C)C',
    'dihydromyrcenol': 'CC(C)(O)CCCC(C)C=C',
    'tetrahydrolinalool': 'CC(C)(O)CCCC(C)CC',
    'hydroxycitronellal': 'OCC(C)CCCC(C)CC=O',
    'cis_3_hexenol': 'OC/C=C\\CCC',
    'trans_2_hexenol': 'OC/C=C/CCC',
    'hexyl_cinnamic_aldehyde': 'O=C/C=C/c1ccccc1CCCCCC',
    'cinnamaldehyde': 'O=C/C=C/c1ccccc1',
    'cinnamyl_alcohol': 'OC/C=C/c1ccccc1',
    'phenylacetaldehyde': 'O=CCc1ccccc1',
    'acetaldehyde': 'CC=O',
    'decanal': 'CCCCCCCCCC=O',
    'undecanal': 'CCCCCCCCCCC=O',
    'dodecanal': 'CCCCCCCCCCCC=O',
    'tridecanal': 'CCCCCCCCCCCCC=O',
    'nonanal': 'CCCCCCCCC=O',
    'octanal': 'CCCCCCCC=O',
    'anethole': 'COc1ccc(/C=C/C)cc1',
    'piperine': 'O=C(/C=C/C=C/c1ccc2OCOc2c1)C3CCCCN3',
    'cuminaldehyde': 'CC(C)c1ccc(C=O)cc1',
}


def pubchem_name_to_smiles(name, retries=2):
    """PubChem PUG REST: 이름 → SMILES"""
    encoded = urllib.parse.quote(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/JSON"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PerfumeGNN/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(0.5)
        except:
            time.sleep(0.5)
    return None


def pubchem_cas_to_smiles(cas_number, retries=2):
    """PubChem: CAS → SMILES"""
    encoded = urllib.parse.quote(cas_number)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/JSON"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PerfumeGNN/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(0.5)
        except:
            time.sleep(0.5)
    return None


def canonicalize(smiles):
    if not HAS_RDKIT or not smiles:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else smiles


def main():
    import database as db_module

    print("=" * 60)
    print("  Phase 1: SMILES 100% Mapping (Multi-Strategy)")
    print("=" * 60)

    db_ings = db_module.get_all_ingredients()
    print(f"\n[1] DB ingredients: {len(db_ings)}")

    # 기존 molecules.json 매핑
    mol_path = os.path.join(DATA_DIR, 'molecules.json')
    existing = {}
    if os.path.exists(mol_path):
        with open(mol_path, 'r', encoding='utf-8') as f:
            for mol in json.load(f):
                for src in mol.get('source_ingredients', []):
                    existing[src] = mol['smiles']
    print(f"[2] Existing molecules.json: {len(existing)}")

    # 이전 실행 로드
    result = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            prev = json.load(f)
            # 이전에 성공한 것만 유지
            for k, v in prev.items():
                if v:
                    result[k] = v
        print(f"[2b] Previous successful: {len(result)}")

    stats = {'manual': 0, 'existing': 0, 'cas': 0, 'pubchem': 0, 'failed': 0}
    failed_list = []
    api_calls = 0

    for i, ing in enumerate(db_ings):
        ing_id = ing.get('id', '')
        name_en = ing.get('name_en', '') or ''
        cas = ing.get('cas_number', '') or ''

        # 이미 성공한 경우
        if ing_id in result:
            continue

        smiles = None

        # 전략 1: 수동 SMILES 테이블
        if ing_id in MANUAL_SMILES:
            smiles = MANUAL_SMILES[ing_id]
            if smiles:
                stats['manual'] += 1
        
        # 전략 2: 기존 molecules.json
        if not smiles and ing_id in existing:
            smiles = existing[ing_id]
            if smiles:
                stats['existing'] += 1
        
        # 전략 3: CAS → PubChem
        if not smiles and cas:
            smiles = pubchem_cas_to_smiles(cas)
            api_calls += 1
            time.sleep(0.2)
            if smiles:
                stats['cas'] += 1
        
        # 전략 4: name_en → PubChem
        if not smiles and name_en:
            smiles = pubchem_name_to_smiles(name_en)
            api_calls += 1
            time.sleep(0.2)
            if not smiles:
                # 괄호 제거 후 재시도
                clean = name_en.split('(')[0].strip()
                if clean != name_en:
                    smiles = pubchem_name_to_smiles(clean)
                    api_calls += 1
                    time.sleep(0.2)
            if not smiles:
                # id → space 변환 후 시도
                clean_id = ing_id.replace('_', ' ')
                smiles = pubchem_name_to_smiles(clean_id)
                api_calls += 1
                time.sleep(0.2)
            if smiles:
                stats['pubchem'] += 1
        
        if smiles:
            result[ing_id] = canonicalize(smiles)
        else:
            result[ing_id] = None
            stats['failed'] += 1
            failed_list.append(f"  {ing_id:35s} ({name_en})")

        if (i + 1) % 100 == 0 or i == len(db_ings) - 1:
            pct = sum(1 for v in result.values() if v) / len(db_ings) * 100
            print(f"  [{i+1}/{len(db_ings)}] coverage={pct:.1f}% api={api_calls} | "
                  f"manual={stats['manual']} exist={stats['existing']} cas={stats['cas']} "
                  f"pub={stats['pubchem']} fail={stats['failed']}")
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    # 최종 저장
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    total_ok = sum(1 for v in result.values() if v)
    total_fail = sum(1 for v in result.values() if not v)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total:   {len(result)}")
    print(f"  Mapped:  {total_ok} ({total_ok/len(result)*100:.1f}%)")
    print(f"  Failed:  {total_fail}")
    print(f"  Sources: manual={stats['manual']}, existing={stats['existing']}, "
          f"cas={stats['cas']}, pubchem={stats['pubchem']}")
    print(f"  API calls: {api_calls}")

    if failed_list:
        print(f"\n  FAILED ({len(failed_list)}):")
        for f_item in failed_list[:50]:
            print(f_item)
        if len(failed_list) > 50:
            print(f"  ... and {len(failed_list)-50} more")

    print(f"{'=' * 60}")
    return total_ok, total_fail


if __name__ == '__main__':
    main()
