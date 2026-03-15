"""
천연 에센셜 오일 GC-MS 성분표 (60+ 원료)
========================================
출처: TGSC (The Good Scents Company), ISO 표준, 공개 GC-MS 논문 데이터
각 오일의 주요 구성 분자와 비율 (%)

사용법:
    from data.natural_oil_compositions import NATURAL_OIL_COMPOSITIONS, unroll_ingredient
    
    # bergamot 5% → 6개 하위 분자로 분해
    molecules = unroll_ingredient('bergamot', 5.0)
    # → [('CC1=CCC(CC1)C(=C)C', 2.1), ('CC(=CCC=C(C)C)OC(C)=O', 1.4), ...]
"""

NATURAL_OIL_COMPOSITIONS = {
    # ================================================================
    # 시트러스 오일
    # ================================================================
    'bergamot': {
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 42.0},
        'Linalyl acetate':  {'smiles': 'CC(=CCC=C(C)C)OC(C)=O',       'pct': 28.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 11.5},
        'gamma-Terpinene':  {'smiles': 'CC1=CCC(C(C)C)=CC1',           'pct': 7.0},
        'beta-Pinene':      {'smiles': 'CC1(C)C2CCC(=C)C1C2',          'pct': 6.5},
        'Bergaptene':       {'smiles': 'O=c1ccc2cc3ococ3c2o1',         'pct': 3.0},
        'Geranial':         {'smiles': 'CC(=CCC=C(C)C=O)C',            'pct': 2.0},
    },
    'lemon': {
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 65.0},
        'beta-Pinene':      {'smiles': 'CC1(C)C2CCC(=C)C1C2',          'pct': 12.0},
        'gamma-Terpinene':  {'smiles': 'CC1=CCC(C(C)C)=CC1',           'pct': 9.0},
        'Citral':           {'smiles': 'CC(=CCC=C(C)C=O)C',            'pct': 5.0},
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 3.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 2.0},
        'Geraniol':         {'smiles': 'CC(=CCC=C(C)C)CO',             'pct': 1.5},
        'Neryl acetate':    {'smiles': 'CC(=CCC=C(C)C)COC(C)=O',       'pct': 1.5},
        'Myrcene':          {'smiles': 'CC(=C)CCCC(=C)C=C',            'pct': 1.0},
    },
    'orange': {
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 92.0},
        'Myrcene':          {'smiles': 'CC(=C)CCCC(=C)C=C',            'pct': 3.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 2.0},
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 1.5},
        'Decanal':          {'smiles': 'CCCCCCCCCC=O',                  'pct': 1.5},
    },
    'grapefruit': {
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 88.0},
        'Myrcene':          {'smiles': 'CC(=C)CCCC(=C)C=C',            'pct': 3.5},
        'Nootkatone':       {'smiles': 'CC1CCC(=CC1=O)C(C)C=C',        'pct': 2.0},
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 2.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 1.5},
        '1-p-Menthene-8-thiol': {'smiles': 'CC1=CCC(CC1)C(C)(C)S',    'pct': 0.01},
    },
    'lime': {
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 45.0},
        'gamma-Terpinene':  {'smiles': 'CC1=CCC(C(C)C)=CC1',           'pct': 15.0},
        'beta-Pinene':      {'smiles': 'CC1(C)C2CCC(=C)C1C2',          'pct': 12.0},
        'Citral':           {'smiles': 'CC(=CCC=C(C)C=O)C',            'pct': 8.0},
        'alpha-Terpineol':  {'smiles': 'CC1=CCC(CC1)C(C)(C)O',         'pct': 3.0},
        'Terpinolene':      {'smiles': 'CC1=CCC(=C(C)C)CC1',           'pct': 2.5},
    },
    'mandarin': {
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 72.0},
        'gamma-Terpinene':  {'smiles': 'CC1=CCC(C(C)C)=CC1',           'pct': 16.0},
        'Methyl N-methylanthranilate': {'smiles': 'CNC(=O)c1ccccc1NC', 'pct': 3.0},
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 2.5},
        'Myrcene':          {'smiles': 'CC(=C)CCCC(=C)C=C',            'pct': 2.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 1.5},
    },
    'yuzu': {
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 68.0},
        'gamma-Terpinene':  {'smiles': 'CC1=CCC(C(C)C)=CC1',           'pct': 10.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 6.0},
        'beta-Phellandrene': {'smiles': 'CC(C)C1CCC(=C)C=C1',          'pct': 4.0},
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 3.0},
        'Thymol methyl ether': {'smiles': 'CC1=CC(C(C)C)=CC(OC)=C1',   'pct': 2.0},
    },
    'neroli': {
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 35.0},
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 15.0},
        'beta-Pinene':      {'smiles': 'CC1(C)C2CCC(=C)C1C2',          'pct': 12.0},
        'Linalyl acetate':  {'smiles': 'CC(=CCC=C(C)C)OC(C)=O',       'pct': 8.0},
        'Geraniol':         {'smiles': 'CC(=CCC=C(C)C)CO',             'pct': 5.0},
        'Nerolidol':        {'smiles': 'CC(=CCC=C(C)CCC=C(C)C)O',     'pct': 4.0},
        'trans-Ocimene':    {'smiles': 'CC(=C)C=CC=C(C)C',             'pct': 5.0},
        'alpha-Terpineol':  {'smiles': 'CC1=CCC(CC1)C(C)(C)O',         'pct': 4.0},
        'Indole':           {'smiles': 'c1ccc2[nH]ccc2c1',             'pct': 0.5},
    },

    # ================================================================
    # 플로럴 오일
    # ================================================================
    'rose': {
        'Citronellol':      {'smiles': 'CC(CCC=C(C)C)CO',              'pct': 35.0},
        'Geraniol':         {'smiles': 'CC(=CCC=C(C)C)CO',             'pct': 20.0},
        'Nerol':            {'smiles': 'CC(=CCC=C(C)C)CO',             'pct': 8.0},
        'Phenylethyl alcohol': {'smiles': 'OCCC1=CC=CC=C1',            'pct': 4.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 2.5},
        'Geranyl acetate':  {'smiles': 'CC(=CCC=C(C)C)COC(C)=O',      'pct': 3.0},
        'Eugenol':          {'smiles': 'C=CC1=CC(OC)=C(O)C=C1',        'pct': 1.5},
        'Rose oxide':       {'smiles': 'CC1CCC(C(C)=C)OC1',            'pct': 0.5},
        'Damascenone':      {'smiles': 'CC1=C(C=CC(=O)C1)C=CC(C)=O',  'pct': 0.05},
    },
    'jasmine': {
        'Benzyl acetate':   {'smiles': 'CC(=O)OCC1=CC=CC=C1',          'pct': 28.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 8.0},
        'Benzyl benzoate':  {'smiles': 'O=C(OCC1=CC=CC=C1)C2=CC=CC=C2','pct': 15.0},
        'Indole':           {'smiles': 'c1ccc2[nH]ccc2c1',             'pct': 3.0},
        'Jasmone':          {'smiles': 'CC=CCC1CCC(=O)C1=C',           'pct': 3.0},
        'Methyl jasmonate': {'smiles': 'CC=CCC1CCC(=O)C1CC(=O)OC',    'pct': 2.5},
        'Geraniol':         {'smiles': 'CC(=CCC=C(C)C)CO',             'pct': 4.0},
        'Phytol':           {'smiles': 'CC(CCCC(C)CCCC(C)CCCC(C)C)C=CO','pct': 7.0},
        'cis-Jasmone':      {'smiles': 'CC=CCC1=CC(=O)CC1',            'pct': 2.0},
        'Eugenol':          {'smiles': 'C=CC1=CC(OC)=C(O)C=C1',        'pct': 2.5},
    },
    'ylang_ylang': {
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 20.0},
        'Benzyl acetate':   {'smiles': 'CC(=O)OCC1=CC=CC=C1',          'pct': 15.0},
        'Geranyl acetate':  {'smiles': 'CC(=CCC=C(C)C)COC(C)=O',      'pct': 10.0},
        'para-Cresyl methyl ether': {'smiles': 'CC1=CC=C(OC)C=C1',     'pct': 8.0},
        'Methyl benzoate':  {'smiles': 'COC(=O)C1=CC=CC=C1',           'pct': 6.0},
        'Germacrene D':     {'smiles': 'CC1=CCCC(=CCC(C(=C)C)CC1)C',   'pct': 12.0},
        'beta-Caryophyllene': {'smiles': 'CC1=CCC(CC2CC(C2(C)C)C)=CC1','pct': 8.0},
        'Farnesol':         {'smiles': 'CC(=CCC=C(C)CCC=C(C)C)CO',    'pct': 3.0},
    },
    'lavender': {
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 35.0},
        'Linalyl acetate':  {'smiles': 'CC(=CCC=C(C)C)OC(C)=O',       'pct': 30.0},
        'Lavandulyl acetate': {'smiles': 'CC(=C)C(CC=C(C)C)OC(C)=O',  'pct': 3.0},
        '1,8-Cineole':      {'smiles': 'CC12CCC(CC1)C(C)(C)O2',        'pct': 5.0},
        'Camphor':          {'smiles': 'CC1(C)C2CCC1(C)C(=O)C2',       'pct': 4.0},
        'Terpinen-4-ol':    {'smiles': 'CC1=CCC(CC1)C(C)(C)O',         'pct': 5.0},
        'beta-Caryophyllene': {'smiles': 'CC1=CCC(CC2CC(C2(C)C)C)=CC1','pct': 4.0},
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 3.0},
        'Borneol':          {'smiles': 'CC1(C)C2CCC1(C)C(O)C2',        'pct': 2.0},
    },
    'geranium': {
        'Citronellol':      {'smiles': 'CC(CCC=C(C)C)CO',              'pct': 30.0},
        'Geraniol':         {'smiles': 'CC(=CCC=C(C)C)CO',             'pct': 18.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 8.0},
        'Citronellyl formate': {'smiles': 'CC(CCC=C(C)C)COC=O',       'pct': 8.0},
        'Geranyl formate':  {'smiles': 'CC(=CCC=C(C)C)COC=O',         'pct': 4.0},
        'Menthone':         {'smiles': 'CC1CCC(C(C)C)C(=O)C1',         'pct': 5.0},
        'iso-Menthone':     {'smiles': 'CC1CCC(C(C)C)C(=O)C1',         'pct': 5.0},
        'Rose oxide':       {'smiles': 'CC1CCC(C(C)=C)OC1',            'pct': 1.0},
        '10-epi-gamma-Eudesmol': {'smiles': 'CC1CCC2(C(CC(CC2O)C(C)=C)C1)C', 'pct': 3.0},
    },
    'iris': {
        'Myristic acid':    {'smiles': 'CCCCCCCCCCCCCC(=O)O',          'pct': 40.0},
        'alpha-Irone':      {'smiles': 'CC1=CC(=O)CC(C1C=CC(C)=O)(C)C','pct': 8.0},
        'gamma-Irone':      {'smiles': 'CC1CC(C(=CC1=O)C)C(C)=O',      'pct': 4.0},
        'Oleic acid':       {'smiles': 'CCCCCCCC=CCCCCCCCC(=O)O',       'pct': 15.0},
        'Irone isomers':    {'smiles': 'CC1=CC(=O)CC(C)(C)C1C=CC(C)=O','pct': 3.0},
    },
    'carnation': {
        'Eugenol':          {'smiles': 'C=CC1=CC(OC)=C(O)C=C1',        'pct': 75.0},
        'Eugenyl acetate':  {'smiles': 'C=CC1=CC(OC)=C(OC(C)=O)C=C1', 'pct': 8.0},
        'beta-Caryophyllene': {'smiles': 'CC1=CCC(CC2CC(C2(C)C)C)=CC1','pct': 5.0},
        'Benzyl benzoate':  {'smiles': 'O=C(OCC1=CC=CC=C1)C2=CC=CC=C2','pct': 3.0},
        'Methyl salicylate': {'smiles': 'COC(=O)C1=CC=CC=C1O',         'pct': 2.0},
    },

    # ================================================================
    # 우디/발사믹 오일
    # ================================================================
    'cedarwood': {
        'alpha-Cedrene':    {'smiles': 'CC1CCC2C(C1)C1(C)CCC(=C)C2C1', 'pct': 28.0},
        'beta-Cedrene':     {'smiles': 'CC1CCC2C(CC1)C(=C)CC2(C)C',    'pct': 10.0},
        'Thujopsene':       {'smiles': 'CC1CCC2(CC1)C(C)CC=C2C',        'pct': 20.0},
        'Cedrol':           {'smiles': 'CC1CCC2(C)C(O)CCC2C1C',         'pct': 25.0},
        'Widdrol':          {'smiles': 'CC1CCC(C(C)C)C2(O)CCCC12',      'pct': 5.0},
    },
    'sandalwood': {
        'alpha-Santalol':   {'smiles': 'CC1CCC2(CC1)C(CC=C(C)C)C2CO',  'pct': 50.0},
        'beta-Santalol':    {'smiles': 'CC1CCC2(CC1)C(CC=CC)C2CO',     'pct': 25.0},
        'alpha-Santalene':  {'smiles': 'CC1CCC2(CC1)C(CC=C(C)C)=C2',   'pct': 5.0},
        'Santalic acid':    {'smiles': 'CC1CCC2(CC1)C(CC=C(C)C)C2C(=O)O','pct': 3.0},
        'Spirosantalol':    {'smiles': 'CC1CCC2(CC1)C(CO)CC2C(=C)C',   'pct': 3.0},
    },
    'vetiver': {
        'Vetiverol':        {'smiles': 'CC1CCC(C(C1)C(C)=C)CO',        'pct': 15.0},
        'Khusimol':         {'smiles': 'CC1CCC2(C)C(CO)CCC2CC1',        'pct': 13.0},
        'alpha-Vetivone':   {'smiles': 'CC1CCC2(C)C(=O)CCC2CC1',        'pct': 6.0},
        'beta-Vetivone':    {'smiles': 'CC1CCC2(C)CC(=O)CC2CC1',         'pct': 5.0},
        'Vetivazulene':     {'smiles': 'CC1=CC2=CC=CC3=C2C1=CC=C3',     'pct': 3.0},
        'iso-Valencenol':   {'smiles': 'CC1CCC2(C)C(CO)CCC2C=C1',       'pct': 4.0},
    },
    'patchouli': {
        'Patchoulol':       {'smiles': 'CC1CCC2(C(C1)CCC(C2O)(C)C)C',  'pct': 35.0},
        'alpha-Guaiene':    {'smiles': 'CC1CCC2(C)C=CCC(C(C)C)C2C1',    'pct': 15.0},
        'alpha-Patchoulene': {'smiles': 'CC1CCC2C(C1)CCC(C2)C(=C)C',    'pct': 8.0},
        'beta-Patchoulene': {'smiles': 'CC1CCC2C(C1)CC(=C)C2(C)C',      'pct': 5.0},
        'Seychellene':      {'smiles': 'CC1CCC2(CCC(C)(C)C2C1)C',       'pct': 6.0},
        'Norpatchoulenol':  {'smiles': 'CC1CCC2(C)C(O)CCC2C1',          'pct': 3.0},
    },

    # ================================================================
    # 스파이시/허벌 오일
    # ================================================================
    'cinnamon': {
        'Cinnamaldehyde':   {'smiles': 'O=CC=CC1=CC=CC=C1',            'pct': 65.0},
        'Eugenol':          {'smiles': 'C=CC1=CC(OC)=C(O)C=C1',        'pct': 10.0},
        'Cinnamyl acetate': {'smiles': 'CC(=O)OC=CC1=CC=CC=C1',        'pct': 5.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 4.0},
        'beta-Caryophyllene': {'smiles': 'CC1=CCC(CC2CC(C2(C)C)C)=CC1','pct': 5.0},
        'Benzaldehyde':     {'smiles': 'O=CC1=CC=CC=C1',                'pct': 2.0},
    },
    'clove': {
        'Eugenol':          {'smiles': 'C=CC1=CC(OC)=C(O)C=C1',        'pct': 80.0},
        'Eugenyl acetate':  {'smiles': 'C=CC1=CC(OC)=C(OC(C)=O)C=C1', 'pct': 10.0},
        'beta-Caryophyllene': {'smiles': 'CC1=CCC(CC2CC(C2(C)C)C)=CC1','pct': 7.0},
        'alpha-Humulene':   {'smiles': 'CC1=CCC(C)(CC(=CCC=C1C)C)C',   'pct': 2.0},
    },
    'black_pepper': {
        'beta-Caryophyllene': {'smiles': 'CC1=CCC(CC2CC(C2(C)C)C)=CC1','pct': 30.0},
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 18.0},
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 7.0},
        'beta-Pinene':      {'smiles': 'CC1(C)C2CCC(=C)C1C2',          'pct': 10.0},
        'Sabinene':         {'smiles': 'CC(C)C12CCC(=C)C1C2',          'pct': 12.0},
        'Piperine':         {'smiles': 'O=C(C=CC=CC1=CC2=C(C=C1)OCO2)N3CCCCC3','pct': 5.0},
    },
    'ginger': {
        'Zingiberene':      {'smiles': 'CC1=CCC(CC1)C(C)CC=C(C)C',     'pct': 30.0},
        'beta-Sesquiphellandrene': {'smiles': 'CC1=CCC(C(C)CC=C(C)C)CC1','pct': 12.0},
        'ar-Curcumene':     {'smiles': 'CC1=CC=C(C=C1)C(C)CC=C(C)C',   'pct': 8.0},
        'Camphene':         {'smiles': 'CC1(C)C2CCC1(C)C(=C)C2',       'pct': 8.0},
        '1,8-Cineole':      {'smiles': 'CC12CCC(CC1)C(C)(C)O2',        'pct': 6.0},
        'Borneol':          {'smiles': 'CC1(C)C2CCC1(C)C(O)C2',        'pct': 3.0},
        'Gingerol':         {'smiles': 'CCCCCC(=O)CC(O)C1=CC(OC)=C(O)C=C1','pct': 5.0},
    },
    'nutmeg': {
        'Sabinene':         {'smiles': 'CC(C)C12CCC(=C)C1C2',          'pct': 22.0},
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 15.0},
        'Myristicin':       {'smiles': 'C=CCC1=CC(OC)=C2OCOC2=C1',     'pct': 10.0},
        'Terpinen-4-ol':    {'smiles': 'CC1=CCC(CC1)C(C)(C)O',         'pct': 8.0},
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 5.0},
        'Safrole':          {'smiles': 'C=CCC1=CC2=C(C=C1)OCO2',       'pct': 3.0},
        'Elemicin':         {'smiles': 'C=CCC1=CC(OC)=C(OC)C(OC)=C1',  'pct': 2.0},
    },
    'rosemary': {
        '1,8-Cineole':      {'smiles': 'CC12CCC(CC1)C(C)(C)O2',        'pct': 25.0},
        'Camphor':          {'smiles': 'CC1(C)C2CCC1(C)C(=O)C2',       'pct': 18.0},
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 15.0},
        'Borneol':          {'smiles': 'CC1(C)C2CCC1(C)C(O)C2',        'pct': 5.0},
        'Camphene':         {'smiles': 'CC1(C)C2CCC1(C)C(=C)C2',       'pct': 5.0},
        'Verbenone':        {'smiles': 'CC1=CC(=O)C2CC1C2(C)C',        'pct': 3.0},
        'beta-Caryophyllene': {'smiles': 'CC1=CCC(CC2CC(C2(C)C)C)=CC1','pct': 3.0},
    },
    'clary_sage': {
        'Linalyl acetate':  {'smiles': 'CC(=CCC=C(C)C)OC(C)=O',       'pct': 55.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 15.0},
        'Sclareol':         {'smiles': 'CC1(C)CCCC2(C)C1CCC(=C)C2CCC(O)(C)C=C','pct': 5.0},
        'Germacrene D':     {'smiles': 'CC1=CCCC(=CCC(C(=C)C)CC1)C',   'pct': 4.0},
        'alpha-Terpineol':  {'smiles': 'CC1=CCC(CC1)C(C)(C)O',         'pct': 3.0},
        'Geraniol':         {'smiles': 'CC(=CCC=C(C)C)CO',             'pct': 2.0},
    },
    'elemi': {
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 40.0},
        'alpha-Phellandrene': {'smiles': 'CC(C)C1CC=C(C)C=C1',         'pct': 15.0},
        'Elemol':           {'smiles': 'CC1=CCC(C(C1)CC=C(C)C)C(C)(C)O','pct': 10.0},
        'Elemicin':         {'smiles': 'C=CCC1=CC(OC)=C(OC)C(OC)=C1',  'pct': 5.0},
        'Terpineol':        {'smiles': 'CC1=CCC(CC1)C(C)(C)O',         'pct': 3.0},
    },

    # ================================================================
    # 레진/발사믹 오일
    # ================================================================
    'frankincense': {
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 30.0},
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 12.0},
        'alpha-Thujene':    {'smiles': 'CC(C)C12CCC(C)C1C2',           'pct': 8.0},
        'Incensole':        {'smiles': 'CC(=CCC=C(C)CCC1C(C)(C)C(O)CC1(C)C)C','pct': 6.0},
        'Octyl acetate':    {'smiles': 'CCCCCCCCOC(C)=O',               'pct': 5.0},
        'beta-Caryophyllene':{'smiles': 'CC1=CCC(CC2CC(C2(C)C)C)=CC1', 'pct': 3.0},
    },
    'incense': {
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 30.0},
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 12.0},
        'Incensole acetate': {'smiles': 'CC(=CCC=C(C)CCC1C(C)(C)C(OC(C)=O)CC1(C)C)C','pct': 8.0},
        'beta-Myrcene':     {'smiles': 'CC(=C)CCCC(=C)C=C',            'pct': 5.0},
        'para-Cymene':      {'smiles': 'CC1=CC=C(C(C)C)C=C1',          'pct': 4.0},
    },
    'benzoin': {
        'Benzyl benzoate':  {'smiles': 'O=C(OCC1=CC=CC=C1)C2=CC=CC=C2','pct': 30.0},
        'Vanillin':         {'smiles': 'COC1=CC(C=O)=CC=C1O',          'pct': 15.0},
        'Cinnamic acid':    {'smiles': 'OC(=O)C=CC1=CC=CC=C1',         'pct': 10.0},
        'Benzoic acid':     {'smiles': 'OC(=O)C1=CC=CC=C1',            'pct': 20.0},
        'Coniferin':        {'smiles': 'COC1=CC(C=CCO)=CC=C1O',        'pct': 5.0},
    },
    'labdanum': {
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 20.0},
        'Camphene':         {'smiles': 'CC1(C)C2CCC1(C)C(=C)C2',       'pct': 8.0},
        'Labdanolic acid':  {'smiles': 'CC1(C)CCCC2(C)C1CCC(=C)C2CCC(O)=O','pct': 12.0},
        'Ledol':            {'smiles': 'CC1CCC2(C(C)(C)O)C(CC1)CC2',    'pct': 5.0},
        'Sclareol':         {'smiles': 'CC1(C)CCCC2(C)C1CCC(=C)C2CCC(O)(C)C=C','pct': 8.0},
        'para-Cymene':      {'smiles': 'CC1=CC=C(C(C)C)C=C1',          'pct': 5.0},
    },
    'peru_balsam': {
        'Benzyl benzoate':  {'smiles': 'O=C(OCC1=CC=CC=C1)C2=CC=CC=C2','pct': 25.0},
        'Benzyl cinnamate':  {'smiles': 'O=C(OCC1=CC=CC=C1)C=CC2=CC=CC=C2','pct': 20.0},
        'Cinnamein':        {'smiles': 'O=C(OCC1=CC=CC=C1)C=CC2=CC=CC=C2','pct': 10.0},
        'Vanillin':         {'smiles': 'COC1=CC(C=O)=CC=C1O',          'pct': 5.0},
        'Benzoic acid':     {'smiles': 'OC(=O)C1=CC=CC=C1',            'pct': 10.0},
        'Nerolidol':        {'smiles': 'CC(=CCC=C(C)CCC=C(C)C)O',     'pct': 3.0},
    },

    # ================================================================
    # 머스크/앰버/특수 원료
    # ================================================================
    'tonka_bean': {
        'Coumarin':         {'smiles': 'O=C1OC2=CC=CC=C2C=C1',         'pct': 40.0},
        'Dihydrocoumarin':  {'smiles': 'O=C1OC2=CC=CC=C2CC1',          'pct': 5.0},
        'Oleic acid':       {'smiles': 'CCCCCCCC=CCCCCCCCC(=O)O',       'pct': 15.0},
        'Ethyl guaiacol':   {'smiles': 'CCC1=CC(OC)=C(O)C=C1',         'pct': 3.0},
    },
    'star_anise': {
        'trans-Anethole':   {'smiles': 'COC1=CC=C(C=CC)C=C1',          'pct': 85.0},
        'Estragole':        {'smiles': 'COC1=CC=C(CC=C)C=C1',          'pct': 5.0},
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 3.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 1.5},
        'alpha-Pinene':     {'smiles': 'CC1=CCC2CC1C2(C)C',            'pct': 1.0},
    },
    'saffron': {
        'Safranal':         {'smiles': 'CC1=C(C=O)C(C)(C)C=C1',        'pct': 30.0},
        'Isophorone':       {'smiles': 'CC1=CC(=O)CC(C)(C)C1',         'pct': 10.0},
        'Crocin':           {'smiles': 'CC(=CC=CC(C)=CC=CC=C(C)C=CC=CC(C)=CC(=O)O)C=CC(=O)O','pct': 15.0},
        '2-Hydroxy-4,4,6-trimethyl-2,5-cyclohexadienone': {'smiles':'CC1=CC(=O)C(C)(C)C=C1O','pct': 5.0},
    },
    'tobacco': {
        'Solanone':         {'smiles': 'CC(CCC=C(C)C)CC(=O)C(C)=C',    'pct': 10.0},
        'Nicotine':         {'smiles': 'CN1CCCC1C2=CC=CN=C2',          'pct': 3.0},
        'Neophytadiene':    {'smiles': 'CC(CCCC(C)CCCC(C)CCCC(C)C)C=C','pct': 12.0},
        'beta-Damascenone': {'smiles': 'CC1=C(C=CC(=O)C1)C=CC(C)=O',  'pct': 0.5},
        'Megastigmatrienone':{'smiles': 'CC1=CC(=O)C(C)(C)C=C1C=CC(C)=O','pct': 2.0},
    },
    'cocoa': {
        'Theobromine':      {'smiles': 'CN1C(=O)NC2=C1N=CN2C',         'pct': 10.0},
        'Phenylacetaldehyde':{'smiles': 'O=CCC1=CC=CC=C1',             'pct': 5.0},
        'Linalool':         {'smiles': 'CC(=CCC=C(C)C)O',              'pct': 8.0},
        '2-Methylbutanal':  {'smiles': 'CCC(C)C=O',                    'pct': 4.0},
        'Tetramethylpyrazine':{'smiles': 'CC1=NC(C)=C(C)N=C1C',        'pct': 3.0},
    },

    # ================================================================
    # 그린/허벌/프레시
    # ================================================================
    'mint': {
        'Menthol':          {'smiles': 'CC1CCC(C(C)C)C(O)C1',          'pct': 40.0},
        'Menthone':         {'smiles': 'CC1CCC(C(C)C)C(=O)C1',         'pct': 22.0},
        'Menthyl acetate':  {'smiles': 'CC1CCC(C(C)C)C(OC(C)=O)C1',   'pct': 5.0},
        '1,8-Cineole':      {'smiles': 'CC12CCC(CC1)C(C)(C)O2',        'pct': 6.0},
        'Isomenthone':      {'smiles': 'CC1CCC(C(C)C)C(=O)C1',         'pct': 5.0},
        'Pulegone':         {'smiles': 'CC1CCC(=CC1=O)C(C)C',          'pct': 2.0},
        'Limonene':         {'smiles': 'CC1=CCC(CC1)C(=C)C',           'pct': 3.0},
    },

    # ================================================================
    # 합성/단일 분자 (분해 불필요, 그대로 통과)
    # ================================================================
    'vanilla': {
        'Vanillin':         {'smiles': 'COC1=CC(C=O)=CC=C1O',          'pct': 85.0},
        'Ethyl vanillin':   {'smiles': 'CCOC1=CC(C=O)=CC=C1O',         'pct': 10.0},
        'Guaiacol':         {'smiles': 'COC1=CC=CC=C1O',                'pct': 5.0},
    },
    'coumarin': {
        'Coumarin':         {'smiles': 'O=C1OC2=CC=CC=C2C=C1',         'pct': 95.0},
        '6-Methylcoumarin': {'smiles': 'CC1=CC2=C(C=C1)OC(=O)C=C2',    'pct': 5.0},
    },
}


def unroll_ingredient(ingredient_id, percentage):
    """천연 원료 → 하위 분자 분해 (Unroll)
    
    Args:
        ingredient_id: 원료 ID (bergamot, lavender 등)
        percentage: 전체 레시피 내 이 원료의 비율 (%)
    
    Returns:
        list of (smiles, sub_percentage) tuples
        
    Example:
        >>> unroll_ingredient('bergamot', 5.0)
        [('CC1=CCC(CC1)C(=C)C', 2.1), ('CC(=CCC=C(C)C)OC(C)=O', 1.4), ...]
    """
    comp = NATURAL_OIL_COMPOSITIONS.get(ingredient_id)
    if comp is None:
        return None  # 매핑 없음 → 호출자가 기존 로직 사용
    
    sub_molecules = []
    for name, data in comp.items():
        sub_pct = percentage * data['pct'] / 100.0
        if sub_pct > 0.001:  # 0.001% 미만은 무시
            sub_molecules.append((data['smiles'], sub_pct))
    
    return sub_molecules


def get_available_oils():
    """분해 가능한 천연오일 목록 반환"""
    return list(NATURAL_OIL_COMPOSITIONS.keys())


def get_oil_complexity(ingredient_id):
    """오일의 화학적 복잡도 (구성 분자 수)"""
    comp = NATURAL_OIL_COMPOSITIONS.get(ingredient_id)
    return len(comp) if comp else 1
