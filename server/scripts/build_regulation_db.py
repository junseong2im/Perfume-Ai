"""
IFRA 51st Amendment DB + Industry Cost DB + Volatility Physics
=============================================================
Production-grade regulatory/economic/physical constraint data.
"""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# IFRA 51st Amendment Standards (June 2023)
# Category 4 = Fine Fragrance (Eau de Parfum, EDT, etc.)
# Format: CAS -> max allowed % in final formula (Cat 4)
# Sources: IFRA Standards Library, RIFM, industry references
# 'prohibited' = banned entirely
# ============================================================

IFRA_51ST_CAT4 = {
    # === PROHIBITED (complete ban) ===
    '101-86-0': 0,       # alpha-Hexylcinnamaldehyde (genotoxicity concern)
    '4602-84-0': 0,      # Farnesol (prohibited raw material) -> actually restricted
    '90-02-8': 0,        # Salicylaldehyde (prohibited)
    '93-15-2': 0,        # Methyl eugenol (genotoxic carcinogen)
    
    # === SEVERE RESTRICTION (< 0.1%) ===
    '97-53-0': 0.5,      # Eugenol (sensitizer) -> Cat 4: 0.5%
    '106-22-9': 1.5,     # Citronellol (sensitizer) -> Cat 4: 1.5%
    '106-24-1': 1.8,     # Geraniol (sensitizer) -> Cat 4: 1.8%
    '107-75-5': 0.5,     # Hydroxycitronellal (strong sensitizer) -> Cat 4: 0.5%
    '104-55-2': 0.07,    # Cinnamaldehyde (strong sensitizer)
    '91-64-5': 0.31,     # Coumarin (hepatotoxicity) -> Cat 4: 0.31%
    '140-67-0': 0.01,    # Estragole (genotoxic) -> Cat 4: 0.01%
    '5392-40-5': 2.5,    # Citral (sensitizer) -> Cat 4: 2.5%
    '5989-27-5': 4.0,    # d-Limonene (oxidation sensitizer) -> Cat 4: 4%
    '127-91-3': 4.0,     # beta-Pinene -> Cat 4: 4%
    '80-56-8': 4.0,      # alpha-Pinene -> Cat 4: 4%
    '78-70-6': 6.8,      # Linalool (with peroxide limit) -> Cat 4: 6.8%
    '60-12-8': 1.6,      # Phenylethyl alcohol -> Cat 4: 1.6%
    
    # === MODERATE RESTRICTION (0.1% - 2%) ===
    '101-39-3': 0.05,    # alpha-Methylcinnamaldehyde -> Cat 4: 0.05%
    '104-54-1': 1.5,     # Cinnamyl alcohol -> Cat 4: 1.5%
    '100-51-6': 4.0,     # Benzyl alcohol -> Cat 4: 4%
    '120-51-4': 8.5,     # Benzyl benzoate -> Cat 4: 8.5%
    '103-41-3': 0.5,     # Benzyl cinnamate -> Cat 4
    '118-58-1': 0.8,     # Benzyl salicylate -> Cat 4: 0.8%
    '4180-23-8': 1.0,    # trans-Anethole -> Cat 4: 1%
    '98-55-5': 5.0,      # alpha-Terpineol -> Cat 4: 5%
    '105-87-3': 2.6,     # Geranyl acetate -> Cat 4: 2.6%
    '115-95-7': 5.5,     # Linalyl acetate -> Cat 4: 5.5%
    '150-84-5': 4.0,     # Citronellyl acetate -> Cat 4: 4%
    
    # === MILD RESTRICTION (2% - 10%) ===
    '121-33-5': 99.0,    # Vanillin (no restriction basically)
    '121-32-4': 99.0,    # Ethyl vanillin
    '68-12-2': 0.01,     # DMF (prohibited in perfumery)
    '100-52-7': 2.0,     # Benzaldehyde -> Cat 4: 2%
    '98-86-2': 2.5,      # Acetophenone -> Cat 4: 2.5%
    '122-78-1': 0.6,     # Phenylacetaldehyde -> Cat 4: 0.6%
    '111-12-6': 1.0,     # Methyl heptine carbonate -> Cat 4: 1%
    '93-92-5': 3.0,      # alpha-Methylbenzyl acetate -> Cat 4: 3%
    
    # === ALLERGENS (EU Annex III mandatory disclosure) ===
    '5471-51-2': 99.0,   # Raspberry ketone (no IFRA limit)
    '4602-84-0': 0.6,    # Farnesol -> Cat 4: 0.6%
    '69-72-7': 99.0,     # Salicylic acid
    
    # === NITRO MUSKS (mostly prohibited) ===
    '81-14-1': 0,        # Musk ketone (prohibited)
    '81-15-2': 0,        # Musk xylene (prohibited)
    '145-39-1': 0,       # Musk tibetene (prohibited)
    
    # === COMMON SAFE (high or no limit) ===
    '68647-72-3': 99.0,  # Orange terpenes (no specific IFRA limit)
    '8000-25-7': 99.0,   # Rosemary oil
    '8008-99-9': 99.0,   # Garlic oil
    '8007-01-0': 4.0,    # Rose oil -> Cat 4: 4%
    '8000-28-0': 8.0,    # Lavender oil -> Cat 4: 8%
    '8015-01-8': 0.1,    # Ylang ylang oil -> Cat 4: 0.1%
    
    # === ISO E SUPER / HEDIONE / GALAXOLIDE (workhorses) ===
    '54464-57-2': 99.0,  # Iso E Super (no IFRA limit)
    '24851-98-7': 99.0,  # Hedione (no IFRA limit)
    '1222-05-5': 5.0,    # Galaxolide -> Cat 4: 5%
    '81-14-1': 0,        # Musk ketone (prohibited)
    
    # === AROMA CHEMICALS (high limits) ===
    '18479-58-8': 99.0,  # Dihydromyrcenol (no limit)
    '127-51-5': 2.0,     # alpha-Isomethyl ionone -> Cat 4: 2%
    '6259-76-3': 2.0,    # Hexyl salicylate -> Cat 4: 2%
    '89-43-0': 99.0,     # Oxoisophorone (no limit)
    '1205-17-0': 0.04,   # Piperonal (methylenedioxy) -> Cat 4: restricted
    '120-57-0': 0.04,    # Piperonal -> Cat 4: 0.04%
    '123-11-5': 99.0,    # Anisaldehyde (no limit)
    
    # === OAKMOSS / TREEMOSS (classic restricted) ===
    '90028-68-5': 0.1,   # Oakmoss (atranol content concern)
    '90028-67-4': 0.1,   # Treemoss
}

# ============================================================
# Industry Cost DB (USD per kg, approximate 2024 wholesale)
# Sources: Perfumer's Apprentice, Sigma-Aldrich, industry averages
# ============================================================

COST_PER_KG_USD = {
    # === NATURALS (expensive) ===
    'rose_absolute': 8000,
    'rose_essential_oil': 6000,
    'jasmine_absolute': 7500,
    'neroli_oil': 4500,
    'oud_oil': 50000,
    'sandalwood_oil': 2500,
    'vetiver_oil': 300,
    'patchouli_oil': 80,
    'ylang_ylang_oil': 200,
    'lavender_oil': 60,
    'bergamot_oil': 100,
    'lemon_oil': 25,
    'orange_oil': 8,
    'eucalyptus_oil': 20,
    'clove_oil': 15,
    'cinnamon_oil': 30,
    'geranium_oil': 120,
    'iris_absolute': 45000,
    'tuberose_absolute': 12000,
    
    # === SYNTHETIC (cheap, industry workhorses) ===
    'linalool': 15,
    'linalyl_acetate': 18,
    'limonene': 8,
    'geraniol': 25,
    'citronellol': 22,
    'citral': 20,
    'eugenol': 12,
    'vanillin': 15,
    'ethyl_vanillin': 25,
    'coumarin': 18,
    'hedione': 30,
    'iso_e_super': 35,
    'galaxolide': 40,
    'ambroxan': 120,
    'muscone': 200,
    'musk_ketone': 50,      # prohibited but listed for reference
    'dihydromyrcenol': 10,
    'benzyl_acetate': 12,
    'benzyl_benzoate': 8,
    'phenylethyl_alcohol': 20,
    'indole': 45,
    'skatole': 60,
    'carvone': 30,
    'menthol': 20,
    'camphor': 10,
    'cinnamaldehyde': 15,
    'acetophenone': 12,
    'benzaldehyde': 10,
    'phenylacetaldehyde': 25,
    'hydroxycitronellal': 35,
    'lilial': 40,
    'lyral': 45,
    'methyl_salicylate': 8,
    'alpha_ionone': 50,
    'beta_ionone': 55,
    'damascone': 250,
    'damascenone': 800,
    'safranal': 500,
    'geranyl_acetate': 20,
    'citronellyl_acetate': 25,
    'terpineol': 15,
    'cedrol': 35,
    'santalol_alpha': 180,
    'vetiverol': 80,
    'patchoulol': 90,
    'nerolidol': 60,
    
    # === DEFAULT for unknown ===
    '_default': 30,
}

# ============================================================
# Volatility Classification (based on MolWt + LogP heuristics)
# These are literature-standard thresholds
# ============================================================

VOLATILITY_THRESHOLDS = {
    'top':    {'molwt_max': 150, 'logp_max': 3.0},   # Light, volatile
    'middle': {'molwt_range': (150, 250), 'logp_range': (2.0, 5.0)},
    'base':   {'molwt_min': 250, 'logp_min': 4.0},   # Heavy, persistent
}

# Golden balance for fine fragrance (industry standard)
GOLDEN_BALANCE = {
    'top':    {'min_pct': 15, 'max_pct': 30, 'ideal_pct': 20},
    'middle': {'min_pct': 35, 'max_pct': 60, 'ideal_pct': 50},
    'base':   {'min_pct': 15, 'max_pct': 35, 'ideal_pct': 30},
}


def build_ifra_db():
    """Build complete IFRA DB as JSON"""
    db = {}
    for cas, max_pct in IFRA_51ST_CAT4.items():
        db[cas] = {
            'cas': cas,
            'max_pct_cat4': max_pct,
            'prohibited': max_pct == 0,
            'restricted': 0 < max_pct < 99,
        }
    return db


def build_cost_db():
    """Build cost DB as JSON"""
    return dict(COST_PER_KG_USD)


if __name__ == '__main__':
    ifra = build_ifra_db()
    print(f"IFRA 51st DB: {len(ifra)} substances")
    print(f"  Prohibited: {sum(1 for v in ifra.values() if v['prohibited'])}")
    print(f"  Restricted: {sum(1 for v in ifra.values() if v['restricted'])}")
    print(f"  Unrestricted: {sum(1 for v in ifra.values() if not v['prohibited'] and not v['restricted'])}")
    
    costs = build_cost_db()
    print(f"\nCost DB: {len(costs)} entries")
    
    # Save
    out_dir = os.path.join(BASE, '..', 'data', 'pom_upgrade')
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'ifra_51st_cat4.json'), 'w') as f:
        json.dump(ifra, f, indent=2)
    
    with open(os.path.join(out_dir, 'industry_costs.json'), 'w') as f:
        json.dump(costs, f, indent=2)
    
    print("\nSaved: ifra_51st_cat4.json, industry_costs.json")
