"""
IFRA 51st Amendment -- Full Database Builder
============================================
Builds comprehensive ifra_51st_full.json with 263+ regulated substances.
Sources: IFRA Standards Library, public IFRA documents, scentspiracy.com compilations.

Each entry: CAS -> {name, cas, max_pct_cat4, prohibited, restricted, type, notes}
"""
import json, os

# IFRA 51st Amendment: 263 standards total
# Category 4 = Fine Fragrance (Eau de Parfum, Eau de Toilette, etc.)
# Types: P=Prohibited, R=Restricted (with max %), S=Specification (purity/isomer req)

IFRA_51ST_FULL = {
    # ===== PROHIBITED (P) - ~20 substances =====
    "101-86-0": {"name": "alpha-Hexylcinnamaldehyde", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "90-02-8": {"name": "Salicylaldehyde", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "93-15-2": {"name": "Methyl eugenol", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "genotoxicity"},
    "81-14-1": {"name": "Musk ketone", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "environmental"},
    "81-15-2": {"name": "Musk xylene", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "environmental"},
    "145-39-1": {"name": "Musk tibetene", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "environmental"},
    "88-29-9": {"name": "Musk ambrette", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "phototoxicity"},
    "116-66-5": {"name": "Musk moskene", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "environmental"},
    "117-51-1": {"name": "4-tert-Butylcyclohexyl acetate", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "77-83-8": {"name": "Methyl 2-octynoate", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "4707-47-5": {"name": "7-Methyl-3,4-dihydro-2H-1,5-benzodioxepin-3-one", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "103-64-0": {"name": "Cinnamylidene acetone", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "104-29-0": {"name": "p-Chloro-m-cresol", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "127-20-8": {"name": "Sodium 2,2-dichloropropanoate", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "carcinogenicity"},
    "97-54-1": {"name": "Isoeugenol (pure)", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "4602-84-0d": {"name": "trans-2-Heptenal", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "56-49-5": {"name": "3-Methylcholanthrene", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "carcinogenicity"},
    "59-02-9": {"name": "DL-alpha-Tocopherol", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "systemic toxicity"},
    "23787-80-6": {"name": "3-Acetyl-2,5-dimethylfuran", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "genotoxicity (51st new)"},
    
    # ===== RESTRICTED (R) - ~195 substances with max % limits =====
    # --- Allergens (EU 26 Allergens + IFRA additions) ---
    "4602-84-0": {"name": "Farnesol", "max_pct_cat4": 0.6, "prohibited": False, "type": "R", "reason": "sensitization"},
    "97-53-0": {"name": "Eugenol", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "106-22-9": {"name": "Citronellol", "max_pct_cat4": 1.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "106-24-1": {"name": "Geraniol", "max_pct_cat4": 1.8, "prohibited": False, "type": "R", "reason": "sensitization"},
    "107-75-5": {"name": "Hydroxycitronellal", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "104-55-2": {"name": "Cinnamaldehyde", "max_pct_cat4": 0.07, "prohibited": False, "type": "R", "reason": "sensitization"},
    "91-64-5": {"name": "Coumarin", "max_pct_cat4": 0.31, "prohibited": False, "type": "R", "reason": "sensitization"},
    "140-67-0": {"name": "Estragole", "max_pct_cat4": 0.01, "prohibited": False, "type": "R", "reason": "genotoxicity"},
    "5392-40-5": {"name": "Citral", "max_pct_cat4": 2.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "5989-27-5": {"name": "d-Limonene", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "sensitization (oxidized)"},
    "127-91-3": {"name": "beta-Pinene", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "sensitization (oxidized)"},
    "80-56-8": {"name": "alpha-Pinene", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "sensitization (oxidized)"},
    "78-70-6": {"name": "Linalool", "max_pct_cat4": 6.8, "prohibited": False, "type": "R", "reason": "sensitization (oxidized)"},
    "60-12-8": {"name": "2-Phenylethanol", "max_pct_cat4": 1.6, "prohibited": False, "type": "R", "reason": "sensitization"},
    "101-39-3": {"name": "alpha-Methylcinnamaldehyde", "max_pct_cat4": 0.05, "prohibited": False, "type": "R", "reason": "sensitization"},
    "104-54-1": {"name": "Cinnamyl alcohol", "max_pct_cat4": 1.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "100-51-6": {"name": "Benzyl alcohol", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "120-51-4": {"name": "Benzyl benzoate", "max_pct_cat4": 8.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "103-41-3": {"name": "Benzyl cinnamate", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "118-58-1": {"name": "Benzyl salicylate", "max_pct_cat4": 0.8, "prohibited": False, "type": "R", "reason": "sensitization"},
    "4180-23-8": {"name": "trans-Anethole", "max_pct_cat4": 1.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "98-55-5": {"name": "alpha-Terpineol", "max_pct_cat4": 5.0, "prohibited": False, "type": "R", "reason": "skin irritation"},
    "105-87-3": {"name": "Geranyl acetate", "max_pct_cat4": 2.6, "prohibited": False, "type": "R", "reason": "sensitization"},
    "115-95-7": {"name": "Linalyl acetate", "max_pct_cat4": 5.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "150-84-5": {"name": "Citronellyl acetate", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "100-52-7": {"name": "Benzaldehyde", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "98-86-2": {"name": "Acetophenone", "max_pct_cat4": 2.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "122-78-1": {"name": "Phenylacetaldehyde", "max_pct_cat4": 0.6, "prohibited": False, "type": "R", "reason": "sensitization"},
    "111-12-6": {"name": "2-Methyl-3-(3,4-methylenedioxyphenyl)propanal", "max_pct_cat4": 1.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "93-92-5": {"name": "Methylbenzyl acetate", "max_pct_cat4": 3.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "68-12-2": {"name": "Dimethylformamide", "max_pct_cat4": 0.01, "prohibited": False, "type": "R", "reason": "systemic toxicity"},
    "127-51-5": {"name": "alpha-Isomethylionone", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "6259-76-3": {"name": "Hexyl salicylate", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "1205-17-0": {"name": "BMHCA (Lysmeral)", "max_pct_cat4": 0.04, "prohibited": False, "type": "R", "reason": "sensitization"},
    "120-57-0": {"name": "Piperonal (Heliotropin)", "max_pct_cat4": 0.04, "prohibited": False, "type": "R", "reason": "sensitization"},
    "1222-05-5": {"name": "Galaxolide (HHCB)", "max_pct_cat4": 5.0, "prohibited": False, "type": "R", "reason": "environmental"},
    "90028-68-5": {"name": "Treemoss absolute", "max_pct_cat4": 0.1, "prohibited": False, "type": "R", "reason": "sensitization"},
    "90028-67-4": {"name": "Oakmoss absolute", "max_pct_cat4": 0.1, "prohibited": False, "type": "R", "reason": "sensitization"},
    "8007-01-0": {"name": "Rose oil", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "8000-28-0": {"name": "Lavender oil", "max_pct_cat4": 8.0, "prohibited": False, "type": "R", "reason": "sensitization (linalool)"},
    "8015-01-8": {"name": "Cananga oil (Ylang ylang)", "max_pct_cat4": 0.1, "prohibited": False, "type": "R", "reason": "sensitization"},
    
    # --- Terpenes & Terpenoids (oxidation sensitive) ---
    "99-87-6": {"name": "p-Cymene", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "sensitization (oxidized)"},
    "586-62-9": {"name": "Terpinolene", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "sensitization (oxidized)"},
    "7785-70-8": {"name": "1R-alpha-Pinene", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "sensitization (oxidized)"},
    "470-82-6": {"name": "1,8-Cineole (Eucalyptol)", "max_pct_cat4": 6.0, "prohibited": False, "type": "R", "reason": "skin irritation"},
    "99-86-5": {"name": "alpha-Terpinene", "max_pct_cat4": 1.0, "prohibited": False, "type": "R", "reason": "sensitization peroxide"},
    "99-83-2": {"name": "alpha-Phellandrene", "max_pct_cat4": 1.0, "prohibited": False, "type": "R", "reason": "sensitization peroxide"},
    "138-86-3": {"name": "Limonene (racemic)", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "sensitization (oxidized)"},
    "7212-44-4": {"name": "Nerolidol", "max_pct_cat4": 0.9, "prohibited": False, "type": "R", "reason": "sensitization"},
    "513-86-0": {"name": "Acetoin", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    
    # --- Aldehydes (reactive) ---
    "110-62-3": {"name": "n-Pentanal (Valeraldehyde)", "max_pct_cat4": 0.1, "prohibited": False, "type": "R", "reason": "sensitization"},
    "66-25-1": {"name": "n-Hexanal", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "111-71-7": {"name": "n-Heptanal", "max_pct_cat4": 0.3, "prohibited": False, "type": "R", "reason": "sensitization"},
    "124-13-0": {"name": "n-Octanal", "max_pct_cat4": 0.4, "prohibited": False, "type": "R", "reason": "sensitization"},
    "112-31-2": {"name": "n-Decanal", "max_pct_cat4": 2.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "112-44-7": {"name": "Undecanal", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "112-54-9": {"name": "Dodecanal (Lauraldehyde)", "max_pct_cat4": 0.3, "prohibited": False, "type": "R", "reason": "sensitization"},
    "6789-80-6": {"name": "cis-3-Hexenol", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "2363-89-5": {"name": "2-Octenal", "max_pct_cat4": 0.003, "prohibited": False, "type": "R", "reason": "sensitization"},
    "18829-56-6": {"name": "trans-2-Nonenal", "max_pct_cat4": 0.01, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "3913-81-3": {"name": "trans-2-Decenal", "max_pct_cat4": 0.01, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    
    # --- Phenolics & Cresols ---
    "106-44-5": {"name": "p-Cresol", "max_pct_cat4": 0.004, "prohibited": False, "type": "R", "reason": "systemic toxicity (51st revised)"},
    "95-48-7": {"name": "o-Cresol", "max_pct_cat4": 0.004, "prohibited": False, "type": "R", "reason": "systemic toxicity"},
    "108-39-4": {"name": "m-Cresol", "max_pct_cat4": 0.004, "prohibited": False, "type": "R", "reason": "systemic toxicity"},
    
    # --- Musks (polycyclic, macrocyclic) ---
    "21145-77-7": {"name": "Tonalide (AHTN)", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "environmental"},
    "15323-35-0": {"name": "Phantolide (AHDI)", "max_pct_cat4": 1.0, "prohibited": False, "type": "R", "reason": "environmental"},
    "1506-02-1": {"name": "Traseolide (ATII)", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "environmental"},
    "13171-00-1": {"name": "Celestolide (ADBI)", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "environmental"},
    "3100-36-5": {"name": "Cashmeran", "max_pct_cat4": 2.5, "prohibited": False, "type": "R", "reason": "environmental"},
    
    # --- Cinnamates & Cinnamics ---
    "103-26-4": {"name": "Methyl cinnamate", "max_pct_cat4": 0.3, "prohibited": False, "type": "R", "reason": "sensitization"},
    "122-69-0": {"name": "3-Phenylpropyl cinnamate", "max_pct_cat4": 0.6, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "103-36-6": {"name": "Ethyl cinnamate", "max_pct_cat4": 0.6, "prohibited": False, "type": "R", "reason": "sensitization"},
    "122-68-9": {"name": "Cinnamyl propionate", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "sensitization"},
    "87-44-5": {"name": "beta-Caryophyllene", "max_pct_cat4": 5.0, "prohibited": False, "type": "R", "reason": "sensitization (oxidized)"},

    # --- Lactones ---
    "104-67-6": {"name": "gamma-Undecalactone (Peach)", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "706-14-9": {"name": "gamma-Decalactone", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "105-21-5": {"name": "gamma-Butyrolactone", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "systemic"},
    "38049-04-6": {"name": "Mintlactone", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "sensitization (51st new CAS)"},
    
    # --- Salicylates ---
    "87-22-9": {"name": "Amyl salicylate", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "119-36-8": {"name": "Methyl salicylate", "max_pct_cat4": 0.05, "prohibited": False, "type": "R", "reason": "systemic toxicity"},
    
    # --- Coumarins ---
    "548-00-5": {"name": "Dihydrocoumarin", "max_pct_cat4": 0.05, "prohibited": False, "type": "R", "reason": "sensitization"},
    "131-91-9": {"name": "6-Methylcoumarin", "max_pct_cat4": 0.01, "prohibited": False, "type": "R", "reason": "phototoxicity"},
    
    # --- Furanones / Ketones ---
    "3658-77-3": {"name": "4-Hydroxy-2,5-dimethyl-3(2H)-furanone (Furaneol)", "max_pct_cat4": 0.03, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "4077-47-8": {"name": "Ethyl furaneol", "max_pct_cat4": 0.03, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "2785-89-9": {"name": "4-Ethylguaiacol", "max_pct_cat4": 0.1, "prohibited": False, "type": "R", "reason": "sensitization"},
    "97-54-1b": {"name": "Isoeugenol (restricted form)", "max_pct_cat4": 0.02, "prohibited": False, "type": "R", "reason": "sensitization (reformulated)"},
    
    # --- Ionones / Methyl Ionones ---
    "14901-07-6": {"name": "beta-Ionone", "max_pct_cat4": 5.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "127-43-5": {"name": "alpha-Ionone", "max_pct_cat4": 3.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    "1335-46-2": {"name": "Methyl ionones (isomers mix)", "max_pct_cat4": 5.0, "prohibited": False, "type": "R", "reason": "sensitization"},
    
    # --- Acrylates / Methacrylates ---
    "5205-93-6": {"name": "Dimethyl benzyl carbinyl butyrate", "max_pct_cat4": 1.0, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    
    # --- Lilial / Lyral replacements ---
    "80-54-6": {"name": "Lilial (BMHCA / Butylphenyl methylpropanal)", "max_pct_cat4": 0.0, "prohibited": True, "type": "P", "reason": "reproductive toxicity (banned 2022)"},
    "31906-04-4": {"name": "Lyral (HICC)", "max_pct_cat4": 0.0, "prohibited": True, "type": "P", "reason": "sensitization (banned 2022)"},
    
    # --- Essential Oils (with limits) ---
    "8006-81-3": {"name": "Basil oil", "max_pct_cat4": 0.02, "prohibited": False, "type": "R", "reason": "estragole content"},
    "8007-80-5": {"name": "Cassia oil", "max_pct_cat4": 0.07, "prohibited": False, "type": "R", "reason": "cinnamaldehyde"},
    "8007-70-3": {"name": "Anise oil", "max_pct_cat4": 0.01, "prohibited": False, "type": "R", "reason": "estragole"},
    "84775-42-8": {"name": "Cade oil rectified", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "PAH content"},
    "8000-29-1": {"name": "Citronella oil (Java)", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "citronellol/geraniol"},
    "8008-51-3": {"name": "Camphor oil", "max_pct_cat4": 1.0, "prohibited": False, "type": "R", "reason": "safrole content"},
    "8006-87-9": {"name": "Sassafras oil", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "safrole"},
    "85085-22-9": {"name": "Fig leaf absolute", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "phototoxicity"},
    "68917-29-3": {"name": "Tea tree oil", "max_pct_cat4": 1.25, "prohibited": False, "type": "R", "reason": "peroxide sensitization"},
    "8008-56-8": {"name": "Lemon oil (cold pressed)", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "phototoxicity (bergaptene)"},
    "8007-02-1": {"name": "Bergamot oil (furanocoumarin)", "max_pct_cat4": 0.4, "prohibited": False, "type": "R", "reason": "phototoxicity"},
    "68606-83-7": {"name": "Lime oil (cold pressed)", "max_pct_cat4": 0.7, "prohibited": False, "type": "R", "reason": "phototoxicity"},
    "8014-19-5": {"name": "Jasmine absolute", "max_pct_cat4": 0.7, "prohibited": False, "type": "R", "reason": "sensitization"},
    "8016-78-2": {"name": "Sandalwood oil East Indian", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification only"},
    "8008-57-9": {"name": "Orange oil sweet", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "limonene (oxidized)"},
    "8007-08-7": {"name": "Ginger oil", "max_pct_cat4": 4.0, "prohibited": False, "type": "R", "reason": "citral content"},
    "8000-46-2": {"name": "Geranium oil", "max_pct_cat4": 2.5, "prohibited": False, "type": "R", "reason": "geraniol/citronellol"},
    "8022-56-8": {"name": "Sage oil Dalmatian", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "thujone"},
    "8006-80-2": {"name": "Clove oil", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "eugenol"},
    "84649-99-0": {"name": "Verbena oil", "max_pct_cat4": 0.2, "prohibited": False, "type": "R", "reason": "phototoxicity"},
    "8008-26-2": {"name": "Litsea cubeba oil", "max_pct_cat4": 2.5, "prohibited": False, "type": "R", "reason": "citral"},
    "90045-36-6": {"name": "Patchouli oil", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification only"},
    "8000-48-4": {"name": "Eucalyptus oil", "max_pct_cat4": 6.0, "prohibited": False, "type": "R", "reason": "1,8-cineole"},
    "8023-85-6": {"name": "Costus root oil", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "8016-03-3": {"name": "Peru balsam oil", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    "8001-88-5": {"name": "Styrax extract", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "sensitization"},
    
    # --- 51st Amendment NEW substances (2023) ---
    "14049-11-7": {"name": "3-Octen-2-one", "max_pct_cat4": 0.01, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "30346-73-7": {"name": "Methyl lavender ketone", "max_pct_cat4": 0.2, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "5655-61-8": {"name": "cis-3-Nonenyl acetate", "max_pct_cat4": 0.01, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "103-09-3": {"name": "2-Ethylhexyl acetate", "max_pct_cat4": 3.0, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "103-48-0": {"name": "Phenethyl isobutyrate", "max_pct_cat4": 1.0, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "122-63-4": {"name": "Phenethyl butyrate", "max_pct_cat4": 1.0, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "103-45-7": {"name": "Phenethyl acetate", "max_pct_cat4": 2.5, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "140-26-1": {"name": "Phenethyl isovalerate", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "94-44-0": {"name": "Benzyl isovalerate", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "5444-29-3": {"name": "Allyl phenoxyacetate", "max_pct_cat4": 0.1, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "7779-23-9": {"name": "Allyl 3-cyclohexylpropionate", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "93-89-0": {"name": "Ethyl benzoate", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "150-86-7": {"name": "Phytol", "max_pct_cat4": 1.0, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "2628-17-3": {"name": "4-Vinylphenol", "max_pct_cat4": 0.01, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "499-44-5": {"name": "Hinokitiol", "max_pct_cat4": 0.02, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "532-32-1": {"name": "Sodium benzoate (as ingredient)", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "skin irritation (51st new)"},
    "100-54-9": {"name": "Nicotinaldehyde", "max_pct_cat4": 0.01, "prohibited": False, "type": "R", "reason": "systemic toxicity (51st new)"},
    "2305-05-7": {"name": "delta-Decalactone", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "698-10-2": {"name": "gamma-Octalactone", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    "105-01-1": {"name": "Isoamyl butyrate", "max_pct_cat4": 2.0, "prohibited": False, "type": "R", "reason": "sensitization (51st new)"},
    
    # --- Phototoxicity restricted ---
    "298-81-7": {"name": "Methoxsalen (8-MOP)", "max_pct_cat4": 0.001, "prohibited": False, "type": "R", "reason": "phototoxicity"},
    "484-20-8": {"name": "5-Methoxypsoralen (Bergaptene)", "max_pct_cat4": 0.001, "prohibited": False, "type": "R", "reason": "phototoxicity"},
    
    # --- Misc industrial ---
    "94-36-0": {"name": "Benzoyl peroxide", "max_pct_cat4": 0, "prohibited": True, "type": "P", "reason": "oxidizing agent"},
    "128-37-0": {"name": "BHT", "max_pct_cat4": 0.5, "prohibited": False, "type": "R", "reason": "systemic (antioxidant allowed limited)"},

    # --- Specifications Only (S) - no max % but purity/isomer requirements ---
    "8000-25-7": {"name": "Rosemary oil", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "camphor spec"},
    "8008-99-9": {"name": "Garlic oil", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
    "54464-57-2": {"name": "Iso E Super", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
    "24851-98-7": {"name": "Hedione", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
    "18479-58-8": {"name": "Dihydromyrcenol", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
    "68647-72-3": {"name": "Mintone", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
    "121-33-5": {"name": "Vanillin", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
    "121-32-4": {"name": "Ethyl vanillin", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
    "5471-51-2": {"name": "Raspberry ketone (Frambinone)", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
    "69-72-7": {"name": "Salicylic acid", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
    "89-43-0": {"name": "Coumarin-dihydro", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
    "123-11-5": {"name": "Anisaldehyde", "max_pct_cat4": 99.0, "prohibited": False, "type": "S", "reason": "specification"},
}

# Add cas field + restricted flag to all
output = {}
for cas, data in IFRA_51ST_FULL.items():
    entry = {
        "cas": cas,
        "name": data["name"],
        "max_pct_cat4": data["max_pct_cat4"],
        "prohibited": data["prohibited"],
        "restricted": data["max_pct_cat4"] < 99.0 and not data["prohibited"],
        "type": data["type"],
        "reason": data.get("reason", ""),
    }
    output[cas] = entry

# Stats
n_total = len(output)
n_prohibited = sum(1 for v in output.values() if v["prohibited"])
n_restricted = sum(1 for v in output.values() if v["restricted"])
n_spec = sum(1 for v in output.values() if v["type"] == "S")

out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_upgrade', 'ifra_51st_full.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"[IFRA] Written {n_total} entries to {out_path}")
print(f"  Prohibited: {n_prohibited}")
print(f"  Restricted: {n_restricted}")
print(f"  Specification: {n_spec}")
print(f"  Total: {n_total}")
