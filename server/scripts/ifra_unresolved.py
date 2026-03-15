"""
Resolve 206 Unresolvable IFRA CAS
====================================
Strategy: Essential oils/natural extracts → decompose into known chemical constituents

Essential oil = mixture of identifiable molecules with SMILES
→ Apply IFRA restriction to ALL major constituents
→ "If Lavender oil is restricted at 5%, then Linalool + Linalyl acetate (its main parts) must together not exceed 5%"
"""
import json, os, sys, csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

def analyze_unresolved():
    """Categorize the 206 unresolvable IFRA CAS"""
    # Load IFRA
    with open(os.path.join(BASE, 'data', 'pom_upgrade', 'ifra_51st_official.json'), 'r', encoding='utf-8') as f:
        ifra = json.load(f)
    
    # Load resolved CAS
    cache_path = os.path.join(BASE, 'data', 'pom_upgrade', 'pubchem_cas_smiles.json')
    resolved_cas = set()
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
            resolved_cas = {k for k, v in cache.items() if v}
    
    # Build CAS→SMILES from goodscents bridge
    from scripts.ifra_complete import build_complete_cas_smiles
    cas_smi = build_complete_cas_smiles()
    resolved_cas.update(k for k, v in cas_smi.items() if v)
    
    # Find unresolved IFRA CAS
    unresolved = {}
    for cas, entry in ifra.items():
        if cas not in cas_smi or not cas_smi.get(cas):
            unresolved[cas] = entry
    
    print(f"Total IFRA: {len(ifra)}")
    print(f"Resolved: {len(ifra) - len(unresolved)}")
    print(f"Unresolved: {len(unresolved)}")
    
    # Categorize unresolved
    categories = {
        'essential_oil': [],
        'absolute_extract': [],
        'reaction_product': [],
        'mixture_class': [],
        'single_compound': [],
        'other': [],
    }
    
    for cas, entry in unresolved.items():
        name = entry.get('name', '').lower()
        ifra_type = entry.get('type', '')
        
        if any(x in name for x in ['oil', ' oil', 'olie']):
            categories['essential_oil'].append((cas, entry))
        elif any(x in name for x in ['absolute', 'extract', 'concrete', 'resinoid', 'tincture']):
            categories['absolute_extract'].append((cas, entry))
        elif any(x in name for x in ['reaction product', 'products', 'mixture']):
            categories['reaction_product'].append((cas, entry))
        elif any(x in name for x in ['dienal', 'derivatives', 'isomers', 'salts']):
            categories['mixture_class'].append((cas, entry))
        else:
            # Likely a single compound that PubChem couldn't resolve
            categories['single_compound'].append((cas, entry))
    
    for cat, items in categories.items():
        prohibited = sum(1 for _, e in items if e.get('type') == 'P')
        restricted = sum(1 for _, e in items if e.get('type') in ('R', 'RS'))
        print(f"\n  {cat}: {len(items)} ({prohibited}P, {restricted}R)")
        for cas, e in items[:5]:
            print(f"    [{e.get('type','')}] {e.get('name','')[:50]} ({cas})")
        if len(items) > 5:
            print(f"    ... +{len(items)-5} more")
    
    return categories, unresolved

if __name__ == '__main__':
    categories, unresolved = analyze_unresolved()
