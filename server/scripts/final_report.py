"""
FINAL TASKS:
1. Fragrantica 478 real perfume backtest using note matching
2. Verify all system scores are A-grade
"""
import json, os, sys, csv, math, time, re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(os.path.dirname(__file__), '..')

def fragrantica_full_backtest():
    """Backtest against 478 real Fragrantica perfumes"""
    print("=" * 60)
    print("  Fragrantica Full Backtest (478 perfumes)")
    print("=" * 60)
    
    frag_path = os.path.join(BASE, 'data', 'fragrantica_raw.csv')
    with open(frag_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        perfumes = list(reader)
    
    # Load our ingredient DB
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(ing_path, 'r', encoding='utf-8') as f:
        ingredients = json.load(f)
    
    # Build name→ingredient lookup
    name_to_ing = {}
    for ing in ingredients:
        for key in ['name_en', 'id']:
            n = ing.get(key, '').lower().strip()
            if n:
                name_to_ing[n] = ing
    print(f"  Ingredient name index: {len(name_to_ing)}")
    
    # Build note_type distribution from ingredients
    note_map = {}
    for ing in ingredients:
        for key in ['name_en', 'id']:
            n = ing.get(key, '').lower().strip()
            if n and ing.get('note_type'):
                note_map[n] = ing['note_type']
    
    # Process perfumes
    matched = 0
    note_correct = 0 
    note_total = 0
    accord_hits = 0
    accord_total = 0
    
    for perf in perfumes:
        name = perf.get('name', '')
        top_notes_raw = perf.get('top notes', '')
        mid_notes_raw = perf.get('middle notes', '')
        base_notes_raw = perf.get('base notes', '')
        accords_raw = perf.get('main accords', '')
        
        # Parse notes (comma-separated)
        top_notes = [n.strip().lower() for n in top_notes_raw.split(',') if n.strip()]
        mid_notes = [n.strip().lower() for n in mid_notes_raw.split(',') if n.strip()]
        base_notes = [n.strip().lower() for n in base_notes_raw.split(',') if n.strip()]
        accords = [a.strip().lower() for a in accords_raw.split(',') if a.strip()]
        
        all_notes = top_notes + mid_notes + base_notes
        if not all_notes:
            continue
        
        matched += 1
        
        # Check note classification accuracy
        # For each ingredient in top/mid/base, does our DB agree?
        for note_name in top_notes:
            if note_name in note_map:
                note_total += 1
                if note_map[note_name] == 'top':
                    note_correct += 1
        
        for note_name in mid_notes:
            if note_name in note_map:
                note_total += 1
                if note_map[note_name] == 'middle':
                    note_correct += 1
        
        for note_name in base_notes:
            if note_name in note_map:
                note_total += 1
                if note_map[note_name] == 'base':
                    note_correct += 1
    
    note_acc = note_correct / max(1, note_total) * 100
    
    print(f"\n  Perfumes processed: {matched}")
    print(f"  Note classification: {note_correct}/{note_total} ({note_acc:.1f}%)")
    
    # Additional: IFRA Compliance Check
    print(f"\n  --- IFRA Compliance Spot Check ---")
    ifra_violations = 0
    ifra_ok = 0
    for perf in perfumes[:50]:
        top = [n.strip().lower() for n in perf.get('top notes', '').split(',') if n.strip()]
        mid = [n.strip().lower() for n in perf.get('middle notes', '').split(',') if n.strip()]
        base = [n.strip().lower() for n in perf.get('base notes', '').split(',') if n.strip()]
        
        for note in top + mid + base:
            if note in name_to_ing:
                ing = name_to_ing[note]
                if ing.get('ifra_prohibited'):
                    ifra_violations += 1
                    print(f"  VIOLATION: {perf.get('name','')} uses prohibited: {note}")
                else:
                    ifra_ok += 1
    print(f"  IFRA checked: {ifra_ok} OK, {ifra_violations} violations")
    
    return note_correct, note_total, matched

def final_system_report():
    """Generate comprehensive final report"""
    print(f"\n{'='*60}")
    print("  COMPREHENSIVE FINAL REPORT")
    print(f"{'='*60}")
    
    # Load everything
    ing_path = os.path.join(BASE, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = os.path.join(BASE, 'data', 'ingredients.json')
    with open(ing_path, 'r', encoding='utf-8') as f:
        ings = json.load(f)
    
    # Stats
    total = len(ings)
    with_smi = sum(1 for x in ings if x.get('smiles'))
    with_bp = sum(1 for x in ings if x.get('est_boiling_point_c') is not None)
    with_note = sum(1 for x in ings if x.get('note_type'))
    with_price = sum(1 for x in ings if x.get('est_price_usd_kg') is not None)
    ifra_tagged = sum(1 for x in ings if x.get('ifra_cas'))
    ifra_p = sum(1 for x in ings if x.get('ifra_prohibited'))
    ifra_r = sum(1 for x in ings if x.get('ifra_restricted') and not x.get('ifra_prohibited'))
    mixtures = sum(1 for x in ings if x.get('is_mixture'))
    
    # Note distribution
    notes = {}
    for x in ings:
        n = x.get('note_type', 'unknown')
        notes[n] = notes.get(n, 0) + 1
    
    # Source breakdown
    sources = {}
    for x in ings:
        s = x.get('source', 'original')
        sources[s] = sources.get(s, 0) + 1
    
    # Price stats
    prices = [x['est_price_usd_kg'] for x in ings if x.get('est_price_usd_kg')]
    if prices:
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
    
    print(f"\n  {'='*45}")
    print(f"  {'MODULE':<30} {'STATUS':>15}")
    print(f"  {'='*45}")
    print(f"  {'Ingredient DB':<30} {total:>10} entries")
    print(f"  {'  SMILES':<30} {with_smi:>10} ({100*with_smi//total}%)")
    print(f"  {'  Boiling Point':<30} {with_bp:>10} ({100*with_bp//total}%)")
    print(f"  {'  Note Type':<30} {with_note:>10} ({100*with_note//total}%)")
    print(f"  {'  Price Estimate':<30} {with_price:>10} ({100*with_price//total}%)")
    print(f"  {'  Virtual Mixtures':<30} {mixtures:>10}")
    print(f"  {'-'*45}")
    print(f"  {'IFRA Tagged':<30} {ifra_tagged:>10}")
    print(f"  {'IFRA Prohibited':<30} {ifra_p:>10}")
    print(f"  {'IFRA Restricted':<30} {ifra_r:>10}")
    print(f"  {'-'*45}")
    print(f"  {'Note Distribution:':<30}")
    for n, c in sorted(notes.items(), key=lambda x: -x[1]):
        print(f"    {n:<26} {c:>10} ({100*c//total}%)")
    print(f"  {'-'*45}")
    print(f"  {'Sources:':<30}")
    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {s:<26} {c:>10}")
    if prices:
        print(f"  {'-'*45}")
        print(f"  {'Price: avg $':<30} {avg_price:>7.1f}/kg")
        print(f"  {'Price: min $':<30} {min_price:>7.1f}/kg")
        print(f"  {'Price: max $':<30} {max_price:>7.1f}/kg")
    print(f"  {'='*45}")

def main():
    t0 = time.time()
    nc, nt, matched = fragrantica_full_backtest()
    final_system_report()
    print(f"\n  Total time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
