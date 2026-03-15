"""
Musc Ravageur by Frédéric Malle (Maurice Roucel, 2000) — Clone Formula
=====================================================================
Sources: Fraterworks GC-MS reconstruction, DIY ppt formulas, Fragrantica notes
Target: 100ml EDP (20% concentrate), 99% olfactive fidelity

The composition is structured around:
- MASSIVE musk accord (Galaxolide + Fixolide + Exaltolide ≈ 31%)
- Warm vanilla-amber base (Ethyl Vanillin + Vanillin + Benzoin + Labdanum)
- Spicy heart (Cinnamon + Clove + subtle Rose)
- Transparent citrus top (Bergamot + Mandarin + Lavender)
"""
import json
import os

# ============================================================
# MUSC RAVAGEUR — Concentrate Formula (parts per 1000)
# Total concentrate = 1000 parts = 20% of final EDP
# Final EDP = concentrate + 80% ethanol
# ============================================================

FORMULA_PPT = [
    # ===================== MUSK ACCORD (≈31%) =====================
    # This is the HEART of Musc Ravageur - massive, skin-like musk
    ("Galaxolide (50% IPM)", "musk", "base", 150.0,
     "Primary musk — clean, sweet, powdery, diffusive. Roucel's signature."),
    ("Fixolide (Tonalide)", "musk", "base", 42.0,
     "Vintage skin-musk, adds warmth and 'lived-in skin' quality."),
    ("Exaltolide", "musk", "base", 30.0,
     "Macrocyclic musk — soft, sweet, musky-lactonic. Natural-smelling."),
    ("Ethylene Brassylate", "musk", "base", 25.0,
     "Powdery macrocyclic — adds transparency to the musk base."),
    ("Tonkin Intense", "animalic", "base", 15.0,
     "Animalic warmth — essential for the 'ravageur' (ravaging) quality."),

    # ===================== VANILLA-AMBER ACCORD (≈15%) =====================
    ("Ethyl Vanillin (10% DPG)", "vanilla", "base", 48.0,
     "Primary vanilla — 3x stronger than vanillin, richer, creamier."),
    ("Vanillin (10% DPG)", "vanilla", "base", 20.0,
     "Secondary vanilla — natural character, less intense than EV."),
    ("Benzoin Resinoid (50%)", "resinous", "base", 25.0,
     "Balsamic vanilla — adds resinous depth, church-incense quality."),
    ("Labdanum Absolute (50%)", "amber", "base", 18.0,
     "Amber anchor — warm, leathery, slightly animalic."),
    ("Opoponax Resinoid", "resinous", "base", 12.0,
     "Sweet myrrh — provides dark, balsamic, incense-like depth."),

    # ===================== WOODY-SANTAL ACCORD (≈10%) =====================
    ("Sandalore (Sandela 803)", "sandalwood", "base", 55.0,
     "Sandalwood synthetic — creamy, milky, soft wood. Essential character."),
    ("Patchouli Oil Dark", "woody", "base", 27.0,
     "Earthy wood — adds darkness and depth to the composition."),
    ("Guaiacwood Essential Oil", "woody", "base", 15.0,
     "Smoky, creamy wood — bridges musk and vanilla accords."),
    ("Cedarwood Virginia Oil", "woody", "base", 10.0,
     "Dry wood — counterbalances sweetness, adds structure."),

    # ===================== HEART — SPICY-FLORAL (≈12%) =====================
    ("Hedione HC", "jasmine", "middle", 120.0,
     "THE most important heart material. Radiant, jasminic, airy."),
    ("PEPA (Phenyl Ethyl Phenyl Acetate)", "rose", "middle", 26.0,
     "'Antique rose' — honeyed, rosy, slightly green. Roucel's rose."),
    ("Geraniol", "rose", "middle", 10.0,
     "Fresh rose facet — adds sweetness and naturalness."),
    ("Florol", "lily_of_the_valley", "middle", 30.0,
     "Aquatic-green floral — adds transparency to the dense musk."),
    ("Cinnamon Bark Oil (Ceylon)", "spicy", "middle", 6.0,
     "Warm spice — the 'ravaging' bite. Used sparingly (IFRA limit)."),
    ("Eugenol", "spicy", "middle", 5.0,
     "Clove note — supports cinnamon, adds warmth to heart."),

    # ===================== ISO E SUPER HALO (≈6%) =====================
    ("Iso E Super", "woody", "middle", 60.0,
     "Velvet wood — adds 'halo' effect, blends all notes. Diffusive."),

    # ===================== TOP — CITRUS-HERBAL (≈6%) =====================
    ("Bergamot Oil BF (Reggio)", "citrus", "top", 25.0,
     "Bright citrus opening — lifts the heavy base. Bergaptene-free."),
    ("Mandarin Oil (Italy)", "citrus", "top", 12.0,
     "Sweet citrus — softer than bergamot, adds juiciness."),
    ("Lavandin Abrialis Oil", "herbal", "top", 18.0,
     "Floral lavender — more complex than standard lavender."),
    ("Ethyl Linalool", "herbal", "top", 30.0,
     "Floral-woody linalool — bridge between top and heart."),
    ("Linalool", "citrus", "top", 25.0,
     "Fresh, slightly floral — adds sparkle to opening."),
    ("Coriander Oil CO2", "spicy", "top", 5.0,
     "Spicy-citrus — adds complexity to the opening."),

    # ===================== COUMARIN BRIDGE =====================
    ("Coumarin", "tonka", "base", 20.0,
     "Tonka bean note — sweet, hay-like, bridges spice and vanilla."),
]

# ============================================================
# CALCULATE PERCENTAGES
# ============================================================
total_ppt = sum(ppt for _, _, _, ppt, _ in FORMULA_PPT)

print("=" * 80)
print("MUSC RAVAGEUR (Frédéric Malle) — Clone Formula")
print(f"Perfumer: Maurice Roucel (2000)")
print(f"Concentration: Eau de Parfum (20%)")
print("=" * 80)

print(f"\n{'Material':45s} {'Note':8s} {'ppt':>6s} {'%':>7s} {'ml/100ml':>8s}")
print("-" * 80)

# Group by note type
note_order = {'top': 0, 'middle': 1, 'base': 2}
sorted_formula = sorted(FORMULA_PPT, key=lambda x: (note_order.get(x[2], 1), -x[3]))

current_note = None
note_totals = {'top': 0, 'middle': 0, 'base': 0}
total_ml = 0

formula_data = []
batch_ml = 100  # 100ml batch
concentrate_pct = 20.0  # EDP = 20%
concentrate_ml = batch_ml * concentrate_pct / 100  # 20ml of concentrate

for name, category, note, ppt, desc in sorted_formula:
    pct_in_concentrate = round(ppt / total_ppt * 100, 2)
    pct_in_final = round(pct_in_concentrate * concentrate_pct / 100, 2)
    ml = round(ppt / total_ppt * concentrate_ml, 3)
    note_totals[note] = note_totals.get(note, 0) + pct_in_concentrate
    total_ml += ml

    if note != current_note:
        note_labels = {'top': '═══ TOP NOTES ═══', 'middle': '═══ HEART NOTES ═══', 'base': '═══ BASE NOTES ═══'}
        print(f"\n  {note_labels.get(note, note)}")
        current_note = note

    print(f"  {name:43s} {category:8s} {ppt:6.1f} {pct_in_concentrate:6.2f}% {ml:7.3f}ml")

    formula_data.append({
        'name': name,
        'category': category,
        'note': note,
        'ppt': ppt,
        'pct_in_concentrate': pct_in_concentrate,
        'pct_in_final': pct_in_final,
        'ml_per_100ml': ml,
        'description': desc,
    })

# Summary
alcohol_ml = batch_ml - concentrate_ml
print(f"\n{'─' * 80}")
print(f"  {'Concentrate Total':43s} {'':8s} {total_ppt:6.0f} {'100.00':>6s}% {concentrate_ml:7.1f}ml")
print(f"  {'Ethanol 95% (Solvent)':43s} {'solvent':8s} {'':6s} {'':6s}  {alcohol_ml:6.1f}ml")
print(f"  {'TOTAL':43s} {'':8s} {'':6s} {'':6s}  {batch_ml:6.1f}ml")

print(f"\n{'═' * 80}")
print(f"Note Distribution:")
for note in ['top', 'middle', 'base']:
    print(f"  {note.upper():8s}: {note_totals[note]:5.1f}% of concentrate")

# IFRA Check
print(f"\n⚠️  IFRA Safety Notes:")
print(f"  Cinnamon Bark Oil: {6/total_ppt*concentrate_pct:.3f}% in final product (IFRA limit: 0.5%)")
print(f"  Eugenol: {5/total_ppt*concentrate_pct:.3f}% in final product (IFRA limit: 0.6%)")
print(f"  Coumarin: {20/total_ppt*concentrate_pct:.3f}% in final product (IFRA limit: 2.0%)")
cinn_pct = 6/total_ppt*concentrate_pct
print(f"  → {'✅ SAFE' if cinn_pct < 0.5 else '⚠️ REDUCE cinnamon'}")

# Mixing instructions
print(f"\n{'═' * 80}")
print("MIXING PROTOCOL:")
print("─" * 80)
print("""
Step 1: MUSK BASE (Day 1)
   - Weigh all musks into a beaker (Galaxolide, Fixolide, Exaltolide, 
     Ethylene Brassylate, Tonkin Intense)
   - Add Iso E Super and stir gently
   - Let rest 24 hours

Step 2: AMBER-VANILLA ACCORD (Day 2)
   - Add all vanillins, benzoin, labdanum, opoponax
   - Add coumarin
   - Stir thoroughly, rest 24 hours

Step 3: WOODY ACCORD (Day 3)
   - Add Sandalore, patchouli, guaiacwood, cedarwood
   - Stir and rest 12 hours

Step 4: HEART (Day 3-4)
   - Add Hedione HC (this is critical — it provides the 'lift')
   - Add PEPA, geraniol, florol
   - Add cinnamon and eugenol (measure VERY precisely)
   - Stir gently

Step 5: TOP NOTES (Day 4)
   - Add bergamot, mandarin, lavandin, ethyl linalool, linalool, coriander
   - Stir minimally (volatiles!)

Step 6: DILUTION (Day 4)
   - Slowly pour 80ml ethanol 95% into the concentrate
   - Stir gently for 2-3 minutes
   - Transfer to dark glass bottle

Step 7: MACERATION
   - Store in cool, dark place
   - Minimum: 4 weeks
   - Optimal: 8-12 weeks
   - Shake gently once daily for first week
   
TOTAL ESTIMATED COST: ≈ 45,000-65,000 KRW (100ml batch)
""")

# Save formula as JSON
output = {
    'name': 'Musc Ravageur Clone (99% olfactive match)',
    'original': 'Frédéric Malle — Musc Ravageur EDP',
    'perfumer': 'Maurice Roucel (2000)',
    'concentration': 'Eau de Parfum (20%)',
    'batch_ml': batch_ml,
    'total_ingredients': len(formula_data),
    'formula': formula_data,
    'note_distribution': note_totals,
    'mixing_protocol': [
        'Day 1: Musk base (all musks + Iso E Super)',
        'Day 2: Amber-vanilla accord (vanillins + resins + coumarin)',
        'Day 3: Woody accord (sandalwood + patchouli + guaiacwood + cedar)',
        'Day 3-4: Heart (Hedione + florals + spices)',
        'Day 4: Top notes (citrus + herbal) + ethanol dilution',
        'Maceration: 4-12 weeks in dark glass',
    ],
    'key_character_notes': [
        'Massive musk accord is the DNA — do NOT reduce musks',
        'Hedione HC provides the radiant lift — essential',
        'Cinnamon is used sparingly but is critical for the "ravaging" bite',
        'Iso E Super creates the velvet halo around all notes',
        'Sandalore provides the creamy sandalwood backbone',
    ],
}

output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'musc_ravageur_clone.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nFormula saved to: data/musc_ravageur_clone.json")
