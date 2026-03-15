"""
Generate 200+ perfume recipes based on real perfumery knowledge.
Sources: Public perfume note data (Fragrantica-style), classic formulation rules,
         known accords, and professional DIY recipes.
"""
import json, random, sys, os
sys.path.insert(0, '.')
import database as db

# Load real DB ingredients
all_ings = db.get_all_ingredients()
DB_IDS = {i['id'] for i in all_ings}
BY_CAT = {}
BY_NOTE = {'top': [], 'middle': [], 'base': []}
for i in all_ings:
    cat = i.get('category', '')
    nt = i.get('note_type', 'middle')
    BY_CAT.setdefault(cat, []).append(i['id'])
    if nt in BY_NOTE:
        BY_NOTE[nt].append(i['id'])

# ========================================
# KNOWN PERFUME COMPOSITIONS
# Approximate note structures based on
# publicly documented perfume compositions
# ========================================

KNOWN_PERFUMES = [
    # Classic Florals
    {"name": "Classic Aldehyde Floral No.5", "style": "classic", "mood": ["elegant", "romantic"],
     "season": ["all"], "concentration": "EDP",
     "ingredients": [
         {"id": "aldehyde", "note": "top", "pct": 3.0},
         {"id": "neroli", "note": "top", "pct": 2.0},
         {"id": "bergamot", "note": "top", "pct": 2.5},
         {"id": "linalool", "note": "top", "pct": 1.5},
         {"id": "ylang_ylang", "note": "middle", "pct": 3.0},
         {"id": "jasmine", "note": "middle", "pct": 3.5},
         {"id": "rose", "note": "middle", "pct": 2.5},
         {"id": "iris", "note": "middle", "pct": 1.5},
         {"id": "sandalwood", "note": "base", "pct": 2.0},
         {"id": "vetiver", "note": "base", "pct": 1.0},
         {"id": "vanilla", "note": "base", "pct": 1.5},
         {"id": "musk", "note": "base", "pct": 1.5},
     ], "harmony_score": 0.93, "source": "classic_reference"},

    {"name": "Guerlain Shalimar Style", "style": "oriental", "mood": ["sensual", "warm", "luxury"],
     "season": ["winter", "autumn"], "concentration": "EDP",
     "ingredients": [
         {"id": "bergamot", "note": "top", "pct": 5.0},
         {"id": "lemon", "note": "top", "pct": 2.0},
         {"id": "mandarin", "note": "top", "pct": 1.5},
         {"id": "jasmine", "note": "middle", "pct": 2.5},
         {"id": "rose", "note": "middle", "pct": 2.0},
         {"id": "iris", "note": "middle", "pct": 1.5},
         {"id": "patchouli", "note": "middle", "pct": 1.0},
         {"id": "vanilla", "note": "base", "pct": 4.0},
         {"id": "coumarin", "note": "base", "pct": 2.5},
         {"id": "benzoin", "note": "base", "pct": 2.0},
         {"id": "peru_balsam", "note": "base", "pct": 1.5},
         {"id": "leather", "note": "base", "pct": 1.0},
         {"id": "musk", "note": "base", "pct": 0.5},
     ], "harmony_score": 0.95, "source": "classic_reference"},

    {"name": "Sauvage Style Fresh", "style": "fresh", "mood": ["bold", "fresh", "energetic"],
     "season": ["spring", "summer", "all"], "concentration": "EDT",
     "ingredients": [
         {"id": "bergamot", "note": "top", "pct": 4.0},
         {"id": "black_pepper", "note": "top", "pct": 2.0},
         {"id": "grapefruit", "note": "top", "pct": 1.0},
         {"id": "lavender", "note": "middle", "pct": 2.5},
         {"id": "geranium", "note": "middle", "pct": 1.5},
         {"id": "nutmeg", "note": "middle", "pct": 1.0},
         {"id": "elemi", "note": "middle", "pct": 0.5},
         {"id": "ambroxan", "note": "base", "pct": 3.0},
         {"id": "cedarwood", "note": "base", "pct": 2.0},
         {"id": "labdanum", "note": "base", "pct": 1.0},
         {"id": "vetiver", "note": "base", "pct": 1.0},
     ], "harmony_score": 0.90, "source": "modern_reference"},

    {"name": "L'Air du Temps Style", "style": "classic", "mood": ["romantic", "elegant"],
     "season": ["spring", "summer"], "concentration": "EDT",
     "ingredients": [
         {"id": "bergamot", "note": "top", "pct": 3.0},
         {"id": "neroli", "note": "top", "pct": 2.0},
         {"id": "rosemary", "note": "top", "pct": 1.0},
         {"id": "carnation", "note": "middle", "pct": 3.0},
         {"id": "jasmine", "note": "middle", "pct": 2.5},
         {"id": "rose", "note": "middle", "pct": 2.0},
         {"id": "ylang_ylang", "note": "middle", "pct": 1.0},
         {"id": "sandalwood", "note": "base", "pct": 2.0},
         {"id": "musk", "note": "base", "pct": 1.5},
         {"id": "cedarwood", "note": "base", "pct": 1.0},
     ], "harmony_score": 0.91, "source": "classic_reference"},

    {"name": "Bleu de Chanel Style", "style": "fresh", "mood": ["elegant", "bold"],
     "season": ["all"], "concentration": "EDP",
     "ingredients": [
         {"id": "lemon", "note": "top", "pct": 2.5},
         {"id": "bergamot", "note": "top", "pct": 2.0},
         {"id": "mint", "note": "top", "pct": 1.5},
         {"id": "grapefruit", "note": "top", "pct": 1.0},
         {"id": "jasmine", "note": "middle", "pct": 2.0},
         {"id": "nutmeg", "note": "middle", "pct": 1.5},
         {"id": "iso_e_super", "note": "middle", "pct": 2.0},
         {"id": "cedarwood", "note": "base", "pct": 3.0},
         {"id": "sandalwood", "note": "base", "pct": 2.0},
         {"id": "labdanum", "note": "base", "pct": 1.0},
         {"id": "vetiver", "note": "base", "pct": 1.0},
     ], "harmony_score": 0.91, "source": "modern_reference"},

    {"name": "Tobacco Vanille Style", "style": "oriental", "mood": ["warm", "bold", "luxury"],
     "season": ["winter", "autumn"], "concentration": "EDP",
     "ingredients": [
         {"id": "ginger", "note": "top", "pct": 1.5},
         {"id": "star_anise", "note": "top", "pct": 1.0},
         {"id": "tobacco", "note": "middle", "pct": 5.0},
         {"id": "tonka_bean", "note": "middle", "pct": 2.5},
         {"id": "cocoa", "note": "middle", "pct": 2.0},
         {"id": "cinnamon", "note": "middle", "pct": 1.0},
         {"id": "vanilla", "note": "base", "pct": 5.0},
         {"id": "benzoin", "note": "base", "pct": 2.0},
         {"id": "sandalwood", "note": "base", "pct": 1.5},
     ], "harmony_score": 0.92, "source": "niche_reference"},

    {"name": "Aventus Style", "style": "fresh", "mood": ["bold", "elegant"],
     "season": ["spring", "summer"], "concentration": "EDP",
     "ingredients": [
         {"id": "bergamot", "note": "top", "pct": 3.0},
         {"id": "raspberry", "note": "top", "pct": 2.0},
         {"id": "green_apple", "note": "top", "pct": 1.5},
         {"id": "pink_pepper", "note": "top", "pct": 1.0},
         {"id": "patchouli", "note": "middle", "pct": 2.0},
         {"id": "jasmine", "note": "middle", "pct": 1.5},
         {"id": "birch", "note": "middle", "pct": 1.5},
         {"id": "ambroxan", "note": "base", "pct": 2.5},
         {"id": "oakwood", "note": "base", "pct": 2.0},
         {"id": "musk", "note": "base", "pct": 1.5},
         {"id": "vanilla", "note": "base", "pct": 1.0},
     ], "harmony_score": 0.91, "source": "niche_reference"},

    {"name": "Baccarat Rouge Style", "style": "oriental", "mood": ["luxury", "sensual"],
     "season": ["winter", "autumn", "all"], "concentration": "EDP",
     "ingredients": [
         {"id": "saffron", "note": "top", "pct": 2.5},
         {"id": "jasmine", "note": "middle", "pct": 3.0},
         {"id": "hedione", "note": "middle", "pct": 2.0},
         {"id": "ambroxan", "note": "base", "pct": 5.0},
         {"id": "cedarwood", "note": "base", "pct": 3.0},
         {"id": "vanilla", "note": "base", "pct": 2.0},
         {"id": "musk", "note": "base", "pct": 1.5},
         {"id": "benzoin", "note": "base", "pct": 1.0},
     ], "harmony_score": 0.93, "source": "niche_reference"},

    {"name": "Acqua di Gio Style", "style": "fresh", "mood": ["fresh", "clean"],
     "season": ["spring", "summer"], "concentration": "EDT",
     "ingredients": [
         {"id": "bergamot", "note": "top", "pct": 3.0},
         {"id": "lime", "note": "top", "pct": 2.0},
         {"id": "lemon", "note": "top", "pct": 1.5},
         {"id": "calone", "note": "top", "pct": 1.0},
         {"id": "jasmine", "note": "middle", "pct": 2.0},
         {"id": "hedione", "note": "middle", "pct": 2.5},
         {"id": "rosemary", "note": "middle", "pct": 1.0},
         {"id": "cedarwood", "note": "base", "pct": 2.0},
         {"id": "ambroxan", "note": "base", "pct": 1.5},
         {"id": "white_musk", "note": "base", "pct": 1.5},
         {"id": "patchouli", "note": "base", "pct": 1.0},
     ], "harmony_score": 0.89, "source": "classic_reference"},

    {"name": "Black Orchid Style", "style": "oriental", "mood": ["sensual", "exotic", "bold"],
     "season": ["winter", "autumn"], "concentration": "EDP",
     "ingredients": [
         {"id": "raspberry", "note": "top", "pct": 1.5},
         {"id": "bergamot", "note": "top", "pct": 1.0},
         {"id": "mandarin", "note": "top", "pct": 1.0},
         {"id": "frangipani", "note": "middle", "pct": 3.0},
         {"id": "ylang_ylang", "note": "middle", "pct": 2.0},
         {"id": "jasmine", "note": "middle", "pct": 1.5},
         {"id": "clove", "note": "middle", "pct": 1.0},
         {"id": "patchouli", "note": "base", "pct": 3.0},
         {"id": "vanilla", "note": "base", "pct": 2.5},
         {"id": "vetiver", "note": "base", "pct": 1.5},
         {"id": "incense", "note": "base", "pct": 1.5},
         {"id": "sandalwood", "note": "base", "pct": 1.0},
     ], "harmony_score": 0.92, "source": "niche_reference"},
]

# ========================================
# PROFESSIONAL ACCORD TEMPLATES
# Based on real perfumery accords
# ========================================
ACCORDS = {
    "fougere": {"top": ["lavender", "bergamot"], "middle": ["geranium", "clary_sage"], "base": ["coumarin", "oakwood", "vetiver", "musk"]},
    "chypre": {"top": ["bergamot", "lemon"], "middle": ["rose", "jasmine"], "base": ["oakwood", "patchouli", "musk", "labdanum"]},
    "oriental_floral": {"top": ["bergamot", "mandarin"], "middle": ["jasmine", "ylang_ylang", "tuberose"], "base": ["vanilla", "sandalwood", "musk", "benzoin"]},
    "aromatic_fresh": {"top": ["bergamot", "lemon", "mint"], "middle": ["lavender", "rosemary", "sage"], "base": ["cedarwood", "white_musk"]},
    "amber_woody": {"top": ["bergamot", "cardamom"], "middle": ["cinnamon", "rose"], "base": ["labdanum", "vanilla", "sandalwood", "benzoin", "tonka_bean"]},
    "aquatic": {"top": ["bergamot", "lemon", "calone"], "middle": ["hedione", "sea_salt", "lavender"], "base": ["ambroxan", "cedarwood", "white_musk"]},
    "gourmand": {"top": ["bergamot", "cardamom"], "middle": ["coffee", "chocolate", "honey"], "base": ["vanilla", "tonka_bean", "benzoin", "musk"]},
    "green_floral": {"top": ["galbanum", "bergamot", "lemon"], "middle": ["rose", "jasmine", "lily_of_valley"], "base": ["sandalwood", "musk", "cedarwood"]},
    "leather_tobacco": {"top": ["bergamot", "black_pepper"], "middle": ["tobacco", "leather", "incense"], "base": ["vetiver", "labdanum", "vanilla", "musk"]},
    "woody_spicy": {"top": ["cardamom", "ginger", "pink_pepper"], "middle": ["nutmeg", "cinnamon", "iso_e_super"], "base": ["cedarwood", "vetiver", "sandalwood", "ambroxan"]},
    "oud_rose": {"top": ["saffron", "bergamot"], "middle": ["rose", "oud", "geranium"], "base": ["sandalwood", "ambroxan", "musk", "vanilla"]},
    "citrus_woody": {"top": ["bergamot", "grapefruit", "lemon", "lime"], "middle": ["hedione", "geranium"], "base": ["cedarwood", "vetiver", "iso_e_super", "white_musk"]},
    "powdery_floral": {"top": ["bergamot", "pink_pepper"], "middle": ["iris", "violet", "heliotrope"], "base": ["orris_butter", "cashmeran", "vanilla", "white_musk"]},
    "smoky_incense": {"top": ["frankincense", "bergamot"], "middle": ["incense", "myrrh", "labdanum"], "base": ["sandalwood", "benzoin", "vetiver", "musk"]},
    "fruity_floral": {"top": ["peach", "raspberry", "bergamot"], "middle": ["rose", "jasmine", "peony"], "base": ["vanilla", "white_musk", "sandalwood"]},
    "tea_accord": {"top": ["bergamot", "lemon"], "middle": ["black_tea", "green_tea", "rose"], "base": ["cedarwood", "vanilla", "musk"]},
    "coconut_tropical": {"top": ["lime", "bergamot", "mango"], "middle": ["coconut", "frangipani", "jasmine"], "base": ["vanilla", "sandalwood", "white_musk"]},
    "hay_coumarin": {"top": ["bergamot", "lavender"], "middle": ["hay_absolute", "immortelle", "clary_sage"], "base": ["tonka_bean", "coumarin", "vetiver", "musk"]},
}

STYLE_MAP = {
    "fougere": "classic", "chypre": "classic", "oriental_floral": "oriental",
    "aromatic_fresh": "fresh", "amber_woody": "oriental", "aquatic": "fresh",
    "gourmand": "oriental", "green_floral": "fresh", "leather_tobacco": "oriental",
    "woody_spicy": "oriental", "oud_rose": "oriental", "citrus_woody": "fresh",
    "powdery_floral": "classic", "smoky_incense": "oriental", "fruity_floral": "fresh",
    "tea_accord": "classic", "coconut_tropical": "fresh", "hay_coumarin": "classic",
}

MOOD_MAP = {
    "fougere": ["elegant", "calm"], "chypre": ["elegant", "bold"],
    "oriental_floral": ["romantic", "sensual"], "aromatic_fresh": ["fresh", "energetic"],
    "amber_woody": ["warm", "sensual"], "aquatic": ["fresh", "clean"],
    "gourmand": ["cozy", "warm"], "green_floral": ["fresh", "romantic"],
    "leather_tobacco": ["bold", "warm"], "woody_spicy": ["bold", "warm"],
    "oud_rose": ["luxury", "exotic"], "citrus_woody": ["fresh", "elegant"],
    "powdery_floral": ["dreamy", "elegant"], "smoky_incense": ["spiritual", "calm"],
    "fruity_floral": ["playful", "romantic"], "tea_accord": ["calm", "elegant"],
    "coconut_tropical": ["playful", "fresh"], "hay_coumarin": ["calm", "grounding"],
}

SEASON_MAP = {
    "fougere": ["spring", "summer"], "chypre": ["autumn", "spring"],
    "oriental_floral": ["spring", "summer"], "aromatic_fresh": ["summer", "spring"],
    "amber_woody": ["winter", "autumn"], "aquatic": ["summer"],
    "gourmand": ["winter", "autumn"], "green_floral": ["spring", "summer"],
    "leather_tobacco": ["autumn", "winter"], "woody_spicy": ["autumn", "winter"],
    "oud_rose": ["winter", "autumn"], "citrus_woody": ["summer", "spring"],
    "powdery_floral": ["spring", "all"], "smoky_incense": ["autumn", "winter"],
    "fruity_floral": ["spring", "summer"], "tea_accord": ["autumn", "spring"],
    "coconut_tropical": ["summer"], "hay_coumarin": ["autumn"],
}

CONC_MAP = {
    "fougere": "EDT", "chypre": "EDP", "oriental_floral": "EDP",
    "aromatic_fresh": "EDC", "amber_woody": "EDP", "aquatic": "EDT",
    "gourmand": "EDP", "green_floral": "EDT", "leather_tobacco": "EDP",
    "woody_spicy": "EDP", "oud_rose": "Parfum", "citrus_woody": "EDT",
    "powdery_floral": "EDP", "smoky_incense": "EDP", "fruity_floral": "EDT",
    "tea_accord": "EDT", "coconut_tropical": "EDT", "hay_coumarin": "EDP",
}

# Variation names
ADJECTIVES = [
    "Velvet", "Silk", "Golden", "Crystal", "Midnight", "Dawn", "Ivory",
    "Secret", "Wild", "Pure", "Deep", "Soft", "Bright", "Dark", "Noble",
    "Ancient", "Morning", "Evening", "Eternal", "Hidden", "Radiant",
    "Serene", "Royal", "Shadow", "Mystic", "Solar", "Lunar", "Silver"
]

NOUNS = {
    "fougere": ["Fern", "Garden", "Grove", "Meadow"],
    "chypre": ["Forest", "Moss", "Stone", "Earth"],
    "oriental_floral": ["Bloom", "Petal", "Bouquet", "Blossom"],
    "aromatic_fresh": ["Breeze", "Air", "Stream", "Mist"],
    "amber_woody": ["Ember", "Glow", "Flame", "Dusk"],
    "aquatic": ["Wave", "Tide", "Shore", "Rain"],
    "gourmand": ["Honey", "Spice", "Cookie", "Dream"],
    "green_floral": ["Leaf", "Dew", "Spring", "Vine"],
    "leather_tobacco": ["Smoke", "Noir", "Night", "Shadow"],
    "woody_spicy": ["Cedar", "Wood", "Bark", "Root"],
    "oud_rose": ["Throne", "Crown", "Palace", "Dynasty"],
    "citrus_woody": ["Sun", "Zest", "Light", "Spark"],
    "powdery_floral": ["Cloud", "Silk", "Veil", "Powder"],
    "smoky_incense": ["Temple", "Altar", "Ritual", "Prayer"],
    "fruity_floral": ["Kiss", "Blush", "Joy", "Berry"],
    "tea_accord": ["Ceremony", "Cup", "Leaf", "Zen"],
    "coconut_tropical": ["Island", "Beach", "Lagoon", "Paradise"],
    "hay_coumarin": ["Harvest", "Field", "Barn", "Trail"],
}

# Percentage ranges by note type for different concentrations
PCT_GUIDELINES = {
    "EDC": {"top": (2.0, 4.0), "middle": (1.0, 2.5), "base": (0.5, 1.5)},
    "EDT": {"top": (1.5, 3.5), "middle": (1.5, 3.0), "base": (1.0, 2.5)},
    "EDP": {"top": (1.5, 3.0), "middle": (2.0, 4.0), "base": (1.5, 3.5)},
    "Parfum": {"top": (1.0, 2.5), "middle": (2.5, 5.0), "base": (2.0, 5.0)},
}


def gen_recipe_from_accord(accord_name, idx):
    """Generate a recipe from accord template with realistic variations"""
    accord = ACCORDS[accord_name]
    style = STYLE_MAP[accord_name]
    moods = MOOD_MAP[accord_name]
    seasons = SEASON_MAP[accord_name]
    conc = CONC_MAP[accord_name]

    # Sometimes vary concentration
    if random.random() < 0.3:
        conc = random.choice(["EDT", "EDP"])

    pct_range = PCT_GUIDELINES[conc]

    ingredients = []

    # Top notes
    tops = list(accord["top"])
    # Add 0-2 extra top notes from DB
    extra_tops = [t for t in BY_NOTE["top"] if t not in tops and t in DB_IDS]
    if extra_tops and random.random() < 0.4:
        tops.append(random.choice(extra_tops))
    for t in tops:
        if t in DB_IDS:
            pct = round(random.uniform(*pct_range["top"]), 1)
            ingredients.append({"id": t, "note": "top", "pct": pct})

    # Middle notes
    mids = list(accord["middle"])
    extra_mids = [m for m in BY_NOTE["middle"] if m not in mids and m in DB_IDS]
    if extra_mids and random.random() < 0.3:
        mids.append(random.choice(extra_mids))
    for m in mids:
        if m in DB_IDS:
            pct = round(random.uniform(*pct_range["middle"]), 1)
            ingredients.append({"id": m, "note": "middle", "pct": pct})

    # Base notes
    bases = list(accord["base"])
    extra_bases = [b for b in BY_NOTE["base"] if b not in bases and b in DB_IDS]
    if extra_bases and random.random() < 0.3:
        bases.append(random.choice(extra_bases))
    for b in bases:
        if b in DB_IDS:
            pct = round(random.uniform(*pct_range["base"]), 1)
            ingredients.append({"id": b, "note": "base", "pct": pct})

    # Name
    adj = random.choice(ADJECTIVES)
    nouns = NOUNS.get(accord_name, ["Scent"])
    noun = random.choice(nouns)
    name = f"{adj} {noun} #{idx}"

    # Harmony score: base + variation
    base_harmony = 0.75 + random.random() * 0.2  # 0.75~0.95
    # Higher harmony if ingredients come from same/related categories
    cats_used = set()
    for ing in ingredients:
        for db_ing in all_ings:
            if db_ing['id'] == ing['id']:
                cats_used.add(db_ing.get('category', ''))

    if len(cats_used) <= 4:
        base_harmony += 0.03
    harmony = round(min(0.97, base_harmony), 2)

    # Sometimes add extra mood
    extra_moods = ["romantic", "elegant", "bold", "calm", "warm", "fresh", "cozy", "playful"]
    mood_list = list(moods)
    if random.random() < 0.3:
        extra = random.choice(extra_moods)
        if extra not in mood_list:
            mood_list.append(extra)

    return {
        "name": name,
        "style": style,
        "mood": mood_list,
        "season": seasons,
        "concentration": conc,
        "ingredients": ingredients,
        "harmony_score": harmony,
        "source": f"accord_{accord_name}"
    }


def main():
    recipes = list(KNOWN_PERFUMES)
    print(f"Starting with {len(recipes)} known perfume references")

    # Generate 10-12 variations per accord = ~180-216 recipes
    idx = 1
    for accord_name in ACCORDS:
        n_variants = random.randint(10, 12)
        for _ in range(n_variants):
            recipe = gen_recipe_from_accord(accord_name, idx)
            recipes.append(recipe)
            idx += 1

    print(f"Generated {len(recipes)} total recipes")

    # Validate all ingredient IDs
    missing = set()
    total_refs = 0
    for r in recipes:
        for ing in r["ingredients"]:
            total_refs += 1
            if ing["id"] not in DB_IDS:
                missing.add(ing["id"])

    if missing:
        print(f"WARNING: {len(missing)} ingredients not in DB: {missing}")
        # Remove missing ingredients
        for r in recipes:
            r["ingredients"] = [ing for ing in r["ingredients"] if ing["id"] in DB_IDS]

    # Stats
    styles = {}
    moods = {}
    seasons = {}
    concs = {}
    sources = {}
    for r in recipes:
        styles[r["style"]] = styles.get(r["style"], 0) + 1
        concs[r["concentration"]] = concs.get(r["concentration"], 0) + 1
        sources[r.get("source", "unknown")] = sources.get(r.get("source", "unknown"), 0) + 1
        for m in r["mood"]:
            moods[m] = moods.get(m, 0) + 1
        for s in r["season"]:
            seasons[s] = seasons.get(s, 0) + 1

    total_ings = sum(len(r["ingredients"]) for r in recipes)
    print(f"\n=== DATASET STATS ===")
    print(f"Total recipes: {len(recipes)}")
    print(f"Total ingredient references: {total_ings}")
    print(f"Avg ingredients/recipe: {total_ings/len(recipes):.1f}")
    print(f"Styles: {styles}")
    print(f"Concentrations: {concs}")
    print(f"Sources: {sources}")
    print(f"Moods: {dict(sorted(moods.items(), key=lambda x: -x[1]))}")
    print(f"Seasons: {dict(sorted(seasons.items(), key=lambda x: -x[1]))}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'recipe_training_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
