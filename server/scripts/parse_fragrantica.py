"""
Parse Fragrantica CSV and convert to recipe training data,
matching ingredients with our DB.
"""
import csv, json, sys, re
from collections import Counter

sys.path.insert(0, '.')
import database as db

# Load DB ingredient IDs
all_ings = db.get_all_ingredients()
DB_IDS = {i['id'] for i in all_ings}
DB_BY_NAME = {}
for i in all_ings:
    DB_BY_NAME[i['id']] = i
    # Also map by name_ko, aliases
    if i.get('name_ko'):
        DB_BY_NAME[i['name_ko'].lower()] = i

# Manual mapping: Fragrantica note names → our DB IDs
NAME_MAP = {
    # citrus
    'bergamot': 'bergamot', 'lemon': 'lemon', 'orange': 'orange',
    'grapefruit': 'grapefruit', 'lime': 'lime', 'mandarin orange': 'mandarin',
    'tangerine': 'mandarin', 'blood orange': 'orange', 'yuzu': 'yuzu',
    'kumquat': 'kumquat', 'citron': 'lemon', 'bitter orange': 'orange',
    'petitgrain': 'petitgrain', 'neroli': 'neroli', 'lemon verbena': 'lemon',
    
    # floral
    'rose': 'rose', 'jasmine': 'jasmine', 'iris': 'iris', 'violet': 'violet',
    'lily-of-the-valley': 'lily_of_valley', 'tuberose': 'tuberose',
    'ylang-ylang': 'ylang_ylang', 'magnolia': 'magnolia', 'peony': 'peony',
    'gardenia': 'gardenia', 'orchid': 'frangipani', 'orange blossom': 'orange_blossom',
    'carnation': 'carnation', 'freesia': 'freesia', 'cherry blossom': 'cherry_blossom',
    'plumeria': 'plumeria', 'frangipani': 'frangipani', 'lotus': 'lotus',
    'water lily': 'water_lily', 'mimosa': 'mimosa', 'lilac': 'lilac',
    'osmanthus': 'osmanthus', 'heliotrope': 'heliotrope', 'geranium': 'geranium',
    'hibiscus': 'hibiscus', 'hyacinth': 'hyacinth', 'lavender': 'lavender',
    'chamomile': 'chamomile', 'chrysanthemum': 'chrysanthemum',
    'linden blossom': 'linden_blossom', 'tiare flower': 'tiare',
    'wisteria': 'wisteria', 'acacia': 'acacia', 'lily': 'water_lily',
    'honeysuckle': 'honeysuckle', 'daffodil': 'daffodil', 'marigold': 'calendula',
    'hedione': 'hedione', 'damascus rose': 'rose', 'turkish rose': 'rose',
    'bulgarian rose': 'rose', 'tea rose': 'rose', 'may rose': 'rose',
    'wild rose': 'rose',
    
    # woody
    'sandalwood': 'sandalwood', 'cedarwood': 'cedarwood', 'cedar': 'cedarwood',
    'vetiver': 'vetiver', 'patchouli': 'patchouli', 'oud': 'oud',
    'agarwood (oud)': 'oud', 'guaiac wood': 'guaiac_wood', 'birch': 'birch',
    'pine': 'pine_needle', 'pine tree': 'pine_needle', 'cypress': 'cypress',
    'oakmoss': 'oakwood', 'oak': 'oakwood', 'rosewood': 'rosewood',
    'teak': 'teak', 'bamboo': 'bamboo', 'hinoki': 'hinoki',
    'fir': 'fir_balsam', 'juniper': 'juniper_berry', 'juniper berries': 'juniper_berry',
    'mahogany': 'mahogany', 'ebony': 'ebony',
    
    # spicy
    'black pepper': 'black_pepper', 'pepper': 'black_pepper',
    'pink pepper': 'pink_pepper', 'cardamom': 'cardamom', 'cinnamon': 'cinnamon',
    'clove': 'clove', 'ginger': 'ginger', 'nutmeg': 'nutmeg', 'saffron': 'saffron',
    'star anise': 'star_anise', 'cumin': 'cumin', 'coriander': 'coriander',
    'white pepper': 'white_pepper', 'allspice': 'allspice',
    'szechuan pepper': 'szechuan_pepper', 'caraway': 'caraway',
    'turmeric': 'turmeric', 'fennel': 'fennel',
    
    # vanilla/gourmand
    'vanilla': 'vanilla', 'tonka bean': 'tonka_bean', 'cocoa': 'cocoa',
    'chocolate': 'chocolate', 'coffee': 'coffee', 'caramel': 'caramel',
    'praline': 'praline', 'honey': 'honey', 'almond': 'almond',
    'coconut': 'coconut', 'milk': 'milk', 'brown sugar': 'brown_sugar',
    'maple syrup': 'maple', 'rum': 'rum', 'cognac': 'cognac',
    'whiskey': 'whiskey', 'popcorn': 'popcorn_note',
    
    # musk/amber
    'musk': 'musk', 'white musk': 'white_musk', 'ambergris': 'ambergris',
    'amber': 'labdanum', 'ambroxan': 'ambroxan', 'cashmeran': 'cashmeran',
    'cashmere': 'cashmere', 'iso e super': 'iso_e_super',
    
    # resinous
    'benzoin': 'benzoin', 'myrrh': 'myrrh', 'frankincense': 'frankincense',
    'incense': 'incense', 'labdanum': 'labdanum', 'copal': 'copaiba',
    'styrax': 'styrax', 'elemi': 'elemi', 'olibanum': 'frankincense',
    'opoponax': 'opoponax', 'balsam fir': 'fir_balsam',
    
    # fruity
    'peach': 'peach', 'apple': 'green_apple', 'green apple': 'green_apple',
    'pear': 'pear', 'raspberry': 'raspberry', 'blackcurrant': 'raspberry',
    'strawberry': 'strawberry', 'plum': 'plum', 'mango': 'mango',
    'passion fruit': 'passion_fruit', 'lychee': 'lychee', 'fig': 'fig',
    'melon': 'melon', 'cherry': 'cherry', 'blueberry': 'blueberry',
    'blackberry': 'blackberry', 'grape': 'grape', 'pomegranate': 'pomegranate',
    'guava': 'guava', 'papaya': 'papaya', 'watermelon': 'watermelon',
    'apricot': 'apricot', 'pineapple': 'pineapple', 'kiwi': 'kiwi',
    'quince': 'quince', 'cassis': 'raspberry', 'black currant': 'raspberry',
    'red apple': 'green_apple',
    
    # green
    'green tea': 'green_tea', 'tea': 'black_tea', 'black tea': 'black_tea',
    'white tea': 'white_tea', 'mate': 'mate', 'matcha': 'matcha',
    'galbanum': 'galbanum', 'bamboo leaf': 'bamboo_leaf',
    
    # leather/smoky
    'leather': 'leather', 'suede': 'suede', 'tobacco': 'tobacco',
    'birch tar': 'birch_tar', 'smoke': 'campfire',
    
    # synthetic/molecular
    'aldehyde': 'aldehyde', 'aldehydes': 'aldehyde',
    'coumarin': 'coumarin', 'galaxolide': 'galaxolide',
    'linalool': 'linalool', 'calone': 'calone',
    
    # herbal
    'basil': 'basil', 'rosemary': 'rosemary', 'thyme': 'thyme',
    'mint': 'mint', 'sage': 'sage', 'eucalyptus': 'eucalyptus',
    'tarragon': 'tarragon', 'oregano': 'oregano', 'dill': 'dill',
    'artemisia': 'artemisia', 'absinthe': 'absinthe',
    'tea tree': 'tea_tree',
    
    # powdery
    'rice': 'rice', 'rice powder': 'rice_powder', 'silk': 'silk',
    'talcum powder': 'rice_powder',
    
    # other
    'sea notes': 'sea_salt', 'marine notes': 'calone', 'salt': 'sea_salt',
    'seaweed': 'seaweed', 'beeswax': 'beeswax', 'hay': 'hay_absolute',
    'moss': 'oakwood', 'grass': 'grass', 'orris root': 'orris_butter',
    'orris': 'orris_butter', 'tonka': 'tonka_bean',
    'pink pepper': 'pink_pepper', 'vetiver': 'vetiver',
    'patchouli leaf': 'patchouli', 'agarwood': 'oud',
    'virginia cedar': 'cedarwood', 'atlas cedar': 'cedarwood',
    'texan cedarwood': 'cedarwood',
    
    # Additional unmapped notes from Fragrantica
    'vanille': 'vanilla', 'madagascar vanilla': 'vanilla',
    'amalfi lemon': 'lemon', 'italian lemon': 'lemon',
    'pineapple': 'pineapple', 'cacao': 'cocoa',
    'woody notes': 'cedarwood', 'woodsy notes': 'cedarwood',
    'green notes': 'galbanum', 'white flowers': 'jasmine',
    'african orange flower': 'orange_blossom',
    'amberwood': 'sandalwood', 'cashmere wood': 'cashmeran',
    'citruses': 'bergamot', 'litchi': 'lychee',
    'french labdanum': 'labdanum', 'oak moss': 'oakwood',
    'green mandarin': 'mandarin', 'ambrette (musk mallow)': 'ambrette',
    'spices': 'cardamom', 'licorice': 'anise',
    'tahitian vanilla': 'vanilla', 'bourbon vanilla': 'vanilla',
    'pink rose': 'rose', 'red rose': 'rose', 'white rose': 'rose',
    'rose water': 'rose', 'rose absolute': 'rose',
    'musk ketone': 'musk_ketone', 'muscone': 'muscone',
    'iso e super': 'iso_e_super',
    'amber accord': 'labdanum', 'ambrox': 'ambroxan',
    'florals': 'rose', 'fresh spicy notes': 'pink_pepper',
    'dried fruits': 'plum', 'tropical fruits': 'mango',
    'yellow mandarin': 'mandarin', 'sicilian lemon': 'lemon',
    'italian bergamot': 'bergamot', 'calabrian bergamot': 'bergamot',
    'indian sandalwood': 'sandalwood', 'australian sandalwood': 'sandalwood',
    'mysore sandalwood': 'sandalwood',
    'virginia cedarwood': 'cedarwood',
    'haitian vetiver': 'vetiver', 'java vetiver': 'vetiver',
    'taif rose': 'rose', 'damask rose': 'rose',
    'sambac jasmine': 'jasmine', 'indian jasmine': 'jasmine',
    'jasmine sambac': 'jasmine',
    'white cedar': 'white_cedar', 'red cedar': 'cedarwood',
    'pink peppercorn': 'pink_pepper',
    'black vanilla': 'vanilla',
    'tonka absolu': 'tonka_bean', 'tonka absolute': 'tonka_bean',
    'siam benzoin': 'benzoin',
    'laotian oud': 'oud', 'cambodian oud': 'oud', 'indian oud': 'oud',
    'turkish ottoman rose': 'rose',
}


def parse_notes(note_str):
    """Parse note string like "['Bergamot', 'Pink Pepper', 'Orchid']" """
    if not note_str or note_str in ('{}', 'nan', '[]', 'NA', ''):
        return []
    try:
        result = eval(note_str)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return list(result.keys())
    except:
        pass
    return []


def map_note_to_db(note_name):
    """Map Fragrantica note name to our DB ingredient ID"""
    name_lower = note_name.lower().strip()
    # Direct map
    if name_lower in NAME_MAP:
        db_id = NAME_MAP[name_lower]
        if db_id in DB_IDS:
            return db_id
    # Try direct match with DB ID
    slug = name_lower.replace(' ', '_').replace('-', '_')
    if slug in DB_IDS:
        return slug
    return None


def main():
    # Read CSV
    with open('data/fragrantica_raw.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows in CSV: {len(rows)}")
    print(f"Columns: {list(rows[0].keys())}")

    # Show sample
    print("\n--- Sample row ---")
    for k, v in list(rows[0].items()):
        print(f"  {k}: {str(v)[:100]}")

    # Parse and convert
    recipes = []
    unmapped = Counter()
    mapped_count = 0
    skipped_count = 0

    for row in rows:
        name = row.get('name', 'Unknown')
        try:
            rating = float(row.get('rating', '0') or '0')
        except (ValueError, TypeError):
            rating = 3.0
        gender = row.get('for_gender', 'unisex')

        top_notes = parse_notes(row.get('top notes', '{}'))
        mid_notes = parse_notes(row.get('middle notes', '{}'))
        base_notes = parse_notes(row.get('base notes', '{}'))

        if not top_notes and not mid_notes and not base_notes:
            skipped_count += 1
            continue

        ingredients = []
        for note_name in top_notes:
            db_id = map_note_to_db(note_name)
            if db_id:
                ingredients.append({"id": db_id, "note": "top", "pct": 0})
            else:
                unmapped[note_name.lower()] += 1

        for note_name in mid_notes:
            db_id = map_note_to_db(note_name)
            if db_id:
                ingredients.append({"id": db_id, "note": "middle", "pct": 0})
            else:
                unmapped[note_name.lower()] += 1

        for note_name in base_notes:
            db_id = map_note_to_db(note_name)
            if db_id:
                ingredients.append({"id": db_id, "note": "base", "pct": 0})
            else:
                unmapped[note_name.lower()] += 1

        if len(ingredients) < 3:
            skipped_count += 1
            continue

        # Estimate percentages based on note position
        n_top = sum(1 for i in ingredients if i['note'] == 'top')
        n_mid = sum(1 for i in ingredients if i['note'] == 'middle')
        n_base = sum(1 for i in ingredients if i['note'] == 'base')

        total = len(ingredients)
        for ing in ingredients:
            if ing['note'] == 'top':
                ing['pct'] = round(20.0 / max(n_top, 1), 1)  # top ~20%
            elif ing['note'] == 'middle':
                ing['pct'] = round(45.0 / max(n_mid, 1), 1)  # middle ~45%
            else:
                ing['pct'] = round(35.0 / max(n_base, 1), 1)  # base ~35%

        # Determine style/mood from notes
        has_woody = any(i['id'] in BY_CAT_SET.get('woody', set()) for i in ingredients)
        has_floral = any(i['id'] in BY_CAT_SET.get('floral', set()) for i in ingredients)
        has_citrus = any(i['id'] in BY_CAT_SET.get('citrus', set()) for i in ingredients)
        has_spicy = any(i['id'] in BY_CAT_SET.get('spicy', set()) for i in ingredients)
        has_gourmand = any(i['id'] in BY_CAT_SET.get('gourmand', set()) for i in ingredients)

        if has_gourmand or has_spicy:
            style = 'oriental'
        elif has_citrus and not has_floral:
            style = 'fresh'
        elif has_floral:
            style = 'classic'
        else:
            style = 'classic'

        mood = []
        if has_floral: mood.append('romantic')
        if has_citrus: mood.append('fresh')
        if has_woody: mood.append('elegant')
        if has_spicy: mood.append('bold')
        if has_gourmand: mood.append('warm')
        if not mood: mood = ['elegant']

        harmony = round(0.70 + (rating / 5.0) * 0.25, 2)
        harmony = min(0.97, harmony)

        recipe = {
            "name": name[:60],
            "style": style,
            "mood": mood[:3],
            "season": ["all"],
            "concentration": "EDP",
            "ingredients": ingredients,
            "harmony_score": harmony,
            "source": "fragrantica"
        }
        recipes.append(recipe)
        mapped_count += 1

    print(f"\n=== RESULTS ===")
    print(f"Mapped successfully: {mapped_count}")
    print(f"Skipped (no notes or <3 ingredients): {skipped_count}")
    print(f"Unique unmapped notes: {len(unmapped)}")
    print(f"Top 20 unmapped:")
    for name, count in unmapped.most_common(20):
        print(f"  {name}: {count}")

    # Save Fragrantica recipes
    with open('data/fragrantica_recipes.json', 'w', encoding='utf-8') as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(recipes)} Fragrantica recipes to data/fragrantica_recipes.json")

    # Merge with existing generated recipes
    with open('data/recipe_training_data.json', 'r', encoding='utf-8') as f:
        existing = json.load(f)

    merged = existing + recipes
    with open('data/recipe_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Merged: {len(existing)} generated + {len(recipes)} Fragrantica = {len(merged)} total")


# Build category sets
BY_CAT_SET = {}
for i in all_ings:
    cat = i.get('category', '')
    BY_CAT_SET.setdefault(cat, set()).add(i['id'])


if __name__ == '__main__':
    main()
