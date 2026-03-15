import csv

# Check arctander stimuli.csv for CID mapping
with open("data/cache/arctander_1960_stimuli.csv", "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    rows = list(r)
    print(f"Arctander stimuli.csv: {len(rows)} rows")
    print(f"  Cols: {list(rows[0].keys())[:8]}")
    for row in rows[:3]:
        print(f"  {dict(list(row.items())[:6])}")

# Check if behavior Stimulus matches stimuli Stimulus
with open("data/cache/arctander_1960_behavior_1.csv", "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    brows = list(r)
    beh_stim = set(row.get("Stimulus", "") for row in brows)

stim_ids = set(row.get("Stimulus", row.get(list(rows[0].keys())[0], "")) for row in rows)
cid_set = set(row.get("CID", "") for row in rows)
print(f"\nBehavior Stimulus IDs (first 10): {sorted(beh_stim)[:10]}")
print(f"Stimuli IDs (first 10): {sorted(stim_ids)[:10]}")
print(f"Stimuli CIDs (first 10): {sorted(cid_set)[:10]}")
overlap = beh_stim & stim_ids
print(f"Overlap beh/stim: {len(overlap)}")
