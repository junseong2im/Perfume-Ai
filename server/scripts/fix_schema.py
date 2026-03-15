# -*- coding: utf-8 -*-
import sys,os,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
sys.path.insert(0,'.')
import database

conn = database.get_conn()
cur = conn.cursor()

# 1) Check columns
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='ingredients' ORDER BY ordinal_position")
print("=== Current columns ===")
for r in cur.fetchall():
    print(f"  {r[0]:25s} {r[1]}")

# 2) Add missing columns
missing = {
    "odor_descriptors": "TEXT",
    "odor_strength": "INTEGER",
    "price_usd_kg": "DECIMAL",
    "volatility": "VARCHAR(20)",
}
for col, dtype in missing.items():
    try:
        cur.execute(f"ALTER TABLE ingredients ADD COLUMN {col} {dtype}")
        print(f"  + Added {col}")
    except Exception as e:
        conn.rollback()
        if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
            print(f"  = {col} already exists")
        else:
            print(f"  ! {col}: {e}")

# 3) Verify
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='ingredients' ORDER BY ordinal_position")
print("\n=== Final columns ===")
cols = [r[0] for r in cur.fetchall()]
for c in cols:
    print(f"  {c}")
print(f"\nTotal columns: {len(cols)}")
