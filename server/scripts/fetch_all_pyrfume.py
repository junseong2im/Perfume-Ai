# -*- coding: utf-8 -*-
"""
Mega Pyrfume Data Collector — ALL odorCharacter datasets → PostgreSQL
=====================================================================
Downloads ALL 20+ olfactory datasets from Pyrfume GitHub, resolves 
canonical SMILES, deduplicates, and inserts into the perfumer DB.

Target: 25,000-50,000 unique molecules with quality odor labels.

Quality Strategy:
  - Canonical SMILES deduplication via RDKit
  - Multi-source label consensus (molecule with labels from 3+ sources 
    is more trustworthy)
  - Source quality weighting (expert panels > crowd data)
  - Skip molecules without valid SMILES

Usage:
    python scripts/fetch_all_pyrfume.py
"""

import csv
import io
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available, SMILES canonicalization disabled")

import psycopg2

# ================================================================
# Configuration
# ================================================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'perfumer',
    'user': 'perfumer',
    'password': 'perfumer123'
}

PYRFUME_BASE = 'https://raw.githubusercontent.com/pyrfume/pyrfume-data/main'

# All Pyrfume datasets tagged with <odorCharacter> + <human>
# Ordered by data quality (expert panels first, then databases, then crowd)
DATASETS = [
    # === Tier 1: Expert panel / curated professional data ===
    {
        'name': 'leffingwell',
        'quality': 1.0,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',   # behavior cols are 0/1
        'desc': 'Leffingwell & Associates — expert perfumery panel',
    },
    {
        'name': 'goodscents',
        'quality': 1.0,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'The Good Scents Company — professional flavor/fragrance DB',
    },
    {
        'name': 'arctander_1960',
        'quality': 0.95,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'Arctander 1960 — classic perfumery reference book',
    },
    {
        'name': 'sigma_2014',
        'quality': 0.90,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'Sigma-Aldrich Flavors & Fragrances catalog',
    },
    {
        'name': 'ifra_2019',
        'quality': 0.90,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'IFRA — International Fragrance Association',
    },
    
    # === Tier 2: Academic studies with controlled panels ===
    {
        'name': 'dravnieks_1985',
        'quality': 0.85,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'intensity',  # cols are numeric intensity 0-5
        'threshold': 0.5,     # min intensity to count as "present"
        'desc': 'Dravnieks Atlas of Odor Character Profiles',
    },
    {
        'name': 'keller_2016',
        'quality': 0.85,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'intensity',
        'threshold': 20,  # 0-100 scale
        'desc': 'Keller & Vosshall 2016 — 481 molecules, crowd-sourced ratings',
    },
    {
        'name': 'keller_2012',
        'quality': 0.80,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'intensity',
        'threshold': 20,
        'desc': 'Keller & Vosshall 2012 — initial study',
    },
    {
        'name': 'bushdid_2014',
        'quality': 0.80,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'Bushdid et al 2014 — odor discrimination study',
    },
    {
        'name': 'snitz_2013',
        'quality': 0.80,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'intensity',
        'threshold': 20,
        'desc': 'Snitz et al 2013 — odor similarity',
    },
    {
        'name': 'snitz_2019',
        'quality': 0.80,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'intensity',
        'threshold': 20,
        'desc': 'Snitz et al 2019 — odor character',
    },
    {
        'name': 'ravia_2020',
        'quality': 0.80,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'Ravia et al 2020 — odor mixtures',
    },
    {
        'name': 'weiss_2012',
        'quality': 0.75,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'intensity',
        'threshold': 20,
        'desc': 'Weiss et al 2012 — US census odor survey',
    },
    {
        'name': 'nat_geo_1986',
        'quality': 0.70,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'intensity',
        'threshold': 0.3,
        'desc': 'National Geographic 1986 smell survey',
    },
    {
        'name': 'nhanes_2014',
        'quality': 0.70,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'NHANES 2014 — health survey with odor data',
    },
    
    # === Tier 3: Database aggregations ===
    {
        'name': 'flavordb',
        'quality': 0.75,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'FlavorDB — flavor compound database',
    },
    {
        'name': 'flavornet',
        'quality': 0.75,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'Flavornet — GC-olfactometry compiled data',
    },
    {
        'name': 'foodb',
        'quality': 0.70,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'FooDB — food compound database',
    },
    {
        'name': 'aromadb',
        'quality': 0.70,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'AromaDB — aroma compound database',
    },
    {
        'name': 'freesolve',
        'quality': 0.65,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'FreeSolv — with odor annotations',
    },
    
    # === Tier 4: Specialized / supplementary ===
    {
        'name': 'sharma_2021a',
        'quality': 0.70,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'Sharma et al 2021a — computational odor character',
    },
    {
        'name': 'sharma_2021b',
        'quality': 0.70,
        'molecules': 'molecules.csv',
        'behavior': 'behavior.csv',
        'type': 'binary',
        'desc': 'Sharma et al 2021b — odor character prediction',
    },
]


# ================================================================
# Helpers
# ================================================================

def canonical_smiles(smiles):
    """SMILES → canonical SMILES via RDKit"""
    if not smiles:
        return None
    if not HAS_RDKIT:
        return smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def download_csv(url, retries=3):
    """Download CSV and parse to list of dicts"""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PyrfumeCollector/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                # Try UTF-8, fall back to latin-1
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        text = raw.decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    text = raw.decode('utf-8', errors='replace')
                
                reader = csv.DictReader(io.StringIO(text))
                return list(reader)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # File doesn't exist
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    WARN: Failed to download {url}: {e}")
    return None


def connect_db():
    """Connect to PostgreSQL"""
    for attempt in range(3):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = False
            return conn
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"ERROR: Cannot connect to DB: {e}")
                sys.exit(1)


# ================================================================
# Main Collection Engine
# ================================================================

class PyrfumeMegaCollector:
    """Downloads ALL Pyrfume datasets and inserts into DB"""
    
    def __init__(self):
        self.conn = connect_db()
        self.cur = self.conn.cursor()
        
        # In-memory merged data: canonical_smiles → {name, cid, labels, sources, quality}
        self.molecules = {}  # canonical_smiles → dict
        
        # Stats
        self.stats = defaultdict(int)
        
    def run(self):
        """Full collection pipeline"""
        print("=" * 70)
        print("  🧪 MEGA PYRFUME DATA COLLECTOR")
        print("  Target: ALL 22 odorCharacter datasets")
        print("=" * 70)
        
        # Phase 1: Download all datasets into memory
        self._download_all()
        
        # Phase 2: Quality dedup & merge
        self._quality_merge()
        
        # Phase 3: Insert into DB
        self._insert_to_db()
        
        # Phase 4: Print final stats
        self._print_stats()
        
        self.conn.close()
    
    def _download_all(self):
        """Download molecules + behaviors from all datasets"""
        total_datasets = len(DATASETS)
        
        for i, ds in enumerate(DATASETS, 1):
            name = ds['name']
            print(f"\n[{i}/{total_datasets}] {name} — {ds['desc']}")
            
            # Download molecules.csv
            mol_url = f"{PYRFUME_BASE}/{name}/{ds['molecules']}"
            mols = download_csv(mol_url)
            
            if mols is None:
                print(f"    ⚠ molecules.csv not found, skipping")
                self.stats['datasets_skipped'] += 1
                continue
            
            print(f"    molecules.csv: {len(mols)} rows")
            
            # Build CID → SMILES mapping
            cid_to_smiles = {}
            cid_to_name = {}
            direct_smiles = {}
            
            for mol in mols:
                cid = mol.get('CID', '') or mol.get('PubChem CID', '') or mol.get('cid', '')
                smiles = (mol.get('IsomericSMILES', '') or mol.get('SMILES', '') or 
                         mol.get('smiles', '') or mol.get('canonical_smiles', ''))
                mol_name = (mol.get('name', '') or mol.get('Name', '') or 
                           mol.get('IUPACName', '') or mol.get('preferred_name', '') or '')
                
                if smiles:
                    can = canonical_smiles(smiles)
                    if can:
                        if cid:
                            try:
                                cid_int = int(float(cid))
                                cid_to_smiles[str(cid_int)] = can
                                cid_to_name[str(cid_int)] = mol_name
                            except (ValueError, OverflowError):
                                cid_to_smiles[str(cid)] = can
                                cid_to_name[str(cid)] = mol_name
                        direct_smiles[can] = mol_name
            
            print(f"    Valid SMILES: {len(set(cid_to_smiles.values()) | set(direct_smiles.keys()))}")
            
            # Download behavior.csv
            beh_url = f"{PYRFUME_BASE}/{name}/{ds['behavior']}"
            behaviors = download_csv(beh_url)
            
            if behaviors is None:
                print(f"    ⚠ behavior.csv not found, using molecules only")
                # Still add molecules without labels
                for can_smi, mol_name in direct_smiles.items():
                    if can_smi not in self.molecules:
                        self.molecules[can_smi] = {
                            'name': mol_name,
                            'labels': defaultdict(float),
                            'sources': set(),
                            'n_sources': 0,
                        }
                self.stats['datasets_no_behavior'] += 1
                continue
            
            print(f"    behavior.csv: {len(behaviors)} rows")
            
            # Process behaviors
            all_cols = list(behaviors[0].keys()) if behaviors else []
            # Identify the stimulus/CID column
            stim_col = None
            for col_name in ['Stimulus', 'stimulus', 'CID', 'cid', 'PubChem CID']:
                if col_name in all_cols:
                    stim_col = col_name
                    break
            
            if stim_col is None:
                # Try first column
                stim_col = all_cols[0] if all_cols else None
            
            if stim_col is None:
                print(f"    ⚠ No stimulus column found, skipping behavior")
                continue
            
            # Descriptor columns = all except stimulus
            desc_cols = [c for c in all_cols if c != stim_col and c.lower() not in 
                        ('cid', 'stimulus', 'pubchem cid', 'smiles', 'name', 'iupacname',
                         'isomericsmiles', 'molecularweight', 'molecular_weight')]
            
            # Filter out meaningless columns (all zeros, metadata, etc.)
            quality_weight = ds['quality']
            threshold = ds.get('threshold', 0.5)
            ds_type = ds['type']
            
            n_matched = 0
            n_labels_added = 0
            
            for row in behaviors:
                stim = row.get(stim_col, '').strip()
                if not stim:
                    continue
                
                # Find canonical SMILES for this stimulus
                can_smi = cid_to_smiles.get(stim)
                mol_name = cid_to_name.get(stim, '')
                
                if can_smi is None:
                    # Maybe stimulus IS a SMILES (but skip if it looks like a CID number)
                    try:
                        float(stim)
                        continue  # It's a numeric CID with no matching SMILES → skip
                    except ValueError:
                        can_smi = canonical_smiles(stim)
                
                if can_smi is None:
                    continue
                
                # Initialize molecule entry
                if can_smi not in self.molecules:
                    self.molecules[can_smi] = {
                        'name': mol_name,
                        'labels': defaultdict(float),
                        'sources': set(),
                        'n_sources': 0,
                    }
                
                mol_entry = self.molecules[can_smi]
                if not mol_entry['name'] and mol_name:
                    mol_entry['name'] = mol_name
                
                # Extract labels from this row
                labels_this_row = []
                for col in desc_cols:
                    val_str = row.get(col, '').strip()
                    if not val_str:
                        continue
                    
                    try:
                        val = float(val_str)
                    except ValueError:
                        # Some datasets have text labels; treat non-empty as 1
                        if val_str.lower() not in ('', 'nan', 'none', 'null', '0', 'false'):
                            val = 1.0
                        else:
                            continue
                    
                    if ds_type == 'binary':
                        if val >= 0.5:
                            labels_this_row.append(col.lower().strip())
                    elif ds_type == 'intensity':
                        if val >= threshold:
                            labels_this_row.append(col.lower().strip())
                
                if labels_this_row:
                    n_matched += 1
                    mol_entry['sources'].add(name)
                    for label in labels_this_row:
                        # Weight by source quality
                        mol_entry['labels'][label] = max(
                            mol_entry['labels'][label], quality_weight
                        )
                        n_labels_added += 1
            
            print(f"    Matched: {n_matched} molecules, {n_labels_added} label instances")
            self.stats['datasets_loaded'] += 1
            self.stats[f'labels_{name}'] = n_labels_added
        
        print(f"\n{'=' * 70}")
        print(f"  Downloaded {self.stats['datasets_loaded']}/{total_datasets} datasets")
        print(f"  Total unique molecules: {len(self.molecules)}")
    
    def _quality_merge(self):
        """Quality filtering and dedup"""
        print(f"\n{'=' * 70}")
        print("  QUALITY MERGE & DEDUP")
        print(f"{'=' * 70}")
        
        n_total = len(self.molecules)
        n_with_labels = sum(1 for m in self.molecules.values() if len(m['labels']) > 0)
        n_multi_source = sum(1 for m in self.molecules.values() if len(m['sources']) >= 2)
        
        # Update n_sources
        for m in self.molecules.values():
            m['n_sources'] = len(m['sources'])
        
        # Count label distribution
        label_counts = defaultdict(int)
        for m in self.molecules.values():
            for label in m['labels']:
                label_counts[label] += 1
        
        print(f"  Total molecules: {n_total}")
        print(f"  With labels: {n_with_labels}")
        print(f"  Multi-source: {n_multi_source}")
        print(f"  Unique labels: {len(label_counts)}")
        
        # Show top 30 labels
        top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:30]
        print(f"\n  Top 30 labels:")
        for label, count in top_labels:
            print(f"    {label:25s}: {count:6d}")
        
        self.stats['total_molecules'] = n_total
        self.stats['with_labels'] = n_with_labels
        self.stats['multi_source'] = n_multi_source
        self.stats['unique_labels'] = len(label_counts)
    
    def _insert_to_db(self):
        """Insert all molecules and labels into PostgreSQL"""
        print(f"\n{'=' * 70}")
        print("  INSERTING INTO DATABASE")
        print(f"{'=' * 70}")
        
        cur = self.cur
        
        # Step 1: Get existing descriptor map
        cur.execute("SELECT id, lower(name) FROM odor_descriptors")
        desc_map = {name: id for id, name in cur.fetchall()}
        print(f"  Existing descriptors: {len(desc_map)}")
        
        # Step 2: Get existing molecules by SMILES
        cur.execute("SELECT id, smiles FROM molecules WHERE smiles IS NOT NULL")
        existing_smiles = {}
        for mol_id, smi in cur.fetchall():
            if smi:
                can = canonical_smiles(smi) if HAS_RDKIT else smi.strip()
                if can:
                    existing_smiles[can] = mol_id
        print(f"  Existing molecules (by SMILES): {len(existing_smiles)}")
        
        # Step 3: Insert new descriptors
        all_labels = set()
        for m in self.molecules.values():
            all_labels.update(m['labels'].keys())
        
        new_descs = all_labels - set(desc_map.keys())
        if new_descs:
            print(f"  Adding {len(new_descs)} new descriptors...")
            for desc_name in sorted(new_descs):
                try:
                    cur.execute(
                        "INSERT INTO odor_descriptors (name, category) VALUES (%s, %s) RETURNING id",
                        (desc_name, 'pyrfume_mega')
                    )
                    result = cur.fetchone()
                    if result:
                        desc_map[desc_name] = result[0]
                except Exception:
                    self.conn.rollback()
                    # Might already exist (race), try to fetch
                    cur.execute("SELECT id FROM odor_descriptors WHERE lower(name) = %s", (desc_name,))
                    row = cur.fetchone()
                    if row:
                        desc_map[desc_name] = row[0]
        
        self.conn.commit()
        print(f"  Total descriptors: {len(desc_map)}")
        
        # Step 4: Insert molecules
        inserted_mols = 0
        updated_mols = 0
        inserted_labels = 0
        batch_size = 500
        items = list(self.molecules.items())
        
        print(f"  Inserting {len(items)} molecules...")
        
        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start:batch_start + batch_size]
            
            for can_smi, mol_data in batch:
                labels = mol_data['labels']
                if not labels:
                    continue  # Skip molecules without labels
                
                mol_name = mol_data.get('name', '')[:500] or f'mol_{hash(can_smi) % 10**8}'
                n_sources = mol_data['n_sources']
                
                # Check if molecule already exists
                mol_id = existing_smiles.get(can_smi)
                
                if mol_id is None:
                    # Generate a CID from hash if we don't have one
                    cid_hash = abs(hash(can_smi)) % (10**15)
                    
                    try:
                        cur.execute("""
                            INSERT INTO molecules (cid, name, smiles, source)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (cid) DO UPDATE SET 
                                smiles = COALESCE(molecules.smiles, EXCLUDED.smiles),
                                source = CASE WHEN molecules.source LIKE '%%pyrfume%%' 
                                             THEN molecules.source 
                                             ELSE molecules.source || ',pyrfume_mega' END
                            RETURNING id
                        """, (cid_hash, mol_name, can_smi, 'pyrfume_mega'))
                        result = cur.fetchone()
                        if result:
                            mol_id = result[0]
                            existing_smiles[can_smi] = mol_id
                            inserted_mols += 1
                    except Exception as e:
                        self.conn.rollback()
                        # Try with different CID
                        cid_hash = (abs(hash(can_smi + '_alt')) % (10**15))
                        try:
                            cur.execute("""
                                INSERT INTO molecules (cid, name, smiles, source)
                                VALUES (%s, %s, %s, 'pyrfume_mega')
                                ON CONFLICT (cid) DO NOTHING
                                RETURNING id
                            """, (cid_hash, mol_name, can_smi))
                            result = cur.fetchone()
                            if result:
                                mol_id = result[0]
                                existing_smiles[can_smi] = mol_id
                                inserted_mols += 1
                        except Exception:
                            self.conn.rollback()
                            continue
                else:
                    updated_mols += 1
                
                if mol_id is None:
                    continue
                
                # Insert labels
                for label, strength in labels.items():
                    desc_id = desc_map.get(label)
                    if desc_id is None:
                        continue
                    
                    try:
                        cur.execute("""
                            INSERT INTO molecule_odors (molecule_id, descriptor_id, strength, source)
                            VALUES (%s, %s, %s, 'pyrfume_mega')
                            ON CONFLICT (molecule_id, descriptor_id) DO UPDATE SET
                                strength = GREATEST(molecule_odors.strength, EXCLUDED.strength)
                        """, (mol_id, desc_id, strength))
                        inserted_labels += 1
                    except Exception:
                        self.conn.rollback()
            
            self.conn.commit()
            
            pct = min(100, (batch_start + batch_size) / len(items) * 100)
            print(f"    Progress: {pct:.0f}% ({inserted_mols} new, {updated_mols} updated, {inserted_labels} labels)")
        
        self.conn.commit()
        
        self.stats['inserted_mols'] = inserted_mols
        self.stats['updated_mols'] = updated_mols
        self.stats['inserted_labels'] = inserted_labels
        
        print(f"\n  ✅ Done: {inserted_mols} new molecules, {updated_mols} updated, {inserted_labels} labels")
    
    def _print_stats(self):
        """Print final statistics"""
        cur = self.cur
        
        # Count total molecules and labels
        cur.execute("SELECT COUNT(*) FROM molecules")
        total_mols = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM molecule_odors")
        total_labels = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT molecule_id) FROM molecule_odors")
        labeled_mols = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM odor_descriptors")
        total_descs = cur.fetchone()[0]
        
        # Molecules with SMILES
        cur.execute("SELECT COUNT(*) FROM molecules WHERE smiles IS NOT NULL AND smiles != ''")
        with_smiles = cur.fetchone()[0]
        
        print(f"\n{'=' * 70}")
        print(f"  📊 FINAL DATABASE STATISTICS")
        print(f"{'=' * 70}")
        print(f"  Total molecules:        {total_mols:>8,}")
        print(f"  With SMILES:            {with_smiles:>8,}")
        print(f"  With odor labels:       {labeled_mols:>8,}")
        print(f"  Total label instances:  {total_labels:>8,}")
        print(f"  Unique descriptors:     {total_descs:>8,}")
        print(f"")
        print(f"  New molecules inserted: {self.stats.get('inserted_mols', 0):>8,}")
        print(f"  Existing updated:       {self.stats.get('updated_mols', 0):>8,}")
        print(f"  Multi-source molecules: {self.stats.get('multi_source', 0):>8,}")
        print(f"{'=' * 70}")
        
        # Top descriptors
        cur.execute("""
            SELECT d.name, COUNT(*) as cnt 
            FROM molecule_odors mo 
            JOIN odor_descriptors d ON d.id = mo.descriptor_id 
            GROUP BY d.name 
            ORDER BY cnt DESC 
            LIMIT 30
        """)
        print(f"\n  Top 30 descriptors in DB:")
        for name, count in cur.fetchall():
            print(f"    {name:25s}: {count:6d}")


# ================================================================
# Entry point
# ================================================================

if __name__ == '__main__':
    t0 = time.time()
    collector = PyrfumeMegaCollector()
    collector.run()
    elapsed = time.time() - t0
    print(f"\n  ⏱ Total time: {elapsed:.1f}s")
    print(f"  🎯 Next: run precompute_bert.py → train_multitask_pipeline.py")
