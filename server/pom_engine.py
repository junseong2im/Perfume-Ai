"""
POM Engine v3 -- Complete AI Perfumer Engine
=============================================
Step 1: Fragrance DB (name/CAS/SMILES/ODT/POM 256d)
Step 2: Non-linear Mixture (OAV + Weber-Fechner + masking)
Step 3: GA Reverse Engineering (IFRA constraints)
=============================================
10-model MPNN-POM Ensemble (AUROC 0.7890)
"""
import os, sys, json, math, csv, random, sqlite3, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Constants
# ============================================================
TASKS_138 = [
    'alcoholic','aldehydic','alliaceous','almond','amber','animal','anisic',
    'apple','apricot','aromatic','balsamic','banana','beefy','bergamot',
    'berry','bitter','black currant','brandy','burnt','buttery','cabbage',
    'camphoreous','caramellic','cedar','celery','chamomile','cheesy','cherry',
    'chocolate','cinnamon','citrus','clean','clove','cocoa','coconut','coffee',
    'cognac','cooked','cooling','cortex','coumarinic','creamy','cucumber',
    'dairy','dry','earthy','ethereal','fatty','fermented','fishy','floral',
    'fresh','fruit skin','fruity','garlic','gassy','geranium','grape',
    'grapefruit','grassy','green','hawthorn','hay','hazelnut','herbal',
    'honey','hyacinth','jasmin','juicy','ketonic','lactonic','lavender',
    'leafy','leathery','lemon','lily','malty','meaty','medicinal','melon',
    'metallic','milky','mint','muguet','mushroom','musk','musty','natural',
    'nutty','odorless','oily','onion','orange','orangeflower','orris','ozone',
    'peach','pear','phenolic','pine','pineapple','plum','popcorn','potato',
    'powdery','pungent','radish','raspberry','ripe','roasted','rose','rummy',
    'sandalwood','savory','sharp','smoky','soapy','solvent','sour','spicy',
    'strawberry','sulfurous','sweaty','sweet','tea','terpenic','tobacco',
    'tomato','tropical','vanilla','vegetable','vetiver','violet','warm',
    'waxy','weedy','winey','woody'
]

ODOR_DIMS_22 = [
    'floral','citrus','woody','fruity','spicy','herbal',
    'musk','amber','green','warm','balsamic','leather',
    'smoky','earthy','aquatic','powdery','gourmand',
    'animalic','sweet','fresh','aromatic','waxy'
]

_MAP_138_TO_22 = {
    'floral': ['floral','rose','jasmin','violet','lavender','lily','chamomile','orris',
               'muguet','hyacinth','geranium','orangeflower','hawthorn'],
    'citrus': ['citrus','lemon','orange','grapefruit','bergamot'],
    'woody': ['woody','cedar','pine','sandalwood','vetiver','terpenic'],
    'fruity': ['fruity','apple','pear','peach','apricot','cherry','banana','plum',
               'grape','melon','pineapple','strawberry','tropical','berry',
               'black currant','coconut','raspberry','fruit skin','ripe','juicy'],
    'spicy': ['spicy','cinnamon','clove','pungent','sharp'],
    'herbal': ['herbal','mint','tea','hay','leafy','weedy'],
    'musk': ['musk','musty'],
    'amber': ['amber','balsamic'],
    'green': ['green','grassy','cucumber','vegetable'],
    'warm': ['warm','coumarinic','tobacco'],
    'balsamic': ['balsamic','honey','vanilla','caramellic'],
    'leather': ['leathery'],
    'smoky': ['smoky','burnt','roasted','coffee','cocoa','chocolate'],
    'earthy': ['earthy','mushroom','potato','radish'],
    'aquatic': ['ozone','cooling','clean'],
    'powdery': ['powdery','creamy','milky','dry','soapy'],
    'gourmand': ['buttery','popcorn','nutty','hazelnut','almond','chocolate',
                 'caramellic','cocoa','coconut'],
    'animalic': ['animal','fishy','sulfurous','alliaceous','garlic','onion','cheesy',
                 'sweaty','meaty','beefy','cabbage'],
    'sweet': ['sweet','vanilla','honey','caramellic'],
    'fresh': ['fresh','clean','ethereal','solvent','camphoreous'],
    'aromatic': ['aromatic','herbal','medicinal'],
    'waxy': ['waxy','fatty','oily','aldehydic'],
}

_TASK_IDX = {t: i for i, t in enumerate(TASKS_138)}
_MAP_22_IDX = {}
for di, dn in enumerate(ODOR_DIMS_22):
    _MAP_22_IDX[di] = [_TASK_IDX[t] for t in _MAP_138_TO_22.get(dn, []) if t in _TASK_IDX]

# Known SMILES for common perfumery ingredients (hardcoded, no API needed)
_KNOWN_SMILES = {
    'linalool': 'CC(=CCC/C(=C/CO)/C)C',
    'linalyl_acetate': 'CC(=CCC/C(=C/COC(=O)C)/C)C',
    'limonene': 'CC1=CCC(CC1)C(=C)C',
    'geraniol': 'CC(=CCCC(=CC=O)C)C' if False else 'CC(=CCC/C(=C/CO)C)C',
    'citronellol': 'CC(CCC=C(C)C)CCO',
    'citronellal': 'CC(CCC=C(C)C)CC=O',
    'citral': 'CC(=CCC/C(=C/C=O)C)C',
    'vanillin': 'COC1=CC(C=O)=CC=C1O',
    'eugenol': 'COC1=C(O)C=CC(CC=C)=C1',
    'coumarin': 'O=C1OC2=CC=CC=C2C=C1',
    'ionone_alpha': 'CC1=C(C(CC1)(C)C)/C=C/C(=O)C' if False else 'CC(=O)/C=C/C1=C(C)CCCC1(C)C',
    'ionone_beta': 'CC1=CC(=CC(C1)(C)C)/C=C/C(=O)C' if False else 'CC(=O)/C=C/C1C(=CC(CC1)C)(C)C',
    'hedione': 'O=C(OC)CCC1CCC(=CC1)C' if False else 'COC(=O)CCC1CCC(CC1)=CC',
    'iso_e_super': 'CC1(CCCC2(C1CCC(=O)C2)C)CC',
    'galaxolide': 'CC1(C)C2=CC(=CC(=C2OC(C1)C)C)CCC',
    'musk_ketone': 'CC1=CC([N+](=O)[O-])=C(C)C(=C1[N+](=O)[O-])C(C)(C)C',
    'benzyl_acetate': 'CC(=O)OCC1=CC=CC=C1',
    'benzyl_benzoate': 'O=C(OCC1=CC=CC=C1)C2=CC=CC=C2',
    'methyl_salicylate': 'COC(=O)C1=CC=CC=C1O',
    'phenylethyl_alcohol': 'OCCC1=CC=CC=C1',
    'cinnamic_aldehyde': 'O=C/C=C/C1=CC=CC=C1',
    'menthol': 'CC(C)C1CCC(C)CC1O',
    'camphor': 'CC1(C)C2CCC1(C)C(=O)C2',
    'carvone': 'CC(=C)C1CC=C(C)C(=O)C1',
    'thymol': 'CC(C)C1=CC(=C(C=C1)O)C',
    'safranal': 'CC1=C(C=O)C(C)(C)CC=C1',
    'indole': 'C1=CC=C2C(=C1)C=CN2',
    'skatole': 'CC1=CNC2=CC=CC=C21',
    'muscone': 'CC1CCCCCCCCCCCC(=O)CC1',
    'ambroxan': 'CC12CCC(CC1C3CC(C)(O3)C2)C',
    'cedrol': 'CC1CCC2C(C1)C3(CCC(C3CC2(C)C)O)C',
    'santalol_alpha': 'CC(=CCC1C(C1(C)C)CC=C(C)C)CO',
    'vetiverol': 'CC(=C)C1CCC(CC1O)C(=C)C',
    'patchoulol': 'CC1CCC2C(C1)C3(CCCC(C3CC2(C)C)O)C',
    'geranyl_acetate': 'CC(=CCC/C(=C/COC(=O)C)C)C',
    'ethyl_maltol': 'CCC1=C(O)C(=O)C=CO1',
    'damascone_beta': 'CC(=O)/C=C/C1C(=CC(CC1)C)(C)C',
    'methyl_jasmonate': 'COC(=O)C1C(/C=C\\CC)CCC1=O',
    'isoeugenol': 'COC1=C(O)C=CC(/C=C/C)=C1',
    'anethole': 'COC1=CC=C(/C=C/C)C=C1',
    'cinnamate_methyl': 'COC(=O)/C=C/C1=CC=CC=C1',
    'tonalid': 'CC1=CC2=CC(=C1)C(C)(C)CC(C2)C(C)=O',
    'cashmeran': 'CC1(C)C2CC(=O)C1(C)CC2(C)C',
    'acetophenone': 'CC(=O)C1=CC=CC=C1',
    'benzaldehyde': 'O=CC1=CC=CC=C1',
    'phenylacetaldehyde': 'O=CCC1=CC=CC=C1',
    'hydroxycitronellal': 'CC(CCO)CCC(O)CC',
    'lilial': 'CC(C)C1=CC=C(C=C1)CC(C)C=O',
    'lyral': 'CC(CCC=C(C)C)(CC=O)CC=O',
}

# Default ODT values (log scale, from literature / Abraham 2012)
# Higher = more detectable at lower concentration
_DEFAULT_ODT_LOG = {
    'geosmin': 5.0, 'skatole': 4.5, 'indole': 4.0,
    'vanillin': 3.5, 'coumarin': 3.0, 'eugenol': 2.8,
    'cinnamic_aldehyde': 3.2, 'safranal': 3.5,
    'linalool': 2.0, 'geraniol': 1.8, 'citronellol': 1.5,
    'limonene': 1.2, 'menthol': 2.5, 'camphor': 1.8,
    'phenylethyl_alcohol': 1.0, 'benzaldehyde': 2.8,
    'acetophenone': 1.5, 'hedione': 0.5, 'muscone': 3.0,
    'ambroxan': 2.5, 'galaxolide': 1.5, 'iso_e_super': 0.8,
    'methyl_salicylate': 2.0, 'thymol': 2.2,
}

# IFRA restricted/banned categories
_IFRA_RESTRICTED = {
    'oakmoss', 'treemoss', 'nitro_musk', 'musk_ambrette', 'musk_moskene',
    'musk_tibetene', 'methyl_heptine_carbonate', 'fig_leaf_absolute',
    'costus_root', 'peru_balsam',
}

# Essential Oil GC-MS Profiles: name -> {SMILES: fraction}
# From literature / GC-MS analysis (top 3-5 components)
_ESSENTIAL_OIL_PROFILES = {
    'rose': {
        'CC(C)=CCCC(C)CCO': 0.55,           # Citronellol
        'CC(C)=CCC/C(=C/CO)C': 0.20,        # Geraniol
        'OCCC1=CC=CC=C1': 0.15,              # Phenylethyl alcohol
        'CC(C)=CCCC(C)=CCO': 0.10,           # Nerol
    },
    'lavender': {
        'CC(=CCC/C(=C/COC(=O)C)/C)C': 0.40, # Linalyl acetate
        'CC(=CCC/C(=C/CO)/C)C': 0.35,        # Linalool
        'CC1(C)C2CCC1(C)C(=O)C2': 0.10,     # Camphor
        'CC1=CCC(CC1)C(=C)C': 0.08,          # Limonene
        'COC1=CC(C=O)=CC=C1O': 0.07,         # not exact, 1,8-cineole placeholder
    },
    'jasmine': {
        'CC(=CCC/C(=C/CO)/C)C': 0.25,        # Linalool
        'CC(=O)OCC1=CC=CC=C1': 0.20,          # Benzyl acetate
        'C1=CC=C2C(=C1)C=CN2': 0.08,          # Indole
        'COC(=O)CCC1CCC(CC1)=CC': 0.15,      # Hedione (methyl dihydrojasmonate)
        'COC(=O)C1C(/C=C\\CC)CCC1=O': 0.12,  # Methyl jasmonate
    },
    'ylang_ylang': {
        'CC(=CCC/C(=C/CO)/C)C': 0.25,        # Linalool
        'CC(=CCC/C(=C/COC(=O)C)/C)C': 0.15, # Geranyl acetate
        'CC(=O)OCC1=CC=CC=C1': 0.20,          # Benzyl acetate
        'O=C(OCC1=CC=CC=C1)C2=CC=CC=C2': 0.10, # Benzyl benzoate
        'COC1=C(O)C=CC(CC=C)=C1': 0.10,      # Eugenol
    },
    'neroli': {
        'CC(=CCC/C(=C/CO)/C)C': 0.40,        # Linalool
        'CC(=CCC/C(=C/COC(=O)C)/C)C': 0.15, # Linalyl acetate
        'CC1=CCC(CC1)C(=C)C': 0.12,           # Limonene
        'CC(=O)OCC1=CC=CC=C1': 0.08,           # Benzyl acetate
    },
    'bergamot': {
        'CC(=CCC/C(=C/COC(=O)C)/C)C': 0.35, # Linalyl acetate
        'CC(=CCC/C(=C/CO)/C)C': 0.20,        # Linalool
        'CC1=CCC(CC1)C(=C)C': 0.30,           # Limonene
    },
    'geranium': {
        'CC(C)=CCCC(C)CCO': 0.35,             # Citronellol
        'CC(C)=CCC/C(=C/CO)C': 0.20,          # Geraniol
        'CC(C)=CCCC(C)CC=O': 0.10,            # Citronellal
        'CC(=CCC/C(=C/CO)/C)C': 0.10,        # Linalool
        'COC(=O)/C=C/C1=CC=CC=C1': 0.05,     # Methyl cinnamate
    },
    'patchouli': {
        'CC1CCC2C(C1)C3(CCCC(C3CC2(C)C)O)C': 0.40, # Patchoulol
        'CC1CCC2C(C1)C3(CCC(C3CC2(C)C)O)C': 0.20,   # a-Guaiene approx
    },
    'vetiver': {
        'CC(=C)C1CCC(CC1O)C(=C)C': 0.35,     # Vetiverol
    },
    'sandalwood': {
        'CC(=CCC1C(C1(C)C)CC=C(C)C)CO': 0.50, # alpha-Santalol
    },
    'cedarwood': {
        'CC1CCC2C(C1)C3(CCC(C3CC2(C)C)O)C': 0.50, # Cedrol
    },
    'eucalyptus': {
        'CC1(C)C2CCC1(C)C(=O)C2': 0.20,      # Camphor
        'CC1=CCC(CC1)C(=C)C': 0.15,           # Limonene
    },
    'peppermint': {
        'CC(C)C1CCC(C)CC1O': 0.50,            # Menthol
        'CC(=C)C1CC=C(C)C(=O)C1': 0.15,      # Carvone
    },
    'clove': {
        'COC1=C(O)C=CC(CC=C)=C1': 0.80,       # Eugenol
        'CC(=O)OC1=C(OC)C=CC(CC=C)=C1': 0.10, # Eugenyl acetate
    },
    'cinnamon': {
        'O=C/C=C/C1=CC=CC=C1': 0.70,          # Cinnamaldehyde
        'COC1=C(O)C=CC(CC=C)=C1': 0.10,       # Eugenol
    },
    'frankincense': {
        'CC1=CCC(CC1)C(=C)C': 0.25,            # Limonene
        'CC(=CCC/C(=C/CO)/C)C': 0.10,          # Linalool
    },
    'orange': {
        'CC1=CCC(CC1)C(=C)C': 0.90,            # Limonene
    },
    'lemon': {
        'CC1=CCC(CC1)C(=C)C': 0.65,            # Limonene
        'CC(=CCC/C(=C/C=O)C)C': 0.10,          # Citral (neral+geranial)
    },
    'grapefruit': {
        'CC1=CCC(CC1)C(=C)C': 0.85,            # Limonene
    },
    'lime': {
        'CC1=CCC(CC1)C(=C)C': 0.50,            # Limonene
        'CC(=CCC/C(=C/C=O)C)C': 0.08,          # Citral
    },
    'tea_tree': {
        'CC1=CCC(CC1)C(=C)C': 0.10,            # Limonene
        'CC(C)C1=CC(=C(C=C1)O)C': 0.10,       # Thymol-like (terpinen-4-ol placeholder)
    },
    'chamomile': {
        'CC(=CCC/C(=C/CO)/C)C': 0.15,          # Linalool
    },
    'black_pepper': {
        'CC1=CCC(CC1)C(=C)C': 0.25,            # Limonene
    },
    'ginger': {
        'CC1=CCC(CC1)C(=C)C': 0.10,            # Limonene
    },
    'cardamom': {
        'CC(=CCC/C(=C/COC(=O)C)/C)C': 0.35,  # Linalyl acetate (terpinyl acetate placeholder)
        'CC(=CCC/C(=C/CO)/C)C': 0.10,          # Linalool
    },
    'tuberose': {
        'CC(=O)OCC1=CC=CC=C1': 0.15,           # Benzyl acetate
        'COC1=C(O)C=CC(CC=C)=C1': 0.05,       # Eugenol
    },
    'iris': {
        # Orris butter - alpha-isomethyl ionone placeholder
        'CC(=O)/C=C/C1C(=CC(CC1)C)(C)C': 0.30, # beta-Ionone
    },
    'osmanthus': {
        'CC(=CCC/C(=C/CO)/C)C': 0.15,          # Linalool
        'CC(=O)/C=C/C1C(=CC(CC1)C)(C)C': 0.15, # Ionone
    },
    'mimosa': {
        'CC(=CCC/C(=C/CO)/C)C': 0.15,          # Linalool
        'O=CCC1=CC=CC=C1': 0.10,               # Phenylacetaldehyde
    },
}



# ============================================================
# POMEngine v3
# ============================================================
class POMEngine:
    """Complete AI Perfumer Engine
    
    Features:
    1. SMILES -> 138d odor / 256d POM embedding (10-model ensemble)
    2. Fragrance DB: name <-> CAS <-> SMILES <-> ODT <-> POM
    3. Non-linear mixture: OAV + Weber-Fechner + competitive masking
    4. GA reverse engineering: target POM -> optimal recipe
    5. 22d pipeline compatibility
    """
    
    def __init__(self, model_dir=None, device=None):
        self.models = []
        self.n_tasks = len(TASKS_138)
        self._cache = {}
        self._embedding_db = None
        self._fragrance_db = {}  # name -> {smiles, cas, odt_log, category, ...}
        
        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'models', 'openpom_ensemble')
        self.model_dir = model_dir
        self._loaded = False
    
    # ========================================================
    # Core: Model Loading
    # ========================================================
    
    def load(self):
        """Load ensemble models + embedding DB + fragrance DB"""
        if self._loaded:
            return True
        
        # 1. Load models
        try:
            from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
            from openpom.models.mpnn_pom import MPNNPOMModel
            self.featurizer = GraphFeaturizer()
            
            loaded = 0
            for i in range(10):
                exp_dir = os.path.join(self.model_dir, f'experiments_{i+1}')
                if not os.path.exists(os.path.join(exp_dir, 'checkpoint1.pt')):
                    continue
                try:
                    model = MPNNPOMModel(
                        n_tasks=self.n_tasks, batch_size=128,
                        class_imbalance_ratio=[1.0]*self.n_tasks,
                        loss_aggr_type='sum', node_out_feats=100,
                        edge_hidden_feats=75, edge_out_feats=100,
                        num_step_message_passing=5, mpnn_residual=True,
                        message_aggregator_type='sum', mode='classification',
                        number_atom_features=GraphConvConstants.ATOM_FDIM,
                        number_bond_features=GraphConvConstants.BOND_FDIM,
                        n_classes=1, readout_type='set2set',
                        num_step_set2set=3, num_layer_set2set=2,
                        ffn_hidden_list=[392, 392], ffn_embeddings=256,
                        ffn_activation='relu', ffn_dropout_p=0.12,
                        ffn_dropout_at_input_no_act=False, weight_decay=1e-5,
                        self_loop=False, optimizer_name='adam',
                        model_dir=exp_dir, device_name=self.device,
                    )
                    model.restore(model_dir=exp_dir)
                    self.models.append(model)
                    loaded += 1
                except Exception as e:
                    pass
            
            if loaded > 0:
                print(f"  [POM] {loaded} ensemble models loaded (AUROC 0.7890)")
        except ImportError:
            print("  [POM] OpenPOM not available, using pre-computed embeddings")
        
        # 2. Pre-computed embeddings
        # Prefer v2 (expanded) DB, fallback to v1
        emb_path_v2 = os.path.join(self.model_dir, 'pom_embeddings_v2.npz')
        emb_path = emb_path_v2 if os.path.exists(emb_path_v2) else os.path.join(self.model_dir, 'pom_embeddings.npz')
        if os.path.exists(emb_path):
            data = np.load(emb_path, allow_pickle=True)
            smiles_list = [str(s) for s in data['smiles']]
            # Build canonical index for better matching
            smi_idx = {}
            for i, s in enumerate(smiles_list):
                smi_idx[s] = i
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(s, sanitize=True)
                    if mol:
                        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                        smi_idx[can] = i
                except:
                    pass
            self._embedding_db = {
                'embeddings': data['embeddings'],
                'labels': data['labels'],
                'smiles': smiles_list,
                'smiles_idx': smi_idx
            }
            db_ver = 'v2' if 'v2' in emb_path else 'v1'
            print(f"  [EMB] {len(smiles_list)} molecules x 256d loaded ({db_ver}, {len(smi_idx)} index entries)")
        
        # 3. Build fragrance DB
        self._build_fragrance_db()
        
        # 4. Load PairAttentionNet (if available)
        self._pair_attention = None
        self._pair_labels = None
        self._use_attention = False
        pair_dir = os.path.join(os.path.dirname(self.model_dir), 'pair_attention')
        pair_model_path = os.path.join(pair_dir, 'pair_attention_best.pt')
        pair_label_path = os.path.join(pair_dir, 'label_mapping.json')
        
        if os.path.exists(pair_model_path) and os.path.exists(pair_label_path):
            try:
                import torch
                import torch.nn as nn
                
                with open(pair_label_path, 'r') as f:
                    label_data = json.load(f)
                self._pair_labels = label_data.get('label_names', [])
                n_labels = len(self._pair_labels)
                
                # Define model architecture (must match training)
                class _PairAttentionNet(nn.Module):
                    def __init__(self, pom_dim=256, num_labels=109, n_heads=4, hidden=256, dropout=0.15):
                        super().__init__()
                        self.input_proj = nn.Linear(pom_dim, hidden)
                        self.attention = nn.MultiheadAttention(
                            embed_dim=hidden, num_heads=n_heads,
                            dropout=dropout, batch_first=True)
                        self.attn_norm = nn.LayerNorm(hidden)
                        self.ffn = nn.Sequential(
                            nn.Linear(hidden, hidden*2), nn.GELU(),
                            nn.Dropout(dropout), nn.Linear(hidden*2, hidden))
                        self.ffn_norm = nn.LayerNorm(hidden)
                        self.classifier = nn.Sequential(
                            nn.Linear(hidden, 256), nn.GELU(),
                            nn.Dropout(dropout), nn.Linear(256, num_labels))
                    def forward(self, x):
                        h = self.input_proj(x)
                        ao, _ = self.attention(h, h, h)
                        h = self.attn_norm(h + ao)
                        h = self.ffn_norm(h + self.ffn(h))
                        return self.classifier(h.sum(dim=1))
                
                ckpt = torch.load(pair_model_path, map_location='cpu', weights_only=False)
                model = _PairAttentionNet(pom_dim=256, num_labels=n_labels)
                model.load_state_dict(ckpt.get('model_state', ckpt))
                model.eval()
                
                # Move to GPU if available
                self._pair_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(self._pair_device)
                self._pair_attention = model
                self._use_attention = True
                
                auroc = ckpt.get('auroc', 'N/A')
                split = ckpt.get('split_method', 'unknown')
                print(f"  [ATT] PairAttentionNet loaded ({n_labels} labels, AUROC={auroc}, split={split})")
            except Exception as e:
                print(f"  [ATT] PairAttentionNet load failed: {e}")
        
        self._loaded = True
        return True
    
    # ========================================================
    # Step 1: Fragrance Database
    # ========================================================
    
    def _build_fragrance_db(self):
        """Build unified fragrance DB from ingredients.json + Abraham ODT + known SMILES"""
        base_dir = os.path.dirname(__file__)
        
        # Load ingredients.json
        ing_path = os.path.join(base_dir, '..', 'data', 'ingredients.json')
        if not os.path.exists(ing_path):
            ing_path = os.path.join(base_dir, 'data', 'ingredients.json')
        
        ingredients = []
        if os.path.exists(ing_path):
            with open(ing_path, 'r', encoding='utf-8') as f:
                ingredients = json.load(f)
        
        # Process each ingredient
        for ing in ingredients:
            name = ing.get('id', '').lower().strip()
            if not name:
                continue
            
            entry = {
                'name': name,
                'name_en': ing.get('name_en', name),
                'cas': ing.get('cas_number', ''),
                'category': ing.get('category', 'unknown'),
                'note_type': ing.get('note_type', 'middle'),
                'volatility': ing.get('volatility', 5.0),
                'intensity': ing.get('intensity', 5.0),
                'longevity': ing.get('longevity', 3.0),
                'typical_pct': ing.get('typical_pct', 5.0),
                'max_pct': ing.get('max_pct', 15.0),
                'smiles': _KNOWN_SMILES.get(name, ''),
                'odt_log': _DEFAULT_ODT_LOG.get(name, 1.5),  # default medium
                'ifra_restricted': name in _IFRA_RESTRICTED,
                'substitutes': ing.get('substitutes', []),
            }
            self._fragrance_db[name] = entry
        
        # Add known SMILES not in ingredients
        for name, smi in _KNOWN_SMILES.items():
            if name not in self._fragrance_db:
                self._fragrance_db[name] = {
                    'name': name, 'name_en': name.replace('_', ' ').title(),
                    'cas': '', 'category': 'synthetic', 'note_type': 'middle',
                    'volatility': 5, 'intensity': 5, 'longevity': 3,
                    'typical_pct': 5, 'max_pct': 15,
                    'smiles': smi,
                    'odt_log': _DEFAULT_ODT_LOG.get(name, 1.5),
                    'ifra_restricted': False, 'substitutes': [],
                }
        
        # Load Abraham ODT data
        abr_dir = os.path.join(base_dir, 'data', 'pom_data', 'pyrfume_all', 'abraham_2012')
        mol_path = os.path.join(abr_dir, 'molecules.csv')
        beh_path = os.path.join(abr_dir, 'behavior.csv')
        if os.path.exists(mol_path) and os.path.exists(beh_path):
            cid_to_smi = {}
            cid_to_name = {}
            with open(mol_path, 'r', encoding='utf-8', errors='replace') as f:
                for row in csv.DictReader(f):
                    cid = row.get('CID', '')
                    cid_to_smi[cid] = row.get('IsomericSMILES', '')
                    cid_to_name[cid] = row.get('name', '')
            
            abr_count = 0
            with open(beh_path, 'r', encoding='utf-8', errors='replace') as f:
                for row in csv.DictReader(f):
                    cid = row.get('Stimulus', '')
                    odt_str = row.get('Log (1/ODT)', '')
                    smi = cid_to_smi.get(cid, '')
                    name = cid_to_name.get(cid, '').lower().replace(' ', '_')
                    
                    if smi and odt_str:
                        try:
                            odt_log = float(odt_str)
                        except:
                            continue
                        
                        # Update existing or create new
                        if name and name in self._fragrance_db:
                            self._fragrance_db[name]['odt_log'] = odt_log
                            if not self._fragrance_db[name]['smiles']:
                                self._fragrance_db[name]['smiles'] = smi
                        elif smi:
                            # Find by SMILES match
                            found = False
                            for k, v in self._fragrance_db.items():
                                if v.get('smiles') == smi:
                                    v['odt_log'] = odt_log
                                    found = True
                                    break
                            if not found and name:
                                self._fragrance_db[name] = {
                                    'name': name, 'name_en': name.replace('_', ' ').title(),
                                    'cas': '', 'category': 'chemical', 'note_type': 'middle',
                                    'volatility': 5, 'intensity': 5, 'longevity': 3,
                                    'typical_pct': 5, 'max_pct': 15,
                                    'smiles': smi, 'odt_log': odt_log,
                                    'ifra_restricted': False, 'substitutes': [],
                                }
                                abr_count += 1
            
            if abr_count > 0:
                print(f"  [DB] Abraham ODT: {abr_count} molecules added")
        
        # Load XGBoost-predicted ODT (R2=0.71, 5050 molecules)
        odt_pred_path = os.path.join(base_dir, 'data', 'pom_upgrade', 'predicted_odt.json')
        if os.path.exists(odt_pred_path):
            with open(odt_pred_path, 'r') as f:
                odt_pred = json.load(f)
            odt_map = odt_pred.get('odt_map', {})
            upgraded = 0
            for name, entry in self._fragrance_db.items():
                smi = entry.get('smiles', '')
                if smi and smi in odt_map:
                    # Only update if still using default (1.5)
                    if abs(entry.get('odt_log', 1.5) - 1.5) < 0.01:
                        entry['odt_log'] = odt_map[smi]
                        entry['odt_source'] = 'xgboost'
                        upgraded += 1
            print(f"  [DB] XGBoost ODT: {upgraded} upgraded (R2={odt_pred.get('model_r2', 'N/A')})")
            self._odt_predicted = odt_map  # Store full predicted ODT for GA
        
        # Generate Virtual Accords for essential oils
        virtual_count = 0
        for oil_name, profile in _ESSENTIAL_OIL_PROFILES.items():
            if oil_name in self._fragrance_db and not self._fragrance_db[oil_name].get('smiles'):
                # Compute virtual 256d embedding from GC-MS components
                embs, oavs = [], []
                for smi, frac in profile.items():
                    emb = self.predict_embedding(smi)
                    if np.linalg.norm(emb) > 0:
                        odt = self._odt_predicted.get(smi, 1.5) if hasattr(self, '_odt_predicted') else 1.5
                        oav = max(0.1, math.log10(frac * 1e6) + odt)
                        embs.append(emb)
                        oavs.append(oav)
                
                if embs:
                    oavs = np.array(oavs)
                    weights = np.exp(oavs - oavs.max()) / np.sum(np.exp(oavs - oavs.max()))
                    virtual_emb = sum(w * e for w, e in zip(weights, embs))
                    
                    self._fragrance_db[oil_name]['virtual_embedding'] = virtual_emb
                    self._fragrance_db[oil_name]['virtual_accord'] = True
                    virtual_count += 1
        
        if virtual_count > 0:
            print(f"  [DB] Virtual Accords: {virtual_count} essential oils baked")
        
        with_smi = sum(1 for v in self._fragrance_db.values() if v.get('smiles'))
        with_virtual = sum(1 for v in self._fragrance_db.values() if v.get('virtual_accord'))
        print(f"  [DB] Fragrance DB: {len(self._fragrance_db)} entries ({with_smi} SMILES, {with_virtual} virtual)")
    
    def lookup(self, name: str) -> dict:
        """Look up ingredient by name"""
        key = name.lower().strip().replace(' ', '_').replace('-', '_')
        return self._fragrance_db.get(key, None)
    
    def resolve_smiles(self, name_or_smiles: str) -> str:
        """Resolve name/CAS/SMILES to canonical SMILES"""
        # Already SMILES?
        if any(c in name_or_smiles for c in ['=', '(', ')', '#', '/', '\\']):
            return name_or_smiles
        
        # Look up by name
        entry = self.lookup(name_or_smiles)
        if entry and entry.get('smiles'):
            return entry['smiles']
        
        return ''
    
    # ========================================================
    # Core: Predictions
    # ========================================================
    
    def _featurize(self, smiles):
        import deepchem as dc
        try:
            feat = self.featurizer.featurize([smiles])
            if feat is None or len(feat) == 0: return None
            f = feat[0]
            if f is None or not hasattr(f, 'node_features') or f.node_features.shape[0] == 0:
                return None
            return dc.data.NumpyDataset(
                X=np.array([f], dtype=object),
                y=np.zeros((1, self.n_tasks)),
                w=np.ones((1, self.n_tasks)),
                ids=np.array([smiles]))
        except:
            return None
    
    def predict_138d(self, smiles: str) -> np.ndarray:
        """SMILES -> 138d odor probability (10-model average)"""
        if not smiles: return np.zeros(self.n_tasks)
        cache_key = ('138d', smiles)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Pre-computed
        if self._embedding_db:
            idx = self._embedding_db['smiles_idx'].get(smiles)
            if idx is not None:
                result = self._embedding_db['labels'][idx]
                self._cache[cache_key] = result.copy()
                return result.copy()
        
        if not self.models:
            return np.zeros(self.n_tasks)
        
        ds = self._featurize(smiles)
        if ds is None: return np.zeros(self.n_tasks)
        
        preds = []
        for m in self.models:
            try: preds.append(m.predict(ds)[0])
            except: pass
        
        result = np.mean(preds, axis=0) if preds else np.zeros(self.n_tasks)
        if len(self._cache) < 20000: self._cache[cache_key] = result.copy()
        return result
    
    def predict_embedding(self, smiles: str) -> np.ndarray:
        """SMILES -> 256d POM embedding"""
        if not smiles: return np.zeros(256)
        cache_key = ('256d', smiles)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        if self._embedding_db:
            idx = self._embedding_db['smiles_idx'].get(smiles)
            if idx is not None:
                result = self._embedding_db['embeddings'][idx]
                self._cache[cache_key] = result.copy()
                return result.copy()
        
        if not self.models: return np.zeros(256)
        ds = self._featurize(smiles)
        if ds is None: return np.zeros(256)
        
        embs = []
        for m in self.models:
            try:
                e = m.predict_embedding(ds)
                if isinstance(e, list): e = e[0]
                embs.append(e[0])
            except: pass
        
        result = np.mean(embs, axis=0) if embs else np.zeros(256)
        if len(self._cache) < 20000: self._cache[cache_key] = result.copy()
        return result
    
    def predict_22d(self, smiles: str) -> np.ndarray:
        """SMILES -> 22d (pipeline compat)"""
        p = self.predict_138d(smiles)
        v = np.zeros(22)
        for di, si in _MAP_22_IDX.items():
            if si: v[di] = np.mean([p[i] for i in si if i < len(p)])
        return v
    
    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0
    
    @staticmethod
    def perceptual_distance(pred: np.ndarray, target: np.ndarray) -> float:
        """Asymmetric perceptual distance: off-notes get 10x penalty.
        
        Unlike cosine similarity which treats all dimensions equally,
        human perception is asymmetrically sensitive to unpleasant notes.
        A recipe with 137/138 notes matching but 'garlic' creeping in
        is WORSE than one with 130/138 matching and clean profile.
        """
        # Off-note indices (absolute deal-breakers in fine fragrance)
        OFF_NOTE_NAMES = {'animal', 'burnt', 'cabbage', 'fishy', 'garlic',
                          'metallic', 'sulfurous', 'sweaty', 'onion', 'meaty'}
        off_indices = [i for i, t in enumerate(TASKS_138) if t in OFF_NOTE_NAMES]
        
        # Base: cosine similarity
        na, nb = np.linalg.norm(pred), np.linalg.norm(target)
        sim = float(np.dot(pred, target) / (na * nb)) if na > 0 and nb > 0 else 0.0
        
        # Off-note penalty: if recipe introduces stink that target doesn't have
        penalty = 0.0
        for idx in off_indices:
            # pred has off-note but target doesn't (or much less)
            excess = max(0, pred[idx] - max(target[idx], 0.02))
            if excess > 0.03:  # above human detection threshold
                penalty += excess * 10.0  # 1000% amplification
        
        return max(0.0, sim - penalty)
    
    @staticmethod
    def hill_saturation(concentration_pct, odt_log, hill_n=1.5, kd=5000.0):
        """Biological olfactory receptor binding (Hill equation).
        
        At high concentrations receptors saturate (anosmia).
        At low concentrations below ODT, no perception.
        Returns fractional receptor occupancy [0, 1].
        
        Kd=5000 is calibrated for perfumery OAV range:
        - Vanillin 0.1% (ODT_log=3.5): OAV~3.16e8 -> sat~1.0 (correct, very strong)
        - Linalool 10% (ODT_log=0.0): OAV~1e5 -> sat~0.99 (correct, strong)
        - Weak molecule 0.001% (ODT_log=1.5): OAV~316 -> sat~0.35 (below threshold)
        """
        if concentration_pct <= 0:
            return 0.0
        # OAV = concentration / threshold
        odt_ppm = 10 ** (-odt_log)  # threshold in ppm
        conc_ppm = concentration_pct * 1e4  # % -> ppm
        oav = conc_ppm / max(odt_ppm, 1e-10)
        
        if oav <= 0:
            return 0.0
        # Hill equation: response = OAV^n / (OAV^n + Kd)
        oav_n = oav ** hill_n
        return oav_n / (oav_n + kd)
    
    # ========================================================
    # Step 2: Non-linear Mixture Prediction
    # ========================================================
    
    def compute_oav(self, name_or_smiles: str, concentration_pct: float) -> float:
        """Compute Odor Activity Value = concentration / ODT
        
        Returns log10(OAV) following Weber-Fechner law.
        Higher = more perceptually intense at this concentration.
        """
        entry = self.lookup(name_or_smiles)
        odt_log = entry['odt_log'] if entry else 1.5
        
        # ODT = 10^(-odt_log) in ppm
        # OAV = concentration / ODT
        # log(OAV) = log(conc) + odt_log
        if concentration_pct <= 0:
            return 0.0
        
        log_conc = math.log10(concentration_pct * 10000)  # pct -> ppm
        log_oav = log_conc + odt_log
        return max(0.0, log_oav)
    
    def predict_mixture_attention(self, smiles_list, pct_list=None):
        """Predict mixture using PairAttentionNet with receptor saturation defense.
        
        Defenses against N>2 Attention dilution:
        1. Hill equation receptor saturation per molecule
        2. OAV masking: molecules below 1% of max power are excluded
        3. Power-scaled embeddings (not uniform)
        """
        if not self._use_attention or self._pair_attention is None:
            return None
        if len(smiles_list) < 2:
            return None
        
        try:
            import torch
            
            # Step 1: Compute receptor saturation for each molecule
            powers = []
            raw_embs = []
            for i, smi in enumerate(smiles_list):
                emb = self.predict_embedding(smi)
                if np.linalg.norm(emb) == 0:
                    return None
                raw_embs.append(emb)
                
                if pct_list and i < len(pct_list):
                    pct = pct_list[i]
                    entry = self.lookup(smi) or {}
                    odt = entry.get('odt_log', 1.5)
                    power = self.hill_saturation(pct, odt)
                else:
                    power = 0.5  # default mid-range
                powers.append(power)
            
            max_power = max(powers) if powers else 1.0
            if max_power <= 0:
                return None
            
            # Step 2: OAV masking - exclude molecules below 1% of max power
            # This prevents Softmax dilution from inert/weak molecules
            valid_embs = []
            for emb, power in zip(raw_embs, powers):
                if power >= max_power * 0.01:  # above 1% threshold
                    # Scale embedding by receptor occupancy
                    scaled = emb * (power / max_power)
                    valid_embs.append(scaled)
            
            if len(valid_embs) < 2:
                # Not enough active molecules for attention
                return None
            
            # Step 3: Run attention on valid (masked) molecules only
            x = torch.tensor(np.array([valid_embs]), dtype=torch.float32).to(self._pair_device)
            
            with torch.no_grad():
                logits = self._pair_attention(x)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
            # Order ensemble for 2-molecule case
            if len(valid_embs) == 2:
                x_rev = torch.tensor(np.array([[valid_embs[1], valid_embs[0]]]),
                                     dtype=torch.float32).to(self._pair_device)
                with torch.no_grad():
                    probs_rev = torch.sigmoid(self._pair_attention(x_rev)).squeeze(0).cpu().numpy()
                probs = (probs + probs_rev) / 2
            
            return {
                'labels': self._pair_labels,
                'probs': probs,
                'method': 'pair_attention',
                'n_molecules': len(smiles_list),
                'n_active': len(valid_embs),
            }
        except Exception as e:
            return None
    
    def predict_mixture(self, ingredients: list) -> dict:
        """Non-linear mixture prediction with OAV + masking
        
        Uses PairAttentionNet when available (with physical model fallback).
        
        Args:
            ingredients: [{'name': str, 'pct': float}, ...]
                OR [{'smiles': str, 'pct': float}, ...]
                pct = concentration percentage in final formula
        
        Returns:
            {
                'embedding_256d': np.ndarray,
                'prediction_138d': np.ndarray,
                'prediction_22d': np.ndarray,
                'top_notes': list,
                'individual': list,
                'total_pct': float,
                'dominant_ingredient': str,
                'attention_blend_notes': dict (if attention model available)
            }
        """
        if not ingredients:
            return None
        
        # Resolve all ingredients
        resolved = []
        for ing in ingredients:
            name = ing.get('name', '')
            smiles = ing.get('smiles', '')
            pct = ing.get('pct', ing.get('ratio', 5.0))
            
            if not smiles:
                smiles = self.resolve_smiles(name)
            if not smiles and not name:
                continue
            
            entry = self.lookup(name) or {}
            oav = self.compute_oav(name, pct)
            
            emb = self.predict_embedding(smiles) if smiles else np.zeros(256)
            pred = self.predict_138d(smiles) if smiles else np.zeros(self.n_tasks)
            
            resolved.append({
                'name': name or smiles[:20],
                'smiles': smiles,
                'pct': pct,
                'oav': oav,
                'embedding': emb,
                'prediction': pred,
                'note_type': entry.get('note_type', 'middle'),
                'intensity': entry.get('intensity', 5.0),
            })
        
        if not resolved:
            return None
        
        # === Non-linear mixing with receptor saturation ===
        
        # 1. Compute receptor occupancy via Hill equation (replaces naive softmax)
        saturations = []
        for r in resolved:
            sat = self.hill_saturation(r['pct'],
                                       self.lookup(r['name']).get('odt_log', 1.5)
                                       if self.lookup(r['name']) else 1.5)
            saturations.append(sat)
        
        saturations = np.array(saturations)
        sat_sum = saturations.sum()
        
        if sat_sum > 0:
            # Perceptual weight proportional to receptor occupancy
            # NOT softmax -- this preserves biological saturation curve
            perceptual_weights = saturations / sat_sum
        else:
            # Fallback: concentration ratio
            pcts = np.array([r['pct'] for r in resolved])
            perceptual_weights = pcts / pcts.sum() if pcts.sum() > 0 else np.ones(len(resolved)) / len(resolved)
        
        # 2. Mix embeddings with perceptual weights
        mix_emb = np.zeros(256)
        mix_pred = np.zeros(self.n_tasks)
        
        for i, r in enumerate(resolved):
            w = perceptual_weights[i]
            mix_emb += r['embedding'] * w
            mix_pred += r['prediction'] * w
        
        # 3. Apply competitive masking (olfactory suppression)
        dominant_idx = np.argmax(perceptual_weights)
        dominant_pred = resolved[dominant_idx]['prediction']
        dominant_power = saturations[dominant_idx] if len(saturations) > 0 else 0
        
        for j in range(self.n_tasks):
            # Masking strength is proportional to dominant molecule's confidence
            # If dominant strongly predicts NOT this note, suppress weak signals
            if dominant_pred[j] < 0.05 and mix_pred[j] > 0.15:
                # Suppression ratio based on how dominant the leading molecule is
                suppress = 0.3 + 0.5 * (dominant_power / max(sat_sum, 1e-6))
                mix_pred[j] *= (1.0 - suppress)
        
        # 4. Convert to 22d
        mix_22d = np.zeros(22)
        for di, si in _MAP_22_IDX.items():
            if si: mix_22d[di] = np.mean([mix_pred[i] for i in si if i < len(mix_pred)])
        
        # 5. Top notes
        top_idx = np.argsort(mix_pred)[-10:][::-1]
        top_notes = [(TASKS_138[i], float(mix_pred[i])) for i in top_idx if mix_pred[i] > 0.05]
        
        # Individual details
        individual = []
        for i, r in enumerate(resolved):
            top_d = np.argsort(r['prediction'])[-3:][::-1]
            individual.append({
                'name': r['name'],
                'pct': r['pct'],
                'oav': round(r['oav'], 2),
                'perceptual_weight': round(float(perceptual_weights[i]), 4),
                'top_descriptors': [(TASKS_138[j], round(float(r['prediction'][j]), 2))
                                   for j in top_d if r['prediction'][j] > 0.1],
                'note_type': r['note_type'],
            })
        
        # Try Attention model for blend prediction
        attention_result = None
        smiles_for_attn = [r['smiles'] for r in resolved if r['smiles']]
        pcts_for_attn = [r['pct'] for r in resolved if r['smiles']]
        if len(smiles_for_attn) >= 2:
            attention_result = self.predict_mixture_attention(smiles_for_attn, pcts_for_attn)
        
        result = {
            'embedding_256d': mix_emb,
            'prediction_138d': mix_pred,
            'prediction_22d': mix_22d,
            'top_notes': top_notes,
            'individual': individual,
            'total_pct': sum(r['pct'] for r in resolved),
            'dominant_ingredient': resolved[dominant_idx]['name'],
            'method': 'attention+physical' if attention_result else 'physical_oav',
        }
        
        if attention_result:
            # Add top blend notes from Attention model
            attn_probs = attention_result['probs']
            attn_labels = attention_result['labels']
            top_attn_idx = np.argsort(attn_probs)[-10:][::-1]
            result['attention_blend_notes'] = [
                (attn_labels[i], round(float(attn_probs[i]), 3))
                for i in top_attn_idx if attn_probs[i] > 0.05
            ]
        
        return result
    
    # ========================================================
    # Step 3: GA Reverse Engineering
    # ========================================================
    
    def reverse_engineer(self, target, candidates=None, n_components=5,
                        population_size=200, generations=500,
                        exclude_names=None, max_cost_per_kg=None,
                        require_notes=None, enforce_ifra=True,
                        enforce_balance=True, target_budget_usd=None) -> dict:
        """Multi-objective GA: scent similarity + IFRA + cost + volatility
        
        Args:
            target: SMILES string, ingredient name, or 256d vector
            candidates: list of ingredient names (default: all with SMILES)
                       Use for '100-Bottle Lab Lock' mode
            n_components: number of ingredients (3-8)
            population_size: GA population size
            generations: number of GA iterations
            exclude_names: ingredient names to exclude
            max_cost_per_kg: if set, reject recipes above this $/kg
            require_notes: {'top': 1, 'middle': 2, 'base': 1}
            enforce_ifra: True = IFRA death penalty (instant kill)
            enforce_balance: True = Top/Mid/Base golden ratio
            target_budget_usd: target cost per kg (penalty above)
        
        Returns:
            {
                'recipe': [{'name': str, 'pct': float, 'oav': float,
                           'cost_kg': float, 'volatility': str,
                           'ifra_status': str}, ...],
                'similarity': float,
                'cost_per_kg': float,
                'ifra_compliant': bool,
                'balance': {'top': float, 'middle': float, 'base': float},
                'generations_used': int,
            }
        """
        # Resolve target
        if isinstance(target, np.ndarray):
            target_emb = target
        elif isinstance(target, str):
            smi = self.resolve_smiles(target)
            if smi:
                target_emb = self.predict_embedding(smi)
            else:
                return None
        else:
            return None
        
        if np.linalg.norm(target_emb) == 0:
            return None
        
        # Compute target 138d prediction for perceptual distance comparison
        target_pred_138d = None
        if isinstance(target, str):
            target_smi = self.resolve_smiles(target)
            if target_smi:
                target_pred_138d = self.predict_138d(target_smi)
                if np.linalg.norm(target_pred_138d) == 0:
                    target_pred_138d = None
        
        # Build candidate pool
        exclude_set = set(n.lower().replace(' ','_') for n in (exclude_names or []))
        if isinstance(target, str):
            exclude_set.add(target.lower().replace(' ', '_'))
        
        pool = []
        
        # Load IFRA + cost DBs (prefer official 51st Amendment, then full, then legacy)
        ifra_db = {}
        cost_db = {}
        data_dir = os.path.join(os.path.dirname(__file__), 'data', 'pom_upgrade')
        ifra_official_path = os.path.join(data_dir, 'ifra_51st_official.json')
        ifra_full_path = os.path.join(data_dir, 'ifra_51st_full.json')
        ifra_legacy_path = os.path.join(data_dir, 'ifra_51st_cat4.json')
        cost_path = os.path.join(data_dir, 'industry_costs.json')
        if os.path.exists(ifra_official_path):
            with open(ifra_official_path, 'r', encoding='utf-8') as f:
                ifra_db = json.load(f)
        elif os.path.exists(ifra_full_path):
            with open(ifra_full_path, 'r', encoding='utf-8') as f:
                ifra_db = json.load(f)
        elif os.path.exists(ifra_legacy_path):
            with open(ifra_legacy_path, 'r') as f:
                ifra_db = json.load(f)
        if os.path.exists(cost_path):
            with open(cost_path, 'r') as f:
                cost_db = json.load(f)
        default_cost = cost_db.get('_default', 30)
        
        def _get_molecular_descriptors(smiles):
            """Compute MolWt, LogP, TPSA, HBD for QSPR volatility"""
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                mol = Chem.MolFromSmiles(smiles, sanitize=True)
                if mol:
                    return {
                        'molwt': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'tpsa': Descriptors.TPSA(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                    }
            except:
                pass
            return {'molwt': 180.0, 'logp': 3.0, 'tpsa': 30.0, 'hbd': 0, 'hba': 1}
        
        def _classify_volatility(desc):
            """QSPR retention index: MolWt + HBD*25 + TPSA*0.5
            
            Thresholds calibrated to perfumery industry:
            - Top: <160 (citrus, light aldehydes, small alcohols)
            - Middle: 160-230 (florals, spices, most aroma chemicals)
            - Base: >230 (woods, musks, resins, vanillin-class)
            
            Example: Vanillin MW=152, HBD=1, TPSA=46.5
            -> Retention = 152 + 25 + 23.3 = 200.3 -> Middle
            BUT vanillin has phenolic OH with strong H-bonding
            -> HBA=4 adds extra retention: 200.3 + 4*5 = 220.3
            With HBA bonus: Retention > 220 -> Base (correct!)
            """
            mw = desc['molwt']
            tpsa = desc['tpsa']
            hbd = desc['hbd']
            hba = desc.get('hba', 0)
            
            # Core retention + H-bond acceptor bonus (weaker than donor)
            retention = mw + (hbd * 25.0) + (tpsa * 0.5) + (hba * 5.0)
            
            if retention < 160:
                return 'top', retention
            elif retention < 215:
                return 'middle', retention
            else:
                return 'base', retention
        
        def _build_pool_entry(name, entry):
            smi = entry.get('smiles', '')
            if not smi:
                return None
            emb = self.predict_embedding(smi)
            if np.linalg.norm(emb) == 0:
                return None
            
            cas = entry.get('cas', '')
            desc = _get_molecular_descriptors(smi)
            vol_class, retention = _classify_volatility(desc)
            
            # IFRA check
            ifra_info = ifra_db.get(cas, {})
            ifra_max = ifra_info.get('max_pct_cat4', 99.0) if ifra_info else 99.0
            ifra_prohibited = ifra_info.get('prohibited', False) if ifra_info else False
            
            # Cost
            cost_kg = cost_db.get(name, default_cost)
            
            return {
                'name': name, 'smiles': smi, 'embedding': emb,
                'odt_log': entry.get('odt_log', 1.5),
                'note_type': entry.get('note_type', vol_class),
                'cas': cas, 'molwt': desc['molwt'], 'logp': desc['logp'],
                'tpsa': desc['tpsa'], 'hbd': desc['hbd'],
                'retention': round(retention, 1),
                'volatility': vol_class, 'cost_kg': cost_kg,
                'ifra_max_pct': ifra_max, 'ifra_prohibited': ifra_prohibited,
            }
        
        if candidates:
            for name in candidates:
                key = name.lower().replace(' ', '_')
                if key in exclude_set: continue
                entry = self.lookup(key)
                if entry:
                    p = _build_pool_entry(key, entry)
                    if p: pool.append(p)
        else:
            for name, entry in self._fragrance_db.items():
                if name in exclude_set: continue
                cas = entry.get('cas', '')
                # Skip IFRA prohibited if enforcement on
                if enforce_ifra and cas in ifra_db:
                    if ifra_db[cas].get('prohibited', False):
                        continue
                p = _build_pool_entry(name, entry)
                if p: pool.append(p)
        
        if len(pool) < n_components:
            print(f"  [GA] Not enough candidates: {len(pool)} < {n_components}")
            return None
        
        print(f"  [GA] Pool: {len(pool)} candidates, target: {n_components} components")
        print(f"  [GA] Population: {population_size}, Generations: {generations}")
        
        rng = random.Random(42)
        
        # === GA Functions ===
        
        def random_recipe():
            """Create random recipe"""
            indices = rng.sample(range(len(pool)), n_components)
            # Dirichlet-like random ratios
            raw = [rng.random() for _ in range(n_components)]
            total = sum(raw)
            pcts = [r / total * 100 for r in raw]  # percentages summing to 100
            return list(zip(indices, pcts))
        
        def fitness(recipe):
            """Multi-objective: perceptual distance + IFRA + cost + balance"""
            # Hill equation receptor saturation (replaces naive softmax)
            saturations = []
            for idx, pct in recipe:
                p = pool[idx]
                sat = self.hill_saturation(pct, p['odt_log'])
                saturations.append(sat)
            
            saturations = np.array(saturations)
            sat_sum = saturations.sum()
            
            if sat_sum > 0:
                weights = saturations / sat_sum
            else:
                weights = np.ones(len(recipe)) / len(recipe)
            
            # Mix embeddings weighted by receptor occupancy
            mix_emb = np.zeros(256)
            mix_pred = np.zeros(self.n_tasks)
            for i, (idx, pct) in enumerate(recipe):
                mix_emb += pool[idx]['embedding'] * weights[i]
            
            # Compute 138d prediction for off-note checking
            for i, (idx, pct) in enumerate(recipe):
                smi = pool[idx]['smiles']
                pred_i = self.predict_138d(smi)
                mix_pred += pred_i * weights[i]
            
            # Perceptual distance (with off-note penalty)
            sim = self.perceptual_distance(mix_pred, target_pred_138d) if target_pred_138d is not None \
                  else self.cosine_sim(mix_emb, target_emb)
            
            # === IFRA DEATH PENALTY ===
            if enforce_ifra:
                for idx, pct in recipe:
                    p = pool[idx]
                    if p['ifra_prohibited']:
                        return -1.0  # Instant death
                    if pct > p['ifra_max_pct']:
                        return -1.0  # Exceeds IFRA limit -> death
            
            # === COST PENALTY ===
            cost_penalty = 0
            if target_budget_usd or max_cost_per_kg:
                recipe_cost = sum(pool[idx]['cost_kg'] * (pct/100) for idx, pct in recipe)
                budget = target_budget_usd or max_cost_per_kg
                if recipe_cost > budget:
                    overshoot = (recipe_cost - budget) / budget
                    cost_penalty = min(0.3, overshoot * 0.2)  # max 30% penalty
            
            # === VOLATILITY BALANCE PENALTY ===
            balance_penalty = 0
            if enforce_balance:
                vol_pcts = {'top': 0, 'middle': 0, 'base': 0}
                for idx, pct in recipe:
                    vol_pcts[pool[idx]['volatility']] += pct
                total = sum(vol_pcts.values())
                if total > 0:
                    for k in vol_pcts:
                        vol_pcts[k] = vol_pcts[k] / total * 100
                
                # Golden ratio: T=20%, M=50%, B=30%
                ideal = {'top': 20, 'middle': 50, 'base': 30}
                for k in ideal:
                    dev = abs(vol_pcts[k] - ideal[k]) / 100
                    balance_penalty += dev * 0.05  # gentle nudge
            
            # === NOTE DIVERSITY BONUS ===
            note_types = set(pool[idx]['volatility'] for idx, _ in recipe)
            diversity_bonus = len(note_types) * 0.005
            
            return sim + diversity_bonus - cost_penalty - balance_penalty
        
        def crossover(parent1, parent2):
            """Crossover two recipes"""
            split = rng.randint(1, n_components - 1)
            child_indices = [idx for idx, _ in parent1[:split]] + [idx for idx, _ in parent2[split:]]
            child_pcts = [pct for _, pct in parent1[:split]] + [pct for _, pct in parent2[split:]]
            
            # Fix duplicates
            used = set()
            for i in range(len(child_indices)):
                if child_indices[i] in used:
                    available = [j for j in range(len(pool)) if j not in used]
                    if available:
                        child_indices[i] = rng.choice(available)
                used.add(child_indices[i])
            
            # Renormalize
            total = sum(child_pcts)
            child_pcts = [p / total * 100 for p in child_pcts]
            
            return list(zip(child_indices, child_pcts))
        
        def mutate(recipe, mutation_rate=0.3):
            """Mutate recipe"""
            recipe = list(recipe)
            for i in range(len(recipe)):
                if rng.random() < mutation_rate:
                    idx, pct = recipe[i]
                    if rng.random() < 0.5:
                        # Swap ingredient
                        used = {idx2 for idx2, _ in recipe}
                        available = [j for j in range(len(pool)) if j not in used]
                        if available:
                            recipe[i] = (rng.choice(available), pct)
                    else:
                        # Adjust ratio
                        new_pct = max(0.5, pct + rng.gauss(0, 5))
                        recipe[i] = (idx, new_pct)
            
            # Renormalize
            total = sum(p for _, p in recipe)
            return [(idx, p / total * 100) for idx, p in recipe]
        
        # === Evolution ===
        population = [random_recipe() for _ in range(population_size)]
        best_fitness = -1
        best_recipe = None
        stagnation = 0
        
        for gen in range(generations):
            # Evaluate
            scored = [(fitness(r), r) for r in population]
            scored.sort(key=lambda x: -x[0])
            
            if scored[0][0] > best_fitness:
                best_fitness = scored[0][0]
                best_recipe = scored[0][1]
                stagnation = 0
                if (gen + 1) % 50 == 0 or gen == 0:
                    print(f"    Gen {gen+1:4d}: best={best_fitness:.4f}")
            else:
                stagnation += 1
            
            if stagnation >= 80:
                print(f"    Converged at gen {gen+1}")
                break
            
            if best_fitness > 0.995:
                print(f"    Near-perfect match at gen {gen+1}: {best_fitness:.4f}")
                break
            
            # Selection (tournament)
            elite = [r for _, r in scored[:population_size // 5]]
            new_pop = list(elite)
            
            while len(new_pop) < population_size:
                # Tournament selection
                t1 = scored[rng.randint(0, len(scored)//2)]
                t2 = scored[rng.randint(0, len(scored)//2)]
                p1 = t1[1] if t1[0] > t2[0] else t2[1]
                
                t3 = scored[rng.randint(0, len(scored)//2)]
                t4 = scored[rng.randint(0, len(scored)//2)]
                p2 = t3[1] if t3[0] > t4[0] else t4[1]
                
                child = crossover(p1, p2)
                child = mutate(child)
                new_pop.append(child)
            
            population = new_pop
        
        # Build final result
        if best_recipe is None:
            return None
        
        recipe_items = []
        total_cost = 0
        ifra_compliant = True
        vol_pcts = {'top': 0, 'middle': 0, 'base': 0}
        
        for idx, pct in sorted(best_recipe, key=lambda x: -x[1]):
            p = pool[idx]
            cost_contrib = p['cost_kg'] * (pct / 100)
            total_cost += cost_contrib
            vol_pcts[p['volatility']] += pct
            
            # IFRA status
            if p['ifra_prohibited']:
                ifra_status = 'PROHIBITED'
                ifra_compliant = False
            elif pct > p['ifra_max_pct']:
                ifra_status = f"VIOLATION ({pct:.1f}% > {p['ifra_max_pct']}%)"
                ifra_compliant = False
            elif p['ifra_max_pct'] < 99:
                ifra_status = f"OK ({pct:.1f}/{p['ifra_max_pct']}%)"
            else:
                ifra_status = 'unrestricted'
            
            recipe_items.append({
                'name': p['name'],
                'pct': round(pct, 2),
                'oav': round(self.compute_oav(p['name'], pct), 2),
                'note_type': p['note_type'],
                'volatility': p['volatility'],
                'cost_kg': p['cost_kg'],
                'molwt': round(p['molwt'], 1),
                'logp': round(p['logp'], 2),
                'ifra_status': ifra_status,
            })
        
        # Normalize volatility balance
        total_vol = sum(vol_pcts.values())
        if total_vol > 0:
            vol_pcts = {k: round(v / total_vol * 100, 1) for k, v in vol_pcts.items()}
        
        # Predict mixture
        mix_input = [{'name': r['name'], 'pct': r['pct']} for r in recipe_items]
        mixture = self.predict_mixture(mix_input)
        
        # Target top notes
        target_pred = self.predict_138d(self.resolve_smiles(target)) if isinstance(target, str) else None
        target_notes = []
        if target_pred is not None:
            t_idx = np.argsort(target_pred)[-5:][::-1]
            target_notes = [(TASKS_138[i], round(float(target_pred[i]), 2)) for i in t_idx if target_pred[i] > 0.1]
        
        return {
            'recipe': recipe_items,
            'similarity': round(best_fitness, 4),
            'cost_per_kg': round(total_cost, 2),
            'ifra_compliant': ifra_compliant,
            'balance': vol_pcts,
            'target_top_notes': target_notes,
            'recipe_top_notes': mixture['top_notes'][:5] if mixture else [],
            'dominant': mixture['dominant_ingredient'] if mixture else '',
            'method': mixture.get('method', 'physical_oav') if mixture else 'physical_oav',
            'generations_used': gen + 1,
        }
    
    # ========================================================
    # Step 4: Replacement Search 
    # ========================================================
    
    def find_replacements(self, target, top_k=10, exclude=None):
        """Find replacement ingredients by cosine similarity"""
        smi = self.resolve_smiles(target) if isinstance(target, str) else ''
        if not smi and isinstance(target, str) and self._embedding_db:
            return []
        
        target_emb = self.predict_embedding(smi) if smi else target
        if np.linalg.norm(target_emb) == 0: return []
        
        exclude_set = set(exclude or [])
        if isinstance(target, str):
            exclude_set.add(target.lower().replace(' ', '_'))
        
        results = []
        for name, entry in self._fragrance_db.items():
            if name in exclude_set: continue
            esmi = entry.get('smiles', '')
            if not esmi: continue
            
            emb = self.predict_embedding(esmi)
            sim = self.cosine_sim(emb, target_emb)
            if sim > 0:
                results.append({
                    'name': name, 'name_en': entry.get('name_en', name),
                    'similarity': round(sim, 4),
                    'category': entry.get('category', ''),
                    'note_type': entry.get('note_type', ''),
                })
        
        results.sort(key=lambda x: -x['similarity'])
        return results[:top_k]
    
    # ========================================================
    # Step 5: 4D Temporal Evaporation Engine
    # ========================================================
    
    def simulate_temporal(self, ingredients: list,
                          time_points=None,
                          evap_scale=15.0) -> dict:
        """Simulate scent evolution over time (4D perfume dynamics).
        
        Uses QSPR Retention Index to compute per-molecule evaporation
        rate, then re-predicts the 138d scent profile at each timepoint.
        
        Args:
            ingredients: [{'name': str, 'pct': float}, ...]
            time_points: hours to simulate (default: [0, 0.25, 1, 4, 8, 24])
            evap_scale: evaporation rate constant (higher = faster evaporation)
        
        Returns:
            {
                'timeline': {
                    'T+0.0h': {
                        'ingredients': [{'name': str, 'pct': float, 'retention': float}, ...],
                        'top_notes': [(note, prob), ...],
                        'dominant': str,
                        'balance': {'top': %, 'middle': %, 'base': %},
                    },
                    'T+4.0h': { ... },
                },
                'phase_shift': {
                    'opening': [notes at T=0],
                    'heart': [notes at T=1h],
                    'drydown': [notes at T=4h],
                    'sillage': [notes at T=8h],
                },
                'longevity_hours': float,  # when last molecule falls below threshold
            }
        """
        if time_points is None:
            time_points = [0, 0.25, 1, 4, 8, 24]
        
        # Step 1: Compute retention index for each ingredient
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        resolved = []
        for ing in ingredients:
            name = ing.get('name', '')
            pct = ing.get('pct', 0)
            entry = self.lookup(name) or {}
            smiles = entry.get('smiles', '') or self.resolve_smiles(name)
            
            # Compute QSPR retention
            retention = 200.0  # default middle
            try:
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol:
                    mw = Descriptors.MolWt(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    retention = mw + hbd * 25.0 + tpsa * 0.5 + hba * 5.0
            except:
                pass
            
            if retention < 160:
                vol_class = 'top'
            elif retention < 215:
                vol_class = 'middle'
            else:
                vol_class = 'base'
            
            resolved.append({
                'name': name, 'smiles': smiles,
                'initial_pct': pct, 'retention': retention,
                'volatility': vol_class,
                'odt_log': entry.get('odt_log', 1.5),
            })
        
        if not resolved:
            return None
        
        # Step 2: Simulate each timepoint
        timeline = {}
        longevity = 0
        
        for t in time_points:
            # Exponential decay: C(t) = C_0 * exp(-k * t)
            # k = evap_scale / retention (low retention = fast evaporation)
            decayed = []
            for r in resolved:
                k = evap_scale / max(r['retention'], 1.0)
                current_pct = r['initial_pct'] * math.exp(-k * t)
                
                # Below perceptual threshold = evaporated
                if current_pct > 0.001:  # 0.001% threshold
                    decayed.append({
                        'name': r['name'],
                        'pct': round(current_pct, 4),
                        'retention': round(r['retention'], 1),
                        'volatility': r['volatility'],
                        'remaining_pct': round(current_pct / r['initial_pct'] * 100, 1),
                    })
                    longevity = max(longevity, t)
            
            if not decayed:
                timeline[f'T+{t}h'] = {
                    'ingredients': [],
                    'top_notes': [],
                    'dominant': 'evaporated',
                    'balance': {'top': 0, 'middle': 0, 'base': 0},
                    'total_remaining_pct': 0,
                }
                continue
            
            # Renormalize concentrations for weighting
            total = sum(d['pct'] for d in decayed)
            
            # Direct 138d prediction weighted by decayed concentration
            # (NOT Hill-saturated, because temporal tracking needs raw concentration sensitivity)
            mix_pred = np.zeros(self.n_tasks)
            dominant_name = ''
            dominant_pct = 0
            
            for d in decayed:
                w = d['pct'] / total if total > 0 else 1.0 / len(decayed)
                entry = self.lookup(d['name']) or {}
                smiles = entry.get('smiles', '') or self.resolve_smiles(d['name'])
                if smiles:
                    pred_i = self.predict_138d(smiles)
                    mix_pred += pred_i * w
                if d['pct'] > dominant_pct:
                    dominant_pct = d['pct']
                    dominant_name = d['name']
            
            # Extract top notes from this timepoint's prediction
            top_idx = np.argsort(mix_pred)[-5:][::-1]
            top_notes = [(TASKS_138[i], float(mix_pred[i])) for i in top_idx if mix_pred[i] > 0.05]
            
            # Balance at this timepoint
            vol_pcts = {'top': 0, 'middle': 0, 'base': 0}
            for d in decayed:
                vol_pcts[d['volatility']] += d['pct']
            total_vol = sum(vol_pcts.values())
            if total_vol > 0:
                vol_pcts = {k: round(v / total_vol * 100, 1) for k, v in vol_pcts.items()}
            
            timeline[f'T+{t}h'] = {
                'ingredients': decayed,
                'top_notes': top_notes,
                'dominant': dominant_name,
                'balance': vol_pcts,
                'total_remaining_pct': round(total / sum(r['initial_pct'] for r in resolved) * 100, 1),
            }
        
        # Step 3: Extract phase shifts
        phase_names = {0: 'opening', 0.25: 'opening_dry', 1: 'heart',
                       4: 'drydown', 8: 'sillage', 24: 'base_residue'}
        
        phase_shift = {}
        for t in time_points:
            key = f'T+{t}h'
            phase = phase_names.get(t, f't{t}h')
            if key in timeline and timeline[key]['top_notes']:
                phase_shift[phase] = [
                    (n, round(v, 2)) for n, v in timeline[key]['top_notes']
                ]
        
        # Estimate longevity: extrapolate to when strongest molecule hits threshold
        max_retention = max(r['retention'] for r in resolved)
        max_initial = max(r['initial_pct'] for r in resolved)
        if max_retention > 0 and max_initial > 0:
            k_slowest = evap_scale / max_retention
            # Solve: max_initial * exp(-k*t) = 0.001
            if k_slowest > 0:
                longevity_est = math.log(max_initial / 0.001) / k_slowest
            else:
                longevity_est = 999
        else:
            longevity_est = 0
        
        return {
            'timeline': timeline,
            'phase_shift': phase_shift,
            'longevity_hours': round(longevity_est, 1),
            'n_ingredients': len(resolved),
        }


# ============================================================
# Verification Tests
# ============================================================
if __name__ == '__main__':
    t0 = time.time()
    engine = POMEngine()
    engine.load()
    
    print("\n" + "=" * 60)
    print("  POM Engine v3 -- Complete Verification")
    print("=" * 60)
    
    # === Test 1: Fragrance DB Lookup ===
    print("\n--- 1. Fragrance DB Lookup ---")
    for name in ['linalool', 'vanillin', 'hedione', 'iso_e_super', 'rose']:
        entry = engine.lookup(name)
        if entry:
            smi = entry.get('smiles', 'N/A')[:30]
            print(f"  {name:20s}: CAS={entry.get('cas','N/A'):12s} ODT_log={entry.get('odt_log',0):.1f} SMILES={smi}")
        else:
            print(f"  {name:20s}: NOT FOUND")
    
    # === Test 2: OAV Calculation ===
    print("\n--- 2. OAV (Weber-Fechner) ---")
    for name, pct in [('vanillin', 0.1), ('vanillin', 1.0), ('linalool', 5.0),
                       ('hedione', 10.0), ('iso_e_super', 15.0)]:
        oav = engine.compute_oav(name, pct)
        print(f"  {name:20s} @ {pct:5.1f}%: log(OAV)={oav:.2f}")
    
    # === Test 3: Non-linear Mixture ===
    print("\n--- 3. Non-linear Mixture (OAV masking) ---")
    mix = engine.predict_mixture([
        {'name': 'vanillin', 'pct': 0.1},   # Low conc but high ODT -> strong
        {'name': 'linalool', 'pct': 10.0},   # High conc, moderate ODT
        {'name': 'limonene', 'pct': 5.0},    # Medium
    ])
    if mix:
        print(f"  Dominant: {mix['dominant_ingredient']}")
        for ind in mix['individual']:
            print(f"    {ind['name']:15s}: {ind['pct']:5.1f}%  OAV={ind['oav']:5.2f}  weight={ind['perceptual_weight']:.3f}")
        print(f"  Top notes: {[(n,round(v,2)) for n,v in mix['top_notes'][:5]]}")
    
    # === Test 4: 138d Prediction ===
    print("\n--- 4. 138d Prediction ---")
    for name in ['linalool', 'vanillin', 'eugenol']:
        smi = engine.resolve_smiles(name)
        pred = engine.predict_138d(smi)
        top3 = np.argsort(pred)[-3:][::-1]
        descs = ', '.join(f'{TASKS_138[i]}({pred[i]:.2f})' for i in top3)
        print(f"  {name:15s}: {descs}")
    
    # === Test 5: Rose Accord with IFRA + Cost + Balance ===
    print("\n--- 5. GA Multi-Objective (Rose, IFRA+Cost+Balance) ---")
    rose_smiles = _KNOWN_SMILES['geraniol']
    print(f"  Target: Geraniol (rose profile)")
    print(f"  Budget: $30/kg, IFRA enforced, Balance enforced")
    
    result = engine.reverse_engineer(
        target=rose_smiles,
        exclude_names=['rose', 'geraniol', 'citronellol'],
        n_components=5,
        population_size=100,
        generations=200,
        enforce_ifra=True,
        enforce_balance=True,
        target_budget_usd=30,
    )
    
    if result:
        print(f"\n  Similarity:    {result['similarity']}")
        print(f"  Generations:   {result['generations_used']}")
        print(f"  Cost/kg:       ${result['cost_per_kg']}")
        print(f"  IFRA:          {'[OK] COMPLIANT' if result['ifra_compliant'] else '[X] VIOLATION'}")
        print(f"  Balance:       T={result['balance']['top']}% M={result['balance']['middle']}% B={result['balance']['base']}%")
        print(f"  Method:        {result.get('method', 'N/A')}")
        print(f"  Recipe:")
        for item in result['recipe']:
            print(f"    {item['name']:20s}: {item['pct']:5.1f}% "
                  f"[{item['volatility']:6s}] ${item['cost_kg']}/kg "
                  f"MW={item['molwt']} LogP={item['logp']} {item['ifra_status']}")
    else:
        print("  [FAILED] No recipe found")
    
    # === Test 6: Replacement Search ===
    print("\n--- 6. Replacement Search (Linalool alternatives) ---")
    replacements = engine.find_replacements('linalool', top_k=5)
    for i, r in enumerate(replacements):
        print(f"  #{i+1}: {r['name']:20s} sim={r['similarity']:.4f} ({r['category']}, {r['note_type']})")
    
    elapsed = time.time() - t0
    print(f"\n[OK] POM Engine v4 complete ({elapsed:.1f}s)")
