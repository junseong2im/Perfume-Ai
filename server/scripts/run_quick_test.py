import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from odor_engine import ConcentrationModulator, PhysicsMixture, OdorGNN, ODOR_DIMENSIONS

mod = ConcentrationModulator()
mixer = PhysicsMixture()
gnn = OdorGNN(device='cpu')

passed = 0
total = 0

# Test 1: Concentration modulation
print('=== Test 1: ConcentrationModulator ===')
test_vec = np.zeros(22)
test_vec[ODOR_DIMENSIONS.index('floral')] = 0.8
test_vec[ODOR_DIMENSIONS.index('sweet')] = 0.4

fl_idx = ODOR_DIMENSIONS.index('floral')
results = {}
for conc in [0.01, 0.1, 1.0, 3.0, 5.0, 10.0, 20.0]:
    r = mod.modulate(test_vec.copy(), conc)
    fl = r[fl_idx]
    results[conc] = fl
    print(f'  conc={conc:6.2f}% -> floral={fl:.3f}')

# Check: monotonic increase from 0.1 to 5%
total += 1
if results[0.1] < results[1.0] < results[3.0] < results[5.0]:
    print('  [PASS] Monotonic increase 0.1-5%')
    passed += 1
else:
    print('  [FAIL] Not monotonic')

# Check: subthreshold at very low conc
total += 1
if results[0.01] < 0.05:
    print('  [PASS] Subthreshold detection')
    passed += 1
else:
    print('  [FAIL] Subthreshold too high')

# Check: masking at 20%
total += 1
if results[20.0] < results[5.0]:
    print('  [PASS] Masking at high concentration')
    passed += 1
else:
    print('  [FAIL] No masking effect')

# Test 2: High-concentration inversion
print('\n=== Test 2: High-Conc Inversion ===')
inv_vec = np.zeros(22)
inv_vec[ODOR_DIMENSIONS.index('floral')] = 0.7
inv_vec[ODOR_DIMENSIONS.index('earthy')] = 0.1
r_inv = mod.modulate(inv_vec.copy(), 8.0)
earthy_new = r_inv[ODOR_DIMENSIONS.index('earthy')]
total += 1
if earthy_new > 0.1:
    print(f'  [PASS] floral->earthy inversion: earthy 0.1 -> {earthy_new:.3f}')
    passed += 1
else:
    print(f'  [FAIL] No inversion: earthy={earthy_new:.3f}')

# Test 3: Synergy (floral+musk -> powdery boost)
print('\n=== Test 3: PhysicsMixture Synergy ===')
va = np.zeros(22)
va[ODOR_DIMENSIONS.index('floral')] = 0.8
va[ODOR_DIMENSIONS.index('sweet')] = 0.3
vb = np.zeros(22)
vb[ODOR_DIMENSIONS.index('musk')] = 0.8
vb[ODOR_DIMENSIONS.index('warm')] = 0.3
r3, a3 = mixer.mix(np.array([va,vb]), np.array([3.0,3.0]), return_analysis=True)
for d,v in a3['dominant_dims']:
    print(f'  {d}: {v:.3f}')
powdery_idx = ODOR_DIMENSIONS.index('powdery')
total += 1
if r3[powdery_idx] > 0.01:
    print(f'  [PASS] Powdery synergy boost: {r3[powdery_idx]:.3f}')
    passed += 1
else:
    print(f'  [FAIL] No powdery boost: {r3[powdery_idx]:.3f}')

# Test 4: Antagonism (fresh+smoky -> fresh suppressed)
print('\n=== Test 4: PhysicsMixture Antagonism ===')
vc = np.zeros(22)
vc[ODOR_DIMENSIONS.index('fresh')] = 0.8
vc[ODOR_DIMENSIONS.index('citrus')] = 0.5
vd = np.zeros(22)
vd[ODOR_DIMENSIONS.index('smoky')] = 0.8
vd[ODOR_DIMENSIONS.index('woody')] = 0.3
r4, a4 = mixer.mix(np.array([vc,vd]), np.array([3.0,3.0]), return_analysis=True)
for d,v in a4['dominant_dims']:
    print(f'  {d}: {v:.3f}')
fresh_val = r4[ODOR_DIMENSIONS.index('fresh')]
total += 1
if fresh_val < 0.6:
    print(f'  [PASS] Fresh suppressed: 0.8 -> {fresh_val:.3f}')
    passed += 1
else:
    print(f'  [FAIL] Fresh not suppressed: {fresh_val:.3f}')

for ia in a4['interactions']:
    print(f'  interaction: type={ia["type"]} synergy={ia["synergy_score"]:.3f}')

# Test 5: Masking (strong musk vs weak citrus)
print('\n=== Test 5: Masking ===')
ve = np.zeros(22)
ve[ODOR_DIMENSIONS.index('musk')] = 0.9
ve[ODOR_DIMENSIONS.index('warm')] = 0.5
vf = np.zeros(22)
vf[ODOR_DIMENSIONS.index('citrus')] = 0.3
vf[ODOR_DIMENSIONS.index('fresh')] = 0.2
r5, a5 = mixer.mix(np.array([ve,vf]), np.array([8.0,1.0]), return_analysis=True)
citrus_val = r5[ODOR_DIMENSIONS.index('citrus')]
total += 1
if citrus_val < 0.15:
    print(f'  [PASS] Citrus masked by musk: {citrus_val:.3f}')
    passed += 1
else:
    print(f'  [FAIL] Citrus not masked enough: {citrus_val:.3f}')

# Test 6: Full GNN pipeline
print('\n=== Test 6: Full Pipeline (linalool + vanillin) ===')
v1 = gnn.encode('CC(=CCC/C(=C/CO)/C)C')  # linalool
v2 = gnn.encode('O=Cc1ccc(O)c(OC)c1')     # vanillin
mod1 = mod.modulate(v1, 3.0)
mod2 = mod.modulate(v2, 2.0)
rmix = mixer.mix(np.array([mod1,mod2]), np.array([3.0,2.0]))
top = [(ODOR_DIMENSIONS[i], round(float(rmix[i]),3)) for i in np.argsort(rmix)[::-1][:5]]
print(f'  top dims: {top}')
total += 1
if len(top) > 0 and top[0][1] > 0:
    print(f'  [PASS] Pipeline produces valid output')
    passed += 1
else:
    print(f'  [FAIL] Pipeline output empty')

print(f'\n=== RESULTS: {passed}/{total} passed ===')
sys.exit(0 if passed == total else 1)
