"""Simple Before vs After benchmark"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging; logging.disable(logging.CRITICAL)
import numpy as np
import json

print("=== BENCHMARK START ===")

# 1. Evaporation
from biophysics_simulator import ThermodynamicsEngine
from data.natural_oil_compositions import unroll_ingredient
thermo = ThermodynamicsEngine()

single = thermo.simulate_evaporation(['CC(=CCC=C(C)C)CO'], [5.0], duration_hours=8)
sub = unroll_ingredient('bergamot', 5.0)
multi = thermo.simulate_evaporation([s for s,_ in sub], [c for _,c in sub], duration_hours=8)
t_b = len(single.get('transitions',[]))
t_a = len(multi.get('transitions',[]))
l_b = single['longevity_hours']
l_a = multi['longevity_hours']
print("EVAP_TRANSITIONS: %d -> %d (%.1fx)" % (t_b, t_a, t_a/max(t_b,1)))
print("EVAP_LONGEVITY: %.1f -> %.1f" % (l_b, l_a))

# 2. Proxy Reward
from scripts.proxy_reward import train_proxy, extract_recipe_features
try:
    with open('data/recipe_training_data.json', encoding='utf-8') as f: recipes = json.load(f)
except:
    with open('data/recipe_training_data.json', encoding='utf-8-sig') as f: recipes = json.load(f)
model = train_proxy(epochs=80)
errs = []
for r in recipes[:100]:
    feat = extract_recipe_features(r)
    pred = model.predict(feat)
    errs.append(abs(r.get('harmony_score',0.85)-pred))
mae = np.mean(errs)
w5 = sum(1 for e in errs if e<0.05)/len(errs)*100
w10 = sum(1 for e in errs if e<0.10)/len(errs)*100
print("PROXY_MAE: %.4f" % mae)
print("PROXY_5PCT: %.0f%%" % w5)
print("PROXY_10PCT: %.0f%%" % w10)

# 3. PMI
from scripts.commercial_prior import CommercialPrior
import random; random.seed(42)
prior = CommercialPrior()
s_scores = [prior.plausibility_score([i['id'] for i in r.get('ingredients',[])]) for r in recipes[:20]]
all_ids = list(prior.ingredient_freq.keys())
r_scores = [prior.plausibility_score(random.sample(all_ids, min(8,len(all_ids)))) for _ in range(20)]
pairs = [(s,1) for s in s_scores]+[(s,0) for s in r_scores]
pairs.sort(key=lambda x:-x[0])
tp=auc_s=0
for sc,lb in pairs:
    if lb==1: tp+=1
    else: auc_s+=tp
auc=auc_s/(20*20)
print("PMI_SUCCESS: %.3f" % np.mean(s_scores))
print("PMI_RANDOM: %.3f" % np.mean(r_scores))
print("PMI_AUC: %.3f" % auc)

# 4. Receptors
from biophysics_simulator import VirtualNose
nose = VirtualNose()
rb = nose.smell(['CC(=CCC=C(C)C)CO'], [5.0])
ra = nose.smell([s for s,_ in sub], [c for _,c in sub])
print("NOSE_ACTIVE: %d -> %d" % (rb['active_receptors'], ra['active_receptors']))

print("=== BENCHMARK END ===")
