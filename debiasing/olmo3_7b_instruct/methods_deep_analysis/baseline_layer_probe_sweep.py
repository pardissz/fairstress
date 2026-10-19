import json
import os
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
from lib.hf_backend import load_model_and_tokenizer, all_layer_last_token_hidden_states
from lib import concept_fit
CALIB = os.path.join(PROJECT_ROOT, 'external_data/calibration_set.jsonl')
OUT_PATH = os.path.join(HERE, 'baseline_layer_probe_sweep.json')
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31]

def probe_acc(x, z, seed=0):
    X = x.numpy().astype(np.float64)
    Z = z.numpy().astype(np.int64)
    (X_tr, X_te, Z_tr, Z_te) = train_test_split(X, Z, test_size=0.2, random_state=seed, stratify=Z)
    clf = LogisticRegression(max_iter=2000).fit(X_tr, Z_tr)
    return {'train_acc': clf.score(X_tr, Z_tr), 'test_acc': clf.score(X_te, Z_te), 'majority_baseline': max(Z_te.mean(), 1 - Z_te.mean())}

def main():
    (model, tok) = load_model_and_tokenizer()
    calib_items = concept_fit.load_jsonl(CALIB)
    prompts = [it['prompt'] for it in calib_items]
    z = torch.tensor([1 if it['minority_slot'] == 'A' else 0 for it in calib_items])
    print(f'[probe-sweep] extracting hidden states at {len(LAYERS)} layers ...', flush=True)
    x_by_layer = all_layer_last_token_hidden_states(model, tok, prompts, LAYERS)
    results = {}
    for layer in LAYERS:
        r = probe_acc(x_by_layer[layer], z)
        results[layer] = r
        print(f"  layer={layer:2d} test_acc={r['test_acc']:.4f} (majority_baseline={r['majority_baseline']:.4f})", flush=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[done] wrote {OUT_PATH}', flush=True)
if __name__ == '__main__':
    main()
