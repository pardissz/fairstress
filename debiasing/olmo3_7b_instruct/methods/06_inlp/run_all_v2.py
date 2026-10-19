import json
import os
import sys
import time
import numpy as np
import torch
from sklearn.svm import LinearSVC
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer
from lib.hooks import ProjectionHook
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import eval_on_calib, run_and_score, write_jsonl
from lib import concept_fit
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
CALIB = 'external_data/calibration_set.jsonl'
RESP_DIR = 'firmcore_eval/responses'
LAYERS = [8, 16, 24]
N_ITERS = [4, 8]
MAX_FIT_ITEMS = 4000

def fit_inlp_svm(x: torch.Tensor, z: torch.Tensor, n_iter: int, min_norm=1e-06):
    X = x.numpy().astype(np.float64)
    Z = z.numpy().astype(np.int64)
    ws = []
    for _ in range(n_iter):
        if len(np.unique(Z)) < 2:
            break
        clf = LinearSVC(max_iter=20000, dual='auto', tol=0.001).fit(X, Z)
        w = clf.coef_.reshape(-1)
        norm = np.linalg.norm(w)
        if norm < min_norm:
            break
        w = w / norm
        ws.append(w)
        X = X - np.outer(X @ w, w)
    return ws

def inlp_apply_fn(ws, device, dtype):
    if not ws:
        return lambda hs: hs
    W = torch.tensor(np.stack(ws), dtype=torch.float32, device=device)

    def apply_fn(hs):
        (b, s, h) = hs.shape
        flat = hs.reshape(-1, h)
        for i in range(W.shape[0]):
            w = W[i]
            flat = flat - (flat @ w).unsqueeze(-1) * w
        return flat.reshape(b, s, h).to(dtype)
    return apply_fn

def main():
    calib_items = load_jsonl(CALIB)
    firmcore_items = load_jsonl(FIRMCORE)
    fit_cap = MAX_FIT_ITEMS
    if os.environ.get('SMOKE_N'):
        n = int(os.environ['SMOKE_N'])
        (calib_items, firmcore_items) = (calib_items[:n], firmcore_items[:n])
        fit_cap = 200
    print(f'calib={len(calib_items)} firmcore={len(firmcore_items)}', flush=True)
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print('[fit] extracting firm_fit features at all layers ...', flush=True)
    (x_firm_by_layer, z_firm, _) = concept_fit.concept_a_firm_multilayer(model, tok, LAYERS, max_items=fit_cap)
    print('[fit] extracting bib features at all layers ...', flush=True)
    (x_bib_by_layer, z_bib, _) = concept_fit.concept_a_bib_multilayer(model, tok, LAYERS, max_items=fit_cap)
    projections = {}
    for layer in LAYERS:
        for n_iter in N_ITERS:
            print(f'[fit] inlp_v2 firm L{layer} k={n_iter}', flush=True)
            projections['firm', layer, n_iter] = fit_inlp_svm(x_firm_by_layer[layer], z_firm, n_iter)
            print(f'[fit] inlp_v2 bib L{layer} k={n_iter}', flush=True)
            projections['bib', layer, n_iter] = fit_inlp_svm(x_bib_by_layer[layer], z_bib, n_iter)
    calib_scores = {}
    t0 = time.time()
    for ((corpus, layer, n_iter), ws) in projections.items():
        key = f'{corpus}_L{layer}_k{n_iter}'
        print(f'[calib] combo={key} (n_ws={len(ws)})', flush=True)
        apply_fn = inlp_apply_fn(ws, device, dtype)
        hook = ProjectionHook(apply_fn)
        s = eval_on_calib(model, tok, calib_items, hook, layer)
        calib_scores[key] = s
        print(f"  -> dag={s['dag']} gsr={s['gsr']} badness={s['badness']} refusal={s['refusal_rate']}", flush=True)
    best_key = min(calib_scores, key=lambda k: calib_scores[k]['badness'] if calib_scores[k]['badness'] is not None else 1000000000.0)
    parts = best_key.split('_')
    best_corpus = parts[0]
    best_layer = int(parts[1][1:])
    best_n_iter = int(parts[2][1:])
    sweep_log = {'combos_tried': list(calib_scores.keys()), 'calibration_scores': calib_scores, 'chosen_combo': best_key, 'chosen_badness': calib_scores[best_key]['badness'], 'elapsed_sec': time.time() - t0}
    with open(os.path.join(HERE, 'sweep_log_v2.json'), 'w') as f:
        json.dump(sweep_log, f, indent=2)
    print(f'[calib] chosen combo = {best_key}', flush=True)
    print(f'[firmcore] running chosen combo {best_key} on full firmCore ...', flush=True)
    best_ws = projections[best_corpus, best_layer, best_n_iter]
    apply_fn = inlp_apply_fn(best_ws, device, dtype)
    hook = ProjectionHook(apply_fn)
    out_name = f'06_inlp_v2_{best_key}_firmcore.jsonl'
    out_path = os.path.join(RESP_DIR, out_name)
    if os.path.exists(out_path) and sum((1 for _ in open(out_path))) >= len(firmcore_items):
        print(f'  [skip] {out_name} already complete', flush=True)
    else:
        resp = run_and_score(model, tok, firmcore_items, hook, best_layer)
        write_jsonl(out_path, resp)
        print(f'  wrote {len(resp)} -> {out_path}', flush=True)
    chosen_config = {'method': '06_inlp_v2', 'chosen_combo': best_key, 'chosen_corpus': best_corpus, 'chosen_layer': best_layer, 'chosen_n_iter': best_n_iter, 'firmcore_response_file': out_name}
    with open(os.path.join(HERE, 'chosen_config_v2.json'), 'w') as f:
        json.dump(chosen_config, f, indent=2)
    print('[done] method 6 v2 complete', flush=True)
if __name__ == '__main__':
    main()
