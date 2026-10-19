import json
import os
import sys
import time
import torch
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer
from lib.hooks import AddVectorHook
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import eval_on_calib, run_and_score, write_jsonl
from lib import concept_fit
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
CALIB = 'external_data/calibration_set.jsonl'
RESP_DIR = 'firmcore_eval/responses'
LAYERS = [8, 16, 24]
ALPHAS = [-8, -4, -2, 2, 4, 8]
MAX_FIT_ITEMS = 4000

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
    print('[fit] extracting contrastive answer-token features at all layers ...', flush=True)
    (x_min_by_layer, x_maj_by_layer, _) = concept_fit.concept_b_caa(model, tok, LAYERS, max_items=fit_cap)
    raw_vecs = {}
    for layer in LAYERS:
        x_min = x_min_by_layer[layer]
        x_maj = x_maj_by_layer[layer]
        v = (x_min.mean(0) - x_maj.mean(0)).to(device=device, dtype=torch.float32)
        raw_vecs[layer] = v
        print(f'  layer={layer} ||v_raw||={v.norm().item():.3f}', flush=True)
    calib_scores = {}
    t0 = time.time()
    for layer in LAYERS:
        v = raw_vecs[layer]
        for alpha in ALPHAS:
            key = f'L{layer}_a{alpha}'
            print(f'[calib] combo={key}', flush=True)
            hook = AddVectorHook(v, float(alpha))
            s = eval_on_calib(model, tok, calib_items, hook, layer)
            calib_scores[key] = s
            print(f"  -> dag={s['dag']} gsr={s['gsr']} badness={s['badness']} refusal={s['refusal_rate']}", flush=True)
    best_key = min(calib_scores, key=lambda k: calib_scores[k]['badness'] if calib_scores[k]['badness'] is not None else 1000000000.0)
    best_layer = int(best_key.split('_')[0][1:])
    best_alpha = int(best_key.split('_a')[1])
    sweep_log = {'combos_tried': list(calib_scores.keys()), 'calibration_scores': calib_scores, 'chosen_combo': best_key, 'chosen_badness': calib_scores[best_key]['badness'], 'elapsed_sec': time.time() - t0}
    with open(os.path.join(HERE, 'sweep_log_v2.json'), 'w') as f:
        json.dump(sweep_log, f, indent=2)
    print(f'[calib] chosen combo = {best_key}', flush=True)
    print(f'[firmcore] running chosen combo {best_key} on full firmCore ...', flush=True)
    v = raw_vecs[best_layer]
    hook = AddVectorHook(v, float(best_alpha))
    out_name = f'07_caa_v2_{best_key}_firmcore.jsonl'
    out_path = os.path.join(RESP_DIR, out_name)
    if os.path.exists(out_path) and sum((1 for _ in open(out_path))) >= len(firmcore_items):
        print(f'  [skip] {out_name} already complete', flush=True)
    else:
        resp = run_and_score(model, tok, firmcore_items, hook, best_layer)
        write_jsonl(out_path, resp)
        print(f'  wrote {len(resp)} -> {out_path}', flush=True)
    chosen_config = {'method': '07_caa_v2', 'chosen_combo': best_key, 'chosen_layer': best_layer, 'chosen_alpha': best_alpha, 'firmcore_response_file': out_name}
    with open(os.path.join(HERE, 'chosen_config_v2.json'), 'w') as f:
        json.dump(chosen_config, f, indent=2)
    print('[done] method 7 v2 complete', flush=True)
if __name__ == '__main__':
    main()
