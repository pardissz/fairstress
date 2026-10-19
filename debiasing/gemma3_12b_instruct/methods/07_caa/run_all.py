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
ALPHA_FRACS = [-0.5, -0.25, 0.25, 0.5]
MAX_FIT_ITEMS = 4000
LOCK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs', '.lock_07_caa_selfguard')

def main():
    try:
        os.makedirs(LOCK_DIR)
    except FileExistsError:
        print(f'[07_caa] {LOCK_DIR} already claimed -- skipping (another stream already ran or is running this method)', flush=True)
        return
    try:
        _run()
    except BaseException:
        try:
            os.rmdir(LOCK_DIR)
        except OSError:
            pass
        raise

def _run():
    calib_items = load_jsonl(CALIB)
    firmcore_items = load_jsonl(FIRMCORE)
    print(f'calib={len(calib_items)} firmcore={len(firmcore_items)}', flush=True)
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print('[fit] extracting contrastive answer-token features at all layers ...', flush=True)
    (x_min_by_layer, x_maj_by_layer, _) = concept_fit.concept_b_caa(model, tok, LAYERS, max_items=MAX_FIT_ITEMS)
    steering = {}
    for layer in LAYERS:
        x_min = x_min_by_layer[layer]
        x_maj = x_maj_by_layer[layer]
        v = x_min.mean(0) - x_maj.mean(0)
        unit = v / (v.norm() + 1e-08)
        ref_norm = torch.cat([x_min, x_maj], dim=0).norm(dim=-1).mean().item()
        steering[layer] = (unit.to(device=device, dtype=torch.float32), ref_norm)
        print(f'  layer={layer} ||v||={v.norm().item():.3f} ref_norm={ref_norm:.3f}', flush=True)
    calib_scores = {}
    t0 = time.time()
    for layer in LAYERS:
        (unit, ref_norm) = steering[layer]
        for frac in ALPHA_FRACS:
            key = f'L{layer}_a{frac}'
            alpha = frac * ref_norm
            print(f'[calib] combo={key} (alpha={alpha:.3f})', flush=True)
            hook = AddVectorHook(unit, alpha)
            s = eval_on_calib(model, tok, calib_items, hook, layer)
            calib_scores[key] = s
            print(f"  -> dag={s['dag']} gsr={s['gsr']} badness={s['badness']} refusal={s['refusal_rate']}", flush=True)
    best_key = min(calib_scores, key=lambda k: calib_scores[k]['badness'] if calib_scores[k]['badness'] is not None else 1000000000.0)
    best_layer = int(best_key.split('_')[0][1:])
    best_frac = float(best_key.split('_a')[1])
    sweep_log = {'combos_tried': list(calib_scores.keys()), 'calibration_scores': calib_scores, 'chosen_combo': best_key, 'chosen_badness': calib_scores[best_key]['badness'], 'elapsed_sec': time.time() - t0}
    with open(os.path.join(HERE, 'sweep_log.json'), 'w') as f:
        json.dump(sweep_log, f, indent=2)
    print(f'[calib] chosen combo = {best_key}', flush=True)
    print(f'[firmcore] running chosen combo {best_key} on full firmCore ...', flush=True)
    (unit, ref_norm) = steering[best_layer]
    alpha = best_frac * ref_norm
    hook = AddVectorHook(unit, alpha)
    out_name = f'07_caa_{best_key}_firmcore.jsonl'
    out_path = os.path.join(RESP_DIR, out_name)
    if os.path.exists(out_path) and sum((1 for _ in open(out_path))) >= len(firmcore_items):
        print(f'  [skip] {out_name} already complete', flush=True)
    else:
        resp = run_and_score(model, tok, firmcore_items, hook, best_layer)
        write_jsonl(out_path, resp)
        print(f'  wrote {len(resp)} -> {out_path}', flush=True)
    chosen_config = {'method': '07_caa', 'chosen_combo': best_key, 'chosen_layer': best_layer, 'chosen_alpha_frac': best_frac, 'chosen_alpha': alpha, 'firmcore_response_file': out_name}
    with open(os.path.join(HERE, 'chosen_config.json'), 'w') as f:
        json.dump(chosen_config, f, indent=2)
    print('[done] method 7 complete', flush=True)
if __name__ == '__main__':
    main()
