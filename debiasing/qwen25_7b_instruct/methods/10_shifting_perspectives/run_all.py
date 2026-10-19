import json
import os
import sys
import time
from collections import defaultdict
import torch
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer
from lib.hooks import AddVectorHook
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import eval_on_calib, run_and_score, write_jsonl, select_best_combo
from lib import concept_fit
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
CALIB = 'external_data/calibration_set.jsonl'
RESP_DIR = 'firmcore_eval/responses'
LAYERS = [8, 16, 24]
ALPHA_FRACS = [-0.5, -0.25, 0.25, 0.5]
MAX_FIT_ITEMS_PER_AXIS = 30 if os.environ.get('SMOKE_N') else 800

def load_fit_items_by_axis():
    items = []
    for p in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        items.extend(concept_fit.load_jsonl(p))
    by_axis = defaultdict(list)
    for it in items:
        by_axis[it['contrast_category']].append(it)
    return by_axis

@torch.no_grad()
def axis_direction(model, tok, axis_items, layer):
    items = axis_items[:MAX_FIT_ITEMS_PER_AXIS]
    minority_ans = [f"Candidate {it['minority_slot']}" for it in items]
    majority_other = {'A': 'B', 'B': 'A'}
    majority_ans = [f"Candidate {majority_other[it['minority_slot']]}" for it in items]
    x_min = concept_fit._appended_answer_hidden_states(model, tok, items, minority_ans, [layer])[layer]
    x_maj = concept_fit._appended_answer_hidden_states(model, tok, items, majority_ans, [layer])[layer]
    v = concept_fit.contrastive_direction(x_min, x_maj, method='pca')
    ref_norm = torch.cat([x_min, x_maj], dim=0).norm(dim=-1).mean().item()
    return (v, ref_norm)
LOCK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs', '.lock_10_shifting_perspectives_selfguard')

def main():
    try:
        os.makedirs(LOCK_DIR)
    except FileExistsError:
        print(f'[10_shifting_perspectives] {LOCK_DIR} already claimed -- skipping (another stream already ran or is running this method)', flush=True)
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
    if os.environ.get('SMOKE_N'):
        n = int(os.environ['SMOKE_N'])
        (calib_items, firmcore_items) = (calib_items[:n], firmcore_items[:n])
    print(f'calib={len(calib_items)} firmcore={len(firmcore_items)}', flush=True)
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    by_axis = load_fit_items_by_axis()
    print(f'[fit] {len(by_axis)} contrast_category axes: {list(by_axis.keys())}', flush=True)
    combined_by_layer = {}
    for layer in LAYERS:
        (vecs, ref_norms) = ([], [])
        for (axis, axis_items) in by_axis.items():
            if len(axis_items) < 20:
                continue
            (v, ref_norm) = axis_direction(model, tok, axis_items, layer)
            vecs.append(v)
            ref_norms.append(ref_norm)
        combined = torch.stack(vecs).mean(0)
        combined = combined / (combined.norm() + 1e-08)
        ref_norm = sum(ref_norms) / len(ref_norms)
        combined_by_layer[layer] = (combined.to(device=device, dtype=torch.float32), ref_norm)
        print(f'  layer={layer} n_axes={len(vecs)} ref_norm={ref_norm:.3f}', flush=True)
    calib_scores = {}
    t0 = time.time()
    for layer in LAYERS:
        (unit, ref_norm) = combined_by_layer[layer]
        for frac in ALPHA_FRACS:
            key = f'L{layer}_a{frac}'
            alpha = frac * ref_norm
            print(f'[calib] combo={key} (alpha={alpha:.3f})', flush=True)
            hook = AddVectorHook(unit, alpha)
            s = eval_on_calib(model, tok, calib_items, hook, layer)
            calib_scores[key] = s
            print(f"  -> dag={s['dag']} gsr={s['gsr']} badness={s['badness']} refusal={s['refusal_rate']} frac_majority_choice={s['frac_majority_choice']}", flush=True)
    (best_key, all_degenerate) = select_best_combo(calib_scores)
    if all_degenerate:
        print("[WARNING] every combo's calibration responses were content-blind-degenerate (frac_majority_choice > 0.85) -- picking least-bad anyway, but this run's result should not be trusted without further investigation.", flush=True)
    best_layer = int(best_key.split('_')[0][1:])
    best_frac = float(best_key.split('_a')[1])
    sweep_log = {'all_degenerate': all_degenerate, 'combos_tried': list(calib_scores.keys()), 'calibration_scores': calib_scores, 'chosen_combo': best_key, 'chosen_badness': calib_scores[best_key]['badness'], 'elapsed_sec': time.time() - t0}
    with open(os.path.join(HERE, 'sweep_log.json'), 'w') as f:
        json.dump(sweep_log, f, indent=2)
    print(f'[calib] chosen combo = {best_key}', flush=True)
    print(f'[firmcore] running chosen combo {best_key} on full firmCore ...', flush=True)
    (unit, ref_norm) = combined_by_layer[best_layer]
    alpha = best_frac * ref_norm
    hook = AddVectorHook(unit, alpha)
    out_name = f'10_shifting_perspectives_{best_key}_firmcore.jsonl'
    out_path = os.path.join(RESP_DIR, out_name)
    if os.path.exists(out_path) and sum((1 for _ in open(out_path))) >= len(firmcore_items):
        print(f'  [skip] {out_name} already complete', flush=True)
    else:
        resp = run_and_score(model, tok, firmcore_items, hook, best_layer)
        write_jsonl(out_path, resp)
        print(f'  wrote {len(resp)} -> {out_path}', flush=True)
    chosen_config = {'method': '10_shifting_perspectives', 'chosen_combo': best_key, 'chosen_layer': best_layer, 'chosen_alpha_frac': best_frac, 'chosen_alpha': alpha, 'firmcore_response_file': out_name}
    with open(os.path.join(HERE, 'chosen_config.json'), 'w') as f:
        json.dump(chosen_config, f, indent=2)
    print('[done] method 10 complete', flush=True)
if __name__ == '__main__':
    main()
