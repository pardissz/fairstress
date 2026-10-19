import json
import os
import sys
import time
import torch
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer, last_token_hidden_states
from lib.hooks import ConditionalAddVectorHook
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import eval_on_calib, run_and_score, write_jsonl, select_best_combo
from lib import concept_fit
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
CALIB = 'external_data/calibration_set.jsonl'
RESP_DIR = 'firmcore_eval/responses'
LAYERS = [8, 16, 24]
ALPHA_FRAC = 0.5
MAX_FIT_ITEMS = 2000
THRESHOLD_PERCENTILES = [25, 50, 75, 90]

def gate_similarity(h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    c_sq = (c * c).sum() + 1e-08
    proj = (h @ c / c_sq).unsqueeze(-1) * c
    proj = torch.tanh(proj)
    h_norm = h.norm(dim=-1) + 1e-08
    proj_norm = proj.norm(dim=-1) + 1e-08
    return (h * proj).sum(-1) / (h_norm * proj_norm)
LOCK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs', '.lock_08_cast_selfguard')

def main():
    try:
        os.makedirs(LOCK_DIR)
    except FileExistsError:
        print(f'[08_cast] {LOCK_DIR} already claimed -- skipping (another stream already ran or is running this method)', flush=True)
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
    fit_cap = MAX_FIT_ITEMS
    if os.environ.get('SMOKE_N'):
        n = int(os.environ['SMOKE_N'])
        (calib_items, firmcore_items) = (calib_items[:n], firmcore_items[:n])
        fit_cap = 60
    print(f'calib={len(calib_items)} firmcore={len(firmcore_items)}', flush=True)
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    print('[fit] extracting condition-vector features (signaling_level >=2 vs ==0) ...', flush=True)
    (x_cond_pos, x_cond_neg) = concept_fit.condition_vector_firmcore_multilayer(model, tok, LAYERS, max_items_per_class=fit_cap)
    print('[fit] extracting behavior-vector (contrastive answer-token) features ...', flush=True)
    (x_min_by_layer, x_maj_by_layer, _) = concept_fit.concept_b_caa(model, tok, LAYERS, max_items=fit_cap)
    per_layer = {}
    for layer in LAYERS:
        c = concept_fit.contrastive_direction(x_cond_pos[layer], x_cond_neg[layer], method='pca')
        v_raw = concept_fit.contrastive_direction(x_min_by_layer[layer], x_maj_by_layer[layer], method='pca')
        v_unit = v_raw / (v_raw.norm() + 1e-08)
        ref_norm = torch.cat([x_min_by_layer[layer], x_maj_by_layer[layer]], dim=0).norm(dim=-1).mean().item()
        alpha = ALPHA_FRAC * ref_norm
        c = c.to(device=device, dtype=torch.float32)
        v_unit = v_unit.to(device=device, dtype=torch.float32)
        calib_prompts = [it['prompt'] for it in calib_items[:500]]
        h_sample = last_token_hidden_states(model, tok, calib_prompts, layer).to(device)
        sims = gate_similarity(h_sample.float(), c).cpu()
        thresholds = [torch.quantile(sims, p / 100.0).item() for p in THRESHOLD_PERCENTILES]
        print(f'  layer={layer} alpha={alpha:.3f} sim_percentiles={list(zip(THRESHOLD_PERCENTILES, thresholds))}', flush=True)
        per_layer[layer] = (c, v_unit, alpha, thresholds)
    calib_scores = {}
    hooks_by_key = {}
    t0 = time.time()
    for layer in LAYERS:
        (c, v_unit, alpha, thresholds) = per_layer[layer]
        for (pct, thr) in zip(THRESHOLD_PERCENTILES, thresholds):
            key = f'L{layer}_p{pct}'
            print(f'[calib] combo={key} (threshold={thr:.3f})', flush=True)
            hook = ConditionalAddVectorHook(c, thr, v_unit, alpha)
            s = eval_on_calib(model, tok, calib_items, hook, layer)
            calib_scores[key] = s
            hooks_by_key[key] = (layer, c, thr, v_unit, alpha)
            print(f"  -> dag={s['dag']} gsr={s['gsr']} badness={s['badness']} refusal={s['refusal_rate']} frac_majority_choice={s['frac_majority_choice']}", flush=True)
    (best_key, all_degenerate) = select_best_combo(calib_scores)
    if all_degenerate:
        print("[WARNING] every combo's calibration responses were content-blind-degenerate (frac_majority_choice > 0.85) -- picking least-bad anyway, but this run's result should not be trusted without further investigation.", flush=True)
    sweep_log = {'combos_tried': list(calib_scores.keys()), 'calibration_scores': calib_scores, 'chosen_combo': best_key, 'chosen_badness': calib_scores[best_key]['badness'], 'all_degenerate': all_degenerate, 'elapsed_sec': time.time() - t0}
    with open(os.path.join(HERE, 'sweep_log.json'), 'w') as f:
        json.dump(sweep_log, f, indent=2)
    print(f'[calib] chosen combo = {best_key}', flush=True)
    print(f'[firmcore] running chosen combo {best_key} on full firmCore ...', flush=True)
    (best_layer, c, thr, v_unit, alpha) = hooks_by_key[best_key]
    hook = ConditionalAddVectorHook(c, thr, v_unit, alpha)
    out_name = f'08_cast_{best_key}_firmcore.jsonl'
    out_path = os.path.join(RESP_DIR, out_name)
    if os.path.exists(out_path) and sum((1 for _ in open(out_path))) >= len(firmcore_items):
        print(f'  [skip] {out_name} already complete', flush=True)
    else:
        resp = run_and_score(model, tok, firmcore_items, hook, best_layer)
        write_jsonl(out_path, resp)
        print(f'  wrote {len(resp)} -> {out_path}', flush=True)
    chosen_config = {'method': '08_cast', 'chosen_combo': best_key, 'chosen_layer': best_layer, 'chosen_threshold': thr, 'chosen_alpha': alpha, 'firmcore_response_file': out_name}
    with open(os.path.join(HERE, 'chosen_config.json'), 'w') as f:
        json.dump(chosen_config, f, indent=2)
    print('[done] method 8 complete', flush=True)
if __name__ == '__main__':
    main()
