import json
import os
import sys
import time
import torch
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer
from lib.hooks import ProjectionHook
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import eval_on_calib, run_and_score, write_jsonl
from lib import concept_fit
from concept_erasure import LeaceEraser
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
CALIB = 'external_data/calibration_set.jsonl'
RESP_DIR = 'firmcore_eval/responses'
LAYERS = [8, 16, 24]
MAX_FIT_ITEMS = 4000

def leace_apply_fn(eraser, device, dtype):
    proj_left = eraser.proj_left.to(device=device, dtype=torch.float32)
    proj_right = eraser.proj_right.to(device=device, dtype=torch.float32)
    bias = eraser.bias.to(device=device, dtype=torch.float32) if eraser.bias is not None else None

    def apply_fn(hs):
        (b, s, h) = hs.shape
        flat = hs.reshape(-1, h)
        delta = flat - bias if bias is not None else flat
        out = flat - delta @ proj_right.mH @ proj_left.mH
        return out.reshape(b, s, h).to(dtype)
    return apply_fn

def main():
    calib_items = load_jsonl(CALIB)
    firmcore_items = load_jsonl(FIRMCORE)
    print(f'calib={len(calib_items)} firmcore={len(firmcore_items)}', flush=True)
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print('[fit] extracting firm_fit features at all layers ...', flush=True)
    (x_firm_by_layer, z_firm, _) = concept_fit.concept_a_firm_multilayer(model, tok, LAYERS, max_items=MAX_FIT_ITEMS)
    print('[fit] extracting bib features at all layers ...', flush=True)
    (x_bib_by_layer, z_bib, _) = concept_fit.concept_a_bib_multilayer(model, tok, LAYERS, max_items=MAX_FIT_ITEMS)
    erasers = {}
    for layer in LAYERS:
        erasers['firm', layer] = LeaceEraser.fit(x_firm_by_layer[layer], z_firm)
        erasers['bib', layer] = LeaceEraser.fit(x_bib_by_layer[layer], z_bib)
    calib_scores = {}
    t0 = time.time()
    for ((corpus, layer), eraser) in erasers.items():
        key = f'{corpus}_L{layer}'
        print(f'[calib] combo={key}', flush=True)
        apply_fn = leace_apply_fn(eraser, device, dtype)
        hook = ProjectionHook(apply_fn)
        s = eval_on_calib(model, tok, calib_items, hook, layer)
        calib_scores[key] = s
        print(f"  -> dag={s['dag']} gsr={s['gsr']} badness={s['badness']} refusal={s['refusal_rate']}", flush=True)
    best_key = min(calib_scores, key=lambda k: calib_scores[k]['badness'] if calib_scores[k]['badness'] is not None else 1000000000.0)
    (best_corpus, best_layer) = best_key.rsplit('_L', 1)
    best_layer = int(best_layer)
    sweep_log = {'combos_tried': list(calib_scores.keys()), 'calibration_scores': calib_scores, 'chosen_combo': best_key, 'chosen_badness': calib_scores[best_key]['badness'], 'elapsed_sec': time.time() - t0}
    with open(os.path.join(HERE, 'sweep_log.json'), 'w') as f:
        json.dump(sweep_log, f, indent=2)
    print(f'[calib] chosen combo = {best_key}', flush=True)
    print(f'[firmcore] running chosen combo {best_key} on full firmCore ...', flush=True)
    best_eraser = erasers[best_corpus, best_layer]
    apply_fn = leace_apply_fn(best_eraser, device, dtype)
    hook = ProjectionHook(apply_fn)
    out_name = f'05_leace_{best_key}_firmcore.jsonl'
    out_path = os.path.join(RESP_DIR, out_name)
    if os.path.exists(out_path) and sum((1 for _ in open(out_path))) >= len(firmcore_items):
        print(f'  [skip] {out_name} already complete', flush=True)
    else:
        resp = run_and_score(model, tok, firmcore_items, hook, best_layer)
        write_jsonl(out_path, resp)
        print(f'  wrote {len(resp)} -> {out_path}', flush=True)
    chosen_config = {'method': '05_leace', 'chosen_combo': best_key, 'chosen_corpus': best_corpus, 'chosen_layer': best_layer, 'firmcore_response_file': out_name}
    with open(os.path.join(HERE, 'chosen_config.json'), 'w') as f:
        json.dump(chosen_config, f, indent=2)
    print('[done] method 5 complete', flush=True)
if __name__ == '__main__':
    main()
