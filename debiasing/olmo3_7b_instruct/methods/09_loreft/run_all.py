import json
import os
import random
import sys
import time
import torch
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer
from lib.hooks import InterventionModuleHook
from lib.loreft import LoreftIntervention, train_loreft
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import eval_on_calib, run_and_score, write_jsonl
from lib.concept_fit import load_jsonl as load_jsonl_cf
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
CALIB = 'external_data/calibration_set.jsonl'
RESP_DIR = 'firmcore_eval/responses'
LAYER = 16
RANKS = [4, 8]
MAX_TRAIN_ITEMS = 6000
EPOCHS = 2
BATCH_SIZE = 16
LR = 0.001

def load_training_items():
    items = []
    for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        for it in load_jsonl_cf(path):
            if it['condition'] == 'GT' and it.get('expected_correct'):
                items.append({'prompt': it['prompt'], 'target': it['expected_correct']})
    random.Random(0).shuffle(items)
    return items[:MAX_TRAIN_ITEMS]

def main():
    calib_items = load_jsonl(CALIB)
    firmcore_items = load_jsonl(FIRMCORE)
    train_items = load_training_items()
    epochs = EPOCHS
    if os.environ.get('SMOKE_N'):
        n = int(os.environ['SMOKE_N'])
        (calib_items, firmcore_items) = (calib_items[:n], firmcore_items[:n])
        train_items = train_items[:40]
        epochs = 1
    print(f'calib={len(calib_items)} firmcore={len(firmcore_items)}', flush=True)
    print(f'[train] {len(train_items)} GT items with expected_correct', flush=True)
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    calib_scores = {}
    modules = {}
    t0 = time.time()
    for rank in RANKS:
        key = f'L{LAYER}_r{rank}'
        print(f'[train] {key} ...', flush=True)
        module = LoreftIntervention(hidden_dim, rank)
        losses = train_loreft(model, tok, module, LAYER, train_items, device, epochs=epochs, batch_size=BATCH_SIZE, lr=LR)
        print(f'  first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f} n_steps={len(losses)}', flush=True)
        modules[key] = module
        hook = InterventionModuleHook(module)
        s = eval_on_calib(model, tok, calib_items, hook, LAYER)
        calib_scores[key] = s
        print(f"[calib] combo={key} -> dag={s['dag']} gsr={s['gsr']} badness={s['badness']} refusal={s['refusal_rate']}", flush=True)
    best_key = min(calib_scores, key=lambda k: calib_scores[k]['badness'] if calib_scores[k]['badness'] is not None else 1000000000.0)
    sweep_log = {'combos_tried': list(calib_scores.keys()), 'calibration_scores': calib_scores, 'chosen_combo': best_key, 'chosen_badness': calib_scores[best_key]['badness'], 'elapsed_sec': time.time() - t0, 'n_train_items': len(train_items), 'epochs': epochs}
    with open(os.path.join(HERE, 'sweep_log.json'), 'w') as f:
        json.dump(sweep_log, f, indent=2)
    print(f'[calib] chosen combo = {best_key}', flush=True)
    print(f'[firmcore] running chosen combo {best_key} on full firmCore ...', flush=True)
    best_module = modules[best_key]
    out_name = f'09_loreft_{best_key}_firmcore.jsonl'
    out_path = os.path.join(RESP_DIR, out_name)
    if os.path.exists(out_path) and sum((1 for _ in open(out_path))) >= len(firmcore_items):
        print(f'  [skip] {out_name} already complete', flush=True)
    else:
        hook = InterventionModuleHook(best_module)
        resp = run_and_score(model, tok, firmcore_items, hook, LAYER)
        write_jsonl(out_path, resp)
        print(f'  wrote {len(resp)} -> {out_path}', flush=True)
    chosen_config = {'method': '09_loreft', 'chosen_combo': best_key, 'chosen_layer': LAYER, 'chosen_rank': int(best_key.split('_r')[1]), 'firmcore_response_file': out_name}
    with open(os.path.join(HERE, 'chosen_config.json'), 'w') as f:
        json.dump(chosen_config, f, indent=2)
    print('[done] method 9 complete', flush=True)
if __name__ == '__main__':
    main()
