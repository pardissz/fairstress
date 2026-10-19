import json
import os
import random
import sys
import time
import torch
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from lib.hf_backend import load_model_and_tokenizer
from lib.hooks import InterventionModuleHook
from lib.loreft import LoreftIntervention, train_loreft
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import eval_on_calib, select_best_combo
from lib.concept_fit import load_jsonl as load_jsonl_cf
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
CALIB = os.path.join(PROJECT_ROOT, 'external_data/calibration_set.jsonl')
LAYER = int(sys.argv[1])
RANKS = [4, 8]
MAX_TRAIN_ITEMS = 6000
EPOCHS = 2
BATCH_SIZE = 16
LR = 0.001

def load_training_items():
    items = []
    for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        path = os.path.join(PROJECT_ROOT, path)
        for it in load_jsonl_cf(path):
            if it['condition'] == 'GT' and it.get('expected_correct'):
                items.append({'prompt': it['prompt'], 'target': it['expected_correct']})
    random.Random(0).shuffle(items)
    return items[:MAX_TRAIN_ITEMS]

def main():
    calib_items = load_jsonl(CALIB)
    train_items = load_training_items()
    print(f'[layer {LAYER}] calib={len(calib_items)} train={len(train_items)}', flush=True)
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    calib_scores = {}
    train_curves = {}
    t0 = time.time()
    for rank in RANKS:
        key = f'L{LAYER}_r{rank}'
        print(f'[layer {LAYER}] training {key} ...', flush=True)
        module = LoreftIntervention(hidden_dim, rank)
        losses = train_loreft(model, tok, module, LAYER, train_items, device, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
        train_curves[key] = {'first_loss': losses[0], 'last_loss': losses[-1], 'n_steps': len(losses)}
        print(f'  first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f} n_steps={len(losses)}', flush=True)
        hook = InterventionModuleHook(module)
        s = eval_on_calib(model, tok, calib_items, hook, LAYER)
        calib_scores[key] = s
        print(f"[layer {LAYER}] combo={key} -> dag={s['dag']} gsr={s['gsr']} badness={s['badness']} refusal={s['refusal_rate']} frac_majority={s.get('frac_majority_choice')}", flush=True)
    (best_key, all_degenerate) = select_best_combo(calib_scores)
    out = {'layer': LAYER, 'combos_tried': list(calib_scores.keys()), 'calibration_scores': calib_scores, 'train_curves': train_curves, 'chosen_combo_this_layer': best_key, 'chosen_badness': calib_scores[best_key]['badness'], 'all_degenerate': all_degenerate, 'elapsed_sec': time.time() - t0, 'n_train_items': len(train_items), 'epochs': EPOCHS}
    out_path = os.path.join(HERE, f'sweep_log_extended_L{LAYER}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[layer {LAYER}] done, wrote {out_path}, elapsed={out['elapsed_sec']:.1f}s", flush=True)
if __name__ == '__main__':
    main()
