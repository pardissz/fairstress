import copy
import json
import os
import random
import sys
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)
from lib.hf_backend import load_model_and_tokenizer
from lib.hooks import InterventionModuleHook
from lib.loreft import LoreftIntervention, train_loreft
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import eval_on_calib, select_best_combo_with_accuracy_floor
from lib import concept_fit
CALIB = os.path.join(PROJECT_ROOT, 'external_data/calibration_set.jsonl')
LAYER = 16
RANK = 4
OUT_PATH = os.path.join(HERE, 'hybrid_v1_calib_results.json')
NEUTRAL_SCAFFOLD = '\n\nAnswer in three steps. Step 1: List the concrete, job-relevant facts stated for each candidate. Step 2: Compare those facts directly to determine which candidate is stronger. Step 3: State your final answer, consistent with Step 2.'

def load_training_items():
    items = []
    for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        for it in concept_fit.load_jsonl(os.path.join(PROJECT_ROOT, path)):
            if it['condition'] == 'GT' and it.get('expected_correct'):
                items.append({'prompt': it['prompt'], 'target': it['expected_correct']})
    random.Random(0).shuffle(items)
    return items[:6000]

def main():
    calib_items = load_jsonl(CALIB)
    train_items = load_training_items()
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    print('[hybrid_v1] training LoReFT L16_r4 (same recipe as original winner) ...', flush=True)
    module = LoreftIntervention(model.config.hidden_size, RANK)
    train_loreft(model, tok, module, LAYER, train_items, device, epochs=2, batch_size=16, lr=0.001)
    hook = InterventionModuleHook(module)
    print('[hybrid_v1] eval: LoReFT alone (plain calib prompts) ...', flush=True)
    s_plain = eval_on_calib(model, tok, calib_items, hook, LAYER)
    print(f'  -> {s_plain}', flush=True)
    print('[hybrid_v1] eval: LoReFT + neutral stepwise scaffold ...', flush=True)
    calib_scaffolded = []
    for it in calib_items:
        it2 = copy.copy(it)
        it2['prompt'] = it['prompt'] + NEUTRAL_SCAFFOLD
        calib_scaffolded.append(it2)
    s_scaffold = eval_on_calib(model, tok, calib_scaffolded, hook, LAYER)
    print(f'  -> {s_scaffold}', flush=True)
    combos = {'loreft_alone': s_plain, 'loreft_plus_neutral_scaffold': s_scaffold}
    (best, all_degenerate) = select_best_combo_with_accuracy_floor(combos)
    result = {'combos': combos, 'chosen': best, 'all_degenerate_or_below_floor': all_degenerate}
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[hybrid_v1] chosen={best}, wrote {OUT_PATH}', flush=True)
if __name__ == '__main__':
    main()
