import json
import os
import re
import sys
import time
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)
from lib.hf_backend import load_model_and_tokenizer, generate_with_hook
from lib.hooks import InterventionModuleHook
from lib.loreft import LoreftIntervention, train_loreft
from lib.repr_edit_common import eval_on_calib
import loreft_lab as lab
VARIANT = sys.argv[1]
LAYER = 16
RANK = 4
NOOP_LAMBDA = 0.01
N_GSM8K = int(os.environ.get('N_GSM8K', 300))
GSM8K_MAX_NEW_TOKENS = 320
GSM8K_BATCH_SIZE = 32
OUT_PATH = os.path.join(HERE, f'variant_results_{VARIANT}.json')
GSM8K_PROMPT_SUFFIX = "\n\nSolve this step by step, then give the final numeric answer on its own line as exactly 'Answer: <number>'."
NUM_RE = re.compile('-?[\\d,]*\\.?\\d+')

def load_gsm8k(n):
    import datasets
    ds = datasets.load_dataset('openai/gsm8k', 'main', split='test')
    items = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        gold = ex['answer'].split('####')[-1].strip().replace(',', '')
        items.append({'question': ex['question'], 'gold': gold})
    return items

def parse_numeric_answer(text):
    m = re.search('[Aa]nswer\\s*:\\s*\\$?(-?[\\d,]*\\.?\\d+)', text)
    raw = m.group(1) if m else NUM_RE.findall(text)[-1] if NUM_RE.findall(text) else None
    if raw is None:
        return None
    try:
        return float(raw.replace(',', ''))
    except ValueError:
        return None

def gsm8k_accuracy(model, tok, items, hook_obj, layer):
    prompts = [it['question'] + GSM8K_PROMPT_SUFFIX for it in items]
    texts = generate_with_hook(model, tok, prompts, hook_obj=hook_obj, layer_idx=layer, batch_size=GSM8K_BATCH_SIZE, max_new_tokens=GSM8K_MAX_NEW_TOKENS)
    (n_correct, n_parsed) = (0, 0)
    for (it, text) in zip(items, texts):
        pred = parse_numeric_answer(text)
        try:
            gold = float(it['gold'])
        except ValueError:
            gold = None
        if pred is not None:
            n_parsed += 1
            if gold is not None and abs(pred - gold) < 0.001:
                n_correct += 1
    return {'n': len(items), 'n_parsed': n_parsed, 'accuracy': n_correct / len(items)}

def main():
    t0 = time.time()
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    use_pressure = VARIANT in ('pressure', 'combined', 'pressure_mixed', 'combined_mixed')
    use_nonlinear = VARIANT in ('nonlinear', 'combined', 'combined_mixed')
    use_noop = VARIANT in ('noop_reg', 'combined', 'combined_mixed')
    pressure_frac = 0.5 if VARIANT in ('pressure_mixed', 'combined_mixed') else 1.0
    print(f'[{VARIANT}] loading training data (pressure={use_pressure}, frac={pressure_frac}) ...', flush=True)
    train_items = lab.load_training_items(with_pressure=use_pressure, pressure_frac=pressure_frac)
    neutral_items = lab.load_neutral_items() if use_noop else None
    module = lab.LoreftInterventionNonlinear(hidden_dim, RANK) if use_nonlinear else LoreftIntervention(hidden_dim, RANK)
    print(f'[{VARIANT}] training (nonlinear={use_nonlinear}, noop_reg={use_noop}) ...', flush=True)
    t_train0 = time.time()
    if use_noop:
        lab.train_loreft_v2(model, tok, module, LAYER, train_items, device, neutral_items=neutral_items, noop_lambda=NOOP_LAMBDA, epochs=2, batch_size=16, lr=0.001)
    else:
        train_loreft(model, tok, module, LAYER, train_items, device, epochs=2, batch_size=16, lr=0.001)
    train_sec = time.time() - t_train0
    hook = InterventionModuleHook(module)
    print(f'[{VARIANT}] training done in {train_sec:.1f}s', flush=True)
    calib_items = lab.load_calibration_items()
    print(f'[{VARIANT}] eval: plain calibration ...', flush=True)
    plain = eval_on_calib(model, tok, calib_items, hook, LAYER)
    print(f'  -> {plain}', flush=True)
    injected_calib_heldout = lab.build_injected_calibration(split='eval')
    print(f'[{VARIANT}] eval: injected calibration, HELD-OUT families (F7/F9), n={len(injected_calib_heldout)} ...', flush=True)
    pressured_heldout = eval_on_calib(model, tok, injected_calib_heldout, hook, LAYER)
    print(f'  -> {pressured_heldout}', flush=True)
    injected_calib_seen = lab.build_injected_calibration(split='train')
    print(f'[{VARIANT}] eval: injected calibration, SEEN families (7 non-held-out), n={len(injected_calib_seen)} ...', flush=True)
    pressured_seen = eval_on_calib(model, tok, injected_calib_seen, hook, LAYER)
    print(f'  -> {pressured_seen}', flush=True)
    gsm8k_items = load_gsm8k(N_GSM8K)
    print(f'[{VARIANT}] eval: GSM8K hook OFF ...', flush=True)
    gsm8k_off = gsm8k_accuracy(model, tok, gsm8k_items, hook_obj=None, layer=LAYER)
    print(f'  -> {gsm8k_off}', flush=True)
    print(f'[{VARIANT}] eval: GSM8K hook ON ...', flush=True)
    gsm8k_on = gsm8k_accuracy(model, tok, gsm8k_items, hook_obj=hook, layer=LAYER)
    print(f'  -> {gsm8k_on}', flush=True)
    result = {'variant': VARIANT, 'layer': LAYER, 'rank': RANK, 'train_sec': train_sec, 'plain_calibration': plain, 'pressure_calibration_heldout_families': pressured_heldout, 'pressure_calibration_seen_families': pressured_seen, 'pressure_accuracy_drop_vs_plain_heldout': plain.get('gt_accuracy') - pressured_heldout.get('gt_accuracy') if plain.get('gt_accuracy') is not None and pressured_heldout.get('gt_accuracy') is not None else None, 'pressure_accuracy_drop_vs_plain_seen': plain.get('gt_accuracy') - pressured_seen.get('gt_accuracy') if plain.get('gt_accuracy') is not None and pressured_seen.get('gt_accuracy') is not None else None, 'gsm8k': {'hook_off': gsm8k_off, 'hook_on': gsm8k_on, 'accuracy_drop': gsm8k_off['accuracy'] - gsm8k_on['accuracy']}, 'total_elapsed_sec': time.time() - t0}
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[{VARIANT}] done, wrote {OUT_PATH}', flush=True)
if __name__ == '__main__':
    main()
