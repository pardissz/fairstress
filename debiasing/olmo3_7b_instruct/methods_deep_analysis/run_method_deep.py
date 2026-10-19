import json
import os
import re
import sys
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)
from lib.hf_backend import load_model_and_tokenizer, last_token_hidden_states, generate_with_hook
from lib.hooks import register_hook, remove_hooks
import refit
METHOD = sys.argv[1]
N_GSM8K = int(os.environ.get('N_GSM8K', 300))
GSM8K_MAX_NEW_TOKENS = 320
GSM8K_BATCH_SIZE = 32
OUT_PATH = os.path.join(HERE, f'results_{METHOD}.json')
GSM8K_PROMPT_SUFFIX = "\n\nSolve this step by step, then give the final numeric answer on its own line as exactly 'Answer: <number>'."

def load_gsm8k(n):
    import datasets
    ds = datasets.load_dataset('openai/gsm8k', 'main', split='test')
    items = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        gold = ex['answer'].split('####')[-1].strip().replace(',', '')
        items.append({'question': ex['question'], 'gold': gold})
    return items
NUM_RE = re.compile('-?[\\d,]*\\.?\\d+')

def parse_numeric_answer(text):
    m = re.search('[Aa]nswer\\s*:\\s*\\$?(-?[\\d,]*\\.?\\d+)', text)
    if m:
        raw = m.group(1)
    else:
        nums = NUM_RE.findall(text)
        if not nums:
            return None
        raw = nums[-1]
    raw = raw.replace(',', '')
    try:
        return float(raw)
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
    return {'n': len(items), 'n_parsed': n_parsed, 'parse_rate': n_parsed / len(items), 'accuracy': n_correct / len(items), 'accuracy_of_parsed': n_correct / n_parsed if n_parsed else None}

def extract_probe_features(model, tok, prompts, layer, hook_obj=None):
    handle = register_hook(model, layer, hook_obj) if hook_obj is not None else None
    try:
        x = last_token_hidden_states(model, tok, prompts, layer)
    finally:
        if handle is not None:
            remove_hooks([handle])
    return x

def probe_test_accuracy(x, z, seed=0):
    X = x.numpy().astype(np.float64)
    Z = z.numpy().astype(np.int64)
    (X_tr, X_te, Z_tr, Z_te) = train_test_split(X, Z, test_size=0.2, random_state=seed, stratify=Z)
    clf = LogisticRegression(max_iter=2000).fit(X_tr, Z_tr)
    train_acc = clf.score(X_tr, Z_tr)
    test_acc = clf.score(X_te, Z_te)
    majority_baseline = max(Z_te.mean(), 1 - Z_te.mean())
    return {'train_acc': train_acc, 'test_acc': test_acc, 'majority_baseline': majority_baseline, 'n_train': len(Z_tr), 'n_test': len(Z_te)}

def main():
    t_start = time.time()
    print(f'[{METHOD}] loading model ...', flush=True)
    (model, tok) = load_model_and_tokenizer()
    print(f'[{METHOD}] loading GSM8K (n={N_GSM8K}) ...', flush=True)
    gsm8k_items = load_gsm8k(N_GSM8K)
    print(f'[{METHOD}] loading calibration set ...', flush=True)
    calib_items = refit.load_calib_items()
    calib_prompts = [it['prompt'] for it in calib_items]
    z_calib = torch.tensor([1 if it['minority_slot'] == 'A' else 0 for it in calib_items])
    print(f'[{METHOD}] fitting intervention (refit.build_hook) ...', flush=True)
    t_fit0 = time.time()
    (hook_obj, layer) = refit.build_hook(METHOD, model, tok)
    fit_sec = time.time() - t_fit0
    print(f'[{METHOD}] fit done in {fit_sec:.1f}s, layer={layer}', flush=True)
    results = {'method': METHOD, 'layer': layer, 'fit_sec': fit_sec}
    print(f'[{METHOD}] extracting probe features (hook OFF) ...', flush=True)
    x_off = extract_probe_features(model, tok, calib_prompts, layer, hook_obj=None)
    print(f'[{METHOD}] extracting probe features (hook ON) ...', flush=True)
    x_on = extract_probe_features(model, tok, calib_prompts, layer, hook_obj=hook_obj)
    probe_off = probe_test_accuracy(x_off, z_calib)
    probe_on = probe_test_accuracy(x_on, z_calib)
    results['representation_probe'] = {'hook_off': probe_off, 'hook_on': probe_on, 'test_acc_drop': probe_off['test_acc'] - probe_on['test_acc']}
    print(f"[{METHOD}] probe test_acc off={probe_off['test_acc']:.4f} on={probe_on['test_acc']:.4f}", flush=True)
    print(f'[{METHOD}] GSM8K hook OFF ...', flush=True)
    gsm8k_off = gsm8k_accuracy(model, tok, gsm8k_items, hook_obj=None, layer=layer)
    print(f'[{METHOD}] GSM8K hook OFF -> {gsm8k_off}', flush=True)
    print(f'[{METHOD}] GSM8K hook ON ...', flush=True)
    gsm8k_on = gsm8k_accuracy(model, tok, gsm8k_items, hook_obj=hook_obj, layer=layer)
    print(f'[{METHOD}] GSM8K hook ON -> {gsm8k_on}', flush=True)
    results['gsm8k'] = {'hook_off': gsm8k_off, 'hook_on': gsm8k_on, 'accuracy_drop': gsm8k_off['accuracy'] - gsm8k_on['accuracy']}
    results['total_elapsed_sec'] = time.time() - t_start
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[{METHOD}] done, wrote {OUT_PATH}, elapsed={results['total_elapsed_sec']:.1f}s", flush=True)
if __name__ == '__main__':
    main()
