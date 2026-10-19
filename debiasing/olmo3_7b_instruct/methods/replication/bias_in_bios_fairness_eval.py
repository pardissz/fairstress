import json
import os
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from lib.hf_backend import load_model_and_tokenizer, all_layer_last_token_hidden_states
from lib import concept_fit
from concept_erasure import LeaceEraser
BIB_PATH = os.path.join(PROJECT_ROOT, 'external_data', 'bib_pool.jsonl')
OUT_PATH = os.path.join(HERE, 'bias_in_bios_results.json')
MAX_ITEMS = 4000
FIT_CAP = 4000

def probe_accuracy(X_train, y_train, X_test, y_test):
    if len(set(y_train)) < 2:
        return None
    clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    return clf.score(X_test, y_test)

def main():
    items = concept_fit.load_jsonl(BIB_PATH)[:MAX_ITEMS]
    print(f'bias_in_bios items={len(items)}', flush=True)
    texts = [it['text'] for it in items]
    genders = np.array([int(it['z_label']) for it in items])
    professions = np.array([int(it['profession']) for it in items])
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with open(os.path.join(PROJECT_ROOT, 'methods', '05_leace', 'chosen_config.json')) as f:
        cfg5 = json.load(f)
    with open(os.path.join(PROJECT_ROOT, 'methods', '06_inlp', 'chosen_config.json')) as f:
        cfg6 = json.load(f)
    layers_needed = sorted({cfg5['chosen_layer'], cfg6['chosen_layer']})
    print(f'[extract] raw hidden states at layers {layers_needed} for bib_pool items ...', flush=True)
    x_by_layer = all_layer_last_token_hidden_states(model, tok, texts, layers_needed)
    print('[fit] re-fitting LEACE eraser (05) ...', flush=True)
    l5 = cfg5['chosen_layer']
    if cfg5['chosen_corpus'] == 'firm':
        (xf, zf, _) = concept_fit.concept_a_firm_multilayer(model, tok, [l5], max_items=FIT_CAP)
    else:
        (xf, zf, _) = concept_fit.concept_a_bib_multilayer(model, tok, [l5], max_items=FIT_CAP)
    eraser5 = LeaceEraser.fit(xf[l5], zf)
    print('[fit] re-fitting INLP projection (06) ...', flush=True)
    (l6, k6) = (cfg6['chosen_layer'], cfg6['chosen_n_iter'])
    if cfg6['chosen_corpus'] == 'firm':
        (xf6, zf6, _) = concept_fit.concept_a_firm_multilayer(model, tok, [l6], max_items=FIT_CAP)
    else:
        (xf6, zf6, _) = concept_fit.concept_a_bib_multilayer(model, tok, [l6], max_items=FIT_CAP)
    import importlib.util
    spec = importlib.util.spec_from_file_location('m6_run_all', os.path.join(PROJECT_ROOT, 'methods', '06_inlp', 'run_all.py'))
    m6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m6)
    ws6 = m6.fit_inlp(xf6[l6], zf6, k6)

    def apply_leace(x):
        return eraser5(x)

    def apply_inlp(x):
        out = x.clone()
        for w in ws6:
            w_t = torch.tensor(w, dtype=out.dtype)
            out = out - (out @ w_t).unsqueeze(-1) * w_t
        return out
    results = {}
    for (name, layer, apply_fn) in [('05_leace', l5, apply_leace), ('06_inlp', l6, apply_inlp)]:
        x_raw = x_by_layer[layer]
        x_erased = apply_fn(x_raw)
        idx = np.arange(len(items))
        (train_idx, test_idx) = train_test_split(idx, test_size=0.2, random_state=0, stratify=genders)
        raw_gender_acc = probe_accuracy(x_raw[train_idx].numpy(), genders[train_idx], x_raw[test_idx].numpy(), genders[test_idx])
        erased_gender_acc = probe_accuracy(x_erased[train_idx].numpy(), genders[train_idx], x_erased[test_idx].numpy(), genders[test_idx])
        raw_prof_acc = probe_accuracy(x_raw[train_idx].numpy(), professions[train_idx], x_raw[test_idx].numpy(), professions[test_idx])
        erased_prof_acc = probe_accuracy(x_erased[train_idx].numpy(), professions[train_idx], x_erased[test_idx].numpy(), professions[test_idx])
        results[name] = {'layer': layer, 'raw_gender_probe_acc': raw_gender_acc, 'erased_gender_probe_acc': erased_gender_acc, 'gender_leakage_reduction': raw_gender_acc - erased_gender_acc if raw_gender_acc is not None and erased_gender_acc is not None else None, 'raw_profession_probe_acc': raw_prof_acc, 'erased_profession_probe_acc': erased_prof_acc, 'profession_acc_retained_frac': erased_prof_acc / raw_prof_acc if raw_prof_acc and erased_prof_acc is not None else None}
        print(f'  {name}: {results[name]}', flush=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[done] wrote {OUT_PATH}', flush=True)
if __name__ == '__main__':
    main()
