import json
import os
import random
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
DEBIAS_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, DEBIAS_ROOT)
sys.path.insert(0, os.path.join(DEBIAS_ROOT, 'methods_deep_analysis', 'capability_benchmarks'))
from lib.hf_backend import last_token_hidden_states
from lib.concept_fit import load_jsonl
from data import format_mc_prompt
import gsm8k_util as gk
N_POS = 2000
N_POS_TEST = 300
N_NEG_TEST = 300
LAYER = 16
SEED = 0
if os.environ.get('SMOKE_N'):
    N_POS = 120
    N_POS_TEST = 20
    N_NEG_TEST = 20

def load_positive_prompts(n=N_POS, seed=SEED):
    items = []
    for p in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        items.extend(load_jsonl(os.path.join(DEBIAS_ROOT, p)))
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:n]
    return [it['prompt'] for it in items]

def load_negative_prompts(seed=SEED):
    import datasets
    os.environ.setdefault('HF_HOME', '/shared/storage-01/huggingface')
    prompts = []
    mmlu_val = datasets.load_dataset('cais/mmlu', 'all', split='validation')
    for ex in mmlu_val:
        prompts.append(format_mc_prompt(ex['question'], ex['choices']))
    arc_val = datasets.load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='validation')
    for ex in arc_val:
        prompts.append(format_mc_prompt(ex['question'], ex['choices']['text']))
    gsm8k_train = datasets.load_dataset('openai/gsm8k', 'main', split='train')
    rng = random.Random(seed)
    idxs = list(range(len(gsm8k_train)))
    rng.shuffle(idxs)
    for i in idxs[:1500]:
        prompts.append(gsm8k_train[i]['question'] + gk.ANSWER_SUFFIX)
    rng.shuffle(prompts)
    if os.environ.get('SMOKE_N'):
        prompts = prompts[:160]
    print(f'[gate] negative pool: {len(prompts)} (mmlu_val=1531 arc_val=299 gsm8k_train_sample=1500)', flush=True)
    return prompts

def train_and_validate_gate(model, tok, layer=LAYER, out_dir=HERE):
    pos_all = load_positive_prompts()
    neg_all = load_negative_prompts()
    (pos_train, pos_test) = (pos_all[:-N_POS_TEST], pos_all[-N_POS_TEST:])
    (neg_train, neg_test) = (neg_all[:-N_NEG_TEST], neg_all[-N_NEG_TEST:])
    print(f'[gate] pos train={len(pos_train)} test={len(pos_test)} | neg train={len(neg_train)} test={len(neg_test)}', flush=True)
    print('[gate] extracting hidden states (train) ...', flush=True)
    x_pos_train = last_token_hidden_states(model, tok, pos_train, layer, batch_size=32)
    x_neg_train = last_token_hidden_states(model, tok, neg_train, layer, batch_size=32)
    print('[gate] extracting hidden states (test) ...', flush=True)
    x_pos_test = last_token_hidden_states(model, tok, pos_test, layer, batch_size=32)
    x_neg_test = last_token_hidden_states(model, tok, neg_test, layer, batch_size=32)
    X_train = torch.cat([x_pos_train, x_neg_train], dim=0).numpy()
    y_train = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
    X_test = torch.cat([x_pos_test, x_neg_test], dim=0).numpy()
    y_test = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    probs_test = clf.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds_test)
    auc = roc_auc_score(y_test, probs_test)
    pos_mask = y_test == 1
    pos_recall = preds_test[pos_mask].mean()
    neg_specificity = 1 - preds_test[~pos_mask].mean()
    report = {'layer': layer, 'n_train': len(X_train), 'n_test': len(X_test), 'held_out_accuracy': float(acc), 'held_out_auc': float(auc), 'held_out_pos_recall (fraction of real firmCore-style prompts correctly gated ON)': float(pos_recall), 'held_out_neg_specificity (fraction of capability prompts correctly gated OFF)': float(neg_specificity)}
    print(f'[gate] VALIDATION: {json.dumps(report, indent=2)}', flush=True)
    w = torch.tensor(clf.coef_[0], dtype=torch.float32)
    b = float(clf.intercept_[0])
    torch.save({'w': w, 'b': b, 'layer': layer}, os.path.join(out_dir, 'gate_weights.pt'))
    with open(os.path.join(out_dir, 'gate_validation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return (w, b, report)

def load_gate(out_dir=HERE, device='cuda'):
    d = torch.load(os.path.join(out_dir, 'gate_weights.pt'), map_location=device)
    return (d['w'].to(device), d['b'], d['layer'])
