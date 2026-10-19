import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT_ROOT)
from lib.hf_backend import load_model_and_tokenizer, generate_with_hook, next_token_choice_logprobs_with_hook
from data import load_mmlu, load_arc_challenge, format_mc_prompt, LETTERS
import gsm8k_util as gk
os.chdir(PROJECT_ROOT)

def mc_score(model, tok, items):
    max_n = max((len(it['choices']) for it in items))
    choices = tuple(LETTERS[:max_n])
    prompts = [format_mc_prompt(it['question'], it['choices']) for it in items]
    probs = next_token_choice_logprobs_with_hook(model, tok, prompts, choices=choices, hook_obj=None, layer_idx=None, batch_size=24)
    n_correct = n_parsed = 0
    for (it, p) in zip(items, probs):
        valid = {k: v for (k, v) in p.items() if k in LETTERS[:len(it['choices'])] and v is not None}
        if not valid:
            continue
        pred = max(valid, key=valid.get)
        n_parsed += 1
        if pred == it['gold_letter']:
            n_correct += 1
    n = len(items)
    return {'n': n, 'n_parsed': n_parsed, 'parse_rate': n_parsed / n if n else 0.0, 'accuracy': n_correct / n if n else 0.0}

def main():
    mmlu_items = load_mmlu(n_per_subject=30)
    arc_items = load_arc_challenge()
    gsm8k_items = gk.load_gsm8k(300)
    (model, tok) = load_model_and_tokenizer()
    s_mmlu = mc_score(model, tok, mmlu_items)
    print(f'mmlu -> {s_mmlu}', flush=True)
    s_arc = mc_score(model, tok, arc_items)
    print(f'arc_challenge -> {s_arc}', flush=True)
    prompts = [it['question'] + gk.ANSWER_SUFFIX for it in gsm8k_items]
    texts = generate_with_hook(model, tok, prompts, hook_obj=None, layer_idx=None, batch_size=24, max_new_tokens=320)
    s_gsm8k = gk.score_gsm8k(gsm8k_items, texts)
    print(f'gsm8k -> {s_gsm8k}', flush=True)
    out = {'n_per_subject': 30, 'results': {'mmlu': s_mmlu, 'arc_challenge': s_arc, 'gsm8k': s_gsm8k}}
    with open(os.path.join(HERE, 'results_hf_backend_baseline.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('[done] HF-backend baseline complete', flush=True)
if __name__ == '__main__':
    main()
