import json
from lib.hf_backend import generate_with_hook
from lib.prompt_adapter import extract_subject, parse_choice
from lib import calib_metrics

def run_and_score(model, tok, items, hook_obj, layer_idx, batch_size=64, max_new_tokens=32):
    prompts = [it['prompt'] for it in items]
    texts = generate_with_hook(model, tok, prompts, hook_obj=hook_obj, layer_idx=layer_idx, batch_size=batch_size, max_new_tokens=max_new_tokens)
    results = []
    for (it, raw) in zip(items, texts):
        subj = extract_subject(it['prompt'])
        choice = parse_choice(raw, subj)
        results.append({'item_id': it['item_id'], 'parsed_choice': choice, 'raw_text': raw, 'logprob_A': None, 'logprob_B': None})
    return results

def eval_on_calib(model, tok, calib_items, hook_obj, layer_idx, batch_size=64):
    resp = run_and_score(model, tok, calib_items, hook_obj, layer_idx, batch_size=batch_size)
    s = calib_metrics.score(resp, calib_items)
    non_refused = [r['parsed_choice'] for r in resp if r['parsed_choice'] in ('A', 'B')]
    if non_refused:
        frac_a = non_refused.count('A') / len(non_refused)
        s['frac_majority_choice'] = max(frac_a, 1 - frac_a)
    else:
        s['frac_majority_choice'] = None
    return s

def select_best_combo(calib_scores: dict, degenerate_threshold=0.85):

    def badness_of(k):
        b = calib_scores[k]['badness']
        return b if b is not None else 1000000000.0
    non_degenerate = [k for (k, v) in calib_scores.items() if v.get('frac_majority_choice') is None or v['frac_majority_choice'] <= degenerate_threshold]
    if non_degenerate:
        best = min(non_degenerate, key=badness_of)
        return (best, False)
    best = min(calib_scores, key=badness_of)
    return (best, True)

def select_best_combo_with_accuracy_floor(calib_scores: dict, degenerate_threshold=0.85, min_gt_accuracy=0.7):

    def badness_of(k):
        b = calib_scores[k]['badness']
        return b if b is not None else 1000000000.0
    ok = [k for (k, v) in calib_scores.items() if (v.get('frac_majority_choice') is None or v['frac_majority_choice'] <= degenerate_threshold) and (v.get('gt_accuracy') is None or v['gt_accuracy'] >= min_gt_accuracy)]
    if ok:
        best = min(ok, key=badness_of)
        return (best, False)
    best = min(calib_scores, key=badness_of)
    return (best, True)

def write_jsonl(path, records):
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
