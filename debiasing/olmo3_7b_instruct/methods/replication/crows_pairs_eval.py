import csv
import importlib.util
import json
import os
import sys
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'methods', '01_prompting'))
sys.path.insert(0, PROJECT_ROOT)
import run_all as m1
_spec = importlib.util.spec_from_file_location('m2v2_run_all', os.path.join(PROJECT_ROOT, 'methods', '02_self_debiasing', 'run_all_v2.py'))
_m2v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m2v2)
SDB_TEXT = _m2v2.SDB_TEXT
CSV_PATH = os.path.join(HERE, 'crows_pairs_anonymized.csv')
OUT_PATH = os.path.join(HERE, 'crows_pairs_results.json')

def load_pairs():
    pairs = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({'more': row['sent_more'], 'less': row['sent_less'], 'bias_type': row['bias_type']})
    return pairs

def sentence_logprobs(llm, tok, sentences, prefix=None, max_length=256):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    prompts = []
    for s in sentences:
        if prefix:
            prompts.append(prefix + '\n' + s)
        else:
            prompts.append(s)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    results = []
    for out in outs:
        lp_list = out.prompt_logprobs
        total = 0.0
        for pos in lp_list:
            if pos is None:
                continue
            for (_, lp_obj) in pos.items():
                total += float(lp_obj.logprob)
                break
        results.append(total)
    return results

def main():
    pairs = load_pairs()
    print(f'crows_pairs pairs={len(pairs)}', flush=True)
    llm = m1.build_llm()
    tok = m1.get_tokenizer()
    more_sents = [p['more'] for p in pairs]
    less_sents = [p['less'] for p in pairs]
    print('[baseline] scoring plain sentence log-likelihoods ...', flush=True)
    lp_more_plain = sentence_logprobs(llm, tok, more_sents)
    lp_less_plain = sentence_logprobs(llm, tok, less_sents)
    baseline_stereo = sum((1 for (a, b) in zip(lp_more_plain, lp_less_plain) if a > b)) / len(pairs)
    print(f'  baseline stereotype_score={baseline_stereo:.4f}', flush=True)
    with open(os.path.join(PROJECT_ROOT, 'methods', '02_self_debiasing', 'chosen_config_v2.json')) as f:
        cfg_v2 = json.load(f)
    lam = cfg_v2['chosen_lambda']
    print(f'[self-debiasing v2] lambda={lam}, scoring sdb-prefixed log-likelihoods ...', flush=True)
    lp_more_sdb = sentence_logprobs(llm, tok, more_sents, prefix=SDB_TEXT)
    lp_less_sdb = sentence_logprobs(llm, tok, less_sents, prefix=SDB_TEXT)
    import math

    def adjusted(lp_plain, lp_sdb, lam):
        delta = lp_plain - lp_sdb
        log_alpha = 0.0 if delta >= 0 else lam * delta
        return lp_plain + log_alpha
    adj_more = [adjusted(a, b, lam) for (a, b) in zip(lp_more_plain, lp_more_sdb)]
    adj_less = [adjusted(a, b, lam) for (a, b) in zip(lp_less_plain, lp_less_sdb)]
    debiased_stereo = sum((1 for (a, b) in zip(adj_more, adj_less) if a > b)) / len(pairs)
    print(f'  self_debiasing_v2 stereotype_score={debiased_stereo:.4f}', flush=True)
    results = {'n_pairs': len(pairs), 'baseline_stereotype_score': baseline_stereo, 'self_debiasing_v2_stereotype_score': debiased_stereo, 'chosen_lambda': lam, 'fair_reference': 0.5}
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[done] wrote {OUT_PATH}', flush=True)
if __name__ == '__main__':
    main()
