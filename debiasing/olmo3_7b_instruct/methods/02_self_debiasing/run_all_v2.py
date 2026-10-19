import json
import math
import os
import sys
import time
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '01_prompting')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import run_all as m1
from lib.prompt_adapter import load_jsonl, extract_subject, parse_choice
from lib import calib_metrics
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
CALIB = 'external_data/calibration_set.jsonl'
RESP_DIR = 'firmcore_eval/responses'
SDB_TEXT = 'The following text discriminates against people because of their gender, race, disability, and age:'

def sdb_wrap_v2(prompt: str) -> str:
    return f'{SDB_TEXT}\n{prompt}'

def get_ab_probs(llm, tok, items, batch_size=512, wrap_fn=None):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=64, logprobs=10)
    out_all = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        raw_prompts = [wrap_fn(it['prompt']) if wrap_fn else it['prompt'] for it in batch]
        prompts = m1.chat_wrap(tok, raw_prompts)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for (it, out) in zip(batch, outs):
            gen = out.outputs[0]
            raw = gen.text or ''
            subj = extract_subject(it['prompt'])
            choice = parse_choice(raw, subj)
            (lp_a, lp_b) = m1.extract_ab_logprobs_from_vllm_output(gen)
            p_a = math.exp(lp_a) if lp_a is not None else None
            p_b = math.exp(lp_b) if lp_b is not None else None
            out_all.append({'item_id': it['item_id'], 'raw_text': raw, 'fallback_choice': choice, 'p_A': p_a, 'p_B': p_b})
        if i // batch_size % 10 == 0:
            print(f'    [gen] {i + len(batch)}/{len(items)}', flush=True)
    return out_all

def rescale_and_decide(orig, sdb, lam):
    results = []
    for (o, s) in zip(orig, sdb):
        assert o['item_id'] == s['item_id']
        if o['p_A'] is None or o['p_B'] is None:
            choice = o['fallback_choice']
            results.append({'item_id': o['item_id'], 'parsed_choice': choice, 'raw_text': o['raw_text'], 'logprob_A': None, 'logprob_B': None})
            continue
        (pA, pB) = (o['p_A'], o['p_B'])
        sA = s['p_A'] if s['p_A'] is not None else 0.0
        sB = s['p_B'] if s['p_B'] is not None else 0.0
        dA = pA - sA
        dB = pB - sB
        aA = 1.0 if dA >= 0 else math.exp(lam * dA)
        aB = 1.0 if dB >= 0 else math.exp(lam * dB)
        (aA, aB) = (max(0.01, aA), max(0.01, aB))
        (rA, rB) = (aA * pA, aB * pB)
        total = rA + rB
        if total <= 0:
            choice = o['fallback_choice']
        else:
            (rA, rB) = (rA / total, rB / total)
            choice = 'A' if rA >= rB else 'B'
        results.append({'item_id': o['item_id'], 'parsed_choice': choice, 'raw_text': o['raw_text'], 'logprob_A': math.log(max(rA, 1e-12)) if total > 0 else None, 'logprob_B': math.log(max(rB, 1e-12)) if total > 0 else None})
    return results

def write_jsonl(path, records):
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

def main():
    calib_items = load_jsonl(CALIB)
    firmcore_items = load_jsonl(FIRMCORE)
    if os.environ.get('SMOKE_N'):
        n = int(os.environ['SMOKE_N'])
        (calib_items, firmcore_items) = (calib_items[:n], firmcore_items[:n])
    print(f'calib={len(calib_items)} firmcore={len(firmcore_items)}', flush=True)
    llm = m1.build_llm()
    tok = m1.get_tokenizer()
    print('[calib] generating original + sdb_v2 passes ...', flush=True)
    orig_calib = get_ab_probs(llm, tok, calib_items)
    sdb_calib = get_ab_probs(llm, tok, calib_items, wrap_fn=sdb_wrap_v2)
    lambdas = [10, 50, 100]
    sweep = {}
    for lam in lambdas:
        resp = rescale_and_decide(orig_calib, sdb_calib, lam)
        s = calib_metrics.score(resp, calib_items)
        sweep[str(lam)] = s
        print(f"  lambda={lam} -> dag={s['dag']} gsr={s['gsr']} badness={s['badness']}", flush=True)
    best_lam = min(lambdas, key=lambda L: sweep[str(L)]['badness'] if sweep[str(L)]['badness'] is not None else 1000000000.0)
    sweep_log = {'lambdas_tried': lambdas, 'calibration_scores': sweep, 'chosen_lambda': best_lam}
    with open(os.path.join(HERE, 'sweep_log_v2.json'), 'w') as f:
        json.dump(sweep_log, f, indent=2)
    print(f'[calib] chosen lambda = {best_lam}', flush=True)
    print('[firmcore] generating original + sdb_v2 passes over full firmCore ...', flush=True)
    orig_path = os.path.join(HERE, '_firmcore_orig_probs_v2.json')
    sdb_path = os.path.join(HERE, '_firmcore_sdb_probs_v2.json')
    orig_fc = get_ab_probs(llm, tok, firmcore_items)
    with open(orig_path, 'w') as f:
        json.dump(orig_fc, f)
    sdb_fc = get_ab_probs(llm, tok, firmcore_items, wrap_fn=sdb_wrap_v2)
    with open(sdb_path, 'w') as f:
        json.dump(sdb_fc, f)
    for lam in lambdas:
        resp = rescale_and_decide(orig_fc, sdb_fc, lam)
        out_path = os.path.join(RESP_DIR, f'02_self_debiasing_v2_lambda{lam}_firmcore.jsonl')
        write_jsonl(out_path, resp)
        print(f'  wrote lambda={lam} -> {out_path}', flush=True)
    chosen_config = {'method': '02_self_debiasing_v2', 'chosen_lambda': best_lam, 'firmcore_response_file': f'02_self_debiasing_v2_lambda{best_lam}_firmcore.jsonl', 'sdb_text': SDB_TEXT, 'all_lambda_firmcore_files': {str(L): f'02_self_debiasing_v2_lambda{L}_firmcore.jsonl' for L in lambdas}}
    with open(os.path.join(HERE, 'chosen_config_v2.json'), 'w') as f:
        json.dump(chosen_config, f, indent=2)
    print('[done] method 2 v2 complete', flush=True)
if __name__ == '__main__':
    main()
