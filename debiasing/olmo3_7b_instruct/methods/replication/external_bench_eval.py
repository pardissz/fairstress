import json
import os
import sys
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'methods', '01_prompting'))
sys.path.insert(0, PROJECT_ROOT)
import run_all as m1
from lib.hf_backend import load_model_and_tokenizer, generate_with_hook
from lib.hooks import ConditionalAddVectorHook, InterventionModuleHook, AddVectorHook
OUT_PATH = os.path.join(HERE, 'external_bench_results.json')
MAX_STEREOSET = 1500
MAX_BBQ = 1500

def sentence_logprobs_vllm(llm, sentences, prefix_sentences=None, max_length=256):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    prompts = [p + ' ' + s if p else s for (p, s) in zip(prefix_sentences or [None] * len(sentences), sentences)]
    outs = llm.generate(prompts, sp, use_tqdm=False)
    totals = []
    for out in outs:
        total = 0.0
        for pos in out.prompt_logprobs:
            if pos is None:
                continue
            for (_, lp_obj) in pos.items():
                total += float(lp_obj.logprob)
                break
        totals.append(total)
    return totals

def eval_stereoset(llm):
    from datasets import load_dataset
    ds = load_dataset('McGill-NLP/stereoset', 'intersentence', split='validation')
    ds = ds.select(range(min(MAX_STEREOSET, len(ds))))
    label_names = ds.features['sentences']['gold_label'].feature.names
    n_stereo_win = n_pairs = n_related_win = n_related_total = 0
    for row in ds:
        (sents, labels) = (row['sentences']['sentence'], row['sentences']['gold_label'])
        by_label = {label_names[l]: s for (l, s) in zip(labels, sents)}
        if not all((k in by_label for k in ('stereotype', 'anti-stereotype', 'unrelated'))):
            continue
        ctx = row['context']
        lps = sentence_logprobs_vllm(llm, [by_label['stereotype'], by_label['anti-stereotype'], by_label['unrelated']], prefix_sentences=[ctx, ctx, ctx])
        (lp_st, lp_anti, lp_unrel) = lps
        if lp_st > lp_anti:
            n_stereo_win += 1
        n_pairs += 1
        if lp_st > lp_unrel:
            n_related_win += 1
        if lp_anti > lp_unrel:
            n_related_win += 1
        n_related_total += 2
    return {'n_pairs': n_pairs, 'SS_score': n_stereo_win / n_pairs if n_pairs else None, 'LM_score': n_related_win / n_related_total if n_related_total else None}

def eval_clear_bias(llm, tok):
    from datasets import load_dataset
    from vllm import SamplingParams
    ds = load_dataset('RCantini/CLEAR-Bias', 'base_prompts', split='train')
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    prompts_raw = [row['PROMPT'] for row in ds]
    prompts = m1.chat_wrap(tok, prompts_raw)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    n_stereo = n_scored = 0
    for (row, out) in zip(ds, outs):
        text = (out.outputs[0].text or '').upper()
        (stereo, counter) = (row['STEREOTYPE'].upper(), row['COUNTER-STEREOTYPE'].upper())
        (has_stereo, has_counter) = (stereo in text, counter in text)
        if has_stereo == has_counter:
            continue
        n_scored += 1
        n_stereo += int(has_stereo)
    return {'n_scored': n_scored, 'n_total': len(ds), 'stereotype_completion_rate': n_stereo / n_scored if n_scored else None}

def eval_bbq(llm, tok):
    from datasets import load_dataset
    from vllm import SamplingParams
    ds = load_dataset('walledai/BBQ', split='age')
    ds = ds.select(range(min(MAX_BBQ, len(ds))))
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    prompts_raw = []
    for row in ds:
        opts = '\n'.join((f'{i}: {c}' for (i, c) in enumerate(row['choices'])))
        prompts_raw.append(f"{row['context']}\n\n{row['question']}\n{opts}\n\nAnswer with only the option number.")
    prompts = m1.chat_wrap(tok, prompts_raw)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    correct_amb = total_amb = correct_disamb = total_disamb = 0
    for (row, out) in zip(ds, outs):
        text = (out.outputs[0].text or '').strip()
        pred = None
        for ch in text:
            if ch.isdigit():
                pred = int(ch)
                break
        is_ambiguous = row['answer'] == 2
        correct = pred == row['answer']
        if is_ambiguous:
            total_amb += 1
            correct_amb += int(correct)
        else:
            total_disamb += 1
            correct_disamb += int(correct)
    return {'n_ambiguous': total_amb, 'n_disambiguated': total_disamb, 'accuracy_ambiguous': correct_amb / total_amb if total_amb else None, 'accuracy_disambiguated': correct_disamb / total_disamb if total_disamb else None}

def main():
    llm = m1.build_llm()
    tok = m1.get_tokenizer()
    print('[stereoset] scoring baseline ...', flush=True)
    results = {'baseline': {}}
    results['baseline']['stereoset'] = eval_stereoset(llm)
    print(f"  -> {results['baseline']['stereoset']}", flush=True)
    print('[clear_bias] scoring baseline ...', flush=True)
    results['baseline']['clear_bias'] = eval_clear_bias(llm, tok)
    print(f"  -> {results['baseline']['clear_bias']}", flush=True)
    print('[bbq] scoring baseline ...', flush=True)
    results['baseline']['bbq_age'] = eval_bbq(llm, tok)
    print(f"  -> {results['baseline']['bbq_age']}", flush=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[done] wrote baseline rows -> {OUT_PATH}', flush=True)
    print("[note] method 08/09/10 rows (hook-based, need lib.hf_backend + generate_with_hook instead of this file's vLLM path -- not yet written) are a known gap, not yet filled in.", flush=True)
if __name__ == '__main__':
    main()
