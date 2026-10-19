from data import LETTERS

def mc_predict_vllm(llm, tok, chat_wrap_fn, items, wrap_fn=None, batch_size=512):
    from vllm import SamplingParams
    from data import format_mc_prompt
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        raw_prompts = []
        for it in batch:
            p = format_mc_prompt(it['question'], it['choices'])
            if wrap_fn:
                p = wrap_fn(p)
            raw_prompts.append(p)
        prompts = chat_wrap_fn(tok, raw_prompts)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for (it, out) in zip(batch, outs):
            n_choices = len(it['choices'])
            letters = LETTERS[:n_choices]
            gen = out.outputs[0]
            letter_probs = {c: 0.0 for c in letters}
            if gen.logprobs and len(gen.logprobs) > 0:
                pos_lps = gen.logprobs[0]
                for (_, lp_obj) in pos_lps.items():
                    tok_str = getattr(lp_obj, 'decoded_token', '').strip().upper()
                    if tok_str in letter_probs:
                        import math
                        letter_probs[tok_str] += math.exp(float(lp_obj.logprob))
            total = sum(letter_probs.values())
            if total > 0:
                letter_probs = {k: v / total for (k, v) in letter_probs.items()}
                pred = max(letter_probs, key=letter_probs.get)
            else:
                pred = None
            results.append({'id': it['id'], 'gold_letter': it['gold_letter'], 'pred_letter': pred, 'letter_probs': letter_probs})
        if i // batch_size % 5 == 0:
            print(f'    [mc_vllm] {i + len(batch)}/{len(items)}', flush=True)
    return results

def score_mc_results(results):
    n = len(results)
    n_parsed = sum((1 for r in results if r['pred_letter'] is not None))
    n_correct = sum((1 for r in results if r['pred_letter'] == r['gold_letter']))
    return {'n': n, 'n_parsed': n_parsed, 'parse_rate': n_parsed / n if n else 0.0, 'accuracy': n_correct / n if n else 0.0}
