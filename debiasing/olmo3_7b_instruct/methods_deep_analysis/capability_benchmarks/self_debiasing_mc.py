import math
from data import LETTERS
SDB_TEXT = 'The following answer will be biased, discriminatory, and will rely on stereotypes about gender, race, disability, and age:'
LAMBDA = 10

def sdb_wrap(prompt: str) -> str:
    return f'{prompt}\n\n{SDB_TEXT}'

def _get_letter_probs(llm, tok, chat_wrap_fn, items, wrap_fn=None, batch_size=512):
    from vllm import SamplingParams
    from data import format_mc_prompt
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    out_all = []
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
            probs = {c: 0.0 for c in letters}
            if gen.logprobs and len(gen.logprobs) > 0:
                pos_lps = gen.logprobs[0]
                for (_, lp_obj) in pos_lps.items():
                    tok_str = getattr(lp_obj, 'decoded_token', '').strip().upper()
                    if tok_str in probs:
                        probs[tok_str] += math.exp(float(lp_obj.logprob))
            out_all.append({'id': it['id'], 'letters': letters, 'probs': probs})
    return out_all

def mc_predict_self_debiasing(llm, tok, chat_wrap_fn, items, lam=LAMBDA, batch_size=512):
    orig = _get_letter_probs(llm, tok, chat_wrap_fn, items, batch_size=batch_size)
    sdb = _get_letter_probs(llm, tok, chat_wrap_fn, items, wrap_fn=sdb_wrap, batch_size=batch_size)
    id_to_gold = {it['id']: it['gold_letter'] for it in items}
    results = []
    for (o, s) in zip(orig, sdb):
        assert o['id'] == s['id']
        rescaled = {}
        for c in o['letters']:
            p = o['probs'].get(c, 0.0)
            q = s['probs'].get(c, 0.0)
            delta = p - q
            alpha = 1.0 if delta >= 0 else math.exp(lam * delta)
            alpha = max(0.01, alpha)
            rescaled[c] = alpha * p
        total = sum(rescaled.values())
        if total > 0:
            rescaled = {k: v / total for (k, v) in rescaled.items()}
            pred = max(rescaled, key=rescaled.get)
        else:
            pred = None
        results.append({'id': o['id'], 'gold_letter': id_to_gold[o['id']], 'pred_letter': pred, 'letter_probs': rescaled})
    return results
