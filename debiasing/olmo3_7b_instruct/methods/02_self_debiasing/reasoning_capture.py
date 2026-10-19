import json
import os
import sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '01_prompting')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import run_all as m1
from lib.prompt_adapter import load_jsonl
from lib.reasoning_capture import REASONING_SUFFIX, extract_subject, parse_response
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
OUT_DIR = 'firmcore_eval/reasoning'
OUT_PATH = os.path.join(OUT_DIR, '02_self_debiasing_reasoning.jsonl')

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    firmcore_items = load_jsonl(FIRMCORE)
    if os.environ.get('SMOKE_N'):
        firmcore_items = firmcore_items[:int(os.environ['SMOKE_N'])]
    print(f'firmcore={len(firmcore_items)}', flush=True)
    if os.path.exists(OUT_PATH) and sum((1 for _ in open(OUT_PATH))) >= len(firmcore_items):
        print('[skip] already complete', flush=True)
        return
    from vllm import SamplingParams
    llm = m1.build_llm()
    tok = m1.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    results = []
    batch_size = 512
    for i in range(0, len(firmcore_items), batch_size):
        batch = firmcore_items[i:i + batch_size]
        raw_prompts = [f"{it['prompt']}{REASONING_SUFFIX}" for it in batch]
        prompts = m1.chat_wrap(tok, raw_prompts)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for (it, out) in zip(batch, outs):
            raw = out.outputs[0].text or ''
            subj = extract_subject(it['prompt'])
            (choice, reasoning) = parse_response(raw, subj)
            results.append({'item_id': it['item_id'], 'parsed_choice': choice, 'reasoning': reasoning, 'raw_text': raw})
        if i // batch_size % 5 == 0:
            print(f'  {i + len(batch)}/{len(firmcore_items)}', flush=True)
    with open(OUT_PATH, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {len(results)} -> {OUT_PATH}', flush=True)
    print('[done] method 2 reasoning capture complete', flush=True)
if __name__ == '__main__':
    main()
