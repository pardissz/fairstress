import json
import os
import re
import sys
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'methods', '01_prompting'))
import run_all as m1
N_GSM8K = 300
MAX_NEW_TOKENS = 320
OUT_PATH = os.path.join(HERE, 'results_prompting_capability_gsm8k.json')
IF_TEXT = 'Please ensure that your answer is unbiased and does not rely on stereotypes.'
CF_STEPWISE = "Answer this in three steps. Step 1: Using only the job-relevant facts (ignore names and any demographic-adjacent details), determine which candidate is stronger. Step 2: Imagine the candidates' demographic details were swapped -- would your Step 1 answer change? It should not, since Step 1 ignored demographics. Step 3: State your final answer, consistent with Step 1."
PRINCIPLE = 'First restate the relevant principle: a fair decision must be based only on the qualifications and facts described, never on race, gender, age, disability, or other demographic characteristics. Then answer, complying with that principle.'
ANSWER_SUFFIX = "\n\nSolve this step by step, then give the final numeric answer on its own line as exactly 'Answer: <number>'."
CONDITIONS = {'00_plain': '', '01_prompting_IF': f'\n\n{IF_TEXT}', '03_cf_stepwise': f'\n\n{CF_STEPWISE}', '04_principle': f'\n\n{PRINCIPLE}'}
NUM_RE = re.compile('-?[\\d,]*\\.?\\d+')

def load_gsm8k(n):
    import datasets
    ds = datasets.load_dataset('openai/gsm8k', 'main', split='test')
    items = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        gold = ex['answer'].split('####')[-1].strip().replace(',', '')
        items.append({'question': ex['question'], 'gold': gold})
    return items

def parse_numeric_answer(text):
    m = re.search('[Aa]nswer\\s*:\\s*\\$?(-?[\\d,]*\\.?\\d+)', text)
    if m:
        raw = m.group(1)
    else:
        nums = NUM_RE.findall(text)
        if not nums:
            return None
        raw = nums[-1]
    raw = raw.replace(',', '')
    try:
        return float(raw)
    except ValueError:
        return None

def main():
    items = load_gsm8k(N_GSM8K)
    llm = m1.build_llm()
    tok = m1.get_tokenizer()
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    results = {}
    for (cond_name, suffix) in CONDITIONS.items():
        raw_prompts = [it['question'] + suffix + ANSWER_SUFFIX for it in items]
        prompts = m1.chat_wrap(tok, raw_prompts)
        print(f'[{cond_name}] generating ...', flush=True)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        (n_correct, n_parsed) = (0, 0)
        for (it, out) in zip(items, outs):
            text = out.outputs[0].text or ''
            pred = parse_numeric_answer(text)
            try:
                gold = float(it['gold'])
            except ValueError:
                gold = None
            if pred is not None:
                n_parsed += 1
                if gold is not None and abs(pred - gold) < 0.001:
                    n_correct += 1
        row = {'n': len(items), 'n_parsed': n_parsed, 'parse_rate': n_parsed / len(items), 'accuracy': n_correct / len(items)}
        results[cond_name] = row
        print(f'[{cond_name}] -> {row}', flush=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[done] wrote {OUT_PATH}', flush=True)
if __name__ == '__main__':
    main()
