import re
NUM_RE = re.compile('-?[\\d,]*\\.?\\d+')
ANSWER_SUFFIX = "\n\nSolve this step by step, then give the final numeric answer on its own line as exactly 'Answer: <number>'."

def load_gsm8k(n=300):
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

def score_gsm8k(items, texts):
    (n_correct, n_parsed) = (0, 0)
    for (it, text) in zip(items, texts):
        pred = parse_numeric_answer(text or '')
        try:
            gold = float(it['gold'])
        except ValueError:
            gold = None
        if pred is not None:
            n_parsed += 1
            if gold is not None and abs(pred - gold) < 0.001:
                n_correct += 1
    n = len(items)
    return {'n': n, 'n_parsed': n_parsed, 'parse_rate': n_parsed / n if n else 0.0, 'accuracy': n_correct / n if n else 0.0}
