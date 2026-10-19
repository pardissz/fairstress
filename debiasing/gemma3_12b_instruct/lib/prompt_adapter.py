import sys
from pathlib import Path
AI2_ROOT = Path('./data')
if str(AI2_ROOT) not in sys.path:
    sys.path.insert(0, str(AI2_ROOT))
import firm_run
extract_subject = firm_run.extract_subject
parse_choice = firm_run.parse_choice
strip_thinking = firm_run.strip_thinking
MODEL_PATH = 'google/gemma-3-12b-it'

def load_jsonl(path):
    import json
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def write_jsonl_append(path, records):
    import json
    with open(path, 'a') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

def already_done_ids(path):
    import json
    done = set()
    p = Path(path)
    if not p.exists():
        return done
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)['item_id'])
            except Exception:
                continue
    return done

def extract_ab_logprobs_from_vllm_output(gen):
    lp_a = lp_b = None
    if not (gen.logprobs and gen.token_ids):
        return (lp_a, lp_b)
    for pos_idx in range(len(gen.token_ids) - 1, -1, -1):
        if pos_idx >= len(gen.logprobs):
            continue
        pos_lps = gen.logprobs[pos_idx]
        if not pos_lps:
            continue
        tok_id = gen.token_ids[pos_idx]
        actual = pos_lps.get(tok_id)
        if actual is None:
            for (_, lp_obj) in pos_lps.items():
                if getattr(lp_obj, 'rank', None) == 1:
                    actual = lp_obj
                    break
        if actual is None:
            continue
        actual_tok = getattr(actual, 'decoded_token', '').strip().upper()
        if actual_tok in ('A', 'B'):
            for (_, lp_obj) in pos_lps.items():
                t = getattr(lp_obj, 'decoded_token', '').strip().upper()
                lp = float(lp_obj.logprob)
                if t == 'A':
                    lp_a = lp
                elif t == 'B':
                    lp_b = lp
            break
    return (lp_a, lp_b)
_CHAT_TOK = None

def get_chat_tokenizer():
    global _CHAT_TOK
    if _CHAT_TOK is None:
        from transformers import AutoTokenizer
        _CHAT_TOK = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    return _CHAT_TOK

def apply_chat_template(user_content: str) -> str:
    tok = get_chat_tokenizer()
    return tok.apply_chat_template([{'role': 'user', 'content': user_content}], add_generation_prompt=True, tokenize=False)
