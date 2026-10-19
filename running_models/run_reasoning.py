import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
REASONING_SUFFIX = '\n\nBriefly explain your reasoning in 1-2 sentences, then state your final answer on its own line as exactly "Answer: A" or "Answer: B".'

def add_reasoning_instruction(prompt: str) -> str:
    return prompt.rstrip() + REASONING_SUFFIX
SUBJECT_TERMS = ['Candidate', 'Applicant', 'Patient', 'Defendant', 'Individual', 'Tenant', 'Student', 'Parent', 'User', 'Worker', 'Claimant', 'Resident', 'Client', 'Offender', 'Consumer', 'Employee', 'Veteran']
_THINK_PATTERNS = [re.compile('<\\|thinking\\|>.*?</\\|thinking\\|>', re.DOTALL | re.IGNORECASE), re.compile('<think>.*?</think>', re.DOTALL | re.IGNORECASE), re.compile('<\\|thinking_start\\|>.*?<\\|thinking_end\\|>', re.DOTALL | re.IGNORECASE)]
_ANSWER_PAT = re.compile('(?:^|\\n)\\s*Answer\\s*:?\\s*([AB])\\b', re.IGNORECASE | re.MULTILINE)

def strip_thinking(text: str) -> str:
    for pat in _THINK_PATTERNS:
        text = pat.sub('', text)
    return text.strip()

def extract_subject(prompt: str) -> str:
    for term in SUBJECT_TERMS:
        if f'{term} A' in prompt or f'{term} B' in prompt:
            return term
    return 'Candidate'

def parse_response(raw: str, subject: str) -> tuple[str, str]:
    raw_clean = strip_thinking(raw)
    matches = list(_ANSWER_PAT.finditer(raw_clean))
    if matches:
        last = matches[-1]
        choice = last.group(1).upper()
        reasoning = raw_clean[:last.start()].strip()
        return (choice, reasoning)
    m = re.search('(?:[Aa]nswer(?:\\s+is)?|[Cc]hose|[Cc]hoose|[Ss]elect(?:ed)?|[Pp]ick(?:ed)?|[Rr]ecommend)[:\\s]+([AB])\\b', raw_clean)
    if m:
        choice = m.group(1)
        reasoning = raw_clean[:m.start()].strip()
        return (choice, reasoning)
    all_hits = []
    for term in SUBJECT_TERMS:
        all_hits.extend(re.finditer(f'\\b{re.escape(term)}\\s+([AB])\\b', raw_clean, re.IGNORECASE))
    if all_hits:
        first = min(all_hits, key=lambda m: m.start())
        choice = first.group(1).upper()
        reasoning = raw_clean[:first.start()].strip()
        return (choice, reasoning)
    SENTENCE_START = re.compile('(?:^|[.!?]\\s+|\\n)\\s*([AB])\\b')
    sentence_initial_spans = {m.start(1) for m in SENTENCE_START.finditer(raw_clean)}
    tokens = [t.group().upper() for t in re.finditer('\\b[AB]\\b', raw_clean) if t.start() not in sentence_initial_spans]
    unique = set(tokens)
    if len(unique) == 1:
        return (unique.pop(), raw_clean)
    return ('refused', raw_clean)

def run_vllm(items, model: str, tensor_parallel: int=1, batch_size: int=256, max_tokens: int=256, temperature: float=0.0, gpu_memory_utilization: float=0.85, verbose: bool=True, disable_custom_all_reduce: bool=False, max_model_len: int=None, out_f=None) -> list[dict]:
    from vllm import LLM, SamplingParams
    print(f'  Loading {model} (tp={tensor_parallel}, gpu_mem={gpu_memory_utilization}, disable_custom_all_reduce={disable_custom_all_reduce}, max_model_len={max_model_len}) …', flush=True)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel, trust_remote_code=True, dtype='bfloat16', gpu_memory_utilization=gpu_memory_utilization, disable_custom_all_reduce=disable_custom_all_reduce, max_model_len=max_model_len)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    chat_tok = None
    try:
        from transformers import AutoTokenizer
        chat_tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True)
        if not getattr(chat_tok, 'chat_template', None):
            chat_tok = None
            print(f'  [info] No chat_template for {model}; using raw prompts', flush=True)
        else:
            print(f'  [info] Chat template ready for {model}', flush=True)
    except Exception as e:
        print(f'  [warn] Tokenizer load failed ({e}); using raw prompts', flush=True)

    def apply_template(raw_prompts):
        if chat_tok is None:
            return raw_prompts
        out = []
        for p in raw_prompts:
            try:
                out.append(chat_tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, tokenize=False))
            except Exception:
                out.append(p)
        return out

    def process_batch(batch_items):
        raw_prompts = [add_reasoning_instruction(i['prompt']) for i in batch_items]
        batch_ids = [i['item_id'] for i in batch_items]
        batch_subjects = [extract_subject(i['prompt']) for i in batch_items]
        prompts = apply_template(raw_prompts)
        outputs = llm.generate(prompts, sampling_params)
        batch_results = []
        for (out, iid, subj) in zip(outputs, batch_ids, batch_subjects):
            raw = out.outputs[0].text or ''
            (choice, reasoning) = parse_response(raw, subj)
            batch_results.append({'item_id': iid, 'parsed_choice': choice, 'reasoning': reasoning, 'raw_text': raw})
        return batch_results
    results = []
    n_a = n_b = n_ref = total_done = 0
    buf = []
    for item in items:
        buf.append(item)
        if len(buf) >= batch_size:
            if verbose:
                print(f'  [{total_done:,}] generating {len(buf)} items …', flush=True)
            batch_results = process_batch(buf)
            buf = []
            if out_f is not None:
                for r in batch_results:
                    out_f.write(json.dumps(r, ensure_ascii=False) + '\n')
                out_f.flush()
                n_a += sum((1 for r in batch_results if r['parsed_choice'] == 'A'))
                n_b += sum((1 for r in batch_results if r['parsed_choice'] == 'B'))
                n_ref += sum((1 for r in batch_results if r['parsed_choice'] == 'refused'))
                total_done += len(batch_results)
            else:
                results.extend(batch_results)
                total_done += len(batch_results)
    if buf:
        if verbose:
            print(f'  [{total_done:,}] generating final {len(buf)} items …', flush=True)
        batch_results = process_batch(buf)
        if out_f is not None:
            for r in batch_results:
                out_f.write(json.dumps(r, ensure_ascii=False) + '\n')
            out_f.flush()
            n_a += sum((1 for r in batch_results if r['parsed_choice'] == 'A'))
            n_b += sum((1 for r in batch_results if r['parsed_choice'] == 'B'))
            n_ref += sum((1 for r in batch_results if r['parsed_choice'] == 'refused'))
            total_done += len(batch_results)
        else:
            results.extend(batch_results)
            total_done += len(batch_results)
    if out_f is not None:
        if verbose:
            print(f'  [done] {total_done:,} written — A:{n_a:,} B:{n_b:,} refused:{n_ref:,} ({n_ref / max(total_done, 1):.1%})', flush=True)
        return []
    return results

def main():
    ap = argparse.ArgumentParser(description='FIRM reasoning inference runner')
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tensor-parallel', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    ap.add_argument('--disable-custom-all-reduce', action='store_true', default=False, help="Force plain NCCL all-reduce instead of vLLM's custom/FlashInfer path; workaround for CUDA_ERROR_INVALID_VALUE on FlashInfer all-reduce workspace init with tensor-parallel>1")
    ap.add_argument('--max-model-len', type=int, default=None, help="Cap vLLM's max context length (defaults to the model's own config, e.g. 65536, which forces a large guaranteed KV cache reservation even though our prompts are short). Set this to just above prompt_len + max_tokens to free up GPU memory.")
    ap.add_argument('--max-items', type=int, default=None, help='Cap items (for testing, e.g. --max-items 1000)')
    ap.add_argument('--start-line', type=int, default=0, help='0-indexed first benchmark line (inclusive) to process; for contiguous multi-GPU sharding')
    ap.add_argument('--end-line', type=int, default=None, help='0-indexed last benchmark line (exclusive) to process; None = end of file')
    ap.add_argument('--verbose', action='store_true', default=True)
    args = ap.parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)['item_id'])
                    except Exception:
                        pass
        print(f'[firm_run_reasoning] Resuming: {len(done_ids):,} already done, skipping …')

    def _bench_gen():
        n_seen = n_todo = 0
        with open(args.benchmark) as f:
            for (line_idx, line) in enumerate(f):
                if line_idx < args.start_line:
                    continue
                if args.end_line is not None and line_idx >= args.end_line:
                    break
                if not line.strip():
                    continue
                n_seen += 1
                item = json.loads(line)
                if item['item_id'] in done_ids:
                    continue
                n_todo += 1
                yield item
                if args.max_items and n_todo >= args.max_items:
                    break
        print(f"[firm_run_reasoning] benchmark exhausted: {n_seen:,} scanned, {n_todo:,} generated, {len(done_ids):,} skipped (range [{args.start_line:,}, {(args.end_line if args.end_line is not None else 'EOF')}))", flush=True)
    print(f'[firm_run_reasoning] model={args.model}  max_tokens={args.max_tokens}  batch_size={args.batch_size}  already_done={len(done_ids):,}', flush=True)
    print(f'[firm_run_reasoning] Reasoning suffix: {repr(REASONING_SUFFIX[:80])}', flush=True)
    with open(out_path, 'a') as out_f:
        run_vllm(_bench_gen(), args.model, tensor_parallel=args.tensor_parallel, batch_size=args.batch_size, max_tokens=args.max_tokens, temperature=args.temperature, gpu_memory_utilization=args.gpu_memory_utilization, verbose=args.verbose, disable_custom_all_reduce=args.disable_custom_all_reduce, max_model_len=args.max_model_len, out_f=out_f)
    print(f'\n[firm_run_reasoning] Done → {out_path}', flush=True)
if __name__ == '__main__':
    main()
