import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
SUBJECT_TERMS = ['Candidate', 'Applicant', 'Patient', 'Defendant', 'Individual', 'Tenant', 'Student', 'Parent', 'User', 'Worker', 'Claimant', 'Resident', 'Client', 'Offender', 'Consumer', 'Employee', 'Veteran', 'Consumer', 'Employee', 'Veteran']

def extract_subject(prompt: str) -> str:
    for term in SUBJECT_TERMS:
        if f'{term} A' in prompt or f'{term} B' in prompt:
            return term
    return 'Candidate'
_THINK_PATTERNS = [re.compile('<\\|thinking\\|>.*?</\\|thinking\\|>', re.DOTALL | re.IGNORECASE), re.compile('<think>.*?</think>', re.DOTALL | re.IGNORECASE), re.compile('<\\|thinking_start\\|>.*?<\\|thinking_end\\|>', re.DOTALL | re.IGNORECASE)]

def strip_thinking(text: str) -> str:
    for pat in _THINK_PATTERNS:
        text = pat.sub('', text)
    return text.strip()

def parse_choice(raw: str, subject: str) -> str:
    raw = strip_thinking(raw)
    stripped = raw.strip().rstrip('.,:;')
    if stripped.upper() in ('A', f'{subject.upper()} A'):
        return 'A'
    if stripped.upper() in ('B', f'{subject.upper()} B'):
        return 'B'
    all_hits = []
    for term in SUBJECT_TERMS:
        all_hits.extend(re.finditer(f'\\b{re.escape(term)}\\s+([AB])\\b', raw, re.IGNORECASE))
    if all_hits:
        by_letter = {}
        for m in all_hits:
            by_letter.setdefault(m.group(1).upper(), []).append(m)
        if len(by_letter) == 1:
            return next(iter(by_letter))
        earliest = min(all_hits, key=lambda m: m.start())
        return earliest.group(1).upper()
    m = re.search('(?:[Aa]nswer(?:\\s+is)?|[Cc]hose|[Cc]hoose|[Ss]elect(?:ed)?|[Pp]ick(?:ed)?|[Rr]ecommend)[:\\s]+([AB])\\b', raw)
    if m:
        return m.group(1)
    SENTENCE_START = re.compile('(?:^|[.!?]\\s+|\\n)\\s*([AB])\\b')
    sentence_initial_spans = {m.start(1) for m in SENTENCE_START.finditer(raw)}
    tokens_upper = [t.group().upper() for t in re.finditer('\\b[AB]\\b', raw) if t.start() not in sentence_initial_spans]
    unique = set(tokens_upper)
    if len(unique) == 1:
        return unique.pop()
    return 'refused'

def extract_logprobs_openai(response, subject: str) -> tuple[float | None, float | None]:
    try:
        content_tokens = response.choices[0].logprobs.content
    except (AttributeError, IndexError, TypeError):
        return (None, None)
    if not content_tokens:
        return (None, None)
    for token_pos in reversed(content_tokens):
        generated = token_pos.token.strip().upper()
        if generated not in ('A', 'B'):
            continue
        lp_a = lp_b = None
        for entry in token_pos.top_logprobs:
            t = entry.token.strip().upper()
            if t == 'A':
                lp_a = entry.logprob
            elif t == 'B':
                lp_b = entry.logprob
        return (lp_a, lp_b)
    return (None, None)

def run_openai(items, model: str, api_key: str | None, base_url: str | None=None, max_retries: int=5, verbose: bool=True, out_f=None, max_workers: int=50) -> list[dict]:
    from openai import OpenAI, RateLimitError, APIError
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    client = OpenAI(api_key=api_key or os.environ.get('OPENAI_API_KEY'), base_url=base_url)
    lock = threading.Lock()
    results = []
    counters = {'n_a': 0, 'n_b': 0, 'n_ref': 0, 'total': 0}
    _use_completion_tokens = any((model.startswith(p) for p in ('gpt-5', 'o1', 'o3', 'o4')))
    _no_temperature = model.startswith('gpt-5.5') or any((model.startswith(p) for p in ('o1', 'o3', 'o4')))

    def process_one(item):
        subject = extract_subject(item['prompt'])
        tok_key = 'max_completion_tokens' if _use_completion_tokens else 'max_tokens'
        n_tokens = 500 if _no_temperature else 32
        kwargs = dict(model=model, messages=[{'role': 'user', 'content': item['prompt']}], **{tok_key: n_tokens})
        if not _no_temperature:
            kwargs['temperature'] = 0
        if base_url is None and (not _no_temperature):
            kwargs['logprobs'] = True
            kwargs['top_logprobs'] = 5
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except RateLimitError:
                time.sleep(2 ** attempt)
            except APIError as e:
                if attempt == max_retries - 1:
                    return {'item_id': item['item_id'], 'parsed_choice': 'refused', 'raw_text': f'[API ERROR: {e}]', 'logprob_A': None, 'logprob_B': None}
                time.sleep(1)
        else:
            return None
        raw = resp.choices[0].message.content or ''
        choice = parse_choice(raw, subject)
        (lp_a, lp_b) = extract_logprobs_openai(resp, subject) if base_url is None else (None, None)
        return {'item_id': item['item_id'], 'parsed_choice': choice, 'raw_text': raw, 'logprob_A': lp_a, 'logprob_B': lp_b}

    def flush_rec(rec):
        if rec is None:
            return
        with lock:
            if out_f is not None:
                out_f.write(json.dumps(rec) + '\n')
                out_f.flush()
            else:
                results.append(rec)
            counters['total'] += 1
            c = rec['parsed_choice']
            if c == 'A':
                counters['n_a'] += 1
            elif c == 'B':
                counters['n_b'] += 1
            else:
                counters['n_ref'] += 1
            if verbose and counters['total'] % 1000 == 0:
                t = counters['total']
                print(f"  [{t:,}] A:{counters['n_a']:,} B:{counters['n_b']:,} refused:{counters['n_ref']:,} ({counters['n_ref'] / t:.1%})", flush=True)
    CHUNK = max_workers * 4
    chunk = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for item in items:
            chunk.append(item)
            if len(chunk) >= CHUNK:
                for fut in as_completed((ex.submit(process_one, it) for it in chunk)):
                    flush_rec(fut.result())
                chunk = []
        if chunk:
            for fut in as_completed((ex.submit(process_one, it) for it in chunk)):
                flush_rec(fut.result())
    if verbose:
        t = counters['total']
        print(f"  [done] {t:,} written — A:{counters['n_a']:,} B:{counters['n_b']:,} refused:{counters['n_ref']:,} ({counters['n_ref'] / max(t, 1):.1%})", flush=True)
    return results

def run_anthropic(items: list[dict], model: str, api_key: str | None, max_retries: int=5, verbose: bool=True, out_f=None) -> list[dict]:
    import anthropic as ant
    client = ant.Anthropic(api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'))
    results = []
    n = len(items)
    for (idx, item) in enumerate(items):
        if verbose and idx % 100 == 0:
            print(f'  [{idx}/{n}] ...', flush=True)
        subject = extract_subject(item['prompt'])
        for attempt in range(max_retries):
            try:
                resp = client.messages.create(model=model, max_tokens=32, messages=[{'role': 'user', 'content': item['prompt']}])
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    rec = {'item_id': item['item_id'], 'parsed_choice': 'refused', 'raw_text': f'[ERROR: {e}]', 'logprob_A': None, 'logprob_B': None}
                    if out_f is not None:
                        out_f.write(json.dumps(rec) + '\n')
                        out_f.flush()
                    else:
                        results.append(rec)
                    break
                time.sleep(2 ** attempt)
        else:
            continue
        raw = resp.content[0].text if resp.content else ''
        rec = {'item_id': item['item_id'], 'parsed_choice': parse_choice(raw, subject), 'raw_text': raw, 'logprob_A': None, 'logprob_B': None}
        if out_f is not None:
            out_f.write(json.dumps(rec) + '\n')
            out_f.flush()
        else:
            results.append(rec)
    return results

def _apply_chat_template(model: str, raw_prompts: list[str]) -> list[str]:
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True)
    except Exception as e:
        print(f'  [warn] Could not load tokenizer for chat template ({e}); using raw prompts', flush=True)
        return raw_prompts
    if not getattr(tok, 'chat_template', None):
        print(f'  [info] No chat_template in tokenizer for {model}; using raw prompts', flush=True)
        return raw_prompts
    formatted = []
    for p in raw_prompts:
        try:
            text = tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, tokenize=False)
        except Exception:
            text = p
        formatted.append(text)
    print(f'  [info] Chat template applied ({model}). Example prefix: {repr(formatted[0][:80])}', flush=True)
    return formatted

def run_vllm(items, model: str, tensor_parallel: int=1, batch_size: int=256, max_tokens: int=64, temperature: float=0.0, num_samples: int=1, gpu_memory_utilization: float=0.9, enforce_eager: bool=False, max_num_seqs: int | None=None, apply_chat_template: bool=True, verbose: bool=True, max_model_len: int=4096, speculative_ngram: bool=False, out_f=None) -> list[dict]:
    from vllm import LLM, SamplingParams
    print(f'  Loading {model} with vLLM (tp={tensor_parallel}, gpu_mem={gpu_memory_utilization}) …', flush=True)
    llm_kwargs = dict(model=model, tensor_parallel_size=tensor_parallel, trust_remote_code=True, dtype='bfloat16', gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, max_num_batched_tokens=32768, enforce_eager=enforce_eager)
    if max_num_seqs is not None:
        llm_kwargs['max_num_seqs'] = max_num_seqs
    if speculative_ngram:
        llm_kwargs['speculative_config'] = {'method': 'ngram', 'num_speculative_tokens': 3, 'prompt_lookup_max': 4, 'prompt_lookup_min': 2}
    llm = LLM(**llm_kwargs)
    use_logprobs = max_tokens <= 64 and num_samples == 1 and (temperature == 0.0)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=num_samples, logprobs=10 if use_logprobs else None)
    chat_tok = None
    if apply_chat_template:
        try:
            from transformers import AutoTokenizer
            chat_tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True)
            if not getattr(chat_tok, 'chat_template', None):
                chat_tok = None
                print(f'  [info] No chat_template for {model}; using raw prompts', flush=True)
            else:
                print(f'  [info] Chat template ready for {model}', flush=True)
        except Exception as e:
            chat_tok = None
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
        raw_prompts = [i['prompt'] for i in batch_items]
        batch_ids = [i['item_id'] for i in batch_items]
        batch_subjects = [extract_subject(i['prompt']) for i in batch_items]
        prompts = apply_template(raw_prompts)
        outputs = llm.generate(prompts, sampling_params)
        batch_results = []
        for (out, iid, subj) in zip(outputs, batch_ids, batch_subjects):
            for (s_idx, gen) in enumerate(out.outputs):
                raw = gen.text or ''
                choice = parse_choice(raw, subj)
                lp_a = lp_b = None
                if use_logprobs and gen.logprobs and gen.token_ids:
                    for pos_idx in range(len(gen.token_ids)):
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
                out_id = iid if num_samples == 1 else f'{iid}__s{s_idx}'
                batch_results.append({'item_id': out_id, 'parsed_choice': choice, 'raw_text': raw, 'logprob_A': lp_a, 'logprob_B': lp_b, 'finish_reason': getattr(gen, 'finish_reason', None)})
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
                    out_f.write(json.dumps(r) + '\n')
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
                out_f.write(json.dumps(r) + '\n')
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
    ap = argparse.ArgumentParser(description='FIRM inference runner')
    ap.add_argument('--benchmark', required=True, help='firm_benchmark.jsonl')
    ap.add_argument('--output', required=True, help='Output path, e.g. firm_responses_gpt4omini.jsonl')
    ap.add_argument('--backend', required=True, choices=['openai', 'anthropic', 'vllm', 'openai-compat'], help='Inference backend')
    ap.add_argument('--model', required=True, help='Model name / path (e.g. gpt-4o-mini, allenai/OLMo-3-7B-Instruct)')
    ap.add_argument('--api-key', default=None, help='API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)')
    ap.add_argument('--base-url', default=None, help='Base URL for openai-compat backend (e.g. http://localhost:8000/v1)')
    ap.add_argument('--tensor-parallel', type=int, default=1, help='Tensor parallel size for vLLM (default: 1)')
    ap.add_argument('--batch-size', type=int, default=256, help='Batch size for vLLM (default: 256)')
    ap.add_argument('--max-tokens', type=int, default=64, help='Max output tokens per item (default: 64; use 2048 for Think models)')
    ap.add_argument('--max-model-len', type=int, default=4096, help='vLLM max context length (prompt + max_tokens must fit; default: 4096, raise for Think models with a larger --max-tokens budget)')
    ap.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (default: 0.0 = greedy)')
    ap.add_argument('--num-samples', type=int, default=1, help='Samples per item (default: 1). With N>1 item_ids get __s0..s{N-1} suffix')
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.9, help='Fraction of GPU memory vLLM may use (default: 0.90)')
    ap.add_argument('--enforce-eager', action='store_true', default=False, help='Disable CUDA graph capture (use for models that OOM during graph capture)')
    ap.add_argument('--max-num-seqs', type=int, default=None, help="vLLM max_num_seqs (default: vLLM's own default, 256). Lower this for large models on tight VRAM margins instead of --enforce-eager -- keeps compiled decode graphs (much faster) while shrinking graph-capture memory.")
    ap.add_argument('--speculative-ngram', action='store_true', default=False, help="Draft-free n-gram speculative decoding. Lossless at temperature=0 (proposed tokens are verified against real logits and rejected on mismatch, so output is provably identical) -- pure throughput win, measured ~6-7%% on OLMo-3-7B-Think's free-form reasoning traces (2026-08-29 A/B test, n=2000: 2.571 -> 2.738 items/sec).")
    ap.add_argument('--no-chat-template', action='store_true', default=False, help='Disable chat template formatting (raw prompts passed to vLLM). By default chat template is applied to all vLLM runs.')
    ap.add_argument('--max-items', type=int, default=None, help='Cap number of items (for testing)')
    ap.add_argument('--start-line', type=int, default=0, help='0-indexed first benchmark line (inclusive) to process; for contiguous multi-GPU sharding')
    ap.add_argument('--end-line', type=int, default=None, help='0-indexed benchmark line (exclusive) to stop at; for contiguous multi-GPU sharding')
    ap.add_argument('--verbose', action='store_true', default=True)
    args = ap.parse_args()
    out_path = Path(args.output)
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)['item_id'])
                    except Exception:
                        pass
        print(f'[firm_run] Resuming: {len(done_ids):,} already done, skipping …')
    if args.backend == 'vllm':

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
                    check_id = f"{item['item_id']}__s0" if args.num_samples > 1 else item['item_id']
                    if check_id in done_ids:
                        continue
                    n_todo += 1
                    yield item
                    if args.max_items and n_todo >= args.max_items:
                        break
            print(f"[firm_run] benchmark exhausted: {n_seen:,} scanned, {n_todo:,} generated, {len(done_ids):,} skipped (range [{args.start_line:,}, {(args.end_line if args.end_line is not None else 'EOF')}))", flush=True)
        print(f'[firm_run] vLLM streaming from {args.benchmark} (batch_size={args.batch_size}, {len(done_ids):,} already done)', flush=True)
        with open(out_path, 'a') as out_f:
            run_vllm(_bench_gen(), args.model, tensor_parallel=args.tensor_parallel, batch_size=args.batch_size, max_tokens=args.max_tokens, temperature=args.temperature, num_samples=args.num_samples, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=args.enforce_eager, max_num_seqs=args.max_num_seqs, apply_chat_template=not args.no_chat_template, verbose=args.verbose, max_model_len=args.max_model_len, speculative_ngram=args.speculative_ngram, out_f=out_f)
        print(f'\n[firm_run] Done (streaming) → {out_path}', flush=True)
        return

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
                check_id = f"{item['item_id']}__s0" if args.num_samples > 1 else item['item_id']
                if check_id in done_ids:
                    continue
                n_todo += 1
                yield item
                if args.max_items and n_todo >= args.max_items:
                    break
        print(f"[firm_run] benchmark exhausted: {n_seen:,} scanned, {n_todo:,} sent to model, {len(done_ids):,} skipped (range [{args.start_line:,}, {(args.end_line if args.end_line is not None else 'EOF')}))", flush=True)
    print(f'[firm_run] Streaming [{args.backend}] model={args.model} from {args.benchmark} ({len(done_ids):,} already done)', flush=True)
    with open(out_path, 'a') as out_f:
        if args.backend == 'openai':
            run_openai(_bench_gen(), args.model, args.api_key, verbose=args.verbose, out_f=out_f)
        elif args.backend == 'openai-compat':
            if not args.base_url:
                sys.exit('[firm_run] --base-url required for openai-compat backend')
            run_openai(_bench_gen(), args.model, args.api_key, base_url=args.base_url, verbose=args.verbose, out_f=out_f)
        elif args.backend == 'anthropic':
            run_anthropic(_bench_gen(), args.model, args.api_key, verbose=args.verbose, out_f=out_f)
    print(f'\n[firm_run] Done → {out_path}')
if __name__ == '__main__':
    main()
