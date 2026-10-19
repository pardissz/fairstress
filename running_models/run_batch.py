import argparse, json, math, os, re, sys, time
from pathlib import Path
BATCH_SIZE = 40000
POLL_INTERVAL = 60
SUBJECT_TERMS = ['Candidate', 'Applicant', 'Patient', 'Defendant', 'Individual', 'Tenant', 'Student', 'Parent', 'User', 'Worker', 'Claimant', 'Resident', 'Client', 'Offender']
_THINK_PATTERNS = [re.compile('<\\|thinking\\|>.*?</\\|thinking\\|>', re.DOTALL | re.IGNORECASE), re.compile('<think>.*?</think>', re.DOTALL | re.IGNORECASE)]

def strip_thinking(text):
    for pat in _THINK_PATTERNS:
        text = pat.sub('', text)
    return text.strip()

def extract_subject(prompt):
    for term in SUBJECT_TERMS:
        if f'{term} A' in prompt or f'{term} B' in prompt:
            return term
    return 'Candidate'

def parse_choice(raw, subject):
    raw = strip_thinking(raw)
    stripped = raw.strip().rstrip('.,:;')
    if stripped.upper() in ('A', f'{subject.upper()} A'):
        return 'A'
    if stripped.upper() in ('B', f'{subject.upper()} B'):
        return 'B'
    found = {}
    for letter in ('A', 'B'):
        pat = f'\\b{re.escape(subject)}\\s+{letter}\\b'
        if re.findall(pat, raw, re.IGNORECASE):
            found[letter] = True
    if len(found) == 1:
        return list(found.keys())[0]
    m = re.search('(?:answer(?:\\s+is)?|chose|choose|select(?:ed)?|pick(?:ed)?|recommend)[:\\s]+([AB])\\b', raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    tokens = re.findall('\\b[AB]\\b', raw)
    unique = set((t.upper() for t in tokens))
    if len(unique) == 1:
        return unique.pop()
    return 'refused'

def extract_logprobs(top_logprobs):
    lp_a = lp_b = None
    for tok_pos in reversed(top_logprobs):
        generated = tok_pos['token'].strip().upper()
        if generated not in ('A', 'B'):
            continue
        for entry in tok_pos.get('top_logprobs', []):
            t = entry['token'].strip().upper()
            if t == 'A':
                lp_a = entry['logprob']
            elif t == 'B':
                lp_b = entry['logprob']
        return (lp_a, lp_b)
    return (None, None)

def load_state(state_path):
    if Path(state_path).exists():
        return json.load(open(state_path))
    return {'batches': [], 'merged_ids': []}

def save_state(state_path, state):
    json.dump(state, open(state_path, 'w'), indent=2)

def build_and_upload_batches(items_gen, client, model, state, state_path, work_dir, verbose):
    work_dir.mkdir(parents=True, exist_ok=True)
    submitted_start_ids = {b['start_id'] for b in state['batches']}
    chunk = []
    chunk_idx = len([b for b in state['batches'] if b.get('uploaded')])
    first_item = None

    def upload_chunk(chunk, idx):
        start_id = chunk[0]['item_id']
        if start_id in submitted_start_ids:
            if verbose:
                print(f'  [batch {idx}] already uploaded (start_id={start_id}), skipping', flush=True)
            return
        req_file = work_dir / f'batch_{idx:05d}_requests.jsonl'
        use_completion_tokens = any((model.startswith(p) for p in ('gpt-5', 'o1', 'o3', 'o4')))
        tok_key = 'max_completion_tokens' if use_completion_tokens else 'max_tokens'
        with open(req_file, 'w') as f:
            for item in chunk:
                body = {'model': model, 'messages': [{'role': 'user', 'content': item['prompt']}], 'temperature': 0, tok_key: 32, 'logprobs': True, 'top_logprobs': 5}
                f.write(json.dumps({'custom_id': item['item_id'], 'method': 'POST', 'url': '/v1/chat/completions', 'body': body}) + '\n')
        if verbose:
            print(f'  [batch {idx}] uploading {len(chunk):,} requests ({req_file.stat().st_size / 1000000.0:.1f} MB)...', flush=True)
        with open(req_file, 'rb') as f:
            uploaded = client.files.create(file=f, purpose='batch')
        batch_job = client.batches.create(input_file_id=uploaded.id, endpoint='/v1/chat/completions', completion_window='24h')
        state['batches'].append({'batch_idx': idx, 'start_id': start_id, 'n_items': len(chunk), 'file_id': uploaded.id, 'batch_id': batch_job.id, 'status': batch_job.status, 'uploaded': True, 'completed': False, 'output_file': None})
        save_state(state_path, state)
        if verbose:
            print(f'  [batch {idx}] submitted: {batch_job.id}  status={batch_job.status}', flush=True)
    for item in items_gen:
        chunk.append(item)
        if len(chunk) >= BATCH_SIZE:
            upload_chunk(chunk, chunk_idx)
            chunk_idx += 1
            chunk = []
    if chunk:
        upload_chunk(chunk, chunk_idx)
    if verbose:
        print(f"[firm_run_batch] All chunks uploaded: {len(state['batches'])} batches total", flush=True)

def poll_batches(client, state, state_path, verbose):
    pending = [b for b in state['batches'] if not b['completed']]
    if not pending:
        print('[firm_run_batch] All batches already completed.', flush=True)
        return
    print(f'[firm_run_batch] Polling {len(pending)} pending batches...', flush=True)
    while True:
        all_done = True
        for b in state['batches']:
            if b['completed']:
                continue
            job = client.batches.retrieve(b['batch_id'])
            b['status'] = job.status
            if job.status in ('completed', 'failed', 'cancelled', 'expired'):
                b['completed'] = True
                b['output_file'] = job.output_file_id
                if verbose:
                    print(f"  batch {b['batch_idx']:05d} {b['batch_id']}: {job.status}  output_file={job.output_file_id}", flush=True)
            else:
                all_done = False
                if verbose:
                    done = getattr(job.request_counts, 'completed', '?')
                    total = getattr(job.request_counts, 'total', '?')
                    print(f"  batch {b['batch_idx']:05d} {b['batch_id']}: {job.status}  {done}/{total}", flush=True)
        save_state(state_path, state)
        if all_done:
            break
        print(f'  ...waiting {POLL_INTERVAL}s...', flush=True)
        time.sleep(POLL_INTERVAL)
    print('[firm_run_batch] All batches finished.', flush=True)

def download_and_merge(client, state, state_path, out_path, work_dir, verbose):
    out_path = Path(out_path)
    merged_ids = set(state.get('merged_ids', []))
    if out_path.exists() and (not merged_ids):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    try:
                        merged_ids.add(json.loads(line)['item_id'])
                    except Exception:
                        pass
        state['merged_ids'] = list(merged_ids)
        save_state(state_path, state)
    total_written = len(merged_ids)
    with open(out_path, 'a') as out_f:
        for b in state['batches']:
            if not b['completed'] or not b['output_file']:
                print(f"  [batch {b['batch_idx']:05d}] skipping (not complete)", flush=True)
                continue
            result_file = work_dir / f"batch_{b['batch_idx']:05d}_results.jsonl"
            if not result_file.exists():
                if verbose:
                    print(f"  [batch {b['batch_idx']:05d}] downloading {b['output_file']}...", flush=True)
                content = client.files.content(b['output_file']).content
                result_file.write_bytes(content)
            n_batch = 0
            with open(result_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    item_id = row['custom_id']
                    if item_id in merged_ids:
                        continue
                    resp_body = row.get('response', {}).get('body', {})
                    error = row.get('error')
                    if error or not resp_body:
                        rec = {'item_id': item_id, 'parsed_choice': 'refused', 'raw_text': f'[BATCH ERROR: {error}]', 'logprob_A': None, 'logprob_B': None}
                    else:
                        choices = resp_body.get('choices', [])
                        if not choices:
                            raw = ''
                        else:
                            raw = choices[0].get('message', {}).get('content', '') or ''
                        subject = extract_subject('')
                        choice = parse_choice(raw, 'Candidate')
                        lp_a = lp_b = None
                        try:
                            lp_content = choices[0].get('logprobs', {}).get('content', [])
                            if lp_content:
                                (lp_a, lp_b) = extract_logprobs(lp_content)
                        except Exception:
                            pass
                        rec = {'item_id': item_id, 'parsed_choice': choice, 'raw_text': raw, 'logprob_A': lp_a, 'logprob_B': lp_b}
                    out_f.write(json.dumps(rec) + '\n')
                    out_f.flush()
                    merged_ids.add(item_id)
                    n_batch += 1
                    total_written += 1
            if verbose:
                print(f"  [batch {b['batch_idx']:05d}] merged {n_batch:,} records", flush=True)
    state['merged_ids'] = list(merged_ids)
    save_state(state_path, state)
    print(f'[firm_run_batch] Merge complete: {total_written:,} total records → {out_path}', flush=True)

def main():
    ap = argparse.ArgumentParser(description='FIRM Phase 3 runner via OpenAI Batch API')
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--api-key', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--work-dir', default=None, help='Directory for batch request/result files (default: <output>.batch_work/)')
    ap.add_argument('--status', action='store_true', help='Print batch status and exit')
    ap.add_argument('--poll-only', action='store_true', help='Poll + download without re-uploading')
    ap.add_argument('--max-items', type=int, default=None)
    ap.add_argument('--verbose', action='store_true', default=True)
    args = ap.parse_args()
    import openai
    client = openai.OpenAI(api_key=args.api_key)
    out_path = Path(args.output)
    work_dir = Path(args.work_dir) if args.work_dir else Path(str(out_path) + '.batch_work')
    state_path = work_dir / 'batch_state.json'
    work_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path)
    if args.status:
        print(f"Batches: {len(state['batches'])}")
        for b in state['batches']:
            print(f"  [{b['batch_idx']:05d}] {b['batch_id']}  status={b['status']}  n={b['n_items']:,}  completed={b['completed']}")
        return
    if not args.poll_only:
        done_ids = set(state.get('merged_ids', []))
        if out_path.exists() and (not done_ids):
            with open(out_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            done_ids.add(json.loads(line)['item_id'])
                        except Exception:
                            pass
        submitted_ids = {b['start_id'] for b in state['batches'] if b.get('uploaded')}

        def _gen():
            n_todo = 0
            with open(args.benchmark) as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item['item_id'] in done_ids:
                        continue
                    yield item
                    n_todo += 1
                    if args.max_items and n_todo >= args.max_items:
                        break
        print(f'[firm_run_batch] model={args.model}  benchmark={args.benchmark}', flush=True)
        print(f'[firm_run_batch] {len(done_ids):,} already merged, building batches...', flush=True)
        build_and_upload_batches(_gen(), client, args.model, state, state_path, work_dir, args.verbose)
    poll_batches(client, state, state_path, args.verbose)
    download_and_merge(client, state, state_path, out_path, work_dir, args.verbose)
    print(f'\n[firm_run_batch] Done → {out_path}')
if __name__ == '__main__':
    main()
