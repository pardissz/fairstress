import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.prompt_adapter import MODEL_PATH

def load_model_and_tokenizer(device='cuda'):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.eval()
    return (model, tok)

def chat_prompts(tok, raw_prompts):
    out = []
    for p in raw_prompts:
        try:
            out.append(tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, tokenize=False))
        except Exception:
            out.append(p)
    return out

@torch.no_grad()
def last_token_hidden_states(model, tok, raw_prompts, layer_idx, device='cuda', batch_size=32, max_length=3800):
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    feats = []
    prompts = chat_prompts(tok, raw_prompts)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer_idx + 1][:, -1, :]
        feats.append(hs.float().cpu())
    return torch.cat(feats, dim=0)

@torch.no_grad()
def all_layer_last_token_hidden_states(model, tok, raw_prompts, layer_indices, device='cuda', batch_size=32, max_length=3800):
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    feats = {li: [] for li in layer_indices}
    prompts = chat_prompts(tok, raw_prompts)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
        out = model(**enc, output_hidden_states=True)
        for li in layer_indices:
            feats[li].append(out.hidden_states[li + 1][:, -1, :].float().cpu())
    return {li: torch.cat(v, dim=0) for (li, v) in feats.items()}

@torch.no_grad()
def all_layer_mean_pooled_hidden_states(model, tok, raw_prompts, layer_indices, device='cuda', batch_size=32, max_length=3800):
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    feats = {li: [] for li in layer_indices}
    prompts = chat_prompts(tok, raw_prompts)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
        out = model(**enc, output_hidden_states=True)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        for li in layer_indices:
            hs = out.hidden_states[li + 1].float()
            pooled = (hs * mask).sum(dim=1) / denom
            feats[li].append(pooled.cpu())
    return {li: torch.cat(v, dim=0) for (li, v) in feats.items()}

def sentence_logprob_with_hook(model, tok, sentences, hook_obj=None, layer_idx=None, device='cuda', batch_size=16, max_length=512):
    from lib.hooks import register_hook, remove_hooks
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    handle = register_hook(model, layer_idx, hook_obj) if hook_obj is not None else None
    results = []
    try:
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                out = model(**enc)
            log_probs = torch.log_softmax(out.logits.float(), dim=-1)
            input_ids = enc['input_ids']
            attn = enc['attention_mask'].float()
            shift_logprobs = log_probs[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attn[:, 1:]
            token_logprobs = shift_logprobs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            seq_sum = (token_logprobs * shift_mask).sum(dim=1)
            results.extend(seq_sum.cpu().tolist())
    finally:
        if handle is not None:
            remove_hooks([handle])
    return results

@torch.no_grad()
def next_token_yesno_prob_with_hook(model, tok, raw_prompts, hook_obj=None, layer_idx=None, device='cuda', batch_size=32, max_length=3800):
    from lib.hooks import register_hook, remove_hooks
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    (yes_ids, no_ids) = ([], [])
    for tid in range(len(tok)):
        dec = tok.decode([tid]).strip().lower()
        if dec.startswith('yes'):
            yes_ids.append(tid)
        elif dec.startswith('no'):
            no_ids.append(tid)
    yes_ids_t = torch.tensor(yes_ids, device=device)
    no_ids_t = torch.tensor(no_ids, device=device)
    prompts = chat_prompts(tok, raw_prompts)
    handle = register_hook(model, layer_idx, hook_obj) if hook_obj is not None else None
    results = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
            out = model(**enc)
            probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
            p_yes = probs.index_select(1, yes_ids_t).sum(dim=1)
            p_no = probs.index_select(1, no_ids_t).sum(dim=1)
            total = p_yes + p_no
            p = torch.where(total > 0, p_yes / total, torch.full_like(p_yes, float('nan')))
            results.extend(p.cpu().tolist())
    finally:
        if handle is not None:
            remove_hooks([handle])
    return [None if v != v else v for v in results]

@torch.no_grad()
def next_token_choice_logprobs_with_hook(model, tok, raw_prompts, choices=('A', 'B', 'C', 'D'), hook_obj=None, layer_idx=None, device='cuda', batch_size=32, max_length=3800):
    from lib.hooks import register_hook, remove_hooks
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    choice_ids = {c: [] for c in choices}
    for tid in range(len(tok)):
        dec = tok.decode([tid]).strip().lower()
        for c in choices:
            if dec == c.lower():
                choice_ids[c].append(tid)
    choice_id_tensors = {c: torch.tensor(ids, device=device) if ids else None for (c, ids) in choice_ids.items()}
    prompts = chat_prompts(tok, raw_prompts)
    handle = register_hook(model, layer_idx, hook_obj) if hook_obj is not None else None
    results = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
            out = model(**enc)
            probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
            per_choice = {}
            for (c, ids_t) in choice_id_tensors.items():
                if ids_t is not None and ids_t.numel() > 0:
                    per_choice[c] = probs.index_select(1, ids_t).sum(dim=1)
                else:
                    per_choice[c] = torch.zeros(probs.shape[0], device=device)
            stacked = torch.stack([per_choice[c] for c in choices], dim=1)
            total = stacked.sum(dim=1, keepdim=True)
            norm = torch.where(total > 0, stacked / total, torch.full_like(stacked, float('nan')))
            for row in norm.cpu().tolist():
                results.append({c: None if v != v else v for (c, v) in zip(choices, row)})
    finally:
        if handle is not None:
            remove_hooks([handle])
    return results

@torch.no_grad()
def generate_with_hook(model, tok, raw_prompts, hook_obj=None, layer_idx=None, device='cuda', batch_size=64, max_new_tokens=32):
    from lib.hooks import register_hook, remove_hooks
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = chat_prompts(tok, raw_prompts)
    out_texts = []
    handle = None
    if hook_obj is not None:
        handle = register_hook(model, layer_idx, hook_obj)
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=3800).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            new_tokens = gen[:, enc['input_ids'].shape[1]:]
            texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
            out_texts.extend(texts)
    finally:
        if handle is not None:
            remove_hooks([handle])
    return out_texts
