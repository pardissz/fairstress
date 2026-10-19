import json
import torch
from lib.hf_backend import last_token_hidden_states, all_layer_last_token_hidden_states, all_layer_mean_pooled_hidden_states

def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def concept_a_firm(model, tok, layer_idx, paths=('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'), max_items=None):
    items = []
    for p in paths:
        items.extend(load_jsonl(p))
    if max_items:
        items = items[:max_items]
    prompts = [it['prompt'] for it in items]
    z = torch.tensor([1.0 if it['minority_slot'] == 'A' else 0.0 for it in items])
    x = last_token_hidden_states(model, tok, prompts, layer_idx)
    return (x, z, items)

def concept_a_bib(model, tok, layer_idx, path='external_data/bib_pool.jsonl', max_items=None):
    items = load_jsonl(path)
    if max_items:
        items = items[:max_items]
    prompts = [it['text'] for it in items]
    z = torch.tensor([float(it['z_label']) for it in items])
    x = last_token_hidden_states(model, tok, prompts, layer_idx)
    return (x, z, items)

def concept_a_firm_multilayer(model, tok, layer_indices, paths=('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'), max_items=None, batch_size=32):
    items = []
    for p in paths:
        items.extend(load_jsonl(p))
    if max_items:
        items = items[:max_items]
    prompts = [it['prompt'] for it in items]
    z = torch.tensor([1.0 if it['minority_slot'] == 'A' else 0.0 for it in items])
    x_by_layer = all_layer_last_token_hidden_states(model, tok, prompts, layer_indices, batch_size=batch_size)
    return (x_by_layer, z, items)

def concept_a_bib_multilayer(model, tok, layer_indices, path='external_data/bib_pool.jsonl', max_items=None, batch_size=32):
    items = load_jsonl(path)
    if max_items:
        items = items[:max_items]
    prompts = [it['text'] for it in items]
    z = torch.tensor([float(it['z_label']) for it in items])
    x_by_layer = all_layer_last_token_hidden_states(model, tok, prompts, layer_indices, batch_size=batch_size)
    return (x_by_layer, z, items)

@torch.no_grad()
def _appended_answer_hidden_states(model, tok, items, answer_texts, layer_indices, device='cuda', batch_size=32, max_length=3800):
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    feats = {li: [] for li in layer_indices}
    full_texts = []
    for (it, ans) in zip(items, answer_texts):
        prefix = tok.apply_chat_template([{'role': 'user', 'content': it['prompt']}], add_generation_prompt=True, tokenize=False)
        full_texts.append(prefix + ans)
    for i in range(0, len(full_texts), batch_size):
        batch = full_texts[i:i + batch_size]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
        out = model(**enc, output_hidden_states=True)
        for li in layer_indices:
            feats[li].append(out.hidden_states[li + 1][:, -1, :].float().cpu())
    return {li: torch.cat(v, dim=0) for (li, v) in feats.items()}

def concept_b_caa(model, tok, layer_indices, paths=('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'), max_items=None):
    items = []
    for p in paths:
        items.extend(load_jsonl(p))
    if max_items:
        items = items[:max_items]
    minority_ans = [f"Candidate {it['minority_slot']}" for it in items]
    majority_other = {'A': 'B', 'B': 'A'}
    majority_ans = [f"Candidate {majority_other[it['minority_slot']]}" for it in items]
    x_min = _appended_answer_hidden_states(model, tok, items, minority_ans, layer_indices)
    x_maj = _appended_answer_hidden_states(model, tok, items, majority_ans, layer_indices)
    return (x_min, x_maj, items)

def contrastive_direction(x_pos: torch.Tensor, x_neg: torch.Tensor, method='meandiff'):
    if method == 'meandiff':
        return x_pos.mean(0) - x_neg.mean(0)
    if method == 'pca':
        mu = (x_pos.mean(0) + x_neg.mean(0)) / 2.0
        (cp, cn) = (x_pos - mu, x_neg - mu)
        n = min(len(cp), len(cn))
        interleaved = torch.empty(2 * n, x_pos.shape[1], dtype=torch.float64)
        interleaved[0::2] = cp[:n].double()
        interleaved[1::2] = cn[:n].double()
        (_, _, Vt) = torch.linalg.svd(interleaved, full_matrices=False)
        pc1 = Vt[0].float()
        meandiff = x_pos.mean(0) - x_neg.mean(0)
        if torch.dot(pc1, meandiff) < 0:
            pc1 = -pc1
        return pc1
    raise ValueError(f'unknown method {method!r}')

def condition_vector_firmcore_multilayer(model, tok, layer_indices, firmcore_path='./data/lexicon_scoring/firmCore.jsonl', max_items_per_class=2000, batch_size=32):
    (pos_items, neg_items) = ([], [])
    with open(firmcore_path) as f:
        for line in f:
            d = json.loads(line)
            if d['signaling_level'] >= 2 and len(pos_items) < max_items_per_class:
                pos_items.append(d)
            elif d['signaling_level'] == 0 and len(neg_items) < max_items_per_class:
                neg_items.append(d)
            if len(pos_items) >= max_items_per_class and len(neg_items) >= max_items_per_class:
                break
    pos_prompts = [d['prompt'] for d in pos_items]
    neg_prompts = [d['prompt'] for d in neg_items]
    x_pos = all_layer_mean_pooled_hidden_states(model, tok, pos_prompts, layer_indices, batch_size=batch_size)
    x_neg = all_layer_mean_pooled_hidden_states(model, tok, neg_prompts, layer_indices, batch_size=batch_size)
    return (x_pos, x_neg)
