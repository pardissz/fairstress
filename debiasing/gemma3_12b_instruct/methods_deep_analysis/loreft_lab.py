import copy
import json
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
from lib.hooks import register_hook, remove_hooks
from lib import concept_fit
INJECTIONS_PATH = './data/injections.json'
FIT_PATHS = (os.path.join(PROJECT_ROOT, 'external_data/fit_bbq.jsonl'), os.path.join(PROJECT_ROOT, 'external_data/fit_synth.jsonl'))
CALIB = os.path.join(PROJECT_ROOT, 'external_data/calibration_set.jsonl')
ANSWER_LINE = 'Answer with only "Candidate A" or "Candidate B".'
EVAL_ONLY_FAMILIES = {'F7_emotional_appeals', 'F9_statistical_framing'}

def load_injection_bank(split='train'):
    with open(INJECTIONS_PATH) as f:
        inj = json.load(f)
    by_dir = {'pro_A': [], 'pro_B': []}
    for i in inj:
        if i['class'] != 'directional':
            continue
        is_eval_fam = i['family'] in EVAL_ONLY_FAMILIES
        if split == 'train' and is_eval_fam:
            continue
        if split == 'eval' and (not is_eval_fam):
            continue
        by_dir[i['direction']].append(i['text'])
    return by_dir

def inject_wrong_direction(prompt, expected_correct, rng, bank):
    wrong_letter = 'B' if expected_correct == 'Candidate A' else 'A'
    direction = f'pro_{wrong_letter}'
    text = rng.choice(bank[direction])
    note = f'Note: {text}\n'
    if ANSWER_LINE in prompt:
        return prompt.replace(ANSWER_LINE, note + ANSWER_LINE)
    return prompt.rstrip() + '\n\n' + note + ANSWER_LINE

def load_training_items(with_pressure=False, pressure_frac=1.0, seed=0):
    items = []
    for path in FIT_PATHS:
        for it in concept_fit.load_jsonl(path):
            if it['condition'] == 'GT' and it.get('expected_correct'):
                items.append({'prompt': it['prompt'], 'target': it['expected_correct']})
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:6000]
    if with_pressure:
        bank = load_injection_bank(split='train')
        rng2 = random.Random(seed + 1)
        n_pressure = int(len(items) * pressure_frac)
        out = []
        for (idx, it) in enumerate(items):
            if idx < n_pressure:
                out.append({'prompt': inject_wrong_direction(it['prompt'], it['target'], rng2, bank), 'target': it['target']})
            else:
                out.append(it)
        rng.shuffle(out)
        return out
    return items

def load_neutral_items(n=2000, seed=0):
    items = []
    for path in FIT_PATHS:
        for it in concept_fit.load_jsonl(path):
            if it['condition'] == 'AMB':
                items.append({'prompt': it['prompt']})
    random.Random(seed).shuffle(items)
    return items[:n]

def load_calibration_items():
    return concept_fit.load_jsonl(CALIB)

def build_injected_calibration(seed=0, split='eval'):
    calib = load_calibration_items()
    bank = load_injection_bank(split=split)
    rng = random.Random(seed + 2)
    out = []
    for it in calib:
        if it['condition'] == 'GT' and it.get('expected_correct'):
            it2 = copy.copy(it)
            it2['prompt'] = inject_wrong_direction(it['prompt'], it['expected_correct'], rng, bank)
            out.append(it2)
    return out

class LoreftInterventionNonlinear(nn.Module):

    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.rank = rank
        self.R = nn.Linear(hidden_dim, rank, bias=False)
        nn.init.orthogonal_(self.R.weight)
        self.W = nn.Linear(rank, rank, bias=True)
        self.act = nn.Tanh()

    def forward(self, h):
        Rh = self.R(h)
        Wh = self.act(self.W(Rh))
        delta = Wh - Rh
        return h + delta @ self.R.weight

def train_loreft_v2(model, tok, module, layer_idx, train_items, device, neutral_items=None, noop_lambda=0.0, epochs=2, batch_size=16, lr=0.001, max_length=1024, log_every=50):
    from lib.loreft import _TrainHook
    module.to(device)
    module.train()
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(module.parameters(), lr=lr)
    hook_obj = _TrainHook(module)
    handle = register_hook(model, layer_idx, hook_obj)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_noop = neutral_items is not None and noop_lambda > 0
    train_items = list(train_items)
    neutral_items = list(neutral_items) if use_noop else None
    losses = []
    step = 0
    try:
        for epoch in range(epochs):
            random.shuffle(train_items)
            if use_noop:
                random.shuffle(neutral_items)
            for i in range(0, len(train_items), batch_size):
                batch = train_items[i:i + batch_size]
                (input_ids_list, labels_list, prompt_lens) = ([], [], [])
                for it in batch:
                    prefix = tok.apply_chat_template([{'role': 'user', 'content': it['prompt']}], add_generation_prompt=True, tokenize=False)
                    target = ' ' + it['target']
                    prefix_ids = tok(prefix, add_special_tokens=False)['input_ids']
                    target_ids = tok(target, add_special_tokens=False)['input_ids']
                    ids = prefix_ids + target_ids
                    labs = [-100] * len(prefix_ids) + target_ids
                    input_ids_list.append(ids)
                    labels_list.append(labs)
                    prompt_lens.append(len(prefix_ids) - 1)
                maxlen = min(max_length, max((len(x) for x in input_ids_list)))
                bsz = len(batch)
                input_ids = torch.full((bsz, maxlen), tok.pad_token_id, dtype=torch.long)
                labels = torch.full((bsz, maxlen), -100, dtype=torch.long)
                attn = torch.zeros((bsz, maxlen), dtype=torch.long)
                for (j, (ids, labs)) in enumerate(zip(input_ids_list, labels_list)):
                    (ids, labs) = (ids[:maxlen], labs[:maxlen])
                    input_ids[j, :len(ids)] = torch.tensor(ids)
                    labels[j, :len(labs)] = torch.tensor(labs)
                    attn[j, :len(ids)] = 1
                prompt_lens_c = [min(pl, maxlen - 1) for pl in prompt_lens]
                (input_ids, labels, attn) = (input_ids.to(device), labels.to(device), attn.to(device))
                hook_obj.positions = torch.tensor(prompt_lens_c, dtype=torch.long)
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), shift_labels.reshape(-1), ignore_index=-100)
                loss = ce_loss
                noop_term = None
                if use_noop:
                    nbatch = neutral_items[i // batch_size * batch_size % max(1, len(neutral_items) - batch_size):][:batch_size]
                    if not nbatch:
                        nbatch = neutral_items[:batch_size]
                    n_prefixes = [tok.apply_chat_template([{'role': 'user', 'content': it['prompt']}], add_generation_prompt=True, tokenize=False) for it in nbatch]
                    n_ids = [tok(p, add_special_tokens=False)['input_ids'] for p in n_prefixes]
                    n_maxlen = min(max_length, max((len(x) for x in n_ids)))
                    n_bsz = len(nbatch)
                    n_input_ids = torch.full((n_bsz, n_maxlen), tok.pad_token_id, dtype=torch.long)
                    n_attn = torch.zeros((n_bsz, n_maxlen), dtype=torch.long)
                    n_lens = []
                    for (j, ids) in enumerate(n_ids):
                        ids = ids[:n_maxlen]
                        n_input_ids[j, :len(ids)] = torch.tensor(ids)
                        n_attn[j, :len(ids)] = 1
                        n_lens.append(min(len(ids) - 1, n_maxlen - 1))
                    (n_input_ids, n_attn) = (n_input_ids.to(device), n_attn.to(device))
                    remove_hooks([handle])
                    try:
                        n_out = model(input_ids=n_input_ids, attention_mask=n_attn, output_hidden_states=True)
                    finally:
                        handle = register_hook(model, layer_idx, hook_obj)
                    hs_clean = n_out.hidden_states[layer_idx + 1]
                    ar = torch.arange(n_bsz, device=device)
                    idx_t = torch.tensor(n_lens, device=device)
                    h_clean = hs_clean[ar, idx_t].float()
                    delta = module(h_clean) - h_clean
                    noop_term = delta.pow(2).sum(-1).mean()
                    loss = ce_loss + noop_lambda * noop_term
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(ce_loss.item())
                step += 1
                if step % log_every == 0:
                    extra = f' noop={noop_term.item():.4f}' if noop_term is not None else ''
                    print(f'    [train] step={step} ce_loss={ce_loss.item():.4f}{extra}', flush=True)
    finally:
        remove_hooks([handle])
    module.eval()
    return losses
