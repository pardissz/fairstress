import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.hooks import register_hook, remove_hooks

class LoreftIntervention(nn.Module):

    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.rank = rank
        self.R = nn.Linear(hidden_dim, rank, bias=False)
        nn.init.orthogonal_(self.R.weight)
        self.W = nn.Linear(rank, rank, bias=True)

    def forward(self, h):
        Rh = self.R(h)
        Wh = self.W(Rh)
        delta = Wh - Rh
        return h + delta @ self.R.weight

class _TrainHook:

    def __init__(self, module):
        self.module = module
        self.positions = None

    def __call__(self, mod, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        b = hs.shape[0]
        idx = self.positions.to(hs.device)
        ar = torch.arange(b, device=hs.device)
        gathered = hs[ar, idx]
        new_vals = self.module(gathered.float()).to(hs.dtype)
        new_hs = hs.clone()
        new_hs[ar, idx] = new_vals
        return (new_hs,) + output[1:] if is_tuple else new_hs

def train_loreft(model, tok, module, layer_idx, items, device, epochs=3, batch_size=16, lr=0.001, max_length=1024, log_every=50):
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
    items = list(items)
    losses = []
    step = 0
    try:
        for epoch in range(epochs):
            random.shuffle(items)
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
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
                prompt_lens = [min(pl, maxlen - 1) for pl in prompt_lens]
                (input_ids, labels, attn) = (input_ids.to(device), labels.to(device), attn.to(device))
                hook_obj.positions = torch.tensor(prompt_lens, dtype=torch.long)
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), shift_labels.reshape(-1), ignore_index=-100)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                step += 1
                if step % log_every == 0:
                    print(f'    [loreft train] step={step} loss={loss.item():.4f}', flush=True)
    finally:
        remove_hooks([handle])
    module.eval()
    return losses
