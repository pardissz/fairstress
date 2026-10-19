import json
import os
import sys
from collections import defaultdict
import torch
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer, generate_with_hook
from lib.hooks import AddVectorHook
from lib.prompt_adapter import load_jsonl
from lib.reasoning_capture import REASONING_SUFFIX, extract_subject, parse_response
from lib import concept_fit
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
OUT_DIR = 'firmcore_eval/reasoning'
OUT_PATH = os.path.join(OUT_DIR, '10_shifting_perspectives_reasoning.jsonl')
MAX_FIT_ITEMS_PER_AXIS = 800

def load_fit_items_by_axis():
    items = []
    for p in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        items.extend(concept_fit.load_jsonl(p))
    by_axis = defaultdict(list)
    for it in items:
        by_axis[it['contrast_category']].append(it)
    return by_axis

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, 'chosen_config.json')) as f:
        cfg = json.load(f)
    (layer, alpha) = (cfg['chosen_layer'], cfg['chosen_alpha'])
    print(f'chosen combo: layer={layer} alpha={alpha}', flush=True)
    firmcore_items = load_jsonl(FIRMCORE)
    fit_cap = MAX_FIT_ITEMS_PER_AXIS
    if os.environ.get('SMOKE_N'):
        firmcore_items = firmcore_items[:int(os.environ['SMOKE_N'])]
        fit_cap = 30
    print(f'firmcore={len(firmcore_items)}', flush=True)
    if os.path.exists(OUT_PATH) and sum((1 for _ in open(OUT_PATH))) >= len(firmcore_items):
        print('[skip] already complete', flush=True)
        return
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    print(f'[fit] re-fitting per-axis PCA vectors at layer {layer} ...', flush=True)
    by_axis = load_fit_items_by_axis()
    vecs = []
    for (axis, axis_items) in by_axis.items():
        axis_items = axis_items[:fit_cap]
        if len(axis_items) < 20:
            continue
        minority_ans = [f"Candidate {it['minority_slot']}" for it in axis_items]
        majority_other = {'A': 'B', 'B': 'A'}
        majority_ans = [f"Candidate {majority_other[it['minority_slot']]}" for it in axis_items]
        x_min = concept_fit._appended_answer_hidden_states(model, tok, axis_items, minority_ans, [layer])[layer]
        x_maj = concept_fit._appended_answer_hidden_states(model, tok, axis_items, majority_ans, [layer])[layer]
        vecs.append(concept_fit.contrastive_direction(x_min, x_maj, method='pca'))
    combined = torch.stack(vecs).mean(0)
    unit = (combined / (combined.norm() + 1e-08)).to(device=device, dtype=torch.float32)
    hook = AddVectorHook(unit, alpha)
    results = []
    batch_size = 32
    for i in range(0, len(firmcore_items), batch_size):
        batch = firmcore_items[i:i + batch_size]
        prompts = [it['prompt'] + REASONING_SUFFIX for it in batch]
        texts = generate_with_hook(model, tok, prompts, hook_obj=hook, layer_idx=layer, batch_size=batch_size, max_new_tokens=256)
        for (it, raw) in zip(batch, texts):
            subj = extract_subject(it['prompt'])
            (choice, reasoning) = parse_response(raw, subj)
            results.append({'item_id': it['item_id'], 'parsed_choice': choice, 'reasoning': reasoning, 'raw_text': raw})
        if i // batch_size % 20 == 0:
            print(f'  {i + len(batch)}/{len(firmcore_items)}', flush=True)
    with open(OUT_PATH, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {len(results)} -> {OUT_PATH}', flush=True)
    print('[done] method 10 reasoning capture complete', flush=True)
if __name__ == '__main__':
    main()
