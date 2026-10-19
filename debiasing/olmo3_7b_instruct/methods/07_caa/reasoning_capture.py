import json
import os
import sys
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
OUT_PATH = os.path.join(OUT_DIR, '07_caa_reasoning.jsonl')
MAX_FIT_ITEMS = 4000

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, 'chosen_config.json')) as f:
        cfg = json.load(f)
    (layer, alpha_frac) = (cfg['chosen_layer'], cfg['chosen_alpha_frac'])
    print(f'chosen combo: layer={layer} alpha_frac={alpha_frac}', flush=True)
    firmcore_items = load_jsonl(FIRMCORE)
    fit_cap = MAX_FIT_ITEMS
    if os.environ.get('SMOKE_N'):
        firmcore_items = firmcore_items[:int(os.environ['SMOKE_N'])]
        fit_cap = 200
    print(f'firmcore={len(firmcore_items)}', flush=True)
    if os.path.exists(OUT_PATH) and sum((1 for _ in open(OUT_PATH))) >= len(firmcore_items):
        print('[skip] already complete', flush=True)
        return
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    print(f'[fit] re-fitting steering vector at layer {layer} ...', flush=True)
    (x_min_by_layer, x_maj_by_layer, _) = concept_fit.concept_b_caa(model, tok, [layer], max_items=fit_cap)
    (x_min, x_maj) = (x_min_by_layer[layer], x_maj_by_layer[layer])
    v = x_min.mean(0) - x_maj.mean(0)
    unit = (v / (v.norm() + 1e-08)).to(device=device, dtype=torch.float32)
    ref_norm = torch.cat([x_min, x_maj], dim=0).norm(dim=-1).mean().item()
    alpha = alpha_frac * ref_norm
    print(f'  ||v||={v.norm().item():.3f} ref_norm={ref_norm:.3f} alpha={alpha:.3f}', flush=True)
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
    print('[done] method 7 reasoning capture complete', flush=True)
if __name__ == '__main__':
    main()
