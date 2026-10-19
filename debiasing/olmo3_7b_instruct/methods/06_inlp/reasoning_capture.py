import json
import os
import sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer, generate_with_hook
from lib.hooks import ProjectionHook
from lib.prompt_adapter import load_jsonl
from lib.reasoning_capture import REASONING_SUFFIX, extract_subject, parse_response
from lib import concept_fit
import importlib.util
spec = importlib.util.spec_from_file_location('m6_run_all', os.path.join(HERE, 'run_all.py'))
m6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m6)
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
OUT_DIR = 'firmcore_eval/reasoning'
OUT_PATH = os.path.join(OUT_DIR, '06_inlp_reasoning.jsonl')
MAX_FIT_ITEMS = 4000

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, 'chosen_config.json')) as f:
        cfg = json.load(f)
    (corpus, layer, n_iter) = (cfg['chosen_corpus'], cfg['chosen_layer'], cfg['chosen_n_iter'])
    print(f'chosen combo: corpus={corpus} layer={layer} n_iter={n_iter}', flush=True)
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
    dtype = next(model.parameters()).dtype
    print(f'[fit] re-fitting {corpus} INLP at layer {layer} k={n_iter} ...', flush=True)
    if corpus == 'firm':
        (x_by_layer, z, _) = concept_fit.concept_a_firm_multilayer(model, tok, [layer], max_items=fit_cap)
    else:
        (x_by_layer, z, _) = concept_fit.concept_a_bib_multilayer(model, tok, [layer], max_items=fit_cap)
    ws = m6.fit_inlp(x_by_layer[layer], z, n_iter)
    apply_fn = m6.inlp_apply_fn(ws, device, dtype)
    hook = ProjectionHook(apply_fn)
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
    print('[done] method 6 reasoning capture complete', flush=True)
if __name__ == '__main__':
    main()
