import json
import os
import random
import sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer, generate_with_hook
from lib.hooks import InterventionModuleHook
from lib.loreft import LoreftIntervention, train_loreft
from lib.prompt_adapter import load_jsonl
from lib.reasoning_capture import REASONING_SUFFIX, extract_subject, parse_response
from lib.concept_fit import load_jsonl as load_jsonl_cf
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
OUT_DIR = 'firmcore_eval/reasoning'
OUT_PATH = os.path.join(OUT_DIR, '09_loreft_reasoning.jsonl')
MAX_TRAIN_ITEMS = 6000
EPOCHS = 2
BATCH_SIZE = 16
LR = 0.001

def load_training_items():
    items = []
    for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        for it in load_jsonl_cf(path):
            if it['condition'] == 'GT' and it.get('expected_correct'):
                items.append({'prompt': it['prompt'], 'target': it['expected_correct']})
    random.Random(0).shuffle(items)
    return items[:MAX_TRAIN_ITEMS]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, 'chosen_config.json')) as f:
        cfg = json.load(f)
    (layer, rank) = (cfg['chosen_layer'], cfg['chosen_rank'])
    print(f'chosen combo: layer={layer} rank={rank}', flush=True)
    firmcore_items = load_jsonl(FIRMCORE)
    train_items = load_training_items()
    epochs = EPOCHS
    if os.environ.get('SMOKE_N'):
        firmcore_items = firmcore_items[:int(os.environ['SMOKE_N'])]
        train_items = train_items[:40]
        epochs = 1
    print(f'firmcore={len(firmcore_items)} train={len(train_items)}', flush=True)
    if os.path.exists(OUT_PATH) and sum((1 for _ in open(OUT_PATH))) >= len(firmcore_items):
        print('[skip] already complete', flush=True)
        return
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    print(f'[train] retraining rank={rank} at layer={layer} ...', flush=True)
    module = LoreftIntervention(hidden_dim, rank)
    losses = train_loreft(model, tok, module, layer, train_items, device, epochs=epochs, batch_size=BATCH_SIZE, lr=LR)
    print(f'  first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f}', flush=True)
    hook = InterventionModuleHook(module)
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
    print('[done] method 9 reasoning capture complete', flush=True)
if __name__ == '__main__':
    main()
