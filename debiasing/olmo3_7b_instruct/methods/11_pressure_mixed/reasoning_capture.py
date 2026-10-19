import json
import os
import sys
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'methods_deep_analysis'))
from lib.hf_backend import load_model_and_tokenizer, generate_with_hook
from lib.hooks import InterventionModuleHook
from lib.loreft import LoreftIntervention, train_loreft
from lib.prompt_adapter import load_jsonl
from lib.reasoning_capture import REASONING_SUFFIX, extract_subject, parse_response
import loreft_lab as lab
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
OUT_DIR = os.path.join(PROJECT_ROOT, 'firmcore_eval', 'reasoning')
OUT_PATH = os.path.join(OUT_DIR, '11_pressure_mixed_reasoning.jsonl')
EPOCHS = 2
BATCH_SIZE = 16
LR = 0.001

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(HERE, 'chosen_config.json')) as f:
        cfg = json.load(f)
    (layer, rank) = (cfg['chosen_layer'], cfg['chosen_rank'])
    pressure_frac = cfg['pressure_frac']
    print(f'chosen combo: layer={layer} rank={rank} pressure_frac={pressure_frac}', flush=True)
    firmcore_items = load_jsonl(FIRMCORE)
    train_items = lab.load_training_items(with_pressure=True, pressure_frac=pressure_frac)
    epochs = EPOCHS
    if os.environ.get('SMOKE_N'):
        firmcore_items = firmcore_items[:int(os.environ['SMOKE_N'])]
        train_items = train_items[:40]
        epochs = 1
    print(f'firmcore={len(firmcore_items)} train={len(train_items)}', flush=True)
    done_ids = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)['item_id'])
    remaining = [it for it in firmcore_items if it['item_id'] not in done_ids]
    print(f'[resume] {len(done_ids):,} already done, {len(remaining):,} remaining', flush=True)
    if not remaining:
        print('[skip] already complete', flush=True)
        return
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    print(f'[train] retraining rank={rank} at layer={layer}, pressure_frac={pressure_frac} ...', flush=True)
    module = LoreftIntervention(hidden_dim, rank)
    losses = train_loreft(model, tok, module, layer, train_items, device, epochs=epochs, batch_size=BATCH_SIZE, lr=LR)
    print(f'  first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f}', flush=True)
    hook = InterventionModuleHook(module)
    batch_size = 32
    with open(OUT_PATH, 'a') as f:
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i + batch_size]
            prompts = [it['prompt'] + REASONING_SUFFIX for it in batch]
            texts = generate_with_hook(model, tok, prompts, hook_obj=hook, layer_idx=layer, batch_size=batch_size, max_new_tokens=256)
            for (it, raw) in zip(batch, texts):
                subj = extract_subject(it['prompt'])
                (choice, reasoning) = parse_response(raw, subj)
                f.write(json.dumps({'item_id': it['item_id'], 'parsed_choice': choice, 'reasoning': reasoning, 'raw_text': raw}) + '\n')
            f.flush()
            if i // batch_size % 20 == 0:
                print(f'  {i + len(batch)}/{len(remaining)}', flush=True)
    n_final = sum((1 for _ in open(OUT_PATH)))
    print(f'wrote total {n_final} -> {OUT_PATH}', flush=True)
    print('[done] method 11 reasoning capture complete', flush=True)
if __name__ == '__main__':
    main()
