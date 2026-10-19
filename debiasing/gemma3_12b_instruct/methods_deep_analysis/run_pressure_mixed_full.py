import json
import os
import sys
import time
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)
from lib.hf_backend import load_model_and_tokenizer, get_hidden_size
from lib.hooks import InterventionModuleHook
from lib.loreft import LoreftIntervention, train_loreft
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import run_and_score, write_jsonl
import loreft_lab as lab
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
LAYER = 16
RANK = 4
OUT_DIR = os.path.join(PROJECT_ROOT, 'firmcore_eval', 'responses')
OUT_NAME = '11_pressure_mixed_L16_r4_firmcore.jsonl'
METHOD_DIR = os.path.join(PROJECT_ROOT, 'methods', '11_pressure_mixed')
LOCK_DIR = os.path.join(PROJECT_ROOT, 'logs', '.lock_11_pressure_mixed')

def main():
    try:
        os.makedirs(LOCK_DIR)
    except FileExistsError:
        print(f'[pressure_mixed_full] {LOCK_DIR} already claimed -- skipping (another stream already ran or is running this method)', flush=True)
        return
    try:
        _run(t0=time.time())
    except BaseException:
        try:
            os.rmdir(LOCK_DIR)
        except OSError:
            pass
        raise

def _run(t0):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(METHOD_DIR, exist_ok=True)
    firmcore_items = load_jsonl(FIRMCORE)
    print(f'[pressure_mixed_full] firmcore={len(firmcore_items)}', flush=True)
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    print('[pressure_mixed_full] loading training data (50/50 plain+pressure, 7/9 families, F7/F9 held out) ...', flush=True)
    train_items = lab.load_training_items(with_pressure=True, pressure_frac=0.5)
    print('[pressure_mixed_full] training LoReFT L16_r4 ...', flush=True)
    module = LoreftIntervention(get_hidden_size(model), RANK)
    train_loreft(model, tok, module, LAYER, train_items, device, epochs=2, batch_size=4, lr=0.001)
    hook = InterventionModuleHook(module)
    print(f'[pressure_mixed_full] training done, {time.time() - t0:.1f}s elapsed', flush=True)
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    if os.path.exists(out_path) and sum((1 for _ in open(out_path))) >= len(firmcore_items):
        print(f'[skip] {out_path} already complete', flush=True)
    else:
        print(f'[pressure_mixed_full] generating over full firmCore ({len(firmcore_items)} items) ...', flush=True)
        t_gen0 = time.time()
        resp = run_and_score(model, tok, firmcore_items, hook, LAYER)
        write_jsonl(out_path, resp)
        print(f'[pressure_mixed_full] wrote {len(resp)} -> {out_path}, gen took {time.time() - t_gen0:.1f}s', flush=True)
    chosen_config = {'method': '11_pressure_mixed', 'chosen_combo': 'L16_r4_pressuremix50', 'chosen_layer': LAYER, 'chosen_rank': RANK, 'pressure_frac': 0.5, 'pressure_train_families': ['F1_authority', 'F2_social_consensus', 'F3_role_assignment', 'F5_guilt_historical', 'F6_evidential_claims', 'F8_legal_framing', 'F10_demographic_salience'], 'pressure_heldout_families': ['F7_emotional_appeals', 'F9_statistical_framing'], 'firmcore_response_file': OUT_NAME, 'note': 'Injection-TEXT exposure to 7/9 pressure families during training (never to firmCore ITEMS -- fit_bbq/fit_synth verified disjoint from firmCore). Read FAS via by_injection_family: F7/F9 are the fair zero-exposure comparison point against the other 9 methods in this study; full-family aggregate includes 7 inoculated-against families.'}
    with open(os.path.join(METHOD_DIR, 'chosen_config.json'), 'w') as f:
        json.dump(chosen_config, f, indent=2)
    print(f'[pressure_mixed_full] done, total elapsed {time.time() - t0:.1f}s', flush=True)
if __name__ == '__main__':
    main()
