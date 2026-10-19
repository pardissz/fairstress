import argparse
import json
import os
import subprocess
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
DEBIAS_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
AI2_ROOT = os.path.abspath(os.path.join(DEBIAS_ROOT, '..'))
sys.path.insert(0, DEBIAS_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(DEBIAS_ROOT, 'methods_deep_analysis'))
sys.path.insert(0, os.path.join(DEBIAS_ROOT, 'methods_deep_analysis', 'capability_benchmarks'))
from lib.hf_backend import load_model_and_tokenizer, generate_with_hook, next_token_choice_logprobs_with_hook
from lib.hooks import GatedInterventionModuleHook
from lib.loreft import LoreftIntervention, train_loreft
from lib.prompt_adapter import load_jsonl
from lib.concept_fit import load_jsonl as load_jsonl_cf
from lib.repr_edit_common import write_jsonl
from lib.prompt_adapter import extract_subject, parse_choice
import loreft_lab as lab
import gate as gate_mod
from data import load_mmlu, load_arc_challenge, format_mc_prompt, LETTERS
import gsm8k_util as gk
LAYER = 16
RANK = 4
FIRMCORE = os.path.join(AI2_ROOT, 'lexicon_scoring', 'firmCore.jsonl')

def load_plain_training_items(max_items=6000):
    items = []
    for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        for it in load_jsonl_cf(os.path.join(DEBIAS_ROOT, path)):
            if it['condition'] == 'GT' and it.get('expected_correct'):
                items.append({'prompt': it['prompt'], 'target': it['expected_correct']})
    import random
    random.Random(0).shuffle(items)
    return items[:max_items]

def run_and_score_gated(model, tok, items, hook, layer, batch_size=64, max_new_tokens=32):
    prompts = [it['prompt'] for it in items]
    texts = generate_with_hook(model, tok, prompts, hook_obj=hook, layer_idx=layer, batch_size=batch_size, max_new_tokens=max_new_tokens)
    results = []
    for (it, raw) in zip(items, texts):
        subj = extract_subject(it['prompt'])
        choice = parse_choice(raw, subj)
        results.append({'item_id': it['item_id'], 'parsed_choice': choice, 'raw_text': raw, 'logprob_A': None, 'logprob_B': None})
    return results

def mc_score(model, tok, items, hook, layer):
    max_n_choices = max((len(it['choices']) for it in items))
    choices = tuple(LETTERS[:max_n_choices])
    prompts = [format_mc_prompt(it['question'], it['choices']) for it in items]
    probs = next_token_choice_logprobs_with_hook(model, tok, prompts, choices=choices, hook_obj=hook, layer_idx=layer, batch_size=24)
    n_correct = n_parsed = 0
    for (it, p) in zip(items, probs):
        valid = {k: v for (k, v) in p.items() if k in LETTERS[:len(it['choices'])] and v is not None}
        if not valid:
            continue
        pred = max(valid, key=valid.get)
        n_parsed += 1
        if pred == it['gold_letter']:
            n_correct += 1
    n = len(items)
    return {'n': n, 'n_parsed': n_parsed, 'parse_rate': n_parsed / n if n else 0.0, 'accuracy': n_correct / n if n else 0.0}

def gsm8k_score(model, tok, items, hook, layer):
    prompts = [it['question'] + gk.ANSWER_SUFFIX for it in items]
    texts = generate_with_hook(model, tok, prompts, hook_obj=hook, layer_idx=layer, batch_size=24, max_new_tokens=320)
    return gk.score_gsm8k(items, texts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['plain', 'pressure'], required=True)
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()
    method_name = '12_gated_loreft' if args.variant == 'plain' else '13_gated_pressure_mixed'
    out_dir = os.path.join(HERE, method_name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    os.chdir(DEBIAS_ROOT)
    print(f'[{method_name}] loading model ...', flush=True)
    (model, tok) = load_model_and_tokenizer()
    device = next(model.parameters()).device
    gate_out_dir = out_dir
    gate_weights_path = os.path.join(gate_out_dir, 'gate_weights.pt')
    if os.path.exists(gate_weights_path):
        print(f'[{method_name}] reusing existing gate weights at {gate_weights_path}', flush=True)
        import torch
        d = torch.load(gate_weights_path, map_location=device)
        (gate_w, gate_b, gate_layer) = (d['w'].to(device), d['b'], d['layer'])
        with open(os.path.join(gate_out_dir, 'gate_validation_report.json')) as f:
            gate_report = json.load(f)
    else:
        (gate_w, gate_b, gate_report) = gate_mod.train_and_validate_gate(model, tok, layer=LAYER, out_dir=gate_out_dir)
        gate_layer = LAYER
    if args.variant == 'plain':
        train_items = load_plain_training_items()
        note = 'plain GT-only training, same recipe as methods/09_loreft'
    else:
        train_items = lab.load_training_items(with_pressure=True, pressure_frac=0.5)
        note = '50/50 plain+pressure-injected training, same recipe as methods/11_pressure_mixed (7/9 families, F7/F9 held out)'
    epochs = 2
    if os.environ.get('SMOKE_N'):
        train_items = train_items[:60]
        epochs = 1
    print(f'[{method_name}] training LoReFT ({note}), n_train={len(train_items)} ...', flush=True)
    module = LoreftIntervention(model.config.hidden_size, RANK)
    losses = train_loreft(model, tok, module, LAYER, train_items, device, epochs=epochs, batch_size=16, lr=0.001)
    print(f'[{method_name}] train done: first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f}, {time.time() - t0:.1f}s elapsed', flush=True)
    hook = GatedInterventionModuleHook(module, gate_w, gate_b, threshold=args.threshold)
    firmcore_items = load_jsonl(FIRMCORE)
    if os.environ.get('SMOKE_N'):
        firmcore_items = firmcore_items[:int(os.environ['SMOKE_N'])]
    print(f'[{method_name}] generating over full firmCore ({len(firmcore_items)} items) ...', flush=True)
    (hook.n_fired, hook.n_total) = (0, 0)
    t_gen0 = time.time()
    resp = run_and_score_gated(model, tok, firmcore_items, hook, LAYER)
    firmcore_response_path = os.path.join(out_dir, f'{method_name}_firmcore.jsonl')
    write_jsonl(firmcore_response_path, resp)
    firing_firmcore = {'n_fired': hook.n_fired, 'n_total': hook.n_total, 'rate': hook.n_fired / hook.n_total if hook.n_total else None}
    print(f'[{method_name}] firmCore done, {time.time() - t_gen0:.1f}s, firing={firing_firmcore}', flush=True)
    scores_path = os.path.join(out_dir, f'{method_name}_scores.json')
    n_boot = '50' if os.environ.get('SMOKE_N') else '2000'
    subprocess.run([sys.executable, os.path.join(AI2_ROOT, 'firm_score.py'), '--benchmark', FIRMCORE, '--responses', firmcore_response_path, '--output', scores_path, '--n-boot', n_boot, '--model-name', method_name], check=True)
    with open(scores_path) as f:
        scores = json.load(f)
    fas = scores['metrics']['FAS']['overall']
    dag = scores['metrics']['DAG']['overall']
    gsr = scores['metrics']['GSR']['overall'] if 'overall' in scores['metrics']['GSR'] else scores['metrics']['GSR']
    print(f"[{method_name}] FAS flip_rate_difference = {fas.get('flip_rate_difference (toward_maj − toward_min)')}", flush=True)
    mmlu_items = load_mmlu(n_per_subject=30)
    arc_items = load_arc_challenge()
    gsm8k_items = gk.load_gsm8k(300)
    if os.environ.get('SMOKE_N'):
        mmlu_items = mmlu_items[:20]
        arc_items = arc_items[:20]
        gsm8k_items = gsm8k_items[:20]
    (hook.n_fired, hook.n_total) = (0, 0)
    s_mmlu = mc_score(model, tok, mmlu_items, hook, LAYER)
    firing_mmlu = {'n_fired': hook.n_fired, 'n_total': hook.n_total, 'rate': hook.n_fired / hook.n_total if hook.n_total else None}
    print(f'[{method_name}] mmlu -> {s_mmlu} firing={firing_mmlu}', flush=True)
    (hook.n_fired, hook.n_total) = (0, 0)
    s_arc = mc_score(model, tok, arc_items, hook, LAYER)
    firing_arc = {'n_fired': hook.n_fired, 'n_total': hook.n_total, 'rate': hook.n_fired / hook.n_total if hook.n_total else None}
    print(f'[{method_name}] arc -> {s_arc} firing={firing_arc}', flush=True)
    (hook.n_fired, hook.n_total) = (0, 0)
    s_gsm8k = gsm8k_score(model, tok, gsm8k_items, hook, LAYER)
    firing_gsm8k = {'n_fired': hook.n_fired, 'n_total': hook.n_total, 'rate': hook.n_fired / hook.n_total if hook.n_total else None}
    print(f'[{method_name}] gsm8k -> {s_gsm8k} firing={firing_gsm8k}', flush=True)
    result = {'method': method_name, 'variant_note': note, 'gate_report': gate_report, 'gate_threshold': args.threshold, 'firmcore_scores_FAS_overall': fas, 'firmcore_scores_DAG_overall': dag, 'firmcore_scores_GSR_overall': gsr, 'firing_rate': {'firmcore': firing_firmcore, 'mmlu': firing_mmlu, 'arc_challenge': firing_arc, 'gsm8k': firing_gsm8k}, 'capability': {'mmlu': s_mmlu, 'arc_challenge': s_arc, 'gsm8k': s_gsm8k}, 'total_elapsed_s': time.time() - t0}
    with open(os.path.join(out_dir, 'result_summary.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[{method_name}] DONE, total elapsed {time.time() - t0:.1f}s -> {os.path.join(out_dir, 'result_summary.json')}", flush=True)
if __name__ == '__main__':
    main()
