import importlib.util
import json
import os
import sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE.replace('capability_benchmarks', ''))
from lib.hf_backend import load_model_and_tokenizer, generate_with_hook, next_token_choice_logprobs_with_hook
from lib.hooks import AddVectorHook, ProjectionHook, ConditionalAddVectorHook, InterventionModuleHook
from lib.loreft import LoreftIntervention, train_loreft
from lib import concept_fit
from data import load_mmlu, load_arc_challenge, LETTERS
import gsm8k_util as gk
METHODS_DIR = os.path.join(PROJECT_ROOT, 'methods')

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_cfg(method_dir, fname='chosen_config.json'):
    with open(os.path.join(METHODS_DIR, method_dir, fname)) as f:
        return json.load(f)
FIT_CAP = 4000

def build_05_leace(model, tok):
    cfg = load_cfg('05_leace')
    (corpus, layer) = (cfg['chosen_corpus'], cfg['chosen_layer'])
    from concept_erasure import LeaceEraser
    m5 = load_module(os.path.join(METHODS_DIR, '05_leace', 'run_all.py'), 'm5_run_all')
    if corpus == 'firm':
        (x_by_layer, z, _) = concept_fit.concept_a_firm_multilayer(model, tok, [layer], max_items=FIT_CAP)
    else:
        (x_by_layer, z, _) = concept_fit.concept_a_bib_multilayer(model, tok, [layer], max_items=FIT_CAP)
    eraser = LeaceEraser.fit(x_by_layer[layer], z)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    apply_fn = m5.leace_apply_fn(eraser, device, dtype)
    return (ProjectionHook(apply_fn), layer)

def build_06_inlp(model, tok, v2=False):
    cfg = load_cfg('06_inlp', 'chosen_config_v2.json' if v2 else 'chosen_config.json')
    (corpus, layer, n_iter) = (cfg['chosen_corpus'], cfg['chosen_layer'], cfg['chosen_n_iter'])
    fname = 'run_all_v2.py' if v2 else 'run_all.py'
    m6 = load_module(os.path.join(METHODS_DIR, '06_inlp', fname), 'm6_run_all' + ('_v2' if v2 else ''))
    if corpus == 'firm':
        (x_by_layer, z, _) = concept_fit.concept_a_firm_multilayer(model, tok, [layer], max_items=FIT_CAP)
    else:
        (x_by_layer, z, _) = concept_fit.concept_a_bib_multilayer(model, tok, [layer], max_items=FIT_CAP)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if v2:
        ws = m6.fit_inlp_svm(x_by_layer[layer], z, n_iter)
    else:
        ws = m6.fit_inlp(x_by_layer[layer], z, n_iter)
    apply_fn = m6.inlp_apply_fn(ws, device, dtype)
    return (ProjectionHook(apply_fn), layer)

def build_07_caa(model, tok):
    cfg = load_cfg('07_caa')
    (layer, alpha_frac) = (cfg['chosen_layer'], cfg['chosen_alpha_frac'])
    device = next(model.parameters()).device
    (x_min_by_layer, x_maj_by_layer, _) = concept_fit.concept_b_caa(model, tok, [layer], max_items=FIT_CAP)
    (x_min, x_maj) = (x_min_by_layer[layer], x_maj_by_layer[layer])
    v = x_min.mean(0) - x_maj.mean(0)
    unit = (v / (v.norm() + 1e-08)).to(device=device, dtype=torch.float32)
    ref_norm = torch.cat([x_min, x_maj], dim=0).norm(dim=-1).mean().item()
    alpha = alpha_frac * ref_norm
    return (AddVectorHook(unit, alpha), layer)

def build_08_cast(model, tok):
    cfg = load_cfg('08_cast')
    (layer, threshold, alpha) = (cfg['chosen_layer'], cfg['chosen_threshold'], cfg['chosen_alpha'])
    fit_cap = 2000
    device = next(model.parameters()).device
    (x_cond_pos, x_cond_neg) = concept_fit.condition_vector_firmcore_multilayer(model, tok, [layer], max_items_per_class=fit_cap)
    (x_min_by_layer, x_maj_by_layer, _) = concept_fit.concept_b_caa(model, tok, [layer], max_items=fit_cap)
    c = concept_fit.contrastive_direction(x_cond_pos[layer], x_cond_neg[layer], method='pca')
    c = c.to(device=device, dtype=torch.float32)
    v_raw = concept_fit.contrastive_direction(x_min_by_layer[layer], x_maj_by_layer[layer], method='pca')
    v_unit = (v_raw / (v_raw.norm() + 1e-08)).to(device=device, dtype=torch.float32)
    return (ConditionalAddVectorHook(c, threshold, v_unit, alpha), layer)

def build_09_loreft(model, tok):
    cfg = load_cfg('09_loreft')
    (layer, rank) = (cfg['chosen_layer'], cfg['chosen_rank'])
    device = next(model.parameters()).device
    import random

    def load_training_items():
        items = []
        for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
            for it in concept_fit.load_jsonl(os.path.join(DEBIAS_ROOT, path)):
                if it['condition'] == 'GT' and it.get('expected_correct'):
                    items.append({'prompt': it['prompt'], 'target': it['expected_correct']})
        random.Random(0).shuffle(items)
        return items[:6000]
    train_items = load_training_items()
    module = LoreftIntervention(model.config.hidden_size, rank)
    train_loreft(model, tok, module, layer, train_items, device, epochs=2, batch_size=16, lr=0.001)
    return (InterventionModuleHook(module), layer)

def build_10_shifting_perspectives(model, tok):
    from collections import defaultdict
    cfg = load_cfg('10_shifting_perspectives')
    (layer, alpha) = (cfg['chosen_layer'], cfg['chosen_alpha'])
    fit_cap = 800
    items = []
    for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        items.extend(concept_fit.load_jsonl(os.path.join(DEBIAS_ROOT, path)))
    by_axis = defaultdict(list)
    for it in items:
        by_axis[it['contrast_category']].append(it)
    device = next(model.parameters()).device
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
    return (AddVectorHook(unit, alpha), layer)

def build_11_pressure_mixed(model, tok):
    cfg = load_cfg('11_pressure_mixed')
    (layer, rank, pressure_frac) = (cfg['chosen_layer'], cfg['chosen_rank'], cfg['pressure_frac'])
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'methods_deep_analysis'))
    import loreft_lab as lab
    device = next(model.parameters()).device
    train_items = lab.load_training_items(with_pressure=True, pressure_frac=pressure_frac)
    module = LoreftIntervention(model.config.hidden_size, rank)
    train_loreft(model, tok, module, layer, train_items, device, epochs=2, batch_size=16, lr=0.001)
    return (InterventionModuleHook(module), layer)
DEBIAS_ROOT = PROJECT_ROOT
BUILDERS = {'05_leace': build_05_leace, '06_inlp': lambda m, t: build_06_inlp(m, t, v2=False), '06_inlp_v2': lambda m, t: build_06_inlp(m, t, v2=True), '07_caa': build_07_caa, '08_cast': build_08_cast, '09_loreft': build_09_loreft, '10_shifting_perspectives': build_10_shifting_perspectives, '11_pressure_mixed': build_11_pressure_mixed}

def mc_score(model, tok, items, hook_obj, layer):
    from data import format_mc_prompt
    max_n_choices = max((len(it['choices']) for it in items))
    choices = tuple(LETTERS[:max_n_choices])
    prompts = [format_mc_prompt(it['question'], it['choices']) for it in items]
    probs = next_token_choice_logprobs_with_hook(model, tok, prompts, choices=choices, hook_obj=hook_obj, layer_idx=layer, batch_size=24)
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

def gsm8k_score(model, tok, items, hook_obj, layer):
    prompts = [it['question'] + gk.ANSWER_SUFFIX for it in items]
    texts = generate_with_hook(model, tok, prompts, hook_obj=hook_obj, layer_idx=layer, batch_size=24, max_new_tokens=320)
    return gk.score_gsm8k(items, texts)

def main():
    os.chdir(DEBIAS_ROOT)
    mmlu_items = load_mmlu(n_per_subject=30)
    arc_items = load_arc_challenge()
    gsm8k_items = gk.load_gsm8k(300)
    (model, tok) = load_model_and_tokenizer()
    results_mmlu = {}
    results_arc = {}
    results_gsm8k = {}
    out_dir = HERE

    def flush():
        with open(os.path.join(out_dir, 'results_mmlu.json')) as f:
            existing = json.load(f)
        existing.update(results_mmlu)
        with open(os.path.join(out_dir, 'results_mmlu.json'), 'w') as f:
            json.dump(existing, f, indent=2)
        with open(os.path.join(out_dir, 'results_arc_challenge.json')) as f:
            existing_arc = json.load(f)
        existing_arc.update(results_arc)
        with open(os.path.join(out_dir, 'results_arc_challenge.json'), 'w') as f:
            json.dump(existing_arc, f, indent=2)
        with open(os.path.join(out_dir, 'results_gsm8k_hooked.json'), 'w') as f:
            json.dump(results_gsm8k, f, indent=2)
    for fn in ('results_mmlu.json', 'results_arc_challenge.json'):
        p = os.path.join(out_dir, fn)
        if not os.path.exists(p):
            with open(p, 'w') as f:
                json.dump({}, f)
    for (name, builder) in BUILDERS.items():
        print(f'=== {name} ===', flush=True)
        (hook_obj, layer) = builder(model, tok)
        s_mmlu = mc_score(model, tok, mmlu_items, hook_obj, layer)
        results_mmlu[name] = s_mmlu
        print(f'  mmlu -> {s_mmlu}', flush=True)
        s_arc = mc_score(model, tok, arc_items, hook_obj, layer)
        results_arc[name] = s_arc
        print(f'  arc_challenge -> {s_arc}', flush=True)
        s_gsm8k = gsm8k_score(model, tok, gsm8k_items, hook_obj, layer)
        results_gsm8k[name] = s_gsm8k
        print(f'  gsm8k -> {s_gsm8k}', flush=True)
        flush()
    print('[done] hook-backed methods complete', flush=True)
if __name__ == '__main__':
    main()
