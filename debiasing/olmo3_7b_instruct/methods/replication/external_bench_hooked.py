import json
import os
import sys
import torch
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'methods', '01_prompting'))
sys.path.insert(0, PROJECT_ROOT)
import run_all as m1
from lib.hf_backend import load_model_and_tokenizer, sentence_logprob_with_hook, generate_with_hook
from lib.hooks import ConditionalAddVectorHook, AddVectorHook, InterventionModuleHook
from lib.loreft import LoreftIntervention, train_loreft
from lib import concept_fit
OUT_PATH = os.path.join(HERE, 'external_bench_hooked_results.json')
MAX_STEREOSET = 5 if os.environ.get('SMOKE_N') else 1500
MAX_BBQ = 5 if os.environ.get('SMOKE_N') else 1500
MAX_FIT_ITEMS = 60 if os.environ.get('SMOKE_N') else 2000
_TRAIN_ITEMS = 40 if os.environ.get('SMOKE_N') else 6000
_TRAIN_EPOCHS = 1 if os.environ.get('SMOKE_N') else 2

def eval_stereoset_with_hook(model, tok, hook_obj, layer_idx):
    from datasets import load_dataset
    ds = load_dataset('McGill-NLP/stereoset', 'intersentence', split='validation')
    ds = ds.select(range(min(MAX_STEREOSET, len(ds))))
    label_names = ds.features['sentences']['gold_label'].feature.names
    n_stereo_win = n_pairs = n_related_win = n_related_total = 0
    for row in ds:
        (sents, labels) = (row['sentences']['sentence'], row['sentences']['gold_label'])
        by_label = {label_names[l]: s for (l, s) in zip(labels, sents)}
        if not all((k in by_label for k in ('stereotype', 'anti-stereotype', 'unrelated'))):
            continue
        ctx = row['context']
        texts = [f"{ctx} {by_label['stereotype']}", f"{ctx} {by_label['anti-stereotype']}", f"{ctx} {by_label['unrelated']}"]
        lps = sentence_logprob_with_hook(model, tok, texts, hook_obj=hook_obj, layer_idx=layer_idx, batch_size=3)
        (lp_st, lp_anti, lp_unrel) = lps
        if lp_st > lp_anti:
            n_stereo_win += 1
        n_pairs += 1
        if lp_st > lp_unrel:
            n_related_win += 1
        if lp_anti > lp_unrel:
            n_related_win += 1
        n_related_total += 2
    return {'n_pairs': n_pairs, 'SS_score': n_stereo_win / n_pairs if n_pairs else None, 'LM_score': n_related_win / n_related_total if n_related_total else None}

def eval_clear_bias_with_hook(model, tok, hook_obj, layer_idx):
    from datasets import load_dataset
    ds = load_dataset('RCantini/CLEAR-Bias', 'base_prompts', split='train')
    prompts_raw = [row['PROMPT'] for row in ds]
    texts = generate_with_hook(model, tok, prompts_raw, hook_obj=hook_obj, layer_idx=layer_idx, max_new_tokens=16)
    n_stereo = n_scored = 0
    for (row, text) in zip(ds, texts):
        text = (text or '').upper()
        (stereo, counter) = (row['STEREOTYPE'].upper(), row['COUNTER-STEREOTYPE'].upper())
        (has_stereo, has_counter) = (stereo in text, counter in text)
        if has_stereo == has_counter:
            continue
        n_scored += 1
        n_stereo += int(has_stereo)
    return {'n_scored': n_scored, 'n_total': len(ds), 'stereotype_completion_rate': n_stereo / n_scored if n_scored else None}

def eval_bbq_with_hook(model, tok, hook_obj, layer_idx):
    from datasets import load_dataset
    ds = load_dataset('walledai/BBQ', split='age')
    ds = ds.select(range(min(MAX_BBQ, len(ds))))
    prompts_raw = []
    for row in ds:
        opts = '\n'.join((f'{i}: {c}' for (i, c) in enumerate(row['choices'])))
        prompts_raw.append(f"{row['context']}\n\n{row['question']}\n{opts}\n\nAnswer with only the option number.")
    texts = generate_with_hook(model, tok, prompts_raw, hook_obj=hook_obj, layer_idx=layer_idx, max_new_tokens=16)
    correct_amb = total_amb = correct_disamb = total_disamb = 0
    for (row, text) in zip(ds, texts):
        text = (text or '').strip()
        pred = None
        for ch in text:
            if ch.isdigit():
                pred = int(ch)
                break
        is_ambiguous = row['answer'] == 2
        correct = pred == row['answer']
        if is_ambiguous:
            total_amb += 1
            correct_amb += int(correct)
        else:
            total_disamb += 1
            correct_disamb += int(correct)
    return {'n_ambiguous': total_amb, 'n_disambiguated': total_disamb, 'accuracy_ambiguous': correct_amb / total_amb if total_amb else None, 'accuracy_disambiguated': correct_disamb / total_disamb if total_disamb else None}

def build_cast_hook(model, tok):
    with open(os.path.join(PROJECT_ROOT, 'methods', '08_cast', 'chosen_config.json')) as f:
        cfg = json.load(f)
    (layer, threshold, alpha) = (cfg['chosen_layer'], cfg['chosen_threshold'], cfg['chosen_alpha'])
    (x_cond_pos, x_cond_neg) = concept_fit.condition_vector_firmcore_multilayer(model, tok, [layer], max_items_per_class=MAX_FIT_ITEMS)
    (x_min, x_maj, _) = concept_fit.concept_b_caa(model, tok, [layer], max_items=MAX_FIT_ITEMS)
    device = next(model.parameters()).device
    c = concept_fit.contrastive_direction(x_cond_pos[layer], x_cond_neg[layer], method='pca')
    c = c.to(device=device, dtype=torch.float32)
    v_raw = concept_fit.contrastive_direction(x_min[layer], x_maj[layer], method='pca')
    v_unit = (v_raw / (v_raw.norm() + 1e-08)).to(device=device, dtype=torch.float32)
    return (ConditionalAddVectorHook(c, threshold, v_unit, alpha), layer)

def build_loreft_hook(model, tok):
    import random
    from lib.concept_fit import load_jsonl as load_jsonl_cf
    with open(os.path.join(PROJECT_ROOT, 'methods', '09_loreft', 'chosen_config.json')) as f:
        cfg = json.load(f)
    (layer, rank) = (cfg['chosen_layer'], cfg['chosen_rank'])
    items = []
    for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        for it in load_jsonl_cf(os.path.join(PROJECT_ROOT, path)):
            if it['condition'] == 'GT' and it.get('expected_correct'):
                items.append({'prompt': it['prompt'], 'target': it['expected_correct']})
    random.Random(0).shuffle(items)
    items = items[:_TRAIN_ITEMS]
    device = next(model.parameters()).device
    module = LoreftIntervention(model.config.hidden_size, rank)
    train_loreft(model, tok, module, layer, items, device, epochs=_TRAIN_EPOCHS, batch_size=16, lr=0.001)
    return (InterventionModuleHook(module), layer)

def build_shifting_hook(model, tok):
    from collections import defaultdict
    with open(os.path.join(PROJECT_ROOT, 'methods', '10_shifting_perspectives', 'chosen_config.json')) as f:
        cfg = json.load(f)
    (layer, alpha) = (cfg['chosen_layer'], cfg['chosen_alpha'])
    items = []
    for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
        items.extend(concept_fit.load_jsonl(os.path.join(PROJECT_ROOT, path)))
    by_axis = defaultdict(list)
    for it in items:
        by_axis[it['contrast_category']].append(it)
    device = next(model.parameters()).device
    vecs = []
    axis_cap = 30 if os.environ.get('SMOKE_N') else 800
    for (axis, axis_items) in by_axis.items():
        axis_items = axis_items[:axis_cap]
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

def main():
    (model, tok) = load_model_and_tokenizer()
    results = {}
    for (name, builder) in [('08_cast', build_cast_hook), ('09_loreft', build_loreft_hook), ('10_shifting_perspectives', build_shifting_hook)]:
        print(f'[{name}] building hook ...', flush=True)
        (hook_obj, layer) = builder(model, tok)
        results[name] = {}
        print(f'[{name}] scoring StereoSet ...', flush=True)
        results[name]['stereoset'] = eval_stereoset_with_hook(model, tok, hook_obj, layer)
        print(f"  -> {results[name]['stereoset']}", flush=True)
        print(f'[{name}] scoring CLEAR-Bias ...', flush=True)
        results[name]['clear_bias'] = eval_clear_bias_with_hook(model, tok, hook_obj, layer)
        print(f"  -> {results[name]['clear_bias']}", flush=True)
        print(f'[{name}] scoring BBQ (age) ...', flush=True)
        results[name]['bbq_age'] = eval_bbq_with_hook(model, tok, hook_obj, layer)
        print(f"  -> {results[name]['bbq_age']}", flush=True)
        with open(OUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)
    print(f'[done] wrote {OUT_PATH}', flush=True)
if __name__ == '__main__':
    main()
