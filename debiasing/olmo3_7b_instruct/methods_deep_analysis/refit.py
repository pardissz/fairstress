import json
import os
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
from lib.hooks import AddVectorHook, ProjectionHook, ConditionalAddVectorHook, InterventionModuleHook
from lib import concept_fit
MAX_FIT_ITEMS = 4000
CALIB = os.path.join(PROJECT_ROOT, 'external_data/calibration_set.jsonl')
BIB_POOL = os.path.join(PROJECT_ROOT, 'external_data/bib_pool.jsonl')
FIT_PATHS = (os.path.join(PROJECT_ROOT, 'external_data/fit_bbq.jsonl'), os.path.join(PROJECT_ROOT, 'external_data/fit_synth.jsonl'))

def _chosen_config(method_dir):
    path = os.path.join(PROJECT_ROOT, 'methods', method_dir, 'chosen_config.json')
    with open(path) as f:
        return json.load(f)

def build_hook(method, model, tok):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if method == '05_leace':
        from concept_erasure import LeaceEraser
        cfg = _chosen_config('05_leace')
        layer = cfg['chosen_layer']
        corpus = cfg['chosen_corpus']
        if corpus == 'bib':
            (x, z, _) = concept_fit.concept_a_bib_multilayer(model, tok, [layer], path=BIB_POOL, max_items=MAX_FIT_ITEMS)
        else:
            (x, z, _) = concept_fit.concept_a_firm_multilayer(model, tok, [layer], paths=FIT_PATHS, max_items=MAX_FIT_ITEMS)
        eraser = LeaceEraser.fit(x[layer], z)
        proj_left = eraser.proj_left.to(device=device, dtype=torch.float32)
        proj_right = eraser.proj_right.to(device=device, dtype=torch.float32)
        bias = eraser.bias.to(device=device, dtype=torch.float32) if eraser.bias is not None else None

        def apply_fn(hs):
            (b, s, h) = hs.shape
            flat = hs.reshape(-1, h)
            delta = flat - bias if bias is not None else flat
            out = flat - delta @ proj_right.mH @ proj_left.mH
            return out.reshape(b, s, h).to(dtype)
        return (ProjectionHook(apply_fn), layer)
    if method == '06_inlp':
        cfg = _chosen_config('06_inlp')
        (layer, corpus, n_iter) = (cfg['chosen_layer'], cfg['chosen_corpus'], cfg['chosen_n_iter'])
        if corpus == 'bib':
            (x, z, _) = concept_fit.concept_a_bib_multilayer(model, tok, [layer], path=BIB_POOL, max_items=MAX_FIT_ITEMS)
        else:
            (x, z, _) = concept_fit.concept_a_firm_multilayer(model, tok, [layer], paths=FIT_PATHS, max_items=MAX_FIT_ITEMS)
        X = x[layer].numpy().astype(np.float64)
        Z = z.numpy().astype(np.int64)
        ws = []
        for _ in range(n_iter):
            if len(np.unique(Z)) < 2:
                break
            clf = LogisticRegression(max_iter=2000).fit(X, Z)
            w = clf.coef_.reshape(-1)
            norm = np.linalg.norm(w)
            if norm < 1e-06:
                break
            w = w / norm
            ws.append(w)
            X = X - np.outer(X @ w, w)
        W = torch.tensor(np.stack(ws), dtype=torch.float32, device=device) if ws else None

        def apply_fn(hs):
            (b, s, h) = hs.shape
            flat = hs.reshape(-1, h)
            if W is not None:
                for i in range(W.shape[0]):
                    w = W[i]
                    flat = flat - (flat @ w).unsqueeze(-1) * w
            return flat.reshape(b, s, h).to(dtype)
        return (ProjectionHook(apply_fn), layer)
    if method == '07_caa':
        cfg = _chosen_config('07_caa')
        (layer, alpha) = (cfg['chosen_layer'], cfg['chosen_alpha'])
        (x_min_by_layer, x_maj_by_layer, _) = concept_fit.concept_b_caa(model, tok, [layer], paths=FIT_PATHS, max_items=MAX_FIT_ITEMS)
        v = x_min_by_layer[layer].mean(0) - x_maj_by_layer[layer].mean(0)
        unit = (v / (v.norm() + 1e-08)).to(device=device, dtype=torch.float32)
        return (AddVectorHook(unit, alpha), layer)
    if method == '08_cast':
        cfg = _chosen_config('08_cast')
        (layer, threshold, alpha) = (cfg['chosen_layer'], cfg['chosen_threshold'], cfg['chosen_alpha'])
        (x_cond_pos, x_cond_neg) = concept_fit.condition_vector_firmcore_multilayer(model, tok, [layer], max_items_per_class=2000)
        (x_min_by_layer, x_maj_by_layer, _) = concept_fit.concept_b_caa(model, tok, [layer], paths=FIT_PATHS, max_items=2000)
        c = concept_fit.contrastive_direction(x_cond_pos[layer], x_cond_neg[layer], method='pca')
        v_raw = concept_fit.contrastive_direction(x_min_by_layer[layer], x_maj_by_layer[layer], method='pca')
        v_unit = (v_raw / (v_raw.norm() + 1e-08)).to(device=device, dtype=torch.float32)
        c = c.to(device=device, dtype=torch.float32)
        return (ConditionalAddVectorHook(c, threshold, v_unit, alpha), layer)
    if method == '09_loreft':
        from lib.loreft import LoreftIntervention, train_loreft
        import random
        cfg = _chosen_config('09_loreft')
        (layer, rank) = (cfg['chosen_layer'], cfg['chosen_rank'])
        items = []
        for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
            for it in concept_fit.load_jsonl(os.path.join(PROJECT_ROOT, path)):
                if it['condition'] == 'GT' and it.get('expected_correct'):
                    items.append({'prompt': it['prompt'], 'target': it['expected_correct']})
        random.Random(0).shuffle(items)
        items = items[:6000]
        module = LoreftIntervention(model.config.hidden_size, rank)
        train_loreft(model, tok, module, layer, items, device, epochs=2, batch_size=16, lr=0.001)
        return (InterventionModuleHook(module), layer)
    if method == '10_shifting_perspectives':
        from collections import defaultdict
        cfg = _chosen_config('10_shifting_perspectives')
        (layer, alpha) = (cfg['chosen_layer'], cfg['chosen_alpha'])
        items = []
        for path in ('external_data/fit_bbq.jsonl', 'external_data/fit_synth.jsonl'):
            items.extend(concept_fit.load_jsonl(os.path.join(PROJECT_ROOT, path)))
        by_axis = defaultdict(list)
        for it in items:
            by_axis[it['contrast_category']].append(it)
        vecs = []
        for (axis, axis_items) in by_axis.items():
            if len(axis_items) < 20:
                continue
            axis_items = axis_items[:1000]
            minority_ans = [f"Candidate {it['minority_slot']}" for it in axis_items]
            majority_other = {'A': 'B', 'B': 'A'}
            majority_ans = [f"Candidate {majority_other[it['minority_slot']]}" for it in axis_items]
            x_min = concept_fit._appended_answer_hidden_states(model, tok, axis_items, minority_ans, [layer])[layer]
            x_maj = concept_fit._appended_answer_hidden_states(model, tok, axis_items, majority_ans, [layer])[layer]
            vecs.append(concept_fit.contrastive_direction(x_min, x_maj, method='pca'))
        combined = torch.stack(vecs).mean(0)
        unit = (combined / (combined.norm() + 1e-08)).to(device=device, dtype=torch.float32)
        return (AddVectorHook(unit, alpha), layer)
    raise ValueError(f'unknown method {method!r}')

def load_calib_items():
    return concept_fit.load_jsonl(CALIB)
