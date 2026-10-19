import importlib.util
import os
import sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer
from lib.hooks import AddVectorHook
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import eval_on_calib
from lib import concept_fit
spec = importlib.util.spec_from_file_location('method7_run_all', os.path.join(HERE, 'run_all.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
calib_items = load_jsonl('external_data/calibration_set.jsonl')[:20]
print(f'smoke: calib={len(calib_items)}')
(model, tok) = load_model_and_tokenizer()
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
LAYERS = [8, 16]
print('[fit] contrastive answer-token features (tiny) ...')
(x_min_by_layer, x_maj_by_layer, items) = concept_fit.concept_b_caa(model, tok, LAYERS, max_items=40)
print('  x_min shapes:', {k: v.shape for (k, v) in x_min_by_layer.items()})
print('  x_maj shapes:', {k: v.shape for (k, v) in x_maj_by_layer.items()})
import torch
for layer in LAYERS:
    v = x_min_by_layer[layer].mean(0) - x_maj_by_layer[layer].mean(0)
    unit = (v / (v.norm() + 1e-08)).to(device=device, dtype=torch.float32)
    ref_norm = torch.cat([x_min_by_layer[layer], x_maj_by_layer[layer]], dim=0).norm(dim=-1).mean().item()
    print(f'layer={layer} ||v||={v.norm().item():.3f} ref_norm={ref_norm:.3f}')
    for frac in [-0.5, 0.5]:
        alpha = frac * ref_norm
        hook = AddVectorHook(unit, alpha)
        s = eval_on_calib(model, tok, calib_items, hook, layer, batch_size=20)
        print(f'  layer={layer} frac={frac} alpha={alpha:.2f} score={s}')
print('SMOKE TEST OK')
