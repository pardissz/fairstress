import importlib.util
import os
import sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.hf_backend import load_model_and_tokenizer
from lib.hooks import ProjectionHook
from lib.prompt_adapter import load_jsonl
from lib.repr_edit_common import eval_on_calib
from lib import concept_fit
spec = importlib.util.spec_from_file_location('method6_run_all', os.path.join(HERE, 'run_all.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
calib_items = load_jsonl('external_data/calibration_set.jsonl')[:20]
print(f'smoke: calib={len(calib_items)}')
(model, tok) = load_model_and_tokenizer()
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype
LAYERS = [8, 16]
print('[fit] firm features (tiny) ...')
(x_firm_by_layer, z_firm, items) = concept_fit.concept_a_firm_multilayer(model, tok, LAYERS, max_items=40)
print('  z sum:', z_firm.sum().item(), 'of', len(z_firm))
for layer in LAYERS:
    ws = m.fit_inlp(x_firm_by_layer[layer], z_firm, n_iter=2)
    print(f'layer={layer} n_ws={len(ws)}', [round(float((w ** 2).sum() ** 0.5), 4) for w in ws])
    apply_fn = m.inlp_apply_fn(ws, device, dtype)
    hook = ProjectionHook(apply_fn)
    s = eval_on_calib(model, tok, calib_items, hook, layer, batch_size=20)
    print(f'layer={layer} score={s}')
print('SMOKE TEST OK')
