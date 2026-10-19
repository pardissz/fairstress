import importlib.util
import os
import sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
from lib.prompt_adapter import load_jsonl
from lib import calib_metrics
spec = importlib.util.spec_from_file_location('method3_run_all', os.path.join(HERE, 'run_all.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
calib_items = load_jsonl('external_data/calibration_set.jsonl')[:20]
print(f'smoke: calib={len(calib_items)}')
llm = m.m1.build_llm()
tok = m.m1.get_tokenizer()
for (name, instr) in m.CF_VARIANTS.items():
    resp = m.gen_cot_variant(llm, tok, calib_items, instr, batch_size=20, log_tag=name)
    s = calib_metrics.score(resp, calib_items)
    print(name, 'n_resp=', len(resp), s)
    print('  sample raw_text:', repr(resp[0]['raw_text'])[:200], 'choice=', resp[0]['parsed_choice'])
print('SMOKE TEST OK')
