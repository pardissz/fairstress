import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import run_all as m
from lib.prompt_adapter import load_jsonl
from lib import calib_metrics
calib_items = load_jsonl(m.CALIB)[:30]
firmcore_items = load_jsonl(m.FIRMCORE)[:30]
print(f'smoke: calib={len(calib_items)} firmcore={len(firmcore_items)}')
llm = m.build_llm()
tok = m.get_tokenizer()
for cond in ['Q', 'Q_IF', 'discrim_illegal']:
    resp = m.gen_simple(llm, tok, calib_items, cond, batch_size=32)
    s = calib_metrics.score(resp, calib_items)
    print(cond, 'n_resp=', len(resp), s)
    print('  sample raw_text:', repr(resp[0]['raw_text'])[:120], 'choice=', resp[0]['parsed_choice'])
resp_cot = m.gen_cot(llm, tok, calib_items, batch_size=32)
s = calib_metrics.score(resp_cot, calib_items)
print('Q_IF_CoT', 'n_resp=', len(resp_cot), s)
print('  sample raw_text:', repr(resp_cot[0]['raw_text'])[:200])
print('SMOKE TEST OK')
