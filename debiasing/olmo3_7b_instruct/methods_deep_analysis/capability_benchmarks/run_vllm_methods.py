import json
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'methods', '01_prompting'))
import run_all as m1
from data import load_mmlu, load_arc_challenge
from vllm_mc import mc_predict_vllm, score_mc_results
from self_debiasing_mc import mc_predict_self_debiasing
OUT_DIR = HERE
IF_TEXT = m1.IF_TEXT
CF_STEPWISE = "Answer this in three steps. Step 1: Using only the job-relevant facts (ignore names and any demographic-adjacent details), determine which candidate is stronger. Step 2: Imagine the candidates' demographic details were swapped -- would your Step 1 answer change? It should not, since Step 1 ignored demographics. Step 3: State your final answer, consistent with Step 1."
PRINCIPLE = 'First restate the relevant principle: a fair decision must be based only on the qualifications and facts described, never on race, gender, age, disability, or other demographic characteristics. Then answer, complying with that principle.'
CONDITIONS = {'00_baseline': None, '01_prompting': lambda p: f'{p}\n\n{IF_TEXT}', '03_cot_counterfactual': lambda p: f'{p}\n\n{CF_STEPWISE}', '04_self_critique_revise': lambda p: f'{p}\n\n{PRINCIPLE}'}

def main():
    mmlu_items = load_mmlu(n_per_subject=30)
    arc_items = load_arc_challenge()
    llm = m1.build_llm()
    tok = m1.get_tokenizer()
    results_mmlu = {}
    results_arc = {}
    for (name, wrap_fn) in CONDITIONS.items():
        print(f'=== {name} ===', flush=True)
        r_mmlu = mc_predict_vllm(llm, tok, m1.chat_wrap, mmlu_items, wrap_fn=wrap_fn)
        s_mmlu = score_mc_results(r_mmlu)
        results_mmlu[name] = s_mmlu
        print(f'  mmlu -> {s_mmlu}', flush=True)
        r_arc = mc_predict_vllm(llm, tok, m1.chat_wrap, arc_items, wrap_fn=wrap_fn)
        s_arc = score_mc_results(r_arc)
        results_arc[name] = s_arc
        print(f'  arc_challenge -> {s_arc}', flush=True)
        with open(os.path.join(OUT_DIR, 'results_mmlu.json'), 'w') as f:
            json.dump(results_mmlu, f, indent=2)
        with open(os.path.join(OUT_DIR, 'results_arc_challenge.json'), 'w') as f:
            json.dump(results_arc, f, indent=2)
    name = '02_self_debiasing'
    print(f'=== {name} ===', flush=True)
    r_mmlu = mc_predict_self_debiasing(llm, tok, m1.chat_wrap, mmlu_items)
    s_mmlu = score_mc_results(r_mmlu)
    results_mmlu[name] = s_mmlu
    print(f'  mmlu -> {s_mmlu}', flush=True)
    r_arc = mc_predict_self_debiasing(llm, tok, m1.chat_wrap, arc_items)
    s_arc = score_mc_results(r_arc)
    results_arc[name] = s_arc
    print(f'  arc_challenge -> {s_arc}', flush=True)
    with open(os.path.join(OUT_DIR, 'results_mmlu.json'), 'w') as f:
        json.dump(results_mmlu, f, indent=2)
    with open(os.path.join(OUT_DIR, 'results_arc_challenge.json'), 'w') as f:
        json.dump(results_arc, f, indent=2)
    print('[done] vLLM-backed methods complete', flush=True)
if __name__ == '__main__':
    main()
