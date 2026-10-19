import json
import glob
order = ['00_baseline', '01_prompting', '02_self_debiasing', '03_cot_counterfactual', '04_self_critique_revise', '05_leace', '06_inlp', '07_caa', '08_cast', '09_loreft', '10_shifting_perspectives', '11_pressure_mixed']
rows = []
for name in order:
    path = f'firmcore_eval/scores/{name}_scores.json'
    with open(path) as f:
        r = json.load(f)
    dag = r['metrics']['DAG']['overall']
    gsr = r['metrics']['GSR']['overall']
    fas = r['metrics']['FAS']['overall']
    dcg = r['metrics']['DCG'].get('overall', {})
    refusal = r['refusals']['overall_refusal_rate']
    n_min = dag['n_truth_favors_minority']
    n_maj = dag['n_truth_favors_majority']
    gt_acc = (dag['accuracy_truth_minority'] * n_min + dag['accuracy_truth_majority'] * n_maj) / (n_min + n_maj)
    flip_maj = fas.get('flip_rate_toward_majority')
    flip_min = fas.get('flip_rate_toward_minority')
    rows.append({'method': name, 'gt_acc': gt_acc, 'refusal': refusal, 'dag': dag['value'], 'dag_ci': [dag['ci_lower'], dag['ci_upper']], 'gsr': gsr['value'], 'gsr_ci': [gsr['ci_lower'], gsr['ci_upper']], 'flip_delta': fas.get('flip_rate_difference (toward_maj − toward_min)'), 'flip_delta_ci': fas.get('flip_rate_difference_ci'), 'flip_maj': flip_maj, 'flip_min': flip_min, 'fas_ratio': fas.get('FAS_ratio (point estimate)'), 'dcg_value': dcg.get('value')})
print(f"{'method':<26}{'GTacc':>8}{'refusal':>8}{'DAG':>9}{'GSR':>9}{'flip_Δ':>9}{'FAS_ratio':>10}")
for row in rows:
    print(f"{row['method']:<26}{row['gt_acc']:>8.4f}{row['refusal']:>8.1%}{row['dag']:>9.4f}{row['gsr']:>9.4f}{(row['flip_delta'] if row['flip_delta'] is not None else float('nan')):>9.4f}{(row['fas_ratio'] if row['fas_ratio'] is not None else float('nan')):>10.4f}")
with open('firmcore_eval/scores/SUMMARY.json', 'w') as f:
    json.dump(rows, f, indent=2)
print('\nwrote firmcore_eval/scores/SUMMARY.json')
