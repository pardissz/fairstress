import json
import os
import pickle
import sys
import time
from collections import Counter
sys.path.insert(0, './data/lexicon_scoring')
import lexscore_full as lf
from lexscore import preprocess
sys.path.insert(0, './data/resources')
import idp as idp_module
import pandas as pd
import numpy as np
AI2 = './data'
FIRMCORE = f'{AI2}/lexicon_scoring/firmCore.jsonl'
REASONING_DIR = f'{AI2}/debias_olmo3_7b_instruct/firmcore_eval/reasoning'
OUT_DIR = f'{AI2}/debias_olmo3_7b_instruct/methods_deep_analysis/full_lexicon_idp'
MIN_COUNT = 50
TOP_N = 40
METHODS = ['00_baseline', '01_prompting', '02_self_debiasing', '03_cot_counterfactual', '04_self_critique_revise', '05_leace', '06_inlp', '06_inlp_v2', '07_caa', '08_cast', '09_loreft', '10_shifting_perspectives', '11_pressure_mixed']

def load_firmcore_meta():
    meta = {}
    with open(FIRMCORE) as f:
        for line in f:
            d = json.loads(line)
            meta[d['item_id']] = {'condition': d['condition'], 'signaling_level': d['signaling_level'], 'contrast_id': d['contrast_id'], 'contrast_category': d['contrast_category'], 'minority_slot': d['minority_slot'], 'expected_correct': d.get('expected_correct'), 'has_injection': d.get('has_injection', False)}
    return meta

def main():
    t_start = time.time()
    print('[1/4] loading firmCore metadata...', flush=True)
    meta = load_firmcore_meta()
    print(f'  {len(meta):,} items loaded', flush=True)
    print('[2/4] building full 15-source lexicon...', flush=True)
    (exact_dict, bigram_dict, trie, age_gender_intercept) = lf.build_full_lexicon()
    cats = lf.all_categories_weighted(exact_dict, bigram_dict, trie)
    print(f'  {len(cats)} categories across all sources', flush=True)
    with open(os.path.join(OUT_DIR, 'categories.json'), 'w') as f:
        json.dump(cats, f, indent=2)
    MEAN_TRACK_CATS = ('CONCR_concreteness', 'NRC_VAD_valence', 'NRC_VAD_arousal', 'NRC_VAD_dominance')
    buckets = ['correct', 'incorrect', 'minority_chosen', 'majority_chosen', 'overall']
    lex_acc = {m: {b: {c: [0.0, 0] for c in cats} for b in buckets} for m in METHODS}
    lex_mean_acc = {m: {b: {c: [0.0, 0] for c in MEAN_TRACK_CATS} for b in buckets} for m in METHODS}
    n_words_acc = {m: {b: [0, 0] for b in buckets} for m in METHODS}
    method_counters = {m: Counter() for m in METHODS}
    correct_counters = {'correct': Counter(), 'incorrect': Counter()}
    chosen_counters = {'minority': Counter(), 'majority': Counter()}
    per_method_stats = {}
    print('[3/4] scoring all 13 methods (full lexicon + IDP word counts)...', flush=True)
    for (mi, method) in enumerate(METHODS):
        path = os.path.join(REASONING_DIR, f'{method}_reasoning.jsonl')
        t0 = time.time()
        n_lines = 0
        n_scored = 0
        n_no_meta = 0
        with open(path) as f:
            for line in f:
                n_lines += 1
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                item_id = d['item_id']
                m = meta.get(item_id)
                if m is None:
                    n_no_meta += 1
                    continue
                reasoning = d.get('reasoning') or ''
                parsed_choice = d.get('parsed_choice')
                (sums, match_counts, n_words) = lf.score_doc_weighted(reasoning, exact_dict, bigram_dict, trie, track_matches_for=MEAN_TRACK_CATS)
                sums['AGEGENDER_age'] = sums.get('AGEGENDER_age', 0.0) + age_gender_intercept['age'] * n_words if n_words else sums.get('AGEGENDER_age', 0.0)
                sums['AGEGENDER_gender'] = sums.get('AGEGENDER_gender', 0.0) + age_gender_intercept['gender'] * n_words if n_words else sums.get('AGEGENDER_gender', 0.0)
                chosen_group = None
                if parsed_choice in ('A', 'B'):
                    chosen_group = 'minority' if parsed_choice == m['minority_slot'] else 'majority'
                is_correct = None
                expected = m['expected_correct']
                if expected is not None and parsed_choice in ('A', 'B'):
                    subj = expected.split()[0]
                    is_correct = f'{subj} {parsed_choice}' == expected
                bucket_list = ['overall']
                if is_correct is True:
                    bucket_list.append('correct')
                elif is_correct is False:
                    bucket_list.append('incorrect')
                if chosen_group == 'minority':
                    bucket_list.append('minority_chosen')
                elif chosen_group == 'majority':
                    bucket_list.append('majority_chosen')
                for b in bucket_list:
                    n_words_acc[method][b][0] += n_words
                    n_words_acc[method][b][1] += 1
                    for c in cats:
                        val = sums.get(c, 0.0) / n_words if n_words else 0.0
                        lex_acc[method][b][c][0] += val
                        lex_acc[method][b][c][1] += 1
                    for c in MEAN_TRACK_CATS:
                        mc = match_counts.get(c, 0)
                        if mc:
                            lex_mean_acc[method][b][c][0] += sums.get(c, 0.0) / mc
                            lex_mean_acc[method][b][c][1] += 1
                (toks, nw) = preprocess(reasoning)
                if nw > 0:
                    tok_counter = Counter(toks)
                    method_counters[method].update(tok_counter)
                    if is_correct is True:
                        correct_counters['correct'].update(tok_counter)
                    elif is_correct is False:
                        correct_counters['incorrect'].update(tok_counter)
                    if chosen_group:
                        chosen_counters[chosen_group].update(tok_counter)
                n_scored += 1
        dt = time.time() - t0
        print(f'  [{mi + 1}/13] {method}: {n_scored:,}/{n_lines:,} scored ({n_no_meta} no-meta) in {dt:.1f}s ({n_scored / dt:.0f} docs/s)', flush=True)
    with open(os.path.join(OUT_DIR, 'method_counters.pkl'), 'wb') as f:
        pickle.dump(method_counters, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(OUT_DIR, 'correct_counters.pkl'), 'wb') as f:
        pickle.dump(correct_counters, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(OUT_DIR, 'chosen_counters.pkl'), 'wb') as f:
        pickle.dump(chosen_counters, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('[4/4] building lexicon summary table + IDP reports...', flush=True)
    rows = []
    for method in METHODS:
        row = {'method': method, 'n': n_words_acc[method]['overall'][1]}
        for c in cats:
            mc = lex_acc[method]['correct'][c]
            mi_ = lex_acc[method]['incorrect'][c]
            mmin = lex_acc[method]['minority_chosen'][c]
            mmaj = lex_acc[method]['majority_chosen'][c]
            v_c = mc[0] / mc[1] if mc[1] else None
            v_i = mi_[0] / mi_[1] if mi_[1] else None
            v_min = mmin[0] / mmin[1] if mmin[1] else None
            v_maj = mmaj[0] / mmaj[1] if mmaj[1] else None
            row[f'{c}__mean_correct'] = v_c
            row[f'{c}__mean_incorrect'] = v_i
            row[f'{c}__ratio_incorrect_over_correct'] = v_i / v_c if v_c and v_i is not None and (v_c > 0) else None
            row[f'{c}__mean_min_chosen'] = v_min
            row[f'{c}__mean_maj_chosen'] = v_maj
            row[f'{c}__delta_min_minus_maj'] = v_min - v_maj if v_min is not None and v_maj is not None else None
        for c in MEAN_TRACK_CATS:
            for (b, suffix) in [('correct', 'mean_of_mean_correct'), ('incorrect', 'mean_of_mean_incorrect'), ('minority_chosen', 'mean_of_mean_min'), ('majority_chosen', 'mean_of_mean_maj')]:
                acc = lex_mean_acc[method][b][c]
                row[f'{c}__{suffix}'] = acc[0] / acc[1] if acc[1] else None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'full_lexicon_summary.csv'), index=False)
    print(f'  wrote full_lexicon_summary.csv: {df.shape[0]} rows x {df.shape[1]} cols', flush=True)

    def counters_to_df(cat_counters, min_count=MIN_COUNT):
        d = pd.DataFrame({cat: pd.Series(dict(c)) for (cat, c) in cat_counters.items()}).fillna(0)
        keep = d.sum(axis=1) >= min_count
        return d[keep]

    def report_binary(d, cat_a, cat_b, name):
        delta = idp_module.idp(d[[cat_a, cat_b]], cat_a)
        top_a = delta.head(TOP_N)
        top_b = delta.tail(TOP_N).iloc[::-1]
        out = pd.DataFrame({f'top_{cat_a}_word': top_a.index, f'top_{cat_a}_z': top_a.values, f'top_{cat_b}_word': top_b.index, f'top_{cat_b}_z': top_b.values})
        out.to_csv(os.path.join(OUT_DIR, f'idp_{name}.csv'), index=False)
        return (top_a, top_b)
    d = counters_to_df(correct_counters)
    report_binary(d, 'correct', 'incorrect', 'correct_vs_incorrect')
    print('  wrote idp_correct_vs_incorrect.csv', flush=True)
    d = counters_to_df(chosen_counters)
    report_binary(d, 'minority', 'majority', 'chosen_minority_vs_majority')
    print('  wrote idp_chosen_minority_vs_majority.csv', flush=True)
    baseline_counter = method_counters['00_baseline']
    for method in METHODS:
        if method == '00_baseline':
            continue
        pair_df = counters_to_df({method: method_counters[method], 'BASELINE': baseline_counter})
        if pair_df.empty:
            print(f'  [WARN] {method} vs BASELINE: empty after MIN_COUNT filter', flush=True)
            continue
        delta = idp_module.idp(pair_df, method)
        top_method = delta.head(TOP_N)
        top_baseline = delta.tail(TOP_N).iloc[::-1]
        out = pd.DataFrame({f'top_{method}_word': top_method.index, f'top_{method}_z': top_method.values, 'top_BASELINE_word': top_baseline.index, 'top_BASELINE_z': top_baseline.values})
        out.to_csv(os.path.join(OUT_DIR, f'idp_method_{method}_vs_BASELINE.csv'), index=False)
        print(f'  wrote idp_method_{method}_vs_BASELINE.csv', flush=True)
    dt_total = time.time() - t_start
    print(f'\nTOTAL RUNTIME: {dt_total / 60:.1f} min', flush=True)
if __name__ == '__main__':
    main()
