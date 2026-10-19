import json
import random
from collections import defaultdict
random.seed(777)
CALIB_FRAC = 0.2
CALIB_TARGET = 1200

def load(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def stratified_split(items, frac):
    groups = defaultdict(list)
    for it in items:
        key = (it['contrast_category'], it['condition'])
        groups[key].append(it)
    (fit, calib) = ([], [])
    for (key, rows) in groups.items():
        random.shuffle(rows)
        n_calib = max(1, int(len(rows) * frac))
        calib.extend(rows[:n_calib])
        fit.extend(rows[n_calib:])
    return (fit, calib)

def main():
    bbq = load('external_data/bbq_pool.jsonl')
    synth = load('external_data/synthetic_pool.jsonl')
    (bbq_fit, bbq_calib) = stratified_split(bbq, CALIB_FRAC)
    (synth_fit, synth_calib) = stratified_split(synth, CALIB_FRAC)
    with open('external_data/fit_bbq.jsonl', 'w') as f:
        for r in bbq_fit:
            f.write(json.dumps(r) + '\n')
    with open('external_data/fit_synth.jsonl', 'w') as f:
        for r in synth_fit:
            f.write(json.dumps(r) + '\n')
    calib_pool = bbq_calib + synth_calib
    groups = defaultdict(list)
    for it in calib_pool:
        groups[it['contrast_category'], it['condition']].append(it)
    total = len(calib_pool)
    calib_final = []
    for (key, rows) in groups.items():
        random.shuffle(rows)
        n = max(1, round(len(rows) / total * CALIB_TARGET))
        calib_final.extend(rows[:n])
    random.shuffle(calib_final)
    with open('external_data/calibration_set.jsonl', 'w') as f:
        for r in calib_final:
            f.write(json.dumps(r) + '\n')
    manifest = {}
    for r in bbq_fit + synth_fit:
        manifest[r['item_id']] = 'fit'
    for r in calib_final:
        manifest[r['item_id']] = 'calib'
    with open('external_data/split_manifest.json', 'w') as f:
        json.dump(manifest, f)
    print(f'bbq: fit={len(bbq_fit)} calib_pool={len(bbq_calib)}')
    print(f'synth: fit={len(synth_fit)} calib_pool={len(synth_calib)}')
    print(f'calibration_set.jsonl: {len(calib_final)} items')
    from collections import Counter
    print('calib by (category,condition):', Counter(((r['contrast_category'], r['condition']) for r in calib_final)))
    assert set((r['item_id'] for r in calib_final)).isdisjoint(set((r['item_id'] for r in bbq_fit + synth_fit))), 'FIT/CALIB overlap!'
    print('Disjointness check passed: no item_id appears in both fit and calib.')
if __name__ == '__main__':
    main()
