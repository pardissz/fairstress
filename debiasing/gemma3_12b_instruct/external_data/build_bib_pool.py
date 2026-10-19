import json
import random
from datasets import load_dataset
random.seed(1234)
N_TARGET = 6000
OUT_PATH = 'external_data/bib_pool.jsonl'

def main():
    ds = load_dataset('LabHC/bias_in_bios', split='train')
    idx = list(range(len(ds)))
    random.shuffle(idx)
    idx = idx[:N_TARGET * 2]
    out = []
    for i in idx:
        row = ds[i]
        text = row['hard_text']
        if not text or len(text) < 40:
            continue
        z = int(row['gender'])
        out.append({'item_id': f'bib_{i}', 'source': 'bias_in_bios', 'text': text[:800], 'z_label': z, 'profession': row.get('profession')})
        if len(out) >= N_TARGET:
            break
    with open(OUT_PATH, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    n1 = sum((r['z_label'] for r in out))
    print(f'Wrote {len(out)} items -> {OUT_PATH} (z=1: {n1}, z=0: {len(out) - n1})')
if __name__ == '__main__':
    main()
