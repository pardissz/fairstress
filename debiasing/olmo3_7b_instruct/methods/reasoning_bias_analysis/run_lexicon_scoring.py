import glob
import json
import os
import sys
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
LEXICON_SCORING_DIR = './data/lexicon_scoring'
CONNOTATION_DIR = './data/resources/lexica/CONNOTATION'
FIRMCORE = './data/lexicon_scoring/firmCore.jsonl'
REASONING_DIR = os.path.join(PROJECT_ROOT, 'firmcore_eval', 'reasoning')
OUT_PATH = os.path.join(HERE, 'reasoning_bias_summary.json')
sys.path.insert(0, LEXICON_SCORING_DIR)
import lexscore_full as lf

def load_firmcore_meta():
    meta = {}
    with open(FIRMCORE) as f:
        for line in f:
            d = json.loads(line)
            meta[d['item_id']] = {'condition': d['condition'], 'signaling_level': d['signaling_level'], 'minority_slot': d['minority_slot'], 'contrast_category': d['contrast_category'], 'expected_correct': d.get('expected_correct'), 'injection_resolved': d.get('injection_resolved')}
    return meta

def build_mfd_only_lexicon():
    from MORALF import mfUtil as moralF
    sources = {'MFD': moralF.parseMF()}
    combined = {}
    for (prefix, lex) in sources.items():
        for (w, cats) in lex.items():
            w_cats = lf._as_weighted(cats)
            d = combined.setdefault(w, {})
            for (c, wt) in w_cats.items():
                d[f'{prefix}_{c}'] = wt
    (exact_dict, bigram_dict, trie) = ({}, {}, {})
    for (w, cats) in combined.items():
        if ' ' in w:
            bigram_dict[w] = cats
        elif '*' in w:
            stem = w.replace('*', '')
            if len(stem) < 3:
                continue
            node = trie
            for ch in stem:
                node = node.setdefault(ch, {})
            node.setdefault('$', {}).update(cats)
        else:
            exact_dict[w] = cats
    return (exact_dict, bigram_dict, trie)

def build_connotation_dict(nlp):
    import pandas as pd
    convert_labels = {'=': 'equal', '+': 'pos', '-': 'neg'}
    dfs = []
    for (frame, fname) in [('agency', 'agency_verb.csv'), ('authority', 'authority_verb.csv')]:
        df = pd.read_csv(os.path.join(CONNOTATION_DIR, fname))
        df = df.iloc[:, :2]
        df.columns = ['verb', frame]
        df[frame] = df[frame].apply(lambda x: frame + '_' + convert_labels.get(x, x).lower())
        df.set_index('verb', inplace=True)
        dfs.append(df)
    df = __import__('pandas').concat(dfs, axis=1, sort=True).reset_index().rename(columns={'index': 'verb'})
    words_to_cats = {}
    for (verb, agency_cat, authority_cat) in zip(df['verb'], df.get('agency'), df.get('authority')):
        verb_str = str(verb)
        if ' ' in verb_str:
            continue
        doc = nlp(verb_str)
        lemma = doc[0].lemma_.lower() if len(doc) else verb_str.lower()
        cats = [c for c in (agency_cat, authority_cat) if isinstance(c, str)]
        if cats:
            words_to_cats.setdefault(lemma, []).extend(cats)
    return words_to_cats
CONNOTATION_CATS = ['agency_pos', 'agency_neg', 'agency_equal', 'authority_agent', 'authority_theme', 'authority_equal']

def score_method(path, meta, exact_dict, bigram_dict, trie, nlp, verb2cats):
    rows = []
    texts = []
    infos = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            item_id = d['item_id']
            m = meta.get(item_id)
            if m is None:
                continue
            reasoning = d.get('reasoning') or ''
            infos.append((item_id, d.get('parsed_choice'), m))
            texts.append(reasoning)
    mfd_rows = []
    for ((item_id, choice, m), text) in zip(infos, texts):
        (sums, _match_counts, n_words) = lf.score_doc_weighted(text, exact_dict, bigram_dict, trie)
        row = {k: v / n_words if n_words else 0.0 for (k, v) in sums.items()}
        row['n_words'] = n_words
        mfd_rows.append(row)
    conn_rows = []
    for doc in nlp.pipe(texts, batch_size=256):
        counts = {c: 0 for c in CONNOTATION_CATS}
        n_verbs = 0
        for tok in doc:
            if tok.pos_ == 'VERB':
                n_verbs += 1
                for c in verb2cats.get(tok.lemma_.lower(), ()):
                    counts[c] += 1
        conn_rows.append({c: counts[c] / n_verbs if n_verbs else 0.0 for c in CONNOTATION_CATS})
    for ((item_id, choice, m), mfd_row, conn_row) in zip(infos, mfd_rows, conn_rows):
        chosen_group = None
        if choice in ('A', 'B'):
            chosen_group = 'minority' if choice == m['minority_slot'] else 'majority'
        is_correct = None
        expected = m['expected_correct']
        if expected is not None and choice in ('A', 'B'):
            subj = expected.split()[0]
            is_correct = f'{subj} {choice}' == expected
        row = {'item_id': item_id, 'condition': m['condition'], 'signaling_level': m['signaling_level'], 'chosen_group': chosen_group, 'is_correct': is_correct}
        row.update(mfd_row)
        row.update(conn_row)
        rows.append(row)
    return rows

def summarize(rows):
    import statistics

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None
    vice_correct = [r.get('MFD_Vice', 0.0) for r in rows if r['is_correct'] is True]
    vice_incorrect = [r.get('MFD_Vice', 0.0) for r in rows if r['is_correct'] is False]
    virtue_correct = [r.get('MFD_Virtue', 0.0) for r in rows if r['is_correct'] is True]
    virtue_incorrect = [r.get('MFD_Virtue', 0.0) for r in rows if r['is_correct'] is False]
    fairness = [r.get('MFD_Fairness', 0.0) for r in rows]
    fairness_min = [r.get('MFD_Fairness', 0.0) for r in rows if r['chosen_group'] == 'minority']
    fairness_maj = [r.get('MFD_Fairness', 0.0) for r in rows if r['chosen_group'] == 'majority']
    agency_min = [r.get('agency_pos', 0.0) - r.get('agency_neg', 0.0) for r in rows if r['chosen_group'] == 'minority']
    agency_maj = [r.get('agency_pos', 0.0) - r.get('agency_neg', 0.0) for r in rows if r['chosen_group'] == 'majority']
    (vc, vi) = (mean(vice_correct), mean(vice_incorrect))
    return {'n': len(rows), 'MFD_Vice_mean_correct': vc, 'MFD_Vice_mean_incorrect': vi, 'MFD_Vice_ratio_incorrect_over_correct': vi / vc if vc and vi is not None and (vc > 0) else None, 'MFD_Virtue_mean_correct': mean(virtue_correct), 'MFD_Virtue_mean_incorrect': mean(virtue_incorrect), 'MFD_Fairness_mean_overall': mean(fairness), 'MFD_Fairness_mean_minority_chosen': mean(fairness_min), 'MFD_Fairness_mean_majority_chosen': mean(fairness_maj), 'agency_net_mean_minority_chosen': mean(agency_min), 'agency_net_mean_majority_chosen': mean(agency_maj)}

def main():
    print('[setup] loading firmCore metadata ...', flush=True)
    meta = load_firmcore_meta()
    print(f'  {len(meta)} items', flush=True)
    print('[setup] building MFD lexicon ...', flush=True)
    (exact_dict, bigram_dict, trie) = build_mfd_only_lexicon()
    print('[setup] loading spaCy + connotation frames ...', flush=True)
    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    verb2cats = build_connotation_dict(nlp)
    print(f'  {len(verb2cats)} verb lemmas with connotation labels', flush=True)
    paths = sorted(glob.glob(os.path.join(REASONING_DIR, '*_reasoning.jsonl')))
    print(f'[run] {len(paths)} reasoning files found: {[os.path.basename(p) for p in paths]}', flush=True)
    summary = {}
    for path in paths:
        method = os.path.basename(path).replace('_reasoning.jsonl', '')
        print(f'[score] {method} ...', flush=True)
        rows = score_method(path, meta, exact_dict, bigram_dict, trie, nlp, verb2cats)
        summary[method] = summarize(rows)
        print(f'  -> {summary[method]}', flush=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[done] wrote {OUT_PATH}', flush=True)
if __name__ == '__main__':
    main()
