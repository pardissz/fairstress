import os
import random
os.environ.setdefault('HF_HOME', '/shared/storage-01/huggingface')
LETTERS = 'ABCDEFGH'

def format_mc_prompt(question, choices):
    lines = [question, '']
    for (i, c) in enumerate(choices):
        lines.append(f'({LETTERS[i]}) {c}')
    lines.append('\nAnswer with only the letter of the correct choice.')
    lines.append('Answer:')
    return '\n'.join(lines)

def load_mmlu(n_per_subject=30, seed=0):
    import datasets
    ds = datasets.load_dataset('cais/mmlu', 'all', split='test')
    by_subject = {}
    for ex in ds:
        by_subject.setdefault(ex['subject'], []).append(ex)
    rng = random.Random(seed)
    items = []
    for (subject, exs) in sorted(by_subject.items()):
        rng.shuffle(exs)
        for ex in exs[:n_per_subject]:
            gold_idx = ex['answer']
            items.append({'id': f'mmlu_{subject}_{len(items)}', 'subject': subject, 'question': ex['question'], 'choices': ex['choices'], 'gold_letter': LETTERS[gold_idx]})
    print(f'[mmlu] {len(by_subject)} subjects, {len(items)} items (n_per_subject={n_per_subject})', flush=True)
    return items

def load_arc_challenge():
    import datasets
    ds = datasets.load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
    items = []
    skipped = 0
    for ex in ds:
        labels = ex['choices']['label']
        texts = ex['choices']['text']
        answer_key = ex['answerKey']
        if answer_key not in labels:
            skipped += 1
            continue
        gold_idx = labels.index(answer_key)
        items.append({'id': ex['id'], 'question': ex['question'], 'choices': texts, 'gold_letter': LETTERS[gold_idx]})
    print(f'[arc_challenge] {len(items)} items ({skipped} skipped, no valid answerKey)', flush=True)
    return items
