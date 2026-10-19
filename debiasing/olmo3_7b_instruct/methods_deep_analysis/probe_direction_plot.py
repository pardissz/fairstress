import argparse
import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_PATH = 'allenai/Olmo-3-7B-Instruct'
CALIB_PATH = 'AI2/debias_olmo3_7b_instruct/external_data/calibration_set.jsonl'
(BG, PANEL, INK, MUTED) = ('#0d0d0d', '#151513', '#f5f6f8', '#8b8f97')
(BLUE, ORANGE) = ('#3987e5', '#eb6834')

def load_calibration(path):
    items = [json.loads(l) for l in open(path)]
    prompts = [it['prompt'] for it in items]
    z = np.array([1 if it['minority_slot'] == 'A' else 0 for it in items])
    return (prompts, z)

@torch.no_grad()
def extract_hidden_states(prompts, layer, model_path=MODEL_PATH, batch_size=32, max_length=3800, device='cuda'):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.eval()
    chat_prompts = []
    for p in prompts:
        try:
            chat_prompts.append(tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, tokenize=False))
        except Exception:
            chat_prompts.append(p)
    feats = []
    for i in range(0, len(chat_prompts), batch_size):
        batch = chat_prompts[i:i + batch_size]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer + 1][:, -1, :]
        feats.append(hs.float().cpu().numpy())
        print(f'  [{i + len(batch)}/{len(chat_prompts)}]', flush=True)
    return np.concatenate(feats, axis=0)

def fit_and_project(X, z, seed=0):
    idx = np.arange(len(z))
    (idx_tr, idx_te) = train_test_split(idx, test_size=0.2, random_state=seed, stratify=z)
    (X_tr, X_te) = (X[idx_tr], X[idx_te])
    (z_tr, z_te) = (z[idx_tr], z[idx_te])
    clf = LogisticRegression(max_iter=2000).fit(X_tr, z_tr)
    test_acc = clf.score(X_te, z_te)
    w = clf.coef_[0]
    w_unit = w / np.linalg.norm(w)
    mean_vec = X.mean(axis=0)

    def proj_x(pts):
        return (pts - mean_vec) @ w_unit
    boundary_x = -clf.intercept_[0] / np.linalg.norm(w) - mean_vec @ w_unit
    resid = X - np.outer(X @ w_unit, w_unit)
    pca = PCA(n_components=1, random_state=seed).fit(resid)

    def proj_y(pts):
        r = pts - np.outer(pts @ w_unit, w_unit)
        return pca.transform(r)[:, 0]
    return {'x_tr': proj_x(X_tr), 'y_tr': proj_y(X_tr), 'z_tr': z_tr, 'x_te': proj_x(X_te), 'y_te': proj_y(X_te), 'z_te': z_te, 'boundary_x': boundary_x, 'test_acc': test_acc, 'n_train': len(idx_tr), 'n_test': len(idx_te)}

def plot(res, layer, out_path):
    plt.rcParams['font.family'] = ['IBM Plex Sans', 'DejaVu Sans', 'sans-serif']
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    gs = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05, figure=fig)
    ax_top = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1], sharex=ax_top)
    for ax in (ax_top, ax_main):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color('#333331')
        ax.tick_params(colors=MUTED, labelsize=9)
    (x_te, y_te, z_te) = (res['x_te'], res['y_te'], res['z_te'])
    (x_tr, y_tr, z_tr) = (res['x_tr'], res['y_tr'], res['z_tr'])
    xs = np.linspace(min(x_te.min(), x_tr.min()) - 0.4, max(x_te.max(), x_tr.max()) + 0.4, 400)
    labels = {0: 'minority candidate in slot B', 1: 'minority candidate in slot A'}
    colors = {0: BLUE, 1: ORANGE}
    for cls in (0, 1):
        mask = z_te == cls
        if mask.sum() > 1:
            kde = gaussian_kde(x_te[mask])
            ax_top.fill_between(xs, kde(xs), color=colors[cls], alpha=0.35, lw=0)
            ax_top.plot(xs, kde(xs), color=colors[cls], lw=1.6)
    ax_top.set_yticks([])
    ax_top.set_ylabel('density\n(held-out)', color=MUTED, fontsize=9)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    for cls in (0, 1):
        mask = z_tr == cls
        ax_main.scatter(x_tr[mask], y_tr[mask], s=14, color=colors[cls], alpha=0.12, edgecolors='none', zorder=1)
    for cls in (0, 1):
        mask = z_te == cls
        ax_main.scatter(x_te[mask], y_te[mask], s=34, color=colors[cls], alpha=0.75, edgecolors=BG, linewidths=0.4, zorder=3, label=labels[cls])
    ax_main.axvline(res['boundary_x'], color=INK, lw=1.25, ls='--', alpha=0.7, zorder=2)
    ylim = ax_main.get_ylim()
    ax_main.text(res['boundary_x'], ylim[1], "  probe's own\n  decision boundary", color=INK, fontsize=9, va='top')
    ax_main.set_xlabel(f"projection onto the fitted probe's own decision axis  (layer {layer})", color=MUTED, fontsize=10.5)
    ax_main.set_ylabel('top residual PC — orthogonal spread, for legibility only', color=MUTED, fontsize=10.5)
    ax_main.legend(loc='lower right', frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle("Both classes sit on both sides of the model's own linear boundary", color=INK, fontsize=17, fontweight='bold', x=0.065, ha='left', y=0.965)
    fig.text(0.065, 0.928, f"held-out probe accuracy at layer {layer}: {res['test_acc'] * 100:.1f}%   (majority-class baseline ≈ 51.7%)   ·   n_train={res['n_train']} (faint) · n_test={res['n_test']} (bold, held out)", color=MUTED, fontsize=10.5, ha='left')
    fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches='tight')
    print(f'[done] wrote {out_path}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=22, help="decoder layer to probe (0-31); 22 = best layer in the project's own sweep")
    ap.add_argument('--calib', default=CALIB_PATH)
    ap.add_argument('--model', default=MODEL_PATH)
    ap.add_argument('--out', default='concept_projection.png')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    (prompts, z) = load_calibration(args.calib)
    print(f'[data] {len(prompts)} calibration items, {int(z.sum())} minority-in-A / {int((1 - z).sum())} minority-in-B')
    print(f'[model] extracting layer {args.layer} hidden states from {args.model} ...')
    X = extract_hidden_states(prompts, args.layer, model_path=args.model)
    res = fit_and_project(X, z, seed=args.seed)
    print(f"[probe] held-out test accuracy = {res['test_acc']:.4f}")
    plot(res, args.layer, args.out)
if __name__ == '__main__':
    main()
