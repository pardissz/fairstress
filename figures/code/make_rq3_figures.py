import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
OUT = 'docs/figures'
import os
os.makedirs(OUT, exist_ok=True)

def fig1_master_heatmap():
    df = pd.read_csv('data/generated/round38/master_demographic_framing_map.csv')
    stages = ['pretrain', 'sft', 'dpo_chosen', 'dpo_rejected', 'rlvr']
    stage_labels = ['Pretrain', 'SFT', 'DPO-chosen', 'DPO-rejected', 'RLVR']
    net = pd.DataFrame(index=df['group'], columns=stage_labels, dtype=float)
    n_annot = pd.DataFrame(index=df['group'], columns=stage_labels, dtype=object)
    for (stage, label) in zip(stages, stage_labels):
        (pos_col, neg_col, n_col) = (f'{stage}_pos_pct', f'{stage}_neg_pct', f'{stage}_n')
        pos_num = pd.to_numeric(df[pos_col], errors='coerce')
        neg_num = pd.to_numeric(df[neg_col], errors='coerce')
        n_num = pd.to_numeric(df[n_col], errors='coerce')
        for (idx, row) in df.iterrows():
            (pos, neg, n) = (pos_num.iloc[idx], neg_num.iloc[idx], n_num.iloc[idx])
            if pd.isna(pos) or pd.isna(neg):
                net.loc[row['group'], label] = np.nan
                n_annot.loc[row['group'], label] = 'n/a'
            else:
                net.loc[row['group'], label] = pos - neg
                n_annot.loc[row['group'], label] = f"{net.loc[row['group'], label]:+.1f}\n(n={int(n)})"
    net = net.reindex(net['Pretrain'].sort_values(ascending=False).index)
    n_annot = n_annot.reindex(net.index)
    (fig, ax) = plt.subplots(figsize=(9, 11))
    cmap = sns.diverging_palette(10, 220, s=75, l=50, as_cmap=True)
    sns.heatmap(net.astype(float), annot=n_annot, fmt='', cmap=cmap, center=0, vmin=-40, vmax=60, linewidths=1, linecolor='white', cbar_kws={'label': 'Net framing (positive% − negative%)'}, ax=ax)
    ax.set_title('RQ3 Method 1: Net demographic framing by training stage\n(LLM stance classification, real n shown per cell)', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig1_master_heatmap.pdf')
    plt.savefig(f'{OUT}/fig1_master_heatmap.png', dpi=200)
    plt.close()

def fig2_mfd_vs_stance():
    mfd = pd.read_csv('data/generated/round41/mfd_liwc_training_data_scores.csv')
    stance = pd.read_csv('data/generated/round38/master_demographic_framing_map.csv')
    mfd_pre = mfd[(mfd['corpus'] == 'pretrain') & (mfd['status'] == 'ok')].copy()
    mfd_pre = mfd_pre.sort_values('mfd_vice_virtue_ratio', ascending=False)
    stance_map = dict(zip(stance['axis_group'], stance['pretrain_pos_pct']))
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(13, 9), sharey=False)
    colors = ['#c0392b' if g in ('race:white', 'trans_status:cisgender') else '#7f8c8d' for g in mfd_pre['axis_group']]
    ax1.barh(mfd_pre['axis_group'], mfd_pre['mfd_vice_virtue_ratio'], color=colors)
    ax1.invert_yaxis()
    ax1.set_xlabel('MFD Vice/Virtue ratio (higher = more vice-coded)')
    ax1.set_title('Method 3: MFD/LIWC lexical density\n(pretraining, n=1000 every group)')
    stance_vals = [pd.to_numeric(stance_map.get(g), errors='coerce') for g in mfd_pre['axis_group']]
    ax2.barh(mfd_pre['axis_group'], stance_vals, color=colors)
    ax2.invert_yaxis()
    ax2.set_xlabel('Method 1: % classified PROTECTIVE (pretraining)')
    ax2.set_title('Method 1: LLM stance classification\n(same group order as left panel)')
    plt.suptitle('Cross-method disagreement: which groups rank as most vice-coded (left)\nvs. most protected (right)? White/cisgender highlighted red.', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_mfd_vs_stance.pdf')
    plt.savefig(f'{OUT}/fig2_mfd_vs_stance.png', dpi=200)
    plt.close()

def fig3_word_ranking(group='white', corpus='pretrain', top_n=10):
    df = pd.read_csv('data/generated/round41/complete_word_rankings.csv')
    sub = df[(df['group'] == group) & (df['corpus'] == corpus) & (df['status'] == 'ok')]
    sub = sub.sort_values('rank').head(top_n)
    (fig, ax) = plt.subplots(figsize=(7, 5))
    ax.barh(sub['word'], sub['log_odds_z'], color='#2c3e50')
    ax.invert_yaxis()
    ax.set_xlabel('Monroe (2008) informative-Dirichlet-prior log-odds z-score')
    ax.set_title(f"Complete, unfiltered top-{top_n} words: '{group}', {corpus}\n(n_window_tokens={int(sub['n_window_tokens'].iloc[0]):,})")
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_word_ranking_{group}_{corpus}.pdf')
    plt.savefig(f'{OUT}/fig3_word_ranking_{group}_{corpus}.png', dpi=200)
    plt.close()

def fig4_agent_victim():
    pairings = ['White↔Black', 'Man↔Woman', 'Non-disabled↔Disabled', 'Cisgender↔Transgender']
    agent_majority = [15, 51, 0, 0]
    agent_minority = [0, 16, 0, 0]
    patient_minority = [13, 64, 0, 0]
    patient_majority = [0, 15, 0, 0]
    n_labels = ['n=53\n(full-corpus)', 'n=144\n(full-corpus)', 'n=4\n(source-restricted\ncandidates checked)', 'n=4\n(source-restricted\ncandidates checked)']
    x = np.arange(len(pairings))
    width = 0.2
    (fig, ax) = plt.subplots(figsize=(10, 6.5))
    ax.bar(x - 1.5 * width, agent_majority, width, label='Agent = majority-coded group', color='#c0392b')
    ax.bar(x - 0.5 * width, agent_minority, width, label='Agent = minority-coded group', color='#e67e22')
    ax.bar(x + 0.5 * width, patient_minority, width, label='Patient = minority-coded group', color='#2980b9')
    ax.bar(x + 1.5 * width, patient_majority, width, label='Patient = majority-coded group', color='#27ae60')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}\n{n}' for (p, n) in zip(pairings, n_labels)], fontsize=9)
    ax.set_ylabel('Count of classified records')
    ax.set_title('Method 4: agent/victim role-assignment, full-corpus scan,\nall 4 pairings tested (disability/trans-status: clean null)')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_agent_victim.pdf')
    plt.savefig(f'{OUT}/fig4_agent_victim.png', dpi=200)
    plt.close()

def fig5_refusal_density():
    labels = ['WildGuardMix + WildJailbreak\nrefusal-pattern examples\n(in Dolci-Instruct-SFT)', 'FDEC clusters carrying ANY\n"judge by merit" framing\n(of 28,000 indexed)']
    values = [99338, 9]
    (fig, ax) = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, values, color=['#c0392b', '#2980b9'], log=True)
    ax.set_xlabel('Count (log scale)')
    ax.set_title('Method 5: refusal-pattern volume vs. merit-content volume\n(same indexed SFT+DPO corpus, ~4 orders of magnitude apart)')
    for (bar, v) in zip(bars, values):
        ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2, f'{v:,}', va='center', fontsize=11, fontweight='bold')
    ax.text(9 * 1.15, 1, '  (all 9 are themselves refusals\n   that pivot to merit-language —\n   none are organic decision scenes)', va='top', fontsize=8, color='#555555')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_refusal_density.pdf')
    plt.savefig(f'{OUT}/fig5_refusal_density.png', dpi=200)
    plt.close()

def fig6_filtered_word_counts():
    df = pd.read_csv('data/generated/round38/top_words_by_valence.csv')
    pre = df[(df['stage'] == 'pretrain') & df['word'].notna()]
    groups = ['White people', 'cisgender people', 'men', 'non-disabled people', 'women', 'Black people', 'disabled people', 'transgender people']
    (pos_counts, neg_counts) = ([], [])
    for g in groups:
        sub = pre[pre['group'] == g]
        pos_counts.append((sub['valence'] == 'POSITIVE').sum())
        neg_counts.append((sub['valence'] == 'NEGATIVE').sum())
    x = np.arange(len(groups))
    width = 0.38
    (fig, ax) = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, pos_counts, width, label='Positive-context words', color='#2980b9')
    ax.bar(x + width / 2, neg_counts, width, label='Negative-context words', color='#c0392b')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha='right')
    ax.set_ylabel('Count of words (of top ~20-40 ranked candidates, valence-classified)')
    ax.set_title("Figure 6 (filtered view, contrast with Fig. 3's unfiltered ranking):\ncount of positive- vs. negative-labeled words per group, pretraining")
    ax.set_yticks(range(0, int(max(pos_counts + neg_counts)) + 2))
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig6_filtered_word_counts.pdf')
    plt.savefig(f'{OUT}/fig6_filtered_word_counts.png', dpi=200)
    plt.close()

def fig7_term_list_confound():
    df = pd.read_csv('data/generated/round39/corrected_percentages.csv')
    df = df[df['corpus'] == 'pretrain'].copy()
    full = pd.read_csv('data/generated/round39/corrected_percentages.csv')
    white = full[full['axis_group'] == 'race:white'].copy()
    (fig, axes) = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = df['axis_group'].str.replace('gender:', '').str.replace('age:', '').str.replace('race:', '')
    net_orig = df['pos_original'] - df['neg_original']
    net_clean = df['pos_clean_only'] - df['neg_clean_only']
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, net_orig, width, label='Original (loaded phrase included)', color='#7f8c8d')
    axes[0].bar(x + width / 2, net_clean, width, label='Clean terms only (loaded phrase excluded)', color='#2980b9')
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Net framing (positive% − negative%)')
    axes[0].set_title('Pretraining: all 4 groups with a\nvalence-loaded search term')
    axes[0].legend(fontsize=8, loc='upper left')
    stages = white['corpus'].tolist()
    net_orig_w = (white['pos_original'] - white['neg_original']).tolist()
    net_clean_w = (white['pos_clean_only'] - white['neg_clean_only']).tolist()
    xw = np.arange(len(stages))
    axes[1].bar(xw - width / 2, net_orig_w, width, label='Original', color='#7f8c8d')
    axes[1].bar(xw + width / 2, net_clean_w, width, label='Clean terms only', color='#c0392b')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xticks(xw)
    axes[1].set_xticklabels(stages, rotation=20, ha='right')
    axes[1].set_title('White people, all 4 stages:\ncorrection tightens, does not soften, the pattern')
    axes[1].legend(fontsize=8)
    plt.suptitle('Round 39: is the pattern a search-term artifact? (checked, not assumed)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig7_term_list_confound.pdf')
    plt.savefig(f'{OUT}/fig7_term_list_confound.png', dpi=200)
    plt.close()

def fig8_hand_audit_tally():
    import json
    rows = [json.loads(l) for l in open('data/generated/round40_manual_review/pretrain_and_posttrain_examples.jsonl')]
    rows += [json.loads(l) for l in open('data/generated/round40_manual_review/black_examples.jsonl')]
    dfh = pd.DataFrame(rows)
    dfh = dfh[dfh['framing'].isin(['protective', 'negative', 'neutral'])]
    corpora = ['pretrain', 'sft', 'dpo_chosen', 'dpo_rejected']
    corpus_labels = ['Pretrain', 'SFT', 'DPO-chosen', 'DPO-rejected']
    groups = sorted(dfh['axis_group'].unique())
    (fig, axes) = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), sharey=True)
    colors = {'protective': '#2980b9', 'neutral': '#bdc3c7', 'negative': '#c0392b'}
    for (ax, g) in zip(axes, groups):
        sub = dfh[dfh['axis_group'] == g]
        bottoms = np.zeros(len(corpora))
        for framing in ['protective', 'neutral', 'negative']:
            vals = [((sub['corpus'] == c) & (sub['framing'] == framing)).sum() for c in corpora]
            ax.bar(corpus_labels, vals, bottom=bottoms, color=colors[framing], label=framing)
            bottoms += np.array(vals)
        ax.set_title(g, fontsize=11)
        ax.tick_params(axis='x', rotation=30)
    axes[0].set_ylabel('Count of hand-audited real excerpts')
    axes[-1].legend(loc='upper right', fontsize=9)
    plt.suptitle(f'Round 40: hand-audit tally, {len(dfh):,} real excerpts with full text, 4 groups\n(saved at data/generated/round40_manual_review/, not summarized from labels alone)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig8_hand_audit_tally.pdf')
    plt.savefig(f'{OUT}/fig8_hand_audit_tally.png', dpi=200)
    plt.close()

def fig3b_all_groups_word_ranking():
    df = pd.read_csv('data/generated/round41/complete_word_rankings.csv')
    df_pre = df[df['corpus'] == 'pretrain']
    group_order = ['white', 'black', 'hispanic_latino', 'east_asian', 'south_asian', 'arab', 'christian', 'muslim', 'jewish', 'hindu', 'sikh', 'atheist', 'man', 'woman', 'heterosexual', 'gay', 'lesbian', 'bisexual', 'non_disabled', 'disabled', 'cisgender', 'transgender', 'older', 'younger']
    (fig, axes) = plt.subplots(4, 6, figsize=(26, 17))
    axes = axes.flatten()
    for (i, g) in enumerate(group_order):
        ax = axes[i]
        sub = df_pre[(df_pre['group'] == g) & (df_pre['status'] == 'ok')].sort_values('rank').head(10)
        if len(sub) == 0:
            ax.text(0.5, 0.5, 'insufficient_data\n(pretraining)', ha='center', va='center', fontsize=11, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(g, fontsize=11)
            continue
        ax.barh(sub['word'], sub['log_odds_z'], color='#2c3e50')
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=8.5)
        ax.tick_params(axis='x', labelsize=7.5)
        ax.set_title(g, fontsize=11)
    plt.suptitle('Complete unfiltered top-10 word rankings, ALL 24 groups, pretraining\n(Monroe 2008 informative-Dirichlet-prior log-odds z-score, real ranking, no valence curation)', fontsize=15, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'{OUT}/fig3b_all_groups_word_ranking.pdf', bbox_inches='tight')
    plt.savefig(f'{OUT}/fig3b_all_groups_word_ranking.png', dpi=170, bbox_inches='tight')
    plt.close()

def fig8b_all_groups_tally():
    import json
    rows = [json.loads(l) for l in open('data/generated/round42_all_groups/all_groups_examples.jsonl')]
    dfh = pd.DataFrame(rows)
    dfh = dfh[dfh['framing'].isin(['protective', 'negative', 'neutral'])]
    group_order = ['race:white', 'race:black', 'race:hispanic_latino', 'race:east_asian', 'race:south_asian', 'race:arab', 'religion:christian', 'religion:muslim', 'religion:jewish', 'religion:hindu', 'religion:sikh', 'religion:atheist', 'gender:man', 'gender:woman', 'orientation:heterosexual', 'orientation:gay', 'orientation:lesbian', 'orientation:bisexual', 'disability:non_disabled', 'disability:disabled', 'trans_status:cisgender', 'trans_status:transgender', 'age:older', 'age:younger']
    corpora = ['pretrain', 'sft', 'dpo_chosen', 'dpo_rejected']
    corpus_labels = ['Pre', 'SFT', 'DPO-c', 'DPO-r']
    colors = {'protective': '#2980b9', 'neutral': '#bdc3c7', 'negative': '#c0392b'}
    (fig, axes) = plt.subplots(4, 6, figsize=(24, 16))
    axes = axes.flatten()
    for (i, g) in enumerate(group_order):
        ax = axes[i]
        sub = dfh[dfh['axis_group'] == g]
        bottoms = np.zeros(len(corpora))
        for framing in ['protective', 'neutral', 'negative']:
            vals = [((sub['corpus'] == c) & (sub['framing'] == framing)).sum() for c in corpora]
            ax.bar(corpus_labels, vals, bottom=bottoms, color=colors[framing], width=0.7)
            bottoms += np.array(vals)
        ax.set_title(g, fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ['protective', 'neutral', 'negative']]
    fig.legend(handles, ['protective', 'neutral', 'negative'], loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 1.0))
    plt.suptitle(f'Hand-audit tally, ALL 24 taxonomy groups, {len(dfh):,} real classified excerpts\n(pretrain / SFT / DPO-chosen / DPO-rejected, real text saved at data/generated/round42_all_groups/)', fontsize=14, y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{OUT}/fig8b_all_groups_tally.pdf', bbox_inches='tight')
    plt.savefig(f'{OUT}/fig8b_all_groups_tally.png', dpi=170, bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    fig1_master_heatmap()
    fig2_mfd_vs_stance()
    fig3_word_ranking(group='white', corpus='pretrain')
    fig3_word_ranking(group='disabled', corpus='pretrain')
    fig3b_all_groups_word_ranking()
    fig4_agent_victim()
    fig5_refusal_density()
    fig6_filtered_word_counts()
    fig7_term_list_confound()
    fig8_hand_audit_tally()
    fig8b_all_groups_tally()
    print(f'All figures written to {OUT}/')
