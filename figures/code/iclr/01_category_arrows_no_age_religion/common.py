try:
    import duckdb
except ImportError:
    duckdb = None
import json, os
BASE = './data/lexicon_scoring'
SCORES = f'{BASE}/scores_full'
OUTDIR = f'{BASE}/analysis/cot_deep_investigation'
MODELS = ['OLMo-3-7B-Instruct', 'OLMo-3-7B-Instruct-DPO', 'OLMo-3-7B-Instruct-SFT', 'OLMo-3.1-32B-Instruct', 'Tulu-3-8B-SFT', 'Tulu-3-8B-SFT-no-safety-data', 'Tulu-3-8B', 'Tulu-3-8B-DPO', 'Tulu-3.1-8B', 'Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct']

def glob_for(model):
    return f'{SCORES}/{model}/shard_*.parquet'
TYPE1_CONTRASTS = {'race': ['blackM_whiteM', 'hispanicM_whiteM'], 'sex_gender': ['blackF_blackM', 'whiteF_whiteM', 'whtTransW_whiteF', 'whtTransW_whiteM', 'whtTransM_whiteM', 'blkTransW_blkM', 'hispTransW_hispM'], 'sexuality': ['gay_het', 'lesb_het'], 'race_x_sex': ['blackF_whiteM', 'blackF_whiteF', 'hispanicF_whiteM', 'arabF_whiteM', 'blkTransW_whiteM', 'hispTransW_whiteM', 'blkTransM_whiteM'], 'religion': ['muslimM_christianM', 'muslimF_christianF', 'jewishF_christianF', 'hinduF_christianF', 'sikhM_christianM', 'atheistM_christianM'], 'age': ['older_younger'], 'disability': ['disability_normal']}
TYPE1_LIST = [c for cats in TYPE1_CONTRASTS.values() for c in cats]
CONTRAST_TO_CATEGORY = {c: cat for (cat, cs) in TYPE1_CONTRASTS.items() for c in cs}
assert len(TYPE1_LIST) == 26, len(TYPE1_LIST)
EXCLUDED_CONTRASTS = ['arabF_blackF', 'arabM_hispanicM', 'hinduM_muslimM', 'jewishM_muslimM', 'atheistM_muslimF', 'jewishM_christianF', 'muslimM_jewishF']
assert len(EXCLUDED_CONTRASTS) == 7
DIRECTIONAL_FAMILIES = ['F1_authority', 'F2_social_consensus', 'F3_role_assignment', 'F5_guilt_historical', 'F6_evidential_claims', 'F7_emotional_appeals', 'F8_legal_framing', 'F9_statistical_framing', 'F10_demographic_salience']
NEUTRAL_FAMILIES = {'N1_evaluation_awareness': 'neutral_N1', 'N2_stakes_framing': 'neutral_N2', 'N3_reactance': 'neutral_N3'}
CONTROL_FAMILIES = {'C2_neutral_relevant': 'control_C2C3', 'C3_neutral_irrelevant': 'control_C2C3', 'C4_placebo': 'control_C4'}
PARSING_SQL = "\n, _anchor AS (\n    SELECT *,\n        '__' || condition || '__' AS _anchor_str\n    FROM _raw\n),\n_split AS (\n    SELECT *,\n        strpos(item_id, _anchor_str) AS _apos,\n        length(_anchor_str) AS _alen\n    FROM _anchor\n),\n_parts AS (\n    SELECT *,\n        substr(item_id, 1, _apos - 1) AS scenario_id,\n        substr(item_id, _apos + _alen) AS _remainder\n    FROM _split\n    WHERE _apos > 0\n),\n_tok AS (\n    SELECT *,\n        str_split(_remainder, '__') AS _toks\n    FROM _parts\n),\nparsed AS (\n    SELECT\n        * EXCLUDE (_toks, _remainder, _apos, _alen, _anchor_str),\n        regexp_replace(_toks[1], 'P1$', '') AS slot_p1,\n        _toks[2] AS contrast_id,\n        regexp_replace(_toks[3], '1$', '') AS slot_code,\n        CAST(regexp_replace(_toks[4], '^lvl', '') AS INTEGER) AS parsed_level,\n        array_to_string(list_slice(_toks, 5, len(_toks)), '__') AS inj_id,\n        -- minority-coded slot letter: if slot_p1's code == 'M', slot_p1 IS minority;\n        -- if code == 'J', slot_p1 is majority so the OTHER letter is minority.\n        CASE WHEN regexp_replace(_toks[3], '1$', '') = 'M' THEN regexp_replace(_toks[1], 'P1$', '')\n             WHEN regexp_replace(_toks[1], 'P1$', '') = 'A' THEN 'B'\n             ELSE 'A' END AS minority_letter\n    FROM _tok\n)\n"

def pressure_condition_case_sql():
    directional_cases = '\n'.join((f"        WHEN injection_family = '{fam}' THEN (CASE WHEN injection_direction = 'pro_' || minority_letter THEN 'directional_pro_minority' ELSE 'directional_pro_majority' END)" for fam in DIRECTIONAL_FAMILIES))
    neutral_cases = '\n'.join((f"        WHEN injection_family = '{fam}' THEN '{pc}'" for (fam, pc) in NEUTRAL_FAMILIES.items()))
    control_cases = '\n'.join((f"        WHEN injection_family = '{fam}' THEN '{pc}'" for (fam, pc) in CONTROL_FAMILIES.items()))
    return f"\n    CASE\n        WHEN injection_family IS NULL OR is_baseline THEN 'null'\n{directional_cases}\n{neutral_cases}\n{control_cases}\n        ELSE 'UNMAPPED'\n    END\n    "

def type1_in_clause():
    vals = ','.join((f"'{c}'" for c in TYPE1_LIST))
    return f'({vals})'

def base_cte(glob):
    return f"WITH _raw AS (SELECT * FROM read_parquet('{glob}')){PARSING_SQL}"
_LEX_COLS_CACHE = None

def lexicon_columns(con=None):
    global _LEX_COLS_CACHE
    if _LEX_COLS_CACHE is not None:
        return _LEX_COLS_CACHE
    cache_file = f'{OUTDIR}/scripts/_lexcols.json'
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            _LEX_COLS_CACHE = json.load(f)
        return _LEX_COLS_CACHE
    if con is None:
        con = duckdb.connect()
    meta_cols = {'item_id', 'domain', 'condition', 'signaling_level', 'contrast_category', 'is_baseline', 'chosen_group', 'is_correct', 'injection_family', 'injection_direction', 'parsed_choice', 'n_words'}
    df = con.execute(f"DESCRIBE SELECT * FROM '{glob_for(MODELS[0])}' LIMIT 1".replace('shard_*', 'shard_000')).fetchdf()
    cols = [c for c in df['column_name'].tolist() if c not in meta_cols]
    with open(cache_file, 'w') as f:
        json.dump(cols, f)
    _LEX_COLS_CACHE = cols
    return cols
PREFIX_ORDER = ['LIWC', 'MFD', 'EMFD', 'NRC', 'MIND', 'EMPATH', 'AGENCYCOMM', 'GENDER', 'AGEGENDER', 'CONCR', 'VULN']

def prefix_of(col):
    for p in PREFIX_ORDER:
        if col.startswith(p + '_'):
            return p
    return 'OTHER'
MODEL_COLORS = {'OLMo-3-7B-Instruct': '#4C72B0', 'OLMo-3-7B-Instruct-DPO': '#DD8452', 'OLMo-3-7B-Instruct-SFT': '#55A868', 'OLMo-3.1-32B-Instruct': '#C44E52', 'Tulu-3-8B-SFT': '#8172B2', 'Tulu-3-8B-SFT-no-safety-data': '#937860', 'Tulu-3-8B': '#DA8BC3', 'Tulu-3-8B-DPO': '#8C8C8C', 'Tulu-3.1-8B': '#CCB974', 'Llama-3.1-8B-Instruct': '#64B5CD', 'Qwen2.5-7B-Instruct': '#B4413C'}
QUAL_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000', '#8172B2', '#937860']
if __name__ == '__main__':
    con = duckdb.connect()
    cols = lexicon_columns(con)
    print('n lexicon cols:', len(cols))
    assert len(cols) == 264, len(cols)
