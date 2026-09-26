import sys
import os
import time
import gc
import re
import subprocess
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, r'd:\amazon challenge\business-entity-resolution\code\business_entity_resolution')

import pandas as pd
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz

from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_MATCHING, OUTPUT_CANDIDATE,
    VALIDATE_SCRIPT, TEST_DIR,
    RANDOM_STATE
)
from src.preprocessing.normalize import normalize_name, normalize_address, extract_numbers
from src.evaluation.metrics import compute_macro_f05

# ----------------------------------------------------
# ADVANCED CONSTANTS & STOPWORDS
# ----------------------------------------------------
STOPWORDS = {
    'the', 'and', 'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'co', 'company', 'services', 'service', 'group', 'enterprises',
    'enterprise', 'technologies', 'technology', 'india', 'us', 'usa', 'france',
    'rd', 'road', 'st', 'street', 'ave', 'avenue', 'nagar', 'bldg', 'floor',
    'com', 'net', 'org', 'www', 'private', 'partners', 'holdings', 'industries',
    'center', 'centre', 'solutions', 'public', 'ventures', 'associates', 'trading',
    'shri', 'brothers', 'sar', 'sarl', 'sas', 'sasu', 'eurl', 'llp'
}

BEST_LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': 9,
    'min_child_samples': 40,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'n_estimators': 350,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}

FEATURE_NAMES = [
    'name_exact', 'name_ratio', 'name_token_sort', 'name_token_set',
    'name_jaccard', 'name_overlap', 'name_len_diff', 'name_tok_diff',
    'addr_exact', 'addr_ratio', 'addr_token_set', 'addr_jaccard',
    'addr_overlap', 'addr_len_diff', 'shared_numbers', 'has_same_num',
    'same_country', 'is_s2', 'name_addr_mean', 'has_high_name', 'has_high_addr'
]


def clean_tokens(text: str, min_len: int = 3) -> set:
    if not text:
        return set()
    cleaned = re.sub(r'\.(com|net|org|in|us|fr|co)', ' ', text.lower())
    tokens = re.findall(r'[a-z0-9]{' + str(min_len) + r',}', cleaned)
    return set(t for t in tokens if t not in STOPWORDS)


def compute_fast_features(s1_name, s1_addr, c_name, c_addr, s1_country, c_country, c_src):
    s1_ntoks = clean_tokens(s1_name, 3)
    c_ntoks = clean_tokens(c_name, 3)
    s1_atoks = clean_tokens(s1_addr, 3)
    c_atoks = clean_tokens(c_addr, 3)

    n_inter = len(s1_ntoks & c_ntoks)
    n_union = len(s1_ntoks | c_ntoks)
    name_jaccard = (n_inter / n_union) if n_union > 0 else 0.0

    a_inter = len(s1_atoks & c_atoks)
    a_union = len(s1_atoks | c_atoks)
    addr_jaccard = (a_inter / a_union) if a_union > 0 else 0.0

    n_min = min(len(s1_ntoks), len(c_ntoks))
    name_overlap = (n_inter / n_min) if n_min > 0 else 0.0
    a_min = min(len(s1_atoks), len(c_atoks))
    addr_overlap = (a_inter / a_min) if a_min > 0 else 0.0

    name_ratio = fuzz.ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    name_token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    addr_ratio = fuzz.ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0

    s1_nums = extract_numbers(s1_addr)
    c_nums = extract_numbers(c_addr)
    shared_nums = len(s1_nums & c_nums)
    has_same_num = 1.0 if shared_nums > 0 else 0.0

    return [
        1.0 if s1_name == c_name and s1_name else 0.0,
        name_ratio,
        name_token_sort,
        name_token_set,
        name_jaccard,
        name_overlap,
        abs(len(s1_name) - len(c_name)),
        abs(len(s1_ntoks) - len(c_ntoks)),
        1.0 if s1_addr == c_addr and s1_addr else 0.0,
        addr_ratio,
        addr_token_set,
        addr_jaccard,
        addr_overlap,
        abs(len(s1_addr) - len(c_addr)),
        float(shared_nums),
        has_same_num,
        1.0 if s1_country == c_country else 0.0,
        1.0 if c_src == 'S2' else 0.0,
        0.5 * (name_ratio + addr_ratio),
        1.0 if name_token_set >= 0.85 else 0.0,
        1.0 if addr_token_set >= 0.85 else 0.0
    ]


class EnhancedHighRecallBlocker:
    def __init__(self, max_candidates_per_key: int = 50):
        self.max_candidates_per_key = max_candidates_per_key
        self.exact_name_idx = defaultdict(lambda: defaultdict(list))
        self.compact_name_idx = defaultdict(lambda: defaultdict(list))
        self.token_idx = defaultdict(lambda: defaultdict(list))
        self.exact_addr_idx = defaultdict(lambda: defaultdict(list))
        self.addr_combo_idx = defaultdict(lambda: defaultdict(list))
        self.prefix_idx = defaultdict(lambda: defaultdict(list))

    def fit(self, df: pd.DataFrame, desc: str = "Candidates"):
        total = len(df)
        step = max(1, total // 10)
        for i, row in enumerate(df.itertuples(index=False)):
            if (i + 1) % step == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                print(f"[{desc.upper()} INDEXING] {pct:.1f}% ({i + 1:,} / {total:,})", flush=True)

            cid = row.entity_id
            country = row.country
            name = row.business_name_norm
            addr = row.business_address_norm

            if name:
                self.exact_name_idx[country][name].append(cid)
                comp = re.sub(r'[^a-z0-9]', '', name)
                if len(comp) >= 4:
                    self.compact_name_idx[country][comp].append(cid)
                    self.prefix_idx[country][comp[:5]].append(cid)
                toks = clean_tokens(name, min_len=3)
                for tok in toks:
                    self.token_idx[country][tok].append(cid)

            if addr:
                self.exact_addr_idx[country][addr].append(cid)
                nums = extract_numbers(addr)
                atoks = clean_tokens(addr, min_len=4)
                if nums and atoks:
                    for num in list(nums)[:2]:
                        for atok in list(atoks)[:2]:
                            self.addr_combo_idx[country][f"{num}_{atok}"].append(cid)

    def get_candidates(self, country: str, name: str, addr: str, max_cands: int = 45) -> list:
        cands = set()
        comp = re.sub(r'[^a-z0-9]', '', name) if name else ''

        # 1. Exact Name
        if name:
            cands.update(self.exact_name_idx[country].get(name, [])[:50])

        # 2. Compact Name
        if len(comp) >= 4 and len(cands) < max_cands:
            cands.update(self.compact_name_idx[country].get(comp, [])[:50])

        # 3. Discriminative Tokens
        if name and len(cands) < max_cands:
            toks = clean_tokens(name, min_len=3)
            for tok in toks:
                if len(cands) >= max_cands:
                    break
                m = self.token_idx[country].get(tok, [])
                if len(m) <= self.max_candidates_per_key:
                    cands.update(m)

        # 4. 5-Char Name Prefix
        if len(comp) >= 5 and len(cands) < max_cands:
            m = self.prefix_idx[country].get(comp[:5], [])
            if len(m) <= 35:
                cands.update(m)

        # 5. Address exact & combos
        if addr and len(cands) < max_cands:
            cands.update(self.exact_addr_idx[country].get(addr, [])[:40])
            nums = extract_numbers(addr)
            atoks = clean_tokens(addr, min_len=4)
            if nums and atoks:
                for num in list(nums)[:2]:
                    for atok in list(atoks)[:2]:
                        if len(cands) >= max_cands:
                            break
                        m = self.addr_combo_idx[country].get(f"{num}_{atok}", [])
                        if len(m) <= 30:
                            cands.update(m)

        cand_list = list(cands)
        return cand_list[:max_cands] if len(cand_list) > max_cands else cand_list


def run_fine_tuned_pipeline():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("STARTING COMPLETE FINE-TUNED SOTA PIPELINE (F0.5 = 0.9301 TARGET)", flush=True)
    print("=" * 80, flush=True)

    # ----------------------------------------------------
    # PHASE 1: TRAIN BEST MODEL (CONFIG #4)
    # ----------------------------------------------------
    print("\n[STAGE 1/4] Training Best Fine-Tuned Model on Ground Truth...", flush=True)
    gt_df = pd.read_csv(TRAIN_GT, sep='\t', keep_default_na=False)
    s1_df = pd.read_csv(TRAIN_S1, sep='\t', keep_default_na=False)

    gt_df['match_count'] = gt_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
    sampled_gt = gt_df.groupby('match_count', group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) * 35000 / len(gt_df))), random_state=RANDOM_STATE)
    ).head(35000)

    eval_s1_ids = set(sampled_gt['source1_entity_id'])
    s1_sub = s1_df[s1_df.entity_id.isin(eval_s1_ids)].copy()
    s1_sub['business_name_norm'] = s1_sub['business_name'].apply(normalize_name)
    s1_sub['business_address_norm'] = s1_sub['business_address'].apply(normalize_address)

    gt_map = {}
    all_true = set()
    for _, r in sampled_gt.iterrows():
        sid = r['source1_entity_id']
        m = r['matched_entity_ids']
        if m:
            s = set(x.strip() for x in m.split(',') if x.strip())
            gt_map[sid] = s
            all_true.update(s)
        else:
            gt_map[sid] = set()

    countries = set(s1_sub.country.unique())
    s2 = pd.read_csv(TRAIN_S2, sep='\t', keep_default_na=False)
    s3 = pd.read_csv(TRAIN_S3, sep='\t', keep_default_na=False)
    s2 = s2[s2.country.isin(countries)]
    s3 = s3[s3.country.isin(countries)]

    s2_pos = s2[s2.entity_id.isin(all_true)]
    s2_neg = s2[~s2.entity_id.isin(all_true)].sample(min(120000, len(s2)), random_state=RANDOM_STATE)
    s2_pool = pd.concat([s2_pos, s2_neg]).drop_duplicates('entity_id')

    s3_pos = s3[s3.entity_id.isin(all_true)]
    s3_neg = s3[~s3.entity_id.isin(all_true)].sample(min(120000, len(s3)), random_state=RANDOM_STATE)
    s3_pool = pd.concat([s3_pos, s3_neg]).drop_duplicates('entity_id')

    s2_pool['business_name_norm'] = s2_pool['business_name'].apply(normalize_name)
    s2_pool['business_address_norm'] = s2_pool['business_address'].apply(normalize_address)
    s3_pool['business_name_norm'] = s3_pool['business_name'].apply(normalize_name)
    s3_pool['business_address_norm'] = s3_pool['business_address'].apply(normalize_address)

    cand_pool = pd.concat([s2_pool, s3_pool], ignore_index=True)
    del s2, s3, s2_pool, s3_pool
    gc.collect()

    train_blocker = EnhancedHighRecallBlocker(max_candidates_per_key=50)
    train_blocker.fit(cand_pool, desc="Training Candidates")

    cand_lookup = {}
    for r in cand_pool.itertuples(index=False):
        cand_lookup[r.entity_id] = (
            r.business_name_norm, r.business_address_norm, r.country,
            'S2' if r.entity_id.startswith('S2-') else 'S3'
        )

    all_s1 = list(s1_sub.itertuples(index=False))
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(all_s1)
    split_idx = int(len(all_s1) * 0.75)
    train_s1 = all_s1[:split_idx]
    val_s1 = all_s1[split_idx:]
    val_s1_ids = [r.entity_id for r in val_s1]

    X_train, y_train = [], []
    for r in train_s1:
        sid = r.entity_id
        true_set = gt_map.get(sid, set())
        cands = train_blocker.get_candidates(r.country, r.business_name_norm, r.business_address_norm, max_cands=45)
        for cid in cands:
            cinfo = cand_lookup.get(cid)
            if not cinfo: continue
            cn, ca, cc, csrc = cinfo
            feats = compute_fast_features(r.business_name_norm, r.business_address_norm, cn, ca, r.country, cc, csrc)
            X_train.append(feats)
            y_train.append(1 if cid in true_set else 0)

    X_val, y_val, val_pairs = [], [], []
    for r in val_s1:
        sid = r.entity_id
        true_set = gt_map.get(sid, set())
        cands = train_blocker.get_candidates(r.country, r.business_name_norm, r.business_address_norm, max_cands=45)
        for cid in cands:
            cinfo = cand_lookup.get(cid)
            if not cinfo: continue
            cn, ca, cc, csrc = cinfo
            feats = compute_fast_features(r.business_name_norm, r.business_address_norm, cn, ca, r.country, cc, csrc)
            X_val.append(feats)
            y_val.append(1 if cid in true_set else 0)
            val_pairs.append((sid, cid))

    X_tr = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    y_tr = np.array(y_train)
    X_v = pd.DataFrame(X_val, columns=FEATURE_NAMES)
    y_v = np.array(y_val)

    print(f"Fitting Fine-Tuned LightGBM on {len(X_tr):,} pairs (Positives: {y_tr.sum():,})...", flush=True)
    clf = lgb.LGBMClassifier(**BEST_LGBM_PARAMS)
    clf.fit(X_tr, y_tr)

    val_probs = clf.predict_proba(X_v)[:, 1]
    best_th = 0.60
    best_f05 = -1

    for th in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        pred_map = defaultdict(set)
        for (sid, cid), prob in zip(val_pairs, val_probs):
            if prob >= th:
                pred_map[sid].add(cid)
        metrics = compute_macro_f05(gt_map, pred_map, val_s1_ids)
        f05 = metrics['macro_f05']
        print(f"  Threshold {th:.2f} | Macro F0.5 = {f05:.4f} ({f05*100:.2f}%) | Zero-Match F0.5 = {metrics['zero_match_f05']:.4f}", flush=True)
        if f05 > best_f05:
            best_f05 = f05
            best_th = th

    print(f"CONFIRMED OPTIMAL THRESHOLD: {best_th:.2f} -> MACRO F0.5 = {best_f05:.4f} ({best_f05*100:.2f}%)", flush=True)

    del X_train, y_train, X_val, y_val, X_tr, y_tr, X_v, y_v, cand_pool, train_blocker, cand_lookup
    gc.collect()

    # ----------------------------------------------------
    # PHASE 2: TEST CANDIDATE POOL INDEXING (9.97M)
    # ----------------------------------------------------
    print("\n[STAGE 2/4] Loading and Indexing 9,969,589 Test Candidates into Enhanced Blocker...", flush=True)
    s2_reader = pd.read_csv(TEST_S2, sep='\t', keep_default_na=False, chunksize=1000000)
    s3_reader = pd.read_csv(TEST_S3, sep='\t', keep_default_na=False, chunksize=1000000)

    s2_chunks = []
    print("Reading and normalizing test_source2.tsv...", flush=True)
    for i, chunk in enumerate(s2_reader):
        chunk['business_name_norm'] = chunk['business_name'].apply(normalize_name)
        chunk['business_address_norm'] = chunk['business_address'].apply(normalize_address)
        s2_chunks.append(chunk)
        print(f"  test_source2 chunk {i+1} loaded (1M records)", flush=True)

    s3_chunks = []
    print("Reading and normalizing test_source3.tsv...", flush=True)
    for i, chunk in enumerate(s3_reader):
        chunk['business_name_norm'] = chunk['business_name'].apply(normalize_name)
        chunk['business_address_norm'] = chunk['business_address'].apply(normalize_address)
        s3_chunks.append(chunk)
        print(f"  test_source3 chunk {i+1} loaded (1M records)", flush=True)

    test_cand_df = pd.concat(s2_chunks + s3_chunks, ignore_index=True)
    del s2_chunks, s3_chunks
    gc.collect()

    print(f"Total Test Candidate Pool: {len(test_cand_df):,} records", flush=True)
    test_blocker = EnhancedHighRecallBlocker(max_candidates_per_key=50)
    test_blocker.fit(test_cand_df, desc="Test Candidates")

    test_cand_lookup = {}
    print("Building test candidate lookup table...", flush=True)
    for r in test_cand_df.itertuples(index=False):
        test_cand_lookup[r.entity_id] = (
            r.business_name_norm, r.business_address_norm, r.country,
            'S2' if r.entity_id.startswith('S2-') else 'S3'
        )

    del test_cand_df
    gc.collect()

    # ----------------------------------------------------
    # PHASE 3: STREAMING INFERENCE ON TEST S1 (1.73M)
    # ----------------------------------------------------
    print("\n[STAGE 3/4] Streaming Test S1 (1,732,544 Entities) with Live Progress...", flush=True)
    total_test_s1 = 1732544
    chunk_size = 200000

    s1_reader = pd.read_csv(TEST_S1, sep='\t', keep_default_na=False, chunksize=chunk_size)

    OUTPUT_MATCHING.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MATCHING, 'w', encoding='utf-8') as f_m:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
    with open(OUTPUT_CANDIDATE, 'w', encoding='utf-8') as f_c:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

    total_processed = 0
    total_matches = 0
    total_candidates = 0

    for chunk_idx, chunk in enumerate(s1_reader, 1):
        t_c0 = time.time()
        chunk['business_name_norm'] = chunk['business_name'].apply(normalize_name)
        chunk['business_address_norm'] = chunk['business_address'].apply(normalize_address)

        m_lines = []
        c_lines = []
        all_batch_pairs = []
        batch_slice_map = []

        for row in chunk.itertuples(index=False):
            s1_id = row.entity_id
            country = row.country
            name = row.business_name_norm
            addr = row.business_address_norm

            cands = test_blocker.get_candidates(country, name, addr, max_cands=45)
            c_lines.append(f"{s1_id}\t{','.join(cands)}\n")
            total_candidates += len(cands)

            if not cands:
                batch_slice_map.append((s1_id, 0, 0, []))
                continue

            start_idx = len(all_batch_pairs)
            valid_cids = []
            for cid in cands:
                cinfo = test_cand_lookup.get(cid)
                if not cinfo: continue
                cname, caddr, ccountry, csrc = cinfo
                feats = compute_fast_features(name, addr, cname, caddr, country, ccountry, csrc)
                all_batch_pairs.append(feats)
                valid_cids.append(cid)

            end_idx = len(all_batch_pairs)
            batch_slice_map.append((s1_id, start_idx, end_idx, valid_cids))

        if all_batch_pairs:
            X_batch = pd.DataFrame(all_batch_pairs, columns=FEATURE_NAMES)
            batch_probs = clf.predict_proba(X_batch)[:, 1]

            for s1_id, start, end, cids in batch_slice_map:
                if start == end:
                    m_lines.append(f"{s1_id}\t\n")
                else:
                    pair_probs = batch_probs[start:end]
                    selected = [cid for cid, p in zip(cids, pair_probs) if p >= best_th]
                    uniq_selected = list(dict.fromkeys(selected))
                    m_lines.append(f"{s1_id}\t{','.join(uniq_selected)}\n")
                    total_matches += len(uniq_selected)
        else:
            for s1_id, _, _, _ in batch_slice_map:
                m_lines.append(f"{s1_id}\t\n")

        with open(OUTPUT_MATCHING, 'a', encoding='utf-8') as f_m:
            f_m.writelines(m_lines)
        with open(OUTPUT_CANDIDATE, 'a', encoding='utf-8') as f_c:
            f_c.writelines(c_lines)

        total_processed += len(chunk)
        pct = (total_processed / total_test_s1) * 100
        print(f"[TEST INFERENCE PROGRESS] {pct:.1f}% ({total_processed:,} / {total_test_s1:,} entities in {time.time() - t_c0:.1f}s)", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("ALL TEST PREDICTIONS GENERATED & SAVED SUCCESSFULLY!", flush=True)
    print(f"Total S1 Records Processed: {total_processed:,}", flush=True)
    print(f"Total Candidate Pairs:      {total_candidates:,}", flush=True)
    print(f"Total Matches Predicted:     {total_matches:,}", flush=True)
    print(f"Files written:\n  1. {OUTPUT_MATCHING}\n  2. {OUTPUT_CANDIDATE}", flush=True)
    print("=" * 80, flush=True)

    # ----------------------------------------------------
    # PHASE 4: OFFICIAL SUBMISSION VALIDATION
    # ----------------------------------------------------
    print("\n[STAGE 4/4] Running Official validate_submission.py Verification...", flush=True)
    cmd = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--matching", str(OUTPUT_MATCHING),
        "--candidate", str(OUTPUT_CANDIDATE),
        "--test-dir", str(TEST_DIR)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Validator Output:\n" + res.stdout, flush=True)
    if res.stderr:
        print("Validator Stderr:\n" + res.stderr, flush=True)

    print(f"\nCOMPLETE SOTA FINE-TUNED PIPELINE FINISHED IN {(time.time() - t_start)/60:.1f} MINUTES!", flush=True)


if __name__ == '__main__':
    run_fine_tuned_pipeline()
