import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_MATCHING, OUTPUT_CANDIDATE,
    VALIDATE_SCRIPT, TEST_DIR,
    RANDOM_STATE, LGBM_PARAMS
)
from src.data.loader import load_source_tsv, load_ground_truth, get_ground_truth_dict
from src.preprocessing.normalize import normalize_name, normalize_address
from src.blocking.candidate_generator import MultiBlocker
from src.features.feature_builder import compute_pair_features
from src.evaluation.metrics import compute_macro_f05


def train_and_evaluate(sample_size: int = 40000):
    print("=" * 70)
    print(f"STAGE 1: TRAINING & CALIBRATION (Sample: {sample_size:,} S1 entities)")
    print("=" * 70)

    # 1. Load Ground Truth and S1
    gt_df = load_ground_truth(TRAIN_GT)
    s1_df = load_source_tsv(TRAIN_S1)

    gt_df['match_count'] = gt_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
    sampled_gt = gt_df.groupby('match_count', group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) * sample_size / len(gt_df))), random_state=RANDOM_STATE)
    ).head(sample_size)

    eval_s1_ids = set(sampled_gt['source1_entity_id'])
    sampled_s1 = s1_df[s1_df.entity_id.isin(eval_s1_ids)].copy()

    sampled_s1['business_name_norm'] = sampled_s1['business_name'].apply(normalize_name)
    sampled_s1['business_address_norm'] = sampled_s1['business_address'].apply(normalize_address)

    gt_map = get_ground_truth_dict(sampled_gt)
    all_true_matches = set()
    for s1, matches in gt_map.items():
        all_true_matches.update(matches)

    countries = set(sampled_s1.country.unique())
    print(f"Countries: {countries} | True matching targets: {len(all_true_matches):,}")

    # Load S2 & S3
    s2_df = load_source_tsv(TRAIN_S2)
    s3_df = load_source_tsv(TRAIN_S3)
    s2_df = s2_df[s2_df.country.isin(countries)].copy()
    s3_df = s3_df[s3_df.country.isin(countries)].copy()

    s2_true = s2_df[s2_df.entity_id.isin(all_true_matches)]
    s2_rest = s2_df[~s2_df.entity_id.isin(all_true_matches)].sample(min(150000, len(s2_df)), random_state=RANDOM_STATE)
    s2_pool = pd.concat([s2_true, s2_rest]).drop_duplicates(subset=['entity_id'])

    s3_true = s3_df[s3_df.entity_id.isin(all_true_matches)]
    s3_rest = s3_df[~s3_df.entity_id.isin(all_true_matches)].sample(min(150000, len(s3_df)), random_state=RANDOM_STATE)
    s3_pool = pd.concat([s3_true, s3_rest]).drop_duplicates(subset=['entity_id'])

    s2_pool['business_name_norm'] = s2_pool['business_name'].apply(normalize_name)
    s2_pool['business_address_norm'] = s2_pool['business_address'].apply(normalize_address)
    s3_pool['business_name_norm'] = s3_pool['business_name'].apply(normalize_name)
    s3_pool['business_address_norm'] = s3_pool['business_address'].apply(normalize_address)

    cand_pool = pd.concat([s2_pool, s3_pool], ignore_index=True)
    del s2_df, s3_df, s2_pool, s3_pool
    gc.collect()

    cand_lookup = {}
    for r in cand_pool.itertuples(index=False):
        cand_lookup[r.entity_id] = (
            r.business_name_norm, r.business_address_norm, r.country,
            'S2' if r.entity_id.startswith('S2-') else 'S3'
        )

    # Multi-strategy blocking
    blocker = MultiBlocker(max_candidates_per_key=100)
    blocker.fit(cand_pool)

    s1_train_ids, s1_val_ids = train_test_split(list(eval_s1_ids), test_size=0.20, random_state=RANDOM_STATE)
    s1_train_set = set(s1_train_ids)
    s1_val_set = set(s1_val_ids)

    candidate_pairs = []
    total_val_gt_pairs = sum(len(gt_map[s1]) for s1 in s1_val_set)
    recovered_val_gt_pairs = 0

    for s1_row in sampled_s1.itertuples(index=False):
        s1_id = s1_row.entity_id
        cands = blocker.generate_candidates_for_s1(s1_row, max_total_candidates=80)
        if s1_id in s1_val_set:
            true_set = gt_map.get(s1_id, set())
            recovered_val_gt_pairs += len(true_set & set(cands))
        for c_id in cands:
            candidate_pairs.append((s1_id, c_id))

    blocking_recall = (recovered_val_gt_pairs / total_val_gt_pairs) if total_val_gt_pairs > 0 else 0.0
    print(f"\n[Validation Blocking Recall]: {blocking_recall:.2%} ({recovered_val_gt_pairs:,} / {total_val_gt_pairs:,})")
    print(f"Total candidate pairs: {len(candidate_pairs):,}")

    # Feature computation
    s1_dict = {
        r.entity_id: (r.business_name_norm, r.business_address_norm, r.country)
        for r in sampled_s1.itertuples(index=False)
    }

    feature_rows = []
    labels = []
    split_group = []
    s1_pair_ids = []
    cand_pair_ids = []

    for s1_id, c_id in candidate_pairs:
        s1_name, s1_addr, s1_country = s1_dict[s1_id]
        c_info = cand_lookup.get(c_id)
        if not c_info: continue
        c_name, c_addr, c_country, c_src = c_info

        feats = compute_pair_features(s1_name, c_name, s1_addr, c_addr, s1_country, c_country, c_src)
        is_match = 1 if c_id in gt_map.get(s1_id, set()) else 0

        feature_rows.append(feats)
        labels.append(is_match)
        split_group.append('train' if s1_id in s1_train_set else 'val')
        s1_pair_ids.append(s1_id)
        cand_pair_ids.append(c_id)

    X_df = pd.DataFrame(feature_rows)
    y = np.array(labels)
    splits = np.array(split_group)

    train_mask = (splits == 'train')
    val_mask = (splits == 'val')

    X_train, y_train = X_df[train_mask], y[train_mask]
    X_val, y_val = X_df[val_mask], y[val_mask]

    print(f"Training LightGBM on {len(X_train):,} pairs (Positives: {y_train.sum():,})...")
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

    val_probs = clf.predict_proba(X_val)[:, 1]

    # Threshold tuning
    val_pairs_df = pd.DataFrame({
        's1_id': np.array(s1_pair_ids)[val_mask],
        'cand_id': np.array(cand_pair_ids)[val_mask],
        'prob': val_probs
    })

    best_thresh = 0.60
    best_f05 = -1.0
    for th in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        pred_matches = defaultdict(set)
        for _, r in val_pairs_df[val_pairs_df.prob >= th].iterrows():
            pred_matches[r['s1_id']].add(r['cand_id'])
        res = compute_macro_f05(gt_map, pred_matches, s1_val_ids)
        if res['macro_f05'] > best_f05:
            best_f05 = res['macro_f05']
            best_thresh = th

    print(f"[BEST VALIDATION THRESHOLD]: {best_thresh} -> Macro F0.5 = {best_f05:.4f}")
    return clf, best_thresh


def run_test_inference(clf, threshold: float = 0.60, chunk_size: int = 150000):
    print("\n" + "=" * 70)
    print("STAGE 2: FULL TEST INFERENCE")
    print("=" * 70)

    # 1. Load Test S2 & S3 Candidate Tables
    print("Loading test_source2.tsv and test_source3.tsv...")
    s2_test = load_source_tsv(TEST_S2)
    s3_test = load_source_tsv(TEST_S3)

    print("Normalizing test candidates...")
    s2_test['business_name_norm'] = s2_test['business_name'].apply(normalize_name)
    s2_test['business_address_norm'] = s2_test['business_address'].apply(normalize_address)
    s3_test['business_name_norm'] = s3_test['business_name'].apply(normalize_name)
    s3_test['business_address_norm'] = s3_test['business_address'].apply(normalize_address)

    cand_pool = pd.concat([s2_test, s3_test], ignore_index=True)
    del s2_test, s3_test
    gc.collect()

    print(f"Total Test Candidate Pool: {len(cand_pool):,} records across countries: {set(cand_pool.country.unique())}")

    print("Indexing candidate pool in MultiBlocker...")
    blocker = MultiBlocker(max_candidates_per_key=100)
    blocker.fit(cand_pool)

    cand_lookup = {}
    for r in cand_pool.itertuples(index=False):
        cand_lookup[r.entity_id] = (
            r.business_name_norm, r.business_address_norm, r.country,
            'S2' if r.entity_id.startswith('S2-') else 'S3'
        )

    # 2. Process Test S1 in Stream / Chunks
    print(f"Opening test_source1.tsv and generating predictions with threshold {threshold}...")
    s1_reader = pd.read_csv(
        TEST_S1,
        sep='\t',
        dtype={'entity_id': str, 'business_name': str, 'business_address': str, 'country': str},
        keep_default_na=False,
        chunksize=chunk_size
    )

    # Ensure output directories exist
    OUTPUT_MATCHING.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize output TSVs with exact official headers
    with open(OUTPUT_MATCHING, 'w', encoding='utf-8') as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    with open(OUTPUT_CANDIDATE, 'w', encoding='utf-8') as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    total_s1_processed = 0
    total_candidates_written = 0
    total_matches_predicted = 0

    chunk_idx = 0
    for chunk in s1_reader:
        chunk_idx += 1
        t_c0 = time.time()
        chunk['business_name_norm'] = chunk['business_name'].apply(normalize_name)
        chunk['business_address_norm'] = chunk['business_address'].apply(normalize_address)

        chunk_cand_records = []
        matching_lines = []
        candidate_lines = []

        for s1_row in chunk.itertuples(index=False):
            s1_id = s1_row.entity_id
            s1_name = s1_row.business_name_norm
            s1_addr = s1_row.business_address_norm
            s1_country = s1_row.country

            cands = blocker.generate_candidates_for_s1(s1_row, max_total_candidates=70)
            candidate_lines.append(f"{s1_id}\t{','.join(cands)}\n")
            total_candidates_written += len(cands)

            if not cands:
                matching_lines.append(f"{s1_id}\t\n")
                continue

            # Compute features for candidate set
            pair_feats = []
            cand_ids = []
            for c_id in cands:
                c_info = cand_lookup.get(c_id)
                if not c_info: continue
                c_name, c_addr, c_country, c_src = c_info
                pair_feats.append(compute_pair_features(s1_name, c_name, s1_addr, c_addr, s1_country, c_country, c_src))
                cand_ids.append(c_id)

            if not pair_feats:
                matching_lines.append(f"{s1_id}\t\n")
                continue

            # Batch model scoring
            X_batch = pd.DataFrame(pair_feats)
            probs = clf.predict_proba(X_batch)[:, 1]

            matched = [cid for cid, p in zip(cand_ids, probs) if p >= threshold]
            # Maintain uniqueness and clean format
            unique_matches = list(dict.fromkeys(matched))
            matching_lines.append(f"{s1_id}\t{','.join(unique_matches)}\n")
            total_matches_predicted += len(unique_matches)

        # Append to files
        with open(OUTPUT_MATCHING, 'a', encoding='utf-8') as f_match:
            f_match.writelines(matching_lines)

        with open(OUTPUT_CANDIDATE, 'a', encoding='utf-8') as f_cand:
            f_cand.writelines(candidate_lines)

        total_s1_processed += len(chunk)
        print(f"Chunk {chunk_idx}: processed {len(chunk):,} S1 (Total: {total_s1_processed:,}) in {time.time() - t_c0:.1f}s")

    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY:")
    print(f"Total S1 entities processed: {total_s1_processed:,}")
    print(f"Total candidate pairs: {total_candidates_written:,}")
    print(f"Total matches predicted: {total_matches_predicted:,}")
    print(f"Generated files:\n  1. {OUTPUT_MATCHING}\n  2. {OUTPUT_CANDIDATE}")
    print("=" * 70)


if __name__ == "__main__":
    t_start = time.time()
    clf, best_threshold = train_and_evaluate(sample_size=35000)
    run_test_inference(clf, threshold=best_threshold, chunk_size=200000)
    print(f"Total execution finished in {time.time() - t_start:.1f}s")
