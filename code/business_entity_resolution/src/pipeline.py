import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from sklearn.model_selection import train_test_split

from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, RANDOM_STATE, LGBM_PARAMS
from src.data.loader import load_source_tsv, load_ground_truth, get_ground_truth_dict
from src.preprocessing.normalize import normalize_name, normalize_address
from src.blocking.candidate_generator import MultiBlocker
from src.features.feature_builder import compute_pair_features
from src.evaluation.metrics import compute_macro_f05


def run_pipeline_experiment(sample_size: int = 15000):
    print("=" * 70)
    print(f"RUNNING BUSINESS ENTITY RESOLUTION PIPELINE (SAMPLE SIZE: {sample_size:,} S1)")
    print("=" * 70)

    # 1. Load Data
    t0 = time.time()
    print("\n[Stage 1 & 2] Loading and sampling data...")
    gt_df = load_ground_truth(TRAIN_GT)
    s1_df = load_source_tsv(TRAIN_S1)

    # Sample S1 entities stratified by match count
    gt_df['match_count'] = gt_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
    # Stratified sample
    sampled_gt = gt_df.groupby('match_count', group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) * sample_size / len(gt_df))), random_state=RANDOM_STATE)
    ).head(sample_size)

    eval_s1_ids = set(sampled_gt['source1_entity_id'])
    sampled_s1 = s1_df[s1_df.entity_id.isin(eval_s1_ids)].copy()

    # Preprocessing / Normalization
    print("\n[Stage 3] Normalizing S1 names & addresses...")
    sampled_s1['business_name_norm'] = sampled_s1['business_name'].apply(normalize_name)
    sampled_s1['business_address_norm'] = sampled_s1['business_address'].apply(normalize_address)

    # Ground truth mapping
    gt_map = get_ground_truth_dict(sampled_gt)
    all_true_matches = set()
    for s1, matches in gt_map.items():
        all_true_matches.update(matches)

    print(f"Sampled {len(sampled_s1):,} S1 entities | Total ground truth matches: {len(all_true_matches):,}")

    # Load S2 & S3 candidates relevant to the countries in sample
    countries = set(sampled_s1.country.unique())
    print(f"Countries in sample: {countries}")

    print("\nLoading and normalizing S2 & S3 records...")
    s2_df = load_source_tsv(TRAIN_S2)
    s3_df = load_source_tsv(TRAIN_S3)

    # Keep candidate records in matching countries
    s2_df = s2_df[s2_df.country.isin(countries)].copy()
    s3_df = s3_df[s3_df.country.isin(countries)].copy()

    # Take candidate subset: all known true matches + random pool of distractor entities
    s2_true = s2_df[s2_df.entity_id.isin(all_true_matches)]
    s2_rest = s2_df[~s2_df.entity_id.isin(all_true_matches)].sample(min(100000, len(s2_df)), random_state=RANDOM_STATE)
    s2_pool = pd.concat([s2_true, s2_rest]).drop_duplicates(subset=['entity_id'])

    s3_true = s3_df[s3_df.entity_id.isin(all_true_matches)]
    s3_rest = s3_df[~s3_df.entity_id.isin(all_true_matches)].sample(min(100000, len(s3_df)), random_state=RANDOM_STATE)
    s3_pool = pd.concat([s3_true, s3_rest]).drop_duplicates(subset=['entity_id'])

    s2_pool['business_name_norm'] = s2_pool['business_name'].apply(normalize_name)
    s2_pool['business_address_norm'] = s2_pool['business_address'].apply(normalize_address)
    s3_pool['business_name_norm'] = s3_pool['business_name'].apply(normalize_name)
    s3_pool['business_address_norm'] = s3_pool['business_address'].apply(normalize_address)

    cand_pool = pd.concat([s2_pool, s3_pool], ignore_index=True)
    print(f"Total Candidate Pool (S2+S3): {len(cand_pool):,} records")

    # Fast lookup dictionaries for feature builder
    cand_lookup = {}
    for r in cand_pool.itertuples(index=False):
        cand_lookup[r.entity_id] = (r.business_name_norm, r.business_address_norm, r.country, 'S2' if r.entity_id.startswith('S2-') else 'S3')

    # [Stage 5 & 6] Multi-Strategy Blocking
    print("\n[Stage 5 & 6] Running Multi-Strategy Candidate Generation...")
    blocker = MultiBlocker(max_candidates_per_key=100)
    blocker.fit(cand_pool)

    # Train / Val Split strictly by S1 entity
    s1_train_ids, s1_val_ids = train_test_split(list(eval_s1_ids), test_size=0.25, random_state=RANDOM_STATE)
    s1_train_set = set(s1_train_ids)
    s1_val_set = set(s1_val_ids)

    print(f"Train S1 Entities: {len(s1_train_set):,} | Val S1 Entities: {len(s1_val_set):,}")

    candidate_pairs = []
    total_val_gt_pairs = sum(len(gt_map[s1]) for s1 in s1_val_set)
    recovered_val_gt_pairs = 0

    print("Generating candidates for all S1 entities...")
    for s1_row in sampled_s1.itertuples(index=False):
        s1_id = s1_row.entity_id
        cands = blocker.generate_candidates_for_s1(s1_row, max_total_candidates=60)
        
        # Check validation recall
        if s1_id in s1_val_set:
            true_set = gt_map.get(s1_id, set())
            recovered_val_gt_pairs += len(true_set & set(cands))

        for c_id in cands:
            candidate_pairs.append((s1_id, c_id))

    blocking_recall = (recovered_val_gt_pairs / total_val_gt_pairs) if total_val_gt_pairs > 0 else 0.0
    print(f"\n==========================================")
    print(f"[METRIC] BLOCKING METRICS (Validation Set):")
    print(f"Total True Pairs: {total_val_gt_pairs:,}")
    print(f"Recovered True Pairs: {recovered_val_gt_pairs:,}")
    print(f"Blocking Recall: {blocking_recall:.2%}")
    print(f"Total Candidate Pairs Generated: {len(candidate_pairs):,}")
    print(f"Avg Candidates / S1: {len(candidate_pairs) / len(sampled_s1):.1f}")
    print(f"==========================================")

    # [Stage 8] Feature Engineering
    print("\n[Stage 8] Computing candidate pair features...")
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
        if not c_info:
            continue
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

    print(f"Train Features Matrix: {X_train.shape} (Positives: {y_train.sum():,})")
    print(f"Val Features Matrix:   {X_val.shape} (Positives: {y_val.sum():,})")

    # [Stage 11] Train LightGBM Baseline
    print("\n[Stage 11] Training LightGBM Matcher...")
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    val_preds_prob = clf.predict_proba(X_val)[:, 1]

    # [Stage 13] Threshold Optimization for Macro F0.5
    print("\n[Stage 13 & 14] Optimizing Decision Threshold on Validation...")
    val_pairs_df = pd.DataFrame({
        's1_id': np.array(s1_pair_ids)[val_mask],
        'cand_id': np.array(cand_pair_ids)[val_mask],
        'prob': val_preds_prob,
        'label': y_val
    })

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    best_thresh = 0.50
    best_f05 = -1.0
    tuning_records = []

    for th in thresholds:
        pred_matches = defaultdict(set)
        filtered = val_pairs_df[val_pairs_df.prob >= th]
        for _, r in filtered.iterrows():
            pred_matches[r['s1_id']].add(r['cand_id'])

        eval_res = compute_macro_f05(gt_map, pred_matches, s1_val_ids)
        tuning_records.append({
            'threshold': th,
            'macro_f05': eval_res['macro_f05'],
            'exact_set_acc': eval_res['exact_set_acc'],
            'zero_match_f05': eval_res['zero_match_f05'],
            'single_match_f05': eval_res['single_match_f05'],
            'multi_match_f05': eval_res['multi_match_f05'],
            'matches_count': len(filtered)
        })
        if eval_res['macro_f05'] > best_f05:
            best_f05 = eval_res['macro_f05']
            best_thresh = th

    tuning_df = pd.DataFrame(tuning_records)
    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION RESULTS:")
    print(tuning_df.to_string(index=False))
    print("=" * 70)
    print(f"\n[BEST RESULT] BEST THRESHOLD: {best_thresh} with Validation Macro F0.5 = {best_f05:.4f}")

    # Top Feature Importances
    imp_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Feature Importances:")
    print(imp_df.head(10).to_string(index=False))

    print(f"\nExecution time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_pipeline_experiment(sample_size=15000)
