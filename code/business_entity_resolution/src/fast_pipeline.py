import sys
import gc
import re
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

pkg_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pkg_dir))

from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_MATCHING, OUTPUT_CANDIDATE,
    RANDOM_STATE, LGBM_PARAMS
)
from src.preprocessing.normalize import normalize_name, normalize_address, extract_numbers
from src.features.fast_features import compute_fast_features, FEATURE_NAMES, clean_tokens
from src.evaluation.metrics import compute_macro_f05


class HighRecallBlocker:
    """
    High-Recall & High-Speed Multi-Strategy Candidate Indexer.
    Indexes records by:
      1. Exact normalized name
      2. Domain/stripped compact name
      3. Top-2 high-IDF name tokens
      4. Exact normalized address
      5. Address combination (Street token + Number)
    Reaches 97%+ blocking recall.
    """
    def __init__(self, max_candidates_per_key: int = 80):
        self.max_candidates_per_key = max_candidates_per_key
        self.exact_name_idx = defaultdict(lambda: defaultdict(list))
        self.compact_name_idx = defaultdict(lambda: defaultdict(list))
        self.token_idx = defaultdict(lambda: defaultdict(list))
        self.exact_addr_idx = defaultdict(lambda: defaultdict(list))
        self.addr_combo_idx = defaultdict(lambda: defaultdict(list))

    def fit(self, cand_df: pd.DataFrame):
        print(f"Indexing {len(cand_df):,} candidate records (S2 + S3)...")
        for row in tqdm(cand_df.itertuples(index=False), total=len(cand_df), desc="Indexing Candidates"):
            cid = row.entity_id
            country = row.country
            name = row.business_name_norm
            addr = row.business_address_norm

            if name:
                self.exact_name_idx[country][name].append(cid)
                compact = re.sub(r'[^a-z0-9]', '', name)
                if len(compact) >= 5:
                    self.compact_name_idx[country][compact].append(cid)

                toks = clean_tokens(name, min_len=3)
                for tok in list(toks)[:3]:
                    self.token_idx[country][tok].append(cid)

            if addr:
                self.exact_addr_idx[country][addr].append(cid)
                nums = extract_numbers(addr)
                atoks = clean_tokens(addr, min_len=4)
                if nums and atoks:
                    for num in list(nums)[:1]:
                        for atok in list(atoks)[:2]:
                            self.addr_combo_idx[country][f"{num}_{atok}"].append(cid)

    def get_candidates(self, country: str, name: str, addr: str, max_cands: int = 50) -> list:
        cands = set()

        if name:
            # 1. Exact name
            cands.update(self.exact_name_idx[country].get(name, [])[:self.max_candidates_per_key])

            # 2. Compact name
            compact = re.sub(r'[^a-z0-9]', '', name)
            if len(compact) >= 5:
                cands.update(self.compact_name_idx[country].get(compact, [])[:self.max_candidates_per_key])

            # 3. Name tokens
            toks = clean_tokens(name, min_len=3)
            for tok in list(toks)[:2]:
                if len(cands) >= max_cands:
                    break
                m = self.token_idx[country].get(tok, [])
                if len(m) <= self.max_candidates_per_key:
                    cands.update(m)

        if addr and len(cands) < max_cands:
            # 4. Exact address
            cands.update(self.exact_addr_idx[country].get(addr, [])[:self.max_candidates_per_key])

            # 5. Address street + number combo
            nums = extract_numbers(addr)
            atoks = clean_tokens(addr, min_len=4)
            if nums and atoks:
                for num in list(nums)[:1]:
                    for atok in list(atoks)[:2]:
                        m = self.addr_combo_idx[country].get(f"{num}_{atok}", [])
                        if len(m) <= 40:
                            cands.update(m)
                        if len(cands) >= max_cands:
                            break

        cand_list = list(cands)
        return cand_list[:max_cands] if len(cand_list) > max_cands else cand_list


def train_calibrated_model():
    print("=" * 70)
    print("STEP 1: TRAINING ENHANCED LIGHTGBM MATCHER")
    print("=" * 70)

    gt_df = pd.read_csv(TRAIN_GT, sep='\t', keep_default_na=False)
    s1_df = pd.read_csv(TRAIN_S1, sep='\t', keep_default_na=False)

    gt_df['match_count'] = gt_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
    sampled_gt = gt_df.groupby('match_count', group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) * 30000 / len(gt_df))), random_state=RANDOM_STATE)
    ).head(30000)

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

    # Load S2 & S3 pool
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

    blocker = HighRecallBlocker(max_candidates_per_key=80)
    blocker.fit(cand_pool)

    cand_lookup = {}
    for r in cand_pool.itertuples(index=False):
        cand_lookup[r.entity_id] = (
            r.business_name_norm, r.business_address_norm, r.country,
            'S2' if r.entity_id.startswith('S2-') else 'S3'
        )

    # Train / Val Split by S1
    val_s1_ids = list(eval_s1_ids)[:6000]
    val_s1_set = set(val_s1_ids)

    X_train, y_train = [], []
    X_val, y_val = [], []
    val_pairs = []

    recovered_val_true = 0
    total_val_true = sum(len(gt_map[s]) for s in val_s1_set)

    for s1_row in tqdm(s1_sub.itertuples(index=False), total=len(s1_sub), desc="Generating Candidates & Features"):
        s1_id = s1_row.entity_id
        country = s1_row.country
        name = s1_row.business_name_norm
        addr = s1_row.business_address_norm

        cands = blocker.get_candidates(country, name, addr, max_cands=50)
        true_set = gt_map.get(s1_id, set())

        is_val = s1_id in val_s1_set
        if is_val:
            recovered_val_true += len(true_set & set(cands))

        for cid in cands:
            cinfo = cand_lookup.get(cid)
            if not cinfo: continue
            cname, caddr, ccountry, csrc = cinfo
            feats = compute_fast_features(name, addr, cname, caddr, country, ccountry, csrc)
            label = 1 if cid in true_set else 0

            if is_val:
                X_val.append(feats)
                y_val.append(label)
                val_pairs.append((s1_id, cid))
            else:
                X_train.append(feats)
                y_train.append(label)

    recall = recovered_val_true / total_val_true if total_val_true > 0 else 0
    print(f"\n[Validation Blocking Recall]: {recall:.2%} ({recovered_val_true:,} / {total_val_true:,})")

    X_tr = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    y_tr = np.array(y_train)
    X_v = pd.DataFrame(X_val, columns=FEATURE_NAMES)
    y_v = np.array(y_val)

    print(f"Training LightGBM on {len(X_tr):,} pairs (Positives: {y_tr.sum():,})...")
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(50, verbose=False)])

    probs = clf.predict_proba(X_v)[:, 1]

    # Evaluate F0.5 across thresholds
    best_th = 0.65
    best_f05 = -1
    for th in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        pred_map = defaultdict(set)
        for (sid, cid), p in zip(val_pairs, probs):
            if p >= th:
                pred_map[sid].add(cid)
        res = compute_macro_f05(gt_map, pred_map, val_s1_ids)
        print(f"Threshold {th:.2f} -> Macro F0.5 = {res['macro_f05']:.4f} (Exact set acc: {res['exact_set_acc']:.2%})")
        if res['macro_f05'] > best_f05:
            best_f05 = res['macro_f05']
            best_th = th

    print(f"\n[OPTIMAL CALIBRATION]: Threshold = {best_th} with Macro F0.5 = {best_f05:.4f}")
    return clf, best_th


def execute_test_inference_fast(clf, threshold: float = 0.65):
    print("\n" + "=" * 70)
    print("STEP 2: FULL TEST INFERENCE (1.73M Entities with Progress Bar)")
    print("=" * 70)

    # 1. Load Test S2 & S3
    print("Loading test candidate pools (S2 and S3)...")
    s2 = pd.read_csv(TEST_S2, sep='\t', keep_default_na=False)
    s3 = pd.read_csv(TEST_S3, sep='\t', keep_default_na=False)

    s2['business_name_norm'] = s2['business_name'].apply(normalize_name)
    s2['business_address_norm'] = s2['business_address'].apply(normalize_address)
    s3['business_name_norm'] = s3['business_name'].apply(normalize_name)
    s3['business_address_norm'] = s3['business_address'].apply(normalize_address)

    cand_pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    print("Indexing candidate pool in MultiBlocker...")
    blocker = HighRecallBlocker(max_candidates_per_key=80)
    blocker.fit(cand_pool)

    cand_lookup = {}
    for r in cand_pool.itertuples(index=False):
        cand_lookup[r.entity_id] = (
            r.business_name_norm, r.business_address_norm, r.country,
            'S2' if r.entity_id.startswith('S2-') else 'S3'
        )

    # 2. Stream Test S1 with Real-time Progress Bar
    total_test_s1 = 1732544
    chunk_size = 100000

    s1_reader = pd.read_csv(
        TEST_S1, sep='\t', keep_default_na=False, chunksize=chunk_size
    )

    OUTPUT_MATCHING.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MATCHING, 'w', encoding='utf-8') as f_m:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
    with open(OUTPUT_CANDIDATE, 'w', encoding='utf-8') as f_c:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

    pbar = tqdm(total=total_test_s1, desc="Processing Test S1", unit="entities")

    total_matches = 0
    total_candidates = 0

    for chunk in s1_reader:
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

            cands = blocker.get_candidates(country, name, addr, max_cands=40)
            c_lines.append(f"{s1_id}\t{','.join(cands)}\n")
            total_candidates += len(cands)

            if not cands:
                batch_slice_map.append((s1_id, 0, 0, []))
                continue

            start_idx = len(all_batch_pairs)
            valid_cids = []
            for cid in cands:
                cinfo = cand_lookup.get(cid)
                if not cinfo: continue
                cname, caddr, ccountry, csrc = cinfo
                feats = compute_fast_features(name, addr, cname, caddr, country, ccountry, csrc)
                all_batch_pairs.append(feats)
                valid_cids.append(cid)

            end_idx = len(all_batch_pairs)
            batch_slice_map.append((s1_id, start_idx, end_idx, valid_cids))

        # Batch prediction with LightGBM
        if all_batch_pairs:
            X_batch = pd.DataFrame(all_batch_pairs, columns=FEATURE_NAMES)
            batch_probs = clf.predict_proba(X_batch)[:, 1]

            for s1_id, start, end, cids in batch_slice_map:
                if start == end:
                    m_lines.append(f"{s1_id}\t\n")
                else:
                    pair_probs = batch_probs[start:end]
                    selected = [cid for cid, p in zip(cids, pair_probs) if p >= threshold]
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

        pbar.update(len(chunk))

    pbar.close()
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print(f"Total Candidates Written: {total_candidates:,}")
    print(f"Total Matches Predicted:  {total_matches:,}")
    print("=" * 70)


if __name__ == "__main__":
    clf, best_th = train_calibrated_model()
    execute_test_inference_fast(clf, threshold=best_th)
