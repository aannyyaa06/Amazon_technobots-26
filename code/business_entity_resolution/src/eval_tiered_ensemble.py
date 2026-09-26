import sys
from pathlib import Path
sys.path.insert(0, r'd:\amazon challenge\business-entity-resolution\code\business_entity_resolution')
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import lightgbm as lgb
from rapidfuzz import fuzz

from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, RANDOM_STATE
from src.preprocessing.normalize import normalize_name, normalize_address, extract_numbers
from src.evaluation.metrics import compute_macro_f05
from src.eval_ultra_precision import compute_advanced_features, ADV_FEATURE_NAMES, clean_tokens

def evaluate_multi_stage_ensemble():
    print("=" * 80)
    print("DEVELOPING 98-99.5% SOTA ENSEMBLE ENGINE")
    print("=" * 80)

    gt_df = pd.read_csv(TRAIN_GT, sep='\t', keep_default_na=False)
    s1_df = pd.read_csv(TRAIN_S1, sep='\t', keep_default_na=False)

    gt_df['match_count'] = gt_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
    sampled_gt = gt_df.groupby('match_count', group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) * 20000 / len(gt_df))), random_state=RANDOM_STATE)
    ).head(20000)

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
    s2_neg = s2[~s2.entity_id.isin(all_true)].sample(min(70000, len(s2)), random_state=RANDOM_STATE)
    s2_pool = pd.concat([s2_pos, s2_neg]).drop_duplicates('entity_id')

    s3_pos = s3[s3.entity_id.isin(all_true)]
    s3_neg = s3[~s3.entity_id.isin(all_true)].sample(min(70000, len(s3)), random_state=RANDOM_STATE)
    s3_pool = pd.concat([s3_pos, s3_neg]).drop_duplicates('entity_id')

    s2_pool['business_name_norm'] = s2_pool['business_name'].apply(normalize_name)
    s2_pool['business_address_norm'] = s2_pool['business_address'].apply(normalize_address)
    s3_pool['business_name_norm'] = s3_pool['business_name'].apply(normalize_name)
    s3_pool['business_address_norm'] = s3_pool['business_address'].apply(normalize_address)

    cand_pool = pd.concat([s2_pool, s3_pool], ignore_index=True)

    exact_name_idx = defaultdict(lambda: defaultdict(list))
    compact_name_idx = defaultdict(lambda: defaultdict(list))
    token_idx = defaultdict(lambda: defaultdict(list))
    exact_addr_idx = defaultdict(lambda: defaultdict(list))
    addr_combo_idx = defaultdict(lambda: defaultdict(list))
    prefix_idx = defaultdict(lambda: defaultdict(list))
    cand_lookup = {}

    for r in cand_pool.itertuples(index=False):
        cid = r.entity_id
        c = r.country
        n = r.business_name_norm
        a = r.business_address_norm
        cand_lookup[cid] = (n, a, c, 'S2' if cid.startswith('S2-') else 'S3')

        if n:
            exact_name_idx[c][n].append(cid)
            comp = "".join(ch for ch in n if ch.isalnum())
            if len(comp) >= 4:
                compact_name_idx[c][comp].append(cid)
                prefix_idx[c][comp[:5]].append(cid)
            toks = clean_tokens(n, min_len=3)
            for t in toks:
                token_idx[c][t].append(cid)
        if a:
            exact_addr_idx[c][a].append(cid)
            nums = extract_numbers(a)
            atoks = clean_tokens(a, min_len=4)
            if nums and atoks:
                for num in list(nums)[:2]:
                    for atok in list(atoks)[:2]:
                        addr_combo_idx[c][f"{num}_{atok}"].append(cid)

    all_s1 = list(s1_sub.itertuples(index=False))
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(all_s1)
    split_idx = int(len(all_s1) * 0.70)
    train_s1 = all_s1[:split_idx]
    val_s1 = all_s1[split_idx:]
    val_s1_ids = [r.entity_id for r in val_s1]

    def get_cands(row, max_cands=50):
        c, n, a = row.country, row.business_name_norm, row.business_address_norm
        comp = "".join(ch for ch in n if ch.isalnum())
        cands = set()
        if n: cands.update(exact_name_idx[c].get(n, [])[:50])
        if len(comp) >= 4 and len(cands) < max_cands:
            cands.update(compact_name_idx[c].get(comp, [])[:50])
        for t in clean_tokens(n, min_len=3):
            if len(cands) >= max_cands: break
            m = token_idx[c].get(t, [])
            if len(m) <= 50: cands.update(m)
        if len(comp) >= 5 and len(cands) < max_cands:
            m = prefix_idx[c].get(comp[:5], [])
            if len(m) <= 35: cands.update(m)
        if a and len(cands) < max_cands:
            cands.update(exact_addr_idx[c].get(a, [])[:40])
            nums = extract_numbers(a)
            atoks = clean_tokens(a, min_len=4)
            if nums and atoks:
                for num in list(nums)[:2]:
                    for atok in list(atoks)[:2]:
                        if len(cands) >= max_cands: break
                        m = addr_combo_idx[c].get(f"{num}_{atok}", [])
                        if len(m) <= 30: cands.update(m)
        return list(cands)[:max_cands]

    X_train, y_train = [], []
    for r in train_s1:
        sid = r.entity_id
        true_set = gt_map.get(sid, set())
        for cid in get_cands(r, max_cands=45):
            cinfo = cand_lookup.get(cid)
            if not cinfo: continue
            cn, ca, cc, csrc = cinfo
            feats = compute_advanced_features(r.business_name_norm, r.business_address_norm, cn, ca, r.country, cc, csrc)
            X_train.append(feats)
            y_train.append(1 if cid in true_set else 0)

    X_val, y_val, val_pairs = [], [], []
    for r in val_s1:
        sid = r.entity_id
        true_set = gt_map.get(sid, set())
        for cid in get_cands(r, max_cands=45):
            cinfo = cand_lookup.get(cid)
            if not cinfo: continue
            cn, ca, cc, csrc = cinfo
            feats = compute_advanced_features(r.business_name_norm, r.business_address_norm, cn, ca, r.country, cc, csrc)
            X_val.append(feats)
            y_val.append(1 if cid in true_set else 0)
            val_pairs.append((sid, cid))

    clf = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        learning_rate=0.04,
        num_leaves=140,
        max_depth=10,
        min_child_samples=30,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        n_estimators=450,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    )
    clf.fit(pd.DataFrame(X_train, columns=ADV_FEATURE_NAMES), np.array(y_train))
    val_probs = clf.predict_proba(pd.DataFrame(X_val, columns=ADV_FEATURE_NAMES))[:, 1]

    # Two-stage dual-threshold matching:
    # 1. Tier 1 (High Confidence): Any match with prob >= 0.70 is accepted unconditionally.
    # 2. Tier 2 (Contextual Verification): Any match with prob in [0.45, 0.70) is accepted IF exact address number or postal code matches!
    print("\n--- TWO-STAGE TIERED VERIFICATION RESULTS ---")
    val_pair_map = defaultdict(list)
    for (sid, cid), prob, feats in zip(val_pairs, val_probs, X_val):
        val_pair_map[sid].append((cid, prob, feats))

    for high_th in [0.65, 0.70, 0.75]:
        for low_th in [0.40, 0.45, 0.50]:
            pred_map = defaultdict(set)
            for sid, pairs in val_pair_map.items():
                for cid, prob, feats in pairs:
                    if prob >= high_th:
                        pred_map[sid].add(cid)
                    elif prob >= low_th:
                        # Check shared numbers or postal code match
                        shared_nums = feats[14]
                        same_postal = feats[21]
                        name_ratio = feats[1]
                        if (shared_nums > 0 or same_postal == 1.0) and name_ratio >= 0.60:
                            pred_map[sid].add(cid)
                            
            metrics = compute_macro_f05(gt_map, pred_map, val_s1_ids)
            f05 = metrics['macro_f05']
            print(f"High_Th={high_th:.2f}, Low_Th={low_th:.2f} | Macro F0.5 = {f05:.4f} ({f05*100:.2f}%) | Zero-Match F0.5 = {metrics['zero_match_f05']:.4f} | Exact Set Acc = {metrics['exact_set_acc']:.4f}")

if __name__ == '__main__':
    evaluate_multi_stage_ensemble()
