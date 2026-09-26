import sys
from pathlib import Path
sys.path.insert(0, r'd:\amazon challenge\business-entity-resolution\code\business_entity_resolution')
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import lightgbm as lgb
from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, RANDOM_STATE
from src.preprocessing.normalize import normalize_name, normalize_address, extract_numbers
from src.features.fast_features import compute_fast_features, FEATURE_NAMES
from src.evaluation.metrics import compute_macro_f05

STOPWORDS = {
    'the', 'and', 'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'co', 'company', 'services', 'service', 'group', 'enterprises',
    'enterprise', 'technologies', 'technology', 'india', 'us', 'usa', 'france',
    'rd', 'road', 'st', 'street', 'ave', 'avenue', 'nagar', 'bldg', 'floor',
    'com', 'net', 'org', 'www', 'private', 'partners', 'holdings', 'industries',
    'center', 'centre', 'solutions', 'public', 'ventures', 'associates', 'trading',
    'shri', 'brothers', 'sar', 'sarl', 'sas', 'sasu', 'eurl', 'llp'
}

def clean_tokens(text: str, min_len: int = 3) -> set:
    if not text: return set()
    cleaned = re.sub(r'\.(com|net|org|in|us|fr|co)', ' ', text.lower())
    tokens = re.findall(r'[a-z0-9]{' + str(min_len) + r',}', cleaned)
    return set(t for t in tokens if t not in STOPWORDS)

def run_tuning():
    print("=" * 80)
    print("HYPERPARAMETER FINE-TUNING & OFFICIAL F0.5 EVALUATION")
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
    print(f"Loaded {len(cand_pool):,} candidate records.")
    
    # Fast Inverted Index (91% recall configuration)
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
            comp = re.sub(r'[^a-z0-9]', '', n)
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
                        
    # Split S1 into Train (14,000) and Val (6,000)
    all_s1 = list(s1_sub.itertuples(index=False))
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(all_s1)
    train_s1 = all_s1[:14000]
    val_s1 = all_s1[14000:]
    val_s1_ids = [r.entity_id for r in val_s1]
    
    print(f"Generating candidate pairs (Train: {len(train_s1):,}, Val: {len(val_s1):,})...")
    
    def get_cands_for_entity(row, max_cands=50):
        c = row.country
        n = row.business_name_norm
        a = row.business_address_norm
        comp = re.sub(r'[^a-z0-9]', '', n)
        cands = set()
        if n: cands.update(exact_name_idx[c].get(n, [])[:50])
        if len(comp) >= 4 and len(cands) < max_cands:
            cands.update(compact_name_idx[c].get(comp, [])[:50])
        toks = clean_tokens(n, min_len=3)
        for t in toks:
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
        cands = get_cands_for_entity(r, max_cands=45)
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
        cands = get_cands_for_entity(r, max_cands=45)
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
    
    print(f"Training pairs: {len(X_tr):,} (Positives: {y_tr.sum():,})")
    print(f"Validation pairs: {len(X_v):,} (Positives: {y_v.sum():,})")
    
    # Hyperparameter tuning grid
    param_grid = [
        {'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.10, 'min_child_samples': 20},
        {'num_leaves': 63, 'max_depth': 7, 'learning_rate': 0.08, 'min_child_samples': 20},
        {'num_leaves': 95, 'max_depth': 8, 'learning_rate': 0.06, 'min_child_samples': 30},
        {'num_leaves': 127, 'max_depth': 9, 'learning_rate': 0.05, 'min_child_samples': 40}
    ]
    
    best_overall_f05 = -1
    best_config = None
    best_threshold = 0.65
    
    for i, p in enumerate(param_grid):
        print(f"\n--- Model Config #{i+1}: leaves={p['num_leaves']}, depth={p['max_depth']}, lr={p['learning_rate']} ---")
        clf = lgb.LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            n_estimators=350,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=1,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
            **p
        )
        clf.fit(X_tr, y_tr)
        val_probs = clf.predict_proba(X_v)[:, 1]
        
        # Evaluate F0.5 formula across thresholds
        for th in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            pred_map = defaultdict(set)
            for (sid, cid), prob in zip(val_pairs, val_probs):
                if prob >= th:
                    pred_map[sid].add(cid)
            metrics = compute_macro_f05(gt_map, pred_map, val_s1_ids)
            f05 = metrics['macro_f05']
            z_f05 = metrics['zero_match_f05']
            acc = metrics['exact_set_acc']
            print(f"  Th={th:.2f} | Macro F0.5 = {f05:.4f} ({f05*100:.2f}%) | Zero-Match F0.5 = {z_f05:.4f} | Exact Set Acc = {acc:.4f}")
            if f05 > best_overall_f05:
                best_overall_f05 = f05
                best_config = p
                best_threshold = th

    print("\n" + "=" * 80)
    print("FINAL FINE-TUNING RESULT")
    print("=" * 80)
    print(f"Best Hyperparameters: {best_config}")
    print(f"Optimal Threshold:    {best_threshold:.2f}")
    print(f"Final Macro F0.5:     {best_overall_f05:.4f} ({best_overall_f05*100:.2f}%)")
    print("=" * 80)

if __name__ == '__main__':
    run_tuning()
