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

STOPWORDS = {
    'the', 'and', 'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'co', 'company', 'services', 'service', 'group', 'enterprises',
    'enterprise', 'technologies', 'technology', 'india', 'us', 'usa', 'france',
    'rd', 'road', 'st', 'street', 'ave', 'avenue', 'nagar', 'bldg', 'floor',
    'com', 'net', 'org', 'www', 'private', 'partners', 'holdings', 'industries',
    'center', 'centre', 'solutions', 'public', 'ventures', 'associates', 'trading',
    'shri', 'brothers', 'sar', 'sarl', 'sas', 'sasu', 'eurl', 'llp'
}

ADV_FEATURE_NAMES = [
    'name_exact', 'name_ratio', 'name_token_sort', 'name_token_set',
    'name_jaccard', 'name_overlap', 'name_len_diff', 'name_tok_diff',
    'addr_exact', 'addr_ratio', 'addr_token_set', 'addr_jaccard',
    'addr_overlap', 'addr_len_diff', 'shared_numbers', 'has_same_num',
    'same_country', 'is_s2', 'name_addr_mean', 'has_high_name', 'has_high_addr',
    'same_postal_code', 'postal_mismatch', 'char_3gram_jaccard', 'exact_prefix5'
]

def clean_tokens(text: str, min_len: int = 3) -> set:
    if not text: return set()
    cleaned = re.sub(r'\.(com|net|org|in|us|fr|co)', ' ', text.lower())
    tokens = re.findall(r'[a-z0-9]{' + str(min_len) + r',}', cleaned)
    return set(t for t in tokens if t not in STOPWORDS)

def extract_postal_code(addr: str, country: str) -> str:
    if not addr: return ""
    if country == 'US':
        m = re.findall(r'\b\d{5}\b', addr)
        return m[-1] if m else ""
    elif country == 'India':
        m = re.findall(r'\b[1-9]\d{5}\b', addr)
        return m[-1] if m else ""
    elif country == 'France':
        m = re.findall(r'\b\d{5}\b', addr)
        return m[0] if m else ""
    return ""

def compute_advanced_features(s1_name, s1_addr, c_name, c_addr, s1_country, c_country, c_src):
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

    # Advanced precision signals
    post1 = extract_postal_code(s1_addr, s1_country)
    post2 = extract_postal_code(c_addr, c_country)
    same_postal = 1.0 if (post1 and post2 and post1 == post2) else 0.0
    postal_mismatch = 1.0 if (post1 and post2 and post1 != post2) else 0.0

    # Char 3-gram jaccard
    comp1 = re.sub(r'[^a-z0-9]', '', s1_name)
    comp2 = re.sub(r'[^a-z0-9]', '', c_name)
    ng1 = set(comp1[i:i+3] for i in range(len(comp1)-2)) if len(comp1) >= 3 else set()
    ng2 = set(comp2[i:i+3] for i in range(len(comp2)-2)) if len(comp2) >= 3 else set()
    ng_union = len(ng1 | ng2)
    ng_jaccard = len(ng1 & ng2) / ng_union if ng_union > 0 else 0.0

    exact_prefix5 = 1.0 if (len(comp1) >= 5 and len(comp2) >= 5 and comp1[:5] == comp2[:5]) else 0.0

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
        1.0 if addr_token_set >= 0.85 else 0.0,
        same_postal,
        postal_mismatch,
        ng_jaccard,
        exact_prefix5
    ]

def evaluate_ultra_precision():
    print("=" * 80)
    print("TESTING ULTRA-PRECISION MODEL (TARGET: 98-99.5% MACRO F0.5)")
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

    all_s1 = list(s1_sub.itertuples(index=False))
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(all_s1)
    split_idx = int(len(all_s1) * 0.70)
    train_s1 = all_s1[:split_idx]
    val_s1 = all_s1[split_idx:]
    val_s1_ids = [r.entity_id for r in val_s1]

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
            feats = compute_advanced_features(r.business_name_norm, r.business_address_norm, cn, ca, r.country, cc, csrc)
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
            feats = compute_advanced_features(r.business_name_norm, r.business_address_norm, cn, ca, r.country, cc, csrc)
            X_val.append(feats)
            y_val.append(1 if cid in true_set else 0)
            val_pairs.append((sid, cid))

    X_tr = pd.DataFrame(X_train, columns=ADV_FEATURE_NAMES)
    y_tr = np.array(y_train)
    X_v = pd.DataFrame(X_val, columns=ADV_FEATURE_NAMES)
    y_v = np.array(y_val)

    print(f"Training pairs: {len(X_tr):,} | Validation pairs: {len(X_v):,}")

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
    clf.fit(X_tr, y_tr)
    val_probs = clf.predict_proba(X_v)[:, 1]

    # Evaluate with and without Competitive Margin Pruning
    print("\n" + "=" * 70)
    print("STANDARD CALIBRATION RESULTS:")
    print("=" * 70)
    for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        pred_map = defaultdict(set)
        for (sid, cid), prob in zip(val_pairs, val_probs):
            if prob >= th:
                pred_map[sid].add(cid)
        metrics = compute_macro_f05(gt_map, pred_map, val_s1_ids)
        print(f"Th={th:.2f} | Macro F0.5 = {metrics['macro_f05']:.4f} ({metrics['macro_f05']*100:.2f}%) | Zero-Match F0.5 = {metrics['zero_match_f05']:.4f} | Exact Set Acc = {metrics['exact_set_acc']:.4f}")

    print("\n" + "=" * 70)
    print("POST-PROCESSING: COMPETITIVE MARGIN PRUNING (ELIMINATING FALSE MERGES):")
    print("=" * 70)
    # Competitive margin pruning: for any entity, if max probability >= 0.80, prune any candidate with prob < (max_prob - 0.25)
    entity_pair_probs = defaultdict(list)
    for (sid, cid), prob in zip(val_pairs, val_probs):
        entity_pair_probs[sid].append((cid, prob))

    for th in [0.65, 0.70, 0.75]:
        for margin in [0.20, 0.25, 0.30]:
            pred_map = defaultdict(set)
            for sid, pairs in entity_pair_probs.items():
                if not pairs: continue
                max_p = max(p for _, p in pairs)
                for cid, p in pairs:
                    if p >= th and (p >= (max_p - margin)):
                        pred_map[sid].add(cid)
            metrics = compute_macro_f05(gt_map, pred_map, val_s1_ids)
            print(f"Th={th:.2f}, Margin={margin:.2f} | Macro F0.5 = {metrics['macro_f05']:.4f} ({metrics['macro_f05']*100:.2f}%) | Zero-Match F0.5 = {metrics['zero_match_f05']:.4f} | Exact Set Acc = {metrics['exact_set_acc']:.4f}")

if __name__ == '__main__':
    evaluate_ultra_precision()
