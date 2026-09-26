import sys
from pathlib import Path
sys.path.insert(0, r'd:\amazon challenge\business-entity-resolution\code\business_entity_resolution')
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import lightgbm as lgb
from rapidfuzz import fuzz
import gc
import os
import time

from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_MATCHING, OUTPUT_CANDIDATE,
    RANDOM_STATE
)

# -----------------------------------------------------------------------------
# 1. DEEP PREPROCESSING & NORMALIZATION
# -----------------------------------------------------------------------------
LEGAL_TERMS = {
    'pvt', 'ltd', 'limited', 'private', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'cie', 'services', 'service', 'solutions', 'solution',
    'group', 'holdings', 'holding', 'enterprises', 'enterprise', 'industries', 'industry',
    'technologies', 'technology', 'associates', 'associate', 'partners', 'partner',
    'sarl', 'sas', 'sasu', 'eurl', 'gmbh', 'sa', 'ag', 'bv', 'nv', 'shri', 'brothers',
    'm/s', 'dr', 'mr', 'trading', 'traders', 'agency', 'agencies', 'center', 'centre'
}

CITY_ALIASES = {
    'bengaluru': 'bangalore',
    'mumbai': 'bombay',
    'kolkata': 'calcutta',
    'chennai': 'madras',
    'varanasi': 'banaras',
    'prayagraj': 'allahabad',
    'gurugram': 'gurgaon',
    'puducherry': 'pondicherry',
    'kochi': 'cochin',
    'thiruvananthapuram': 'trivandrum',
    'vadodara': 'baroda',
    'vijayawada': 'bezawada',
    'mysuru': 'mysore',
    'visakhapatnam': 'vizag'
}

STREET_ALIASES = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'hwy': 'highway', 'pkway': 'parkway',
    'pkwy': 'parkway', 'ct': 'court', 'pl': 'place', 'sq': 'square',
    'bldg': 'building', 'fl': 'floor', 'flr': 'floor', 'apt': 'apartment',
    'ste': 'suite', 'dept': 'department', 'dist': 'district', 'sec': 'sector',
    'nagar': 'nagar', 'marg': 'marg', 'gali': 'gali', 'chowk': 'chowk',
    'r': 'rue', 'av': 'avenue', 'bd': 'boulevard', 'pl': 'place'
}

def clean_text_advanced(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r'[\r\n\t]+', ' ', t)
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'\.(com|net|org|in|us|fr|co|gov|edu)\b', ' ', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def normalize_business_name_deep(name: str) -> str:
    cleaned = clean_text_advanced(name)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    filtered = [t for t in tokens if t not in LEGAL_TERMS and len(t) > 1]
    if not filtered:
        filtered = tokens
    return " ".join(filtered)

def normalize_address_deep(addr: str) -> str:
    cleaned = clean_text_advanced(addr)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    res = []
    for t in tokens:
        if t in CITY_ALIASES:
            res.append(CITY_ALIASES[t])
        elif t in STREET_ALIASES:
            res.append(STREET_ALIASES[t])
        else:
            res.append(t)
    return " ".join(res)

def extract_digits(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r'\b\d+\b', text))

def extract_pin_or_zip(addr: str, country: str) -> str:
    if not addr:
        return ""
    if country == 'India':
        m = re.findall(r'\b[1-9]\d{5}\b', addr)
        return m[-1] if m else ""
    elif country == 'US':
        m = re.findall(r'\b\d{5}\b', addr)
        return m[-1] if m else ""
    elif country == 'France':
        m = re.findall(r'\b\d{5}\b', addr)
        return m[0] if m else ""
    return ""

def clean_tokens_set(text: str, min_len: int = 3) -> set:
    if not text:
        return set()
    return set(t for t in text.split() if len(t) >= min_len and t not in LEGAL_TERMS)

# -----------------------------------------------------------------------------
# 2. PAIRWISE FEATURE EXTRACTION
# -----------------------------------------------------------------------------
FEATURE_NAMES = [
    'name_exact', 'name_ratio', 'name_token_sort', 'name_token_set',
    'name_jaccard', 'name_overlap', 'name_len_diff', 'name_tok_diff',
    'addr_exact', 'addr_ratio', 'addr_token_set', 'addr_jaccard',
    'addr_overlap', 'addr_len_diff', 'shared_numbers', 'has_same_num',
    'same_country', 'is_s2', 'name_addr_mean', 'has_high_name', 'has_high_addr',
    'same_postal_code', 'postal_mismatch', 'char_3gram_jaccard', 'exact_prefix5'
]

def compute_pairwise_features(s1_name, s1_addr, c_name, c_addr, s1_country, c_country, c_src):
    s1_ntoks = clean_tokens_set(s1_name, 3)
    c_ntoks = clean_tokens_set(c_name, 3)
    s1_atoks = clean_tokens_set(s1_addr, 3)
    c_atoks = clean_tokens_set(c_addr, 3)

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

    s1_nums = extract_digits(s1_addr)
    c_nums = extract_digits(c_addr)
    shared_nums = len(s1_nums & c_nums)
    has_same_num = 1.0 if shared_nums > 0 else 0.0

    post1 = extract_pin_or_zip(s1_addr, s1_country)
    post2 = extract_pin_or_zip(c_addr, c_country)
    same_postal = 1.0 if (post1 and post2 and post1 == post2) else 0.0
    postal_mismatch = 1.0 if (post1 and post2 and post1 != post2) else 0.0

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
        abs(len(s1_name) - len(c_name)) / max(len(s1_name), len(c_name), 1),
        abs(len(s1_ntoks) - len(c_ntoks)) / max(len(s1_ntoks), len(c_ntoks), 1),
        1.0 if s1_addr == c_addr and s1_addr else 0.0,
        addr_ratio,
        addr_token_set,
        addr_jaccard,
        addr_overlap,
        abs(len(s1_addr) - len(c_addr)) / max(len(s1_addr), len(c_addr), 1),
        float(shared_nums),
        has_same_num,
        1.0 if s1_country == c_country else 0.0,
        1.0 if c_src == 'S2' else 0.0,
        (name_token_set + addr_token_set) / 2.0,
        1.0 if name_token_set >= 0.85 else 0.0,
        1.0 if addr_token_set >= 0.85 else 0.0,
        same_postal,
        postal_mismatch,
        ng_jaccard,
        exact_prefix5
    ]

# -----------------------------------------------------------------------------
# 3. TRAINING & CALIBRATION ENGINE
# -----------------------------------------------------------------------------
def train_high_precision_model():
    print("[1/4] Training High-Precision LightGBM Model with Preprocessed Data...")
    gt_df = pd.read_csv(TRAIN_GT, sep='\t', keep_default_na=False)
    s1_df = pd.read_csv(TRAIN_S1, sep='\t', keep_default_na=False)

    gt_df['match_count'] = gt_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
    sampled_gt = gt_df.groupby('match_count', group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) * 35000 / len(gt_df))), random_state=RANDOM_STATE)
    ).head(35000)

    eval_s1_ids = set(sampled_gt['source1_entity_id'])
    s1_sub = s1_df[s1_df.entity_id.isin(eval_s1_ids)].copy()
    s1_sub['name_norm'] = s1_sub['business_name'].apply(normalize_business_name_deep)
    s1_sub['addr_norm'] = s1_sub['business_address'].apply(normalize_address_deep)

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
    s2_neg = s2[~s2.entity_id.isin(all_true)].sample(min(80000, len(s2)), random_state=RANDOM_STATE)
    s2_pool = pd.concat([s2_pos, s2_neg]).drop_duplicates('entity_id')

    s3_pos = s3[s3.entity_id.isin(all_true)]
    s3_neg = s3[~s3.entity_id.isin(all_true)].sample(min(80000, len(s3)), random_state=RANDOM_STATE)
    s3_pool = pd.concat([s3_pos, s3_neg]).drop_duplicates('entity_id')

    s2_pool['name_norm'] = s2_pool['business_name'].apply(normalize_business_name_deep)
    s2_pool['addr_norm'] = s2_pool['business_address'].apply(normalize_address_deep)
    s3_pool['name_norm'] = s3_pool['business_name'].apply(normalize_business_name_deep)
    s3_pool['addr_norm'] = s3_pool['business_address'].apply(normalize_address_deep)

    cand_pool = pd.concat([s2_pool, s3_pool], ignore_index=True)
    del s2, s3, s2_pos, s2_neg, s3_pos, s3_neg; gc.collect()

    exact_name_idx = defaultdict(lambda: defaultdict(list))
    compact_name_idx = defaultdict(lambda: defaultdict(list))
    token_idx = defaultdict(lambda: defaultdict(list))
    prefix_idx = defaultdict(lambda: defaultdict(list))
    cand_lookup = {}

    for r in cand_pool.itertuples(index=False):
        cid = r.entity_id
        c = r.country
        n = r.name_norm
        a = r.addr_norm
        cand_lookup[cid] = (n, a, c, 'S2' if cid.startswith('S2-') else 'S3')

        if n:
            exact_name_idx[c][n].append(cid)
            comp = re.sub(r'[^a-z0-9]', '', n)
            if len(comp) >= 4:
                compact_name_idx[c][comp].append(cid)
                prefix_idx[c][comp[:5]].append(cid)
            toks = clean_tokens_set(n, 3)
            for t in toks:
                token_idx[c][t].append(cid)

    X_train, y_train = [], []
    for r in s1_sub.itertuples(index=False):
        sid = r.entity_id
        c = r.country
        n = r.name_norm
        comp = re.sub(r'[^a-z0-9]', '', n)
        true_set = gt_map.get(sid, set())

        cands = set()
        if n: cands.update(exact_name_idx[c].get(n, [])[:40])
        if len(comp) >= 4 and len(cands) < 50:
            cands.update(compact_name_idx[c].get(comp, [])[:40])
        for t in clean_tokens_set(n, 3):
            if len(cands) >= 50: break
            m = token_idx[c].get(t, [])
            if len(m) <= 40: cands.update(m)
        if len(comp) >= 5 and len(cands) < 50:
            m = prefix_idx[c].get(comp[:5], [])
            if len(m) <= 30: cands.update(m)

        for cid in list(cands)[:50]:
            cinfo = cand_lookup.get(cid)
            if not cinfo: continue
            cn, ca, cc, csrc = cinfo
            feats = compute_pairwise_features(n, r.addr_norm, cn, ca, c, cc, csrc)
            X_train.append(feats)
            y_train.append(1 if cid in true_set else 0)

    clf = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        learning_rate=0.04,
        num_leaves=150,
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
    clf.fit(pd.DataFrame(X_train, columns=FEATURE_NAMES), np.array(y_train))
    print(f"Model trained successfully on {len(X_train):,} pairs.")
    return clf

# -----------------------------------------------------------------------------
# 4. STREAMING TEST INFERENCE WITH 1-TO-1 COMPETITIVE ASSIGNMENT
# -----------------------------------------------------------------------------
def run_sota_test_pipeline(clf, high_th=0.68):
    print("[2/4] Building High-Recall Index for Test Candidate Pool (S2 + S3)...")
    t0 = time.time()

    exact_name_idx = defaultdict(lambda: defaultdict(list))
    compact_name_idx = defaultdict(lambda: defaultdict(list))
    token_idx = defaultdict(lambda: defaultdict(list))
    prefix_idx = defaultdict(lambda: defaultdict(list))
    cand_lookup = {}

    def index_source(path, prefix):
        print(f"  Indexing {prefix} from {path}...")
        for chunk in pd.read_csv(path, sep='\t', chunksize=250000, keep_default_na=False):
            chunk['name_norm'] = chunk['business_name'].apply(normalize_business_name_deep)
            chunk['addr_norm'] = chunk['business_address'].apply(normalize_address_deep)
            for r in chunk.itertuples(index=False):
                cid = r.entity_id
                c = r.country
                n = r.name_norm
                a = r.addr_norm
                cand_lookup[cid] = (n, a, c, prefix)
                if n:
                    exact_name_idx[c][n].append(cid)
                    comp = re.sub(r'[^a-z0-9]', '', n)
                    if len(comp) >= 4:
                        compact_name_idx[c][comp].append(cid)
                        prefix_idx[c][comp[:5]].append(cid)
                    toks = clean_tokens_set(n, 3)
                    for t in toks:
                        token_idx[c][t].append(cid)

    index_source(TEST_S2, 'S2')
    index_source(TEST_S3, 'S3')
    print(f"Indexed {len(cand_lookup):,} candidate records in {time.time()-t0:.1f}s.")

    print(f"[3/4] Streaming Test Source 1 & Scoring Candidates...")
    t_stream = time.time()
    Path(OUTPUT_MATCHING).parent.mkdir(parents=True, exist_ok=True)

    # In streaming mode with 1-to-1 competitive assignment:
    # We maintain target_assigned_best: cid -> (s1_id, max_prob)
    # To handle scale efficiently across chunks, we store matches per chunk,
    # and write candidate pairs directly.
    cand_f = open(OUTPUT_CANDIDATE, 'w', encoding='utf-8')
    cand_f.write("source1_entity_id\tcandidate_entity_ids\n")

    # In-memory predicted matches: sid -> list of (cid, prob)
    predictions_map = defaultdict(list)
    s1_all_ids = []

    chunk_size = 100000
    processed = 0

    for s1_chunk in pd.read_csv(TEST_S1, sep='\t', chunksize=chunk_size, keep_default_na=False):
        s1_chunk['name_norm'] = s1_chunk['business_name'].apply(normalize_business_name_deep)
        s1_chunk['addr_norm'] = s1_chunk['business_address'].apply(normalize_address_deep)

        chunk_pairs = []
        chunk_feats = []

        for r in s1_chunk.itertuples(index=False):
            sid = r.entity_id
            s1_all_ids.append(sid)
            c = r.country
            n = r.name_norm
            a = r.addr_norm
            comp = re.sub(r'[^a-z0-9]', '', n)

            cands = set()
            if n: cands.update(exact_name_idx[c].get(n, [])[:45])
            if len(comp) >= 4 and len(cands) < 50:
                cands.update(compact_name_idx[c].get(comp, [])[:45])
            for t in clean_tokens_set(n, 3):
                if len(cands) >= 50: break
                m = token_idx[c].get(t, [])
                if len(m) <= 40: cands.update(m)
            if len(comp) >= 5 and len(cands) < 50:
                m = prefix_idx[c].get(comp[:5], [])
                if len(m) <= 30: cands.update(m)

            cand_list = list(cands)[:50]
            cand_str = ",".join(cand_list) if cand_list else ""
            cand_f.write(f"{sid}\t{cand_str}\n")

            for cid in cand_list:
                cinfo = cand_lookup.get(cid)
                if not cinfo: continue
                cn, ca, cc, csrc = cinfo
                feats = compute_pairwise_features(n, a, cn, ca, c, cc, csrc)
                chunk_pairs.append((sid, cid))
                chunk_feats.append(feats)

        if chunk_feats:
            X_df = pd.DataFrame(chunk_feats, columns=FEATURE_NAMES)
            probs = clf.predict_proba(X_df)[:, 1]
            for (sid, cid), prob in zip(chunk_pairs, probs):
                if prob >= high_th:
                    predictions_map[sid].append((cid, prob))

        processed += len(s1_chunk)
        print(f"  Processed {processed:,} / 1,732,544 test entities ({time.time()-t_stream:.1f}s)...")

    cand_f.close()
    print("Candidate pairs written successfully.")

    print("[4/4] Applying Global 1-to-1 Competitive Assignment & Writing Matches...")
    # Flat list of all high-confidence predicted pairs: (prob, sid, cid)
    all_pairs = []
    for sid, matches in predictions_map.items():
        for cid, prob in matches:
            all_pairs.append((prob, sid, cid))

    # Sort globally by probability descending
    all_pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_cids = set()
    final_matches = defaultdict(list)

    for prob, sid, cid in all_pairs:
        if cid not in assigned_cids:
            assigned_cids.add(cid)
            final_matches[sid].append(cid)

    # Write matching_results.tsv preserving exact order of TEST_S1
    with open(OUTPUT_MATCHING, 'w', encoding='utf-8') as match_f:
        match_f.write("source1_entity_id\tmatched_entity_ids\n")
        non_empty = 0
        singletons = 0
        for sid in s1_all_ids:
            m_list = final_matches.get(sid, [])
            if m_list:
                match_f.write(f"{sid}\t{','.join(m_list)}\n")
                non_empty += 1
            else:
                match_f.write(f"{sid}\t\n")
                singletons += 1

    print(f"Inference Complete in {time.time()-t0:.1f}s!")
    print(f"Total Rows: {len(s1_all_ids):,}")
    print(f"Entities with Matches: {non_empty:,} ({non_empty/len(s1_all_ids)*100:.2f}%)")
    print(f"Singletons: {singletons:,} ({singletons/len(s1_all_ids)*100:.2f}%)")

if __name__ == '__main__':
    clf = train_high_precision_model()
    run_sota_test_pipeline(clf, high_th=0.68)
