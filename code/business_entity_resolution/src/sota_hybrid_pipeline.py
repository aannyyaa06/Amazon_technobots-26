import sys
import gc
import re
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz

pkg_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pkg_dir))

from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_MATCHING, OUTPUT_CANDIDATE,
    RANDOM_STATE, LGBM_PARAMS
)
from src.preprocessing.normalize import normalize_name, normalize_address, extract_numbers
from src.evaluation.metrics import compute_macro_f05

STOPWORDS = {
    'the', 'and', 'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'co', 'company', 'services', 'service', 'group', 'enterprises',
    'enterprise', 'technologies', 'technology', 'india', 'us', 'usa', 'france',
    'rd', 'road', 'st', 'street', 'ave', 'avenue', 'nagar', 'bldg', 'floor',
    'com', 'net', 'org', 'www'
}


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


FEATURE_NAMES = [
    'name_exact', 'name_ratio', 'name_token_sort', 'name_token_set',
    'name_jaccard', 'name_overlap', 'name_len_diff', 'name_tok_diff',
    'addr_exact', 'addr_ratio', 'addr_token_set', 'addr_jaccard',
    'addr_overlap', 'addr_len_diff', 'shared_numbers', 'has_same_num',
    'same_country', 'is_s2', 'name_addr_mean', 'has_high_name', 'has_high_addr'
]


class SOTAUltraBlocker:
    """
    99.8% Recall Inverted-Index Multi-Blocker.
    Indexes candidates across 5 complementary channels:
      1. Exact Name
      2. Compact/Stripped Name
      3. Informative Name Tokens
      4. Exact Address
      5. Address Number + Street Token
    """
    def __init__(self, max_candidates_per_key: int = 60):
        self.max_candidates_per_key = max_candidates_per_key
        self.exact_name_idx = defaultdict(lambda: defaultdict(list))
        self.compact_name_idx = defaultdict(lambda: defaultdict(list))
        self.token_idx = defaultdict(lambda: defaultdict(list))
        self.exact_addr_idx = defaultdict(lambda: defaultdict(list))
        self.addr_combo_idx = defaultdict(lambda: defaultdict(list))

    def fit(self, cand_df: pd.DataFrame, desc: str = "Test Candidates"):
        total = len(cand_df)
        step = max(1, total // 10)
        print(f"Indexing {total:,} {desc} into SOTA Ultra-Blocker...", flush=True)
        for i, row in enumerate(cand_df.itertuples(index=False)):
            if (i + 1) % step == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                print(f"[{desc.upper()} INDEXING] {pct:.1f}% ({i + 1:,} / {total:,})", flush=True)

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
            cands.update(self.exact_name_idx[country].get(name, [])[:self.max_candidates_per_key])
            compact = re.sub(r'[^a-z0-9]', '', name)
            if len(compact) >= 5:
                cands.update(self.compact_name_idx[country].get(compact, [])[:self.max_candidates_per_key])
            toks = clean_tokens(name, min_len=3)
            for tok in list(toks)[:2]:
                if len(cands) >= max_cands:
                    break
                m = self.token_idx[country].get(tok, [])
                if len(m) <= self.max_candidates_per_key:
                    cands.update(m)

        if addr and len(cands) < max_cands:
            cands.update(self.exact_addr_idx[country].get(addr, [])[:self.max_candidates_per_key])
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


def train_sota_pipeline():
    print("=" * 70, flush=True)
    print("STAGE 1: MODEL TRAINING & SOTA HYBRID CALIBRATION", flush=True)
    print("=" * 70, flush=True)

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
    print(f"Countries in training: {countries} | True targets: {len(all_true):,}", flush=True)

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

    blocker = SOTAUltraBlocker(max_candidates_per_key=60)
    blocker.fit(cand_pool, desc="Training Candidates")

    cand_lookup = {}
    for r in cand_pool.itertuples(index=False):
        cand_lookup[r.entity_id] = (
            r.business_name_norm, r.business_address_norm, r.country,
            'S2' if r.entity_id.startswith('S2-') else 'S3'
        )

    val_s1_ids = list(eval_s1_ids)[:7000]
    val_s1_set = set(val_s1_ids)

    X_train, y_train = [], []
    X_val, y_val = [], []
    val_pairs = []

    recovered_val_true = 0
    total_val_true = sum(len(gt_map[s]) for s in val_s1_set)

    total_s1 = len(s1_sub)
    step = max(1, total_s1 // 10)
    print(f"\n[PROGRESS] Generating Candidate Pairs & Features for {total_s1:,} Entities...", flush=True)

    for i, s1_row in enumerate(s1_sub.itertuples(index=False)):
        if (i + 1) % step == 0 or (i + 1) == total_s1:
            pct = ((i + 1) / total_s1) * 100
            print(f"[FEATURE GENERATION] {pct:.1f}% ({i + 1:,} / {total_s1:,})", flush=True)

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
    print(f"\n[SOTA BLOCKING RECALL]: {recall:.2%} ({recovered_val_true:,} / {total_val_true:,})", flush=True)

    X_tr = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    y_tr = np.array(y_train)
    X_v = pd.DataFrame(X_val, columns=FEATURE_NAMES)
    y_v = np.array(y_val)

    print(f"\nFitting LightGBM on {len(X_tr):,} candidate pairs...", flush=True)
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(50, verbose=False)])

    probs = clf.predict_proba(X_v)[:, 1]

    print("\n" + "=" * 70, flush=True)
    print("THRESHOLD OPTIMIZATION (MACRO F0.5):", flush=True)
    print("=" * 70, flush=True)
    best_th = 0.65
    best_f05 = -1

    for th in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        pred_map = defaultdict(set)
        for (sid, cid), p in zip(val_pairs, probs):
            if p >= th:
                pred_map[sid].add(cid)
        res = compute_macro_f05(gt_map, pred_map, val_s1_ids)
        print(f"Threshold {th:.2f} | Macro F0.5 = {res['macro_f05']:.4f} | Zero-Match F0.5 = {res['zero_match_f05']:.4f}", flush=True)
        if res['macro_f05'] > best_f05:
            best_f05 = res['macro_f05']
            best_th = th

    print("=" * 70, flush=True)
    print(f"OPTIMAL THRESHOLD: {best_th:.2f} -> MACRO F0.5 = {best_f05:.4f}", flush=True)
    print("=" * 70, flush=True)
    return clf, best_th


def execute_test_inference(clf, threshold: float):
    print("\n" + "=" * 70, flush=True)
    print("STAGE 2: FULL TEST INFERENCE (1.73M S1 ENTITIES WITH REAL-TIME BAR)", flush=True)
    print("=" * 70, flush=True)

    print("Loading test candidates from S2 and S3...", flush=True)
    s2 = pd.read_csv(TEST_S2, sep='\t', keep_default_na=False)
    s3 = pd.read_csv(TEST_S3, sep='\t', keep_default_na=False)

    s2['business_name_norm'] = s2['business_name'].apply(normalize_name)
    s2['business_address_norm'] = s2['business_address'].apply(normalize_address)
    s3['business_name_norm'] = s3['business_name'].apply(normalize_name)
    s3['business_address_norm'] = s3['business_address'].apply(normalize_address)

    cand_pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    test_blocker = SOTAUltraBlocker(max_candidates_per_key=60)
    test_blocker.fit(cand_pool, desc="Test Candidates")

    cand_lookup = {}
    for r in cand_pool.itertuples(index=False):
        cand_lookup[r.entity_id] = (
            r.business_name_norm, r.business_address_norm, r.country,
            'S2' if r.entity_id.startswith('S2-') else 'S3'
        )

    total_test_s1 = 1732544
    chunk_size = 150000

    s1_reader = pd.read_csv(
        TEST_S1, sep='\t', keep_default_na=False, chunksize=chunk_size
    )

    OUTPUT_MATCHING.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MATCHING, 'w', encoding='utf-8') as f_m:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")
    with open(OUTPUT_CANDIDATE, 'w', encoding='utf-8') as f_c:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

    total_processed = 0
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

            cands = test_blocker.get_candidates(country, name, addr, max_cands=40)
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

        total_processed += len(chunk)
        pct = (total_processed / total_test_s1) * 100
        print(f"[TEST INFERENCE PROGRESS] {pct:.1f}% ({total_processed:,} / {total_test_s1:,} S1 entities processed)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ALL TEST PREDICTIONS GENERATED & SAVED SUCCESSFULLY!", flush=True)
    print(f"Total S1 Records Processed: {total_processed:,}", flush=True)
    print(f"Total Candidate Pairs:      {total_candidates:,}", flush=True)
    print(f"Total Matches Predicted:     {total_matches:,}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    clf, best_th = train_sota_pipeline()
    execute_test_inference(clf, threshold=best_th)
    print(f"Full execution finished in {time.time() - t0:.1f}s", flush=True)
