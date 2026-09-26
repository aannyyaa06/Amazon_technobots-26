import sys
from pathlib import Path
sys.path.insert(0, r'd:\amazon challenge\business-entity-resolution\code\business_entity_resolution')
import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, RANDOM_STATE
from src.preprocessing.normalize import normalize_name, normalize_address, extract_numbers

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

def run_diagnostics():
    print("=" * 80)
    print("STEP 2 & 3: BLOCKER RECALL DIAGNOSTICS & MISSED PAIR ANALYSIS")
    print("=" * 80)
    
    # 1. Sample 10,000 validation entities with ground truth
    gt_df = pd.read_csv(TRAIN_GT, sep='\t', keep_default_na=False)
    s1_df = pd.read_csv(TRAIN_S1, sep='\t', keep_default_na=False)
    
    # Stratified sample
    gt_df['match_count'] = gt_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
    sampled_gt = gt_df.groupby('match_count', group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) * 10000 / len(gt_df))), random_state=RANDOM_STATE)
    ).head(10000)
    
    eval_s1_ids = set(sampled_gt['source1_entity_id'])
    s1_sub = s1_df[s1_df.entity_id.isin(eval_s1_ids)].copy()
    s1_sub['business_name_norm'] = s1_sub['business_name'].apply(normalize_name)
    s1_sub['business_address_norm'] = s1_sub['business_address'].apply(normalize_address)
    
    gt_map = {}
    all_true = set()
    total_true_pairs = 0
    for _, r in sampled_gt.iterrows():
        sid = r['source1_entity_id']
        m = r['matched_entity_ids']
        if m:
            s = set(x.strip() for x in m.split(',') if x.strip())
            gt_map[sid] = s
            all_true.update(s)
            total_true_pairs += len(s)
        else:
            gt_map[sid] = set()
            
    print(f"Evaluated S1 Entities: {len(eval_s1_ids):,} | Total True Match Pairs: {total_true_pairs:,}")
    
    countries = set(s1_sub.country.unique())
    s2 = pd.read_csv(TRAIN_S2, sep='\t', keep_default_na=False)
    s3 = pd.read_csv(TRAIN_S3, sep='\t', keep_default_na=False)
    s2 = s2[s2.country.isin(countries)]
    s3 = s3[s3.country.isin(countries)]
    
    # Positive pool + negative pool
    s2_pos = s2[s2.entity_id.isin(all_true)]
    s2_neg = s2[~s2.entity_id.isin(all_true)].sample(min(80000, len(s2)), random_state=RANDOM_STATE)
    s2_pool = pd.concat([s2_pos, s2_neg]).drop_duplicates('entity_id')
    
    s3_pos = s3[s3.entity_id.isin(all_true)]
    s3_neg = s3[~s3.entity_id.isin(all_true)].sample(min(80000, len(s3)), random_state=RANDOM_STATE)
    s3_pool = pd.concat([s3_pos, s3_neg]).drop_duplicates('entity_id')
    
    s2_pool['business_name_norm'] = s2_pool['business_name'].apply(normalize_name)
    s2_pool['business_address_norm'] = s2_pool['business_address'].apply(normalize_address)
    s3_pool['business_name_norm'] = s3_pool['business_name'].apply(normalize_name)
    s3_pool['business_address_norm'] = s3_pool['business_address'].apply(normalize_address)
    
    cand_pool = pd.concat([s2_pool, s3_pool], ignore_index=True)
    print(f"Candidate Pool: {len(cand_pool):,} records (contains 100% of available true matches in candidate subset)")
    
    # Build each blocker index separately
    exact_name_idx = defaultdict(lambda: defaultdict(list))
    compact_name_idx = defaultdict(lambda: defaultdict(list))
    token_idx = defaultdict(lambda: defaultdict(list))
    exact_addr_idx = defaultdict(lambda: defaultdict(list))
    addr_combo_idx = defaultdict(lambda: defaultdict(list))
    cand_info = {}
    
    for r in cand_pool.itertuples(index=False):
        cid = r.entity_id
        c = r.country
        n = r.business_name_norm
        a = r.business_address_norm
        cand_info[cid] = (n, a, c)
        
        if n:
            exact_name_idx[c][n].append(cid)
            comp = re.sub(r'[^a-z0-9]', '', n)
            if len(comp) >= 5:
                compact_name_idx[c][comp].append(cid)
            toks = clean_tokens(n, min_len=3)
            for t in list(toks)[:2]:
                token_idx[c][t].append(cid)
        if a:
            exact_addr_idx[c][a].append(cid)
            nums = extract_numbers(a)
            atoks = clean_tokens(a, min_len=4)
            if nums and atoks:
                for num in list(nums)[:1]:
                    for atok in list(atoks)[:1]:
                        addr_combo_idx[c][f"{num}_{atok}"].append(cid)
                        
    # 2. Evaluate Individual Blocker Recalls
    rec_b1 = 0
    rec_b2 = 0
    rec_b3 = 0
    rec_b4 = 0
    rec_b12 = 0
    rec_b123 = 0
    rec_all = 0
    
    missed_pairs = []
    
    s1_dict = {r.entity_id: r for r in s1_sub.itertuples(index=False)}
    
    for sid, true_set in gt_map.items():
        if not true_set:
            continue
        s1_row = s1_dict[sid]
        c = s1_row.country
        n = s1_row.business_name_norm
        a = s1_row.business_address_norm
        
        # B1: Exact Name
        cands_b1 = set(exact_name_idx[c].get(n, [])[:50])
        # B2: Compact Name
        comp = re.sub(r'[^a-z0-9]', '', n)
        cands_b2 = set(compact_name_idx[c].get(comp, [])[:50]) if len(comp) >= 5 else set()
        # B3: Token Index
        cands_b3 = set()
        for t in list(clean_tokens(n, min_len=3))[:2]:
            m = token_idx[c].get(t, [])
            if len(m) <= 50:
                cands_b3.update(m)
        # B4: Address Index
        cands_b4 = set(exact_addr_idx[c].get(a, [])[:50])
        nums = extract_numbers(a)
        atoks = clean_tokens(a, min_len=4)
        if nums and atoks:
            for num in list(nums)[:1]:
                for atok in list(atoks)[:1]:
                    m = addr_combo_idx[c].get(f"{num}_{atok}", [])
                    if len(m) <= 30:
                        cands_b4.update(m)
                        
        cands_b12 = cands_b1 | cands_b2
        cands_b123 = cands_b12 | cands_b3
        cands_all = cands_b123 | cands_b4
        
        rec_b1 += len(true_set & cands_b1)
        rec_b2 += len(true_set & cands_b2)
        rec_b3 += len(true_set & cands_b3)
        rec_b4 += len(true_set & cands_b4)
        rec_b12 += len(true_set & cands_b12)
        rec_b123 += len(true_set & cands_b123)
        rec_all += len(true_set & cands_all)
        
        missed = true_set - cands_all
        for target_id in missed:
            if target_id in cand_info:
                t_name, t_addr, _ = cand_info[target_id]
                missed_pairs.append({
                    's1_id': sid, 's1_name': n, 's1_addr': a,
                    'target_id': target_id, 'target_name': t_name, 'target_addr': t_addr
                })

    print("\n" + "=" * 60)
    print("INDIVIDUAL & CUMULATIVE BLOCKER RECALL TABLE")
    print("=" * 60)
    print(f"{'Blocker Strategy':<30} | {'Recall':>10} | {'Retrieved True Pairs':>20}")
    print("-" * 66)
    print(f"{'Blocker 1 (Exact Name)':<30} | {rec_b1/total_true_pairs*100:>9.2f}% | {rec_b1:>14,} / {total_true_pairs:,}")
    print(f"{'Blocker 2 (Compact Name)':<30} | {rec_b2/total_true_pairs*100:>9.2f}% | {rec_b2:>14,} / {total_true_pairs:,}")
    print(f"{'Blocker 3 (Token Index)':<30} | {rec_b3/total_true_pairs*100:>9.2f}% | {rec_b3:>14,} / {total_true_pairs:,}")
    print(f"{'Blocker 4 (Address Index)':<30} | {rec_b4/total_true_pairs*100:>9.2f}% | {rec_b4:>14,} / {total_true_pairs:,}")
    print("-" * 66)
    print(f"{'Blocker 1 + 2':<30} | {rec_b12/total_true_pairs*100:>9.2f}% | {rec_b12:>14,} / {total_true_pairs:,}")
    print(f"{'Blocker 1 + 2 + 3':<30} | {rec_b123/total_true_pairs*100:>9.2f}% | {rec_b123:>14,} / {total_true_pairs:,}")
    print(f"{'Blocker 1 + 2 + 3 + 4 (All)':<30} | {rec_all/total_true_pairs*100:>9.2f}% | {rec_all:>14,} / {total_true_pairs:,}")
    print("=" * 66)
    
    print(f"\nTotal Missed True Matches: {len(missed_pairs):,}")
    print("\n" + "=" * 80)
    print("SAMPLE MISSED PAIRS & FAILURE PATTERN ANALYSIS (First 25)")
    print("=" * 80)
    
    categories = Counter()
    for i, m in enumerate(missed_pairs[:50]):
        s1_n, tg_n = m['s1_name'], m['target_name']
        s1_a, tg_a = m['s1_addr'], m['target_addr']
        
        # Categorize
        cat = "other"
        s1_words = set(s1_n.split())
        tg_words = set(tg_n.split())
        
        if s1_n == tg_n:
            cat = "name_identical_but_blocked"
        elif s1_words & tg_words:
            cat = "partial_word_overlap_not_first2"
        elif any(w1 in tg_n or w2 in s1_n for w1 in s1_words for w2 in tg_words if len(w1)>3 and len(w2)>3):
            cat = "substring_or_abbreviation"
        elif extract_numbers(s1_a) & extract_numbers(tg_a):
            cat = "same_address_number_different_name"
        else:
            cat = "severe_name_variation_or_alias"
            
        categories[cat] += 1
        if i < 20:
            print(f"[{i+1}] Failure Mode: {cat}")
            print(f"     S1 ({m['s1_id']}):     Name='{s1_n}' | Addr='{s1_a}'")
            print(f"     Target ({m['target_id']}): Name='{tg_n}' | Addr='{tg_a}'\n")

    print("\nInitial Failure Distribution in Sample:")
    for c, cnt in categories.most_common():
        print(f"  - {c}: {cnt}")

if __name__ == '__main__':
    run_diagnostics()
