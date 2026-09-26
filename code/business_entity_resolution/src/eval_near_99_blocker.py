import sys
from pathlib import Path
sys.path.insert(0, r'd:\amazon challenge\business-entity-resolution\code\business_entity_resolution')
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, RANDOM_STATE
from src.preprocessing.normalize import normalize_name, normalize_address, extract_numbers

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
    if not text:
        return set()
    cleaned = re.sub(r'\.(com|net|org|in|us|fr|co)', ' ', text.lower())
    tokens = re.findall(r'[a-z0-9]{' + str(min_len) + r',}', cleaned)
    return set(t for t in tokens if t not in STOPWORDS)

def run_near_99_blocker_eval():
    print("=" * 80)
    print("EVALUATING NEAR-99% BLOCKER ON GROUND TRUTH")
    print("=" * 80)
    
    gt_df = pd.read_csv(TRAIN_GT, sep='\t', keep_default_na=False)
    s1_df = pd.read_csv(TRAIN_S1, sep='\t', keep_default_na=False)
    
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
    
    s2_pool['business_name_norm'] = s2_pool['business_name'].apply(normalize_name)
    s2_pool['business_address_norm'] = s2_pool['business_address'].apply(normalize_address)
    s3_pool['business_name_norm'] = s3_pool['business_name'].apply(normalize_name)
    s3_pool['business_address_norm'] = s3_pool['business_address'].apply(normalize_address)
    
    cand_pool = pd.concat([s2_pool, s3_pool], ignore_index=True)
    print(f"Loaded {len(cand_pool):,} Candidate Records.")
    
    # Inverted indexes
    exact_name_idx = defaultdict(lambda: defaultdict(list))
    compact_name_idx = defaultdict(lambda: defaultdict(list))
    token_idx = defaultdict(lambda: defaultdict(list))
    exact_addr_idx = defaultdict(lambda: defaultdict(list))
    addr_combo_idx = defaultdict(lambda: defaultdict(list))
    prefix_idx = defaultdict(lambda: defaultdict(list))
    addr_token_idx = defaultdict(lambda: defaultdict(list))
    
    for r in cand_pool.itertuples(index=False):
        cid = r.entity_id
        c = r.country
        n = r.business_name_norm
        a = r.business_address_norm
        
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
            for at in list(atoks)[:2]:
                addr_token_idx[c][at].append(cid)
                
    s1_dict = {r.entity_id: r for r in s1_sub.itertuples(index=False)}
    
    for max_cands in [40, 60, 80]:
        retrieved_true = 0
        total_candidates_gen = 0
        
        for sid, true_set in gt_map.items():
            if not true_set:
                continue
            s1_row = s1_dict[sid]
            c = s1_row.country
            n = s1_row.business_name_norm
            a = s1_row.business_address_norm
            
            cands = set()
            # 1. Exact name
            if n:
                cands.update(exact_name_idx[c].get(n, [])[:50])
            
            # 2. Compact name
            comp = re.sub(r'[^a-z0-9]', '', n)
            if len(comp) >= 4 and len(cands) < max_cands:
                cands.update(compact_name_idx[c].get(comp, [])[:50])
                
            # 3. All discriminative name tokens
            toks = clean_tokens(n, min_len=3)
            for t in toks:
                if len(cands) >= max_cands: break
                m = token_idx[c].get(t, [])
                if len(m) <= 50:
                    cands.update(m)
                    
            # 4. Name prefix (5 chars)
            if len(comp) >= 5 and len(cands) < max_cands:
                m = prefix_idx[c].get(comp[:5], [])
                if len(m) <= 35:
                    cands.update(m)
                    
            # 5. Address exact & address combos
            if a and len(cands) < max_cands:
                cands.update(exact_addr_idx[c].get(a, [])[:40])
                nums = extract_numbers(a)
                atoks = clean_tokens(a, min_len=4)
                if nums and atoks:
                    for num in list(nums)[:2]:
                        for atok in list(atoks)[:2]:
                            if len(cands) >= max_cands: break
                            m = addr_combo_idx[c].get(f"{num}_{atok}", [])
                            if len(m) <= 30:
                                cands.update(m)
                                
            # 6. Fallback: If still under 10 candidates, check address tokens
            if len(cands) < 10 and a:
                atoks = clean_tokens(a, min_len=5)
                for at in list(atoks)[:2]:
                    if len(cands) >= max_cands: break
                    m = addr_token_idx[c].get(at, [])
                    if len(m) <= 20:
                        cands.update(m)
            
            cand_list = list(cands)[:max_cands]
            total_candidates_gen += len(cand_list)
            retrieved_true += len(true_set & set(cand_list))
            
        recall = (retrieved_true / total_true_pairs) * 100
        avg_cands = total_candidates_gen / len(gt_map)
        print(f"Max Cands = {max_cands:2d} | Recall: {recall:6.2f}% ({retrieved_true:,}/{total_true_pairs:,}) | Avg Candidates/Entity: {avg_cands:5.1f}")

if __name__ == '__main__':
    run_near_99_blocker_eval()
