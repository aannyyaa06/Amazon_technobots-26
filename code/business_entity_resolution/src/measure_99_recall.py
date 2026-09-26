import sys
from pathlib import Path
sys.path.insert(0, r'd:\amazon challenge\business-entity-resolution\code\business_entity_resolution')
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, RANDOM_STATE
from src.preprocessing.normalize import normalize_name, normalize_address, extract_numbers

def measure_99_recall_and_candidates():
    print("=" * 80)
    print("MEASURING 99.13% BLOCKER RECALL AND CANDIDATE VOLUME")
    print("=" * 80)
    
    gt_df = pd.read_csv(TRAIN_GT, sep='\t', keep_default_na=False)
    s1_df = pd.read_csv(TRAIN_S1, sep='\t', keep_default_na=False)
    
    gt_df['match_count'] = gt_df['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x else 0)
    sampled_gt = gt_df.groupby('match_count', group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) * 5000 / len(gt_df))), random_state=RANDOM_STATE)
    ).head(5000)
    
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
    print(f"Loaded {len(cand_pool):,} candidate records.")
    
    ngram_idx = defaultdict(lambda: defaultdict(list))
    token_idx = defaultdict(lambda: defaultdict(list))
    exact_name_idx = defaultdict(lambda: defaultdict(list))
    addr_num_idx = defaultdict(lambda: defaultdict(list))
    addr_token_idx = defaultdict(lambda: defaultdict(list))
    
    for r in cand_pool.itertuples(index=False):
        c = r.country
        n = r.business_name_norm
        a = r.business_address_norm
        if n:
            exact_name_idx[c][n].append(r.entity_id)
            comp = re.sub(r'[^a-z0-9]', '', n)
            for i in range(len(comp)-2):
                ngram_idx[c][comp[i:i+3]].append(r.entity_id)
            for t in re.findall(r'[a-z0-9]{3,}', n):
                token_idx[c][t].append(r.entity_id)
        if a:
            for num in extract_numbers(a):
                addr_num_idx[c][num].append(r.entity_id)
            for at in re.findall(r'[a-z0-9]{4,}', a):
                addr_token_idx[c][at].append(r.entity_id)
                
    s1_dict = {r.entity_id: r for r in s1_sub.itertuples(index=False)}
    
    for max_cands in [40, 60, 80]:
        covered = 0
        total_candidates = 0
        for sid, tids in gt_map.items():
            s1_r = s1_dict[sid]
            c = s1_r.country
            n1 = s1_r.business_name_norm
            a1 = s1_r.business_address_norm
            comp1 = re.sub(r'[^a-z0-9]', '', n1)
            
            cands = set(exact_name_idx[c].get(n1, []))
            for t in re.findall(r'[a-z0-9]{3,}', n1):
                cands.update(token_idx[c].get(t, [])[:30])
            ngs = [comp1[i:i+3] for i in range(len(comp1)-2)]
            counts = defaultdict(int)
            for ng in ngs:
                for cid in ngram_idx[c].get(ng, []):
                    counts[cid] += 1
            cands.update(cid for cid, cnt in counts.items() if cnt >= 2)
            
            for num in extract_numbers(a1):
                cands.update(addr_num_idx[c].get(num, [])[:20])
            for at in re.findall(r'[a-z0-9]{4,}', a1)[:2]:
                cands.update(addr_token_idx[c].get(at, [])[:10])
                
            cand_list = list(cands)[:max_cands]
            total_candidates += len(cand_list)
            covered += len(tids & set(cand_list))
            
        recall = (covered / total_true_pairs) * 100
        avg_cands = total_candidates / len(gt_map)
        print(f"Max Cands = {max_cands:2d} | Recall: {recall:6.2f}% ({covered:,}/{total_true_pairs:,}) | Avg Candidates/Entity: {avg_cands:5.1f}")

if __name__ == '__main__':
    measure_99_recall_and_candidates()
