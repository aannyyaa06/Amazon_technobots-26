import sys
from pathlib import Path
sys.path.insert(0, r'd:\amazon challenge\business-entity-resolution\code\business_entity_resolution')
import pandas as pd
import numpy as np
from collections import defaultdict
from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, RANDOM_STATE
from src.evaluation.metrics import compute_macro_f05

def inspect_evaluation_ceiling():
    print("=" * 80)
    print("ANALYSIS OF THE THEORETICAL AND PRACTICAL CEILING ON MACRO F0.5")
    print("=" * 80)

    gt_df = pd.read_csv(TRAIN_GT, sep='\t', keep_default_na=False)
    gt_map = {}
    match_counts = []
    for _, r in gt_df.iterrows():
        sid = r['source1_entity_id']
        m = r['matched_entity_ids']
        if m:
            s = set(x.strip() for x in m.split(',') if x.strip())
            gt_map[sid] = s
            match_counts.append(len(s))
        else:
            gt_map[sid] = set()
            match_counts.append(0)

    match_counts = np.array(match_counts)
    total_entities = len(gt_df)
    zero_matches = np.sum(match_counts == 0)
    pos_matches = np.sum(match_counts > 0)

    print(f"Total S1 entities in Ground Truth: {total_entities:,}")
    print(f"Entities with 0 matches (singletons): {zero_matches:,} ({zero_matches/total_entities*100:.2f}%)")
    print(f"Entities with >=1 matches: {pos_matches:,} ({pos_matches/total_entities*100:.2f}%)")
    print(f"Average true matches among positive entities: {np.mean(match_counts[match_counts > 0]):.2f}")
    print(f"Distribution of match counts for positive entities:")
    for k in range(1, 10):
        cnt = np.sum(match_counts == k)
        print(f"  Count = {k}: {cnt:,} ({cnt/pos_matches*100:.2f}%)")
    print(f"  Count >= 10: {np.sum(match_counts >= 10):,} ({np.sum(match_counts >= 10)/pos_matches*100:.2f}%)")

    # Simulation: What happens if precision is 100% (zero false positives),
    # but blocker recall is 90%, 93%, 95%, 98%, 99%, 99.5%, 100%?
    print("\n--- MACRO F0.5 SENSITIVITY TO MATCH RETRIEVAL (ASSUMING 100% PRECISION) ---")
    for recall_rate in [0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99, 0.995, 1.0]:
        f05_scores = []
        for c in match_counts:
            if c == 0:
                f05_scores.append(1.0)
            else:
                # expected retrieved items:
                # If we retrieve k out of c items with 100% precision:
                # Precision = 1.0, Recall = k / c
                # F0.5 = 1.25 * 1.0 * (k/c) / (0.25 * 1.0 + k/c)
                # k ~ Binomial(c, recall_rate)
                # Let's compute expected F0.5 over binomial distribution
                probs = [np.math.comb(c, k) * (recall_rate**k) * ((1-recall_rate)**(c-k)) for k in range(c+1)]
                expected_f = 0.0
                for k, p in enumerate(probs):
                    if k == 0:
                        score = 0.0
                    else:
                        rec = k / c
                        score = (1.25 * 1.0 * rec) / (0.25 * 1.0 + rec)
                    expected_f += p * score
                f05_scores.append(expected_f)
        macro_f = np.mean(f05_scores)
        print(f"Recall = {recall_rate*100:5.1f}% | Simulated Macro F0.5 = {macro_f:.4f} ({macro_f*100:.2f}%)")

if __name__ == '__main__':
    inspect_evaluation_ceiling()
