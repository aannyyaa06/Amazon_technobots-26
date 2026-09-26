from typing import Dict, Set, List
import numpy as np


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """
    Compute F0.5 score for a single Source 1 entity:
    F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
    """
    if len(true_matches) == 0 and len(pred_matches) == 0:
        return 1.0
    if len(true_matches) == 0 and len(pred_matches) > 0:
        return 0.0
    if len(true_matches) > 0 and len(pred_matches) == 0:
        return 0.0

    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0

    p = tp / len(pred_matches)
    r = tp / len(true_matches)

    denom = 0.25 * p + r
    return (1.25 * p * r) / denom if denom > 0 else 0.0


def compute_macro_f05(
    gt_map: Dict[str, Set[str]], 
    pred_map: Dict[str, Set[str]],
    eval_s1_ids: List[str]
) -> Dict[str, float]:
    """
    Compute macro-averaged F0.5 across all evaluated S1 entities,
    including breakdowns for zero-match, single-match, and multi-match entities.
    """
    scores_all = []
    scores_zero = []
    scores_single = []
    scores_multi = []
    exact_matches = 0

    for s1 in eval_s1_ids:
        true_set = gt_map.get(s1, set())
        pred_set = pred_map.get(s1, set())

        f05 = compute_entity_f05(true_set, pred_set)
        scores_all.append(f05)

        if true_set == pred_set:
            exact_matches += 1

        if len(true_set) == 0:
            scores_zero.append(f05)
        elif len(true_set) == 1:
            scores_single.append(f05)
        else:
            scores_multi.append(f05)

    return {
        'macro_f05': float(np.mean(scores_all)) if scores_all else 0.0,
        'exact_set_acc': float(exact_matches / len(eval_s1_ids)) if eval_s1_ids else 0.0,
        'zero_match_f05': float(np.mean(scores_zero)) if scores_zero else 0.0,
        'single_match_f05': float(np.mean(scores_single)) if scores_single else 0.0,
        'multi_match_f05': float(np.mean(scores_multi)) if scores_multi else 0.0,
        'total_evaluated': len(eval_s1_ids)
    }
