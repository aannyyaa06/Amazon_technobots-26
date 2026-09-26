"""Evaluation metrics for business entity resolution.

Primary metric: macro-averaged F0.5 at the Source 1 entity level.
F0.5 weights precision twice as much as recall.

  F0.5 = (1 + 0.5²) × P × R / (0.5² × P + R)
        = 1.25 × P × R / (0.25 × P + R)

A true singleton (GT: empty set) scores:
  - Predicted empty → precision=1, recall=1, F0.5=1
  - Predicted non-empty → precision=0, recall=1 (vacuous), F0.5=0
"""

from __future__ import annotations

from typing import Any


def compute_f05(precision: float, recall: float) -> float:
    """F0.5 for a single entity.

    Both inputs must be in [0, 1]. Returns 0.0 if both are 0.
    """
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return 1.25 * precision * recall / denom


def _per_entity_metrics(
    predicted: set[str],
    ground_truth: set[str],
) -> tuple[float, float, float]:
    """Compute precision, recall, F0.5 for one S1 entity.

    Handles singleton case: if GT is empty, correct iff predicted is empty.
    """
    if not ground_truth:
        # Singleton entity
        if not predicted:
            return 1.0, 1.0, 1.0
        else:
            # False merge
            return 0.0, 1.0, 0.0  # precision=0, recall=vacuous=1, F0.5=0

    tp = len(predicted & ground_truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth)
    f05 = compute_f05(precision, recall)
    return precision, recall, f05


def entity_level_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> dict[str, Any]:
    """Compute macro-averaged F0.5 over all S1 entities.

    Parameters
    ----------
    predictions  : {s1_id: set of predicted match IDs}
    ground_truth : {s1_id: set of true match IDs}

    Both dicts must have the same keys (all S1 entity IDs).

    Returns a summary dict with macro F0.5 and per-entity stats.
    """
    if set(predictions.keys()) != set(ground_truth.keys()):
        missing = set(ground_truth.keys()) - set(predictions.keys())
        extra = set(predictions.keys()) - set(ground_truth.keys())
        if missing:
            # Entities with no prediction → treated as empty set
            for k in missing:
                predictions[k] = set()
        if extra:
            raise ValueError(f"Predictions contain S1 IDs not in ground truth: {sorted(extra)[:5]}")

    f05_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    n_true_singletons_correct = 0
    n_true_singletons_wrong = 0
    n_non_singletons = 0

    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        gt = ground_truth[s1_id]
        p, r, f = _per_entity_metrics(pred, gt)
        f05_scores.append(f)
        precision_scores.append(p)
        recall_scores.append(r)
        if not gt:
            if not pred:
                n_true_singletons_correct += 1
            else:
                n_true_singletons_wrong += 1
        else:
            n_non_singletons += 1

    n = len(f05_scores)
    return {
        "macro_f05": sum(f05_scores) / n if n else 0.0,
        "macro_precision": sum(precision_scores) / n if n else 0.0,
        "macro_recall": sum(recall_scores) / n if n else 0.0,
        "n_entities": n,
        "n_true_singletons": n_true_singletons_correct + n_true_singletons_wrong,
        "n_true_singletons_correctly_empty": n_true_singletons_correct,
        "n_true_singletons_incorrectly_matched": n_true_singletons_wrong,
        "n_non_singleton_entities": n_non_singletons,
    }
