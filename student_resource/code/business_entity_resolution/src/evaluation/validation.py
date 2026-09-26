"""Phase 8 + 10: Entity-level validation and error analysis.

Given scored candidate pairs (with 'match_proba') for the validation split,
applies a threshold, aggregates per S1 entity, computes entity-level F0.5,
and produces an error analysis report.

Key outputs
-----------
- Printed + saved macro F0.5 on the validation set
- output/reports/error_analysis.txt
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ARTIFACTS_DIR, TRAIN_GROUND_TRUTH
from src.data.loader import iter_ground_truth_chunks
from src.evaluation.metrics import entity_level_f05

REPORTS_DIR = ARTIFACTS_DIR.parent / "reports"


def _load_gt_dict(gt_path: Path = TRAIN_GROUND_TRUTH) -> dict[str, set[str]]:
    gt: dict[str, set[str]] = {}
    for chunk in iter_ground_truth_chunks(gt_path):
        for s1_id, matched_raw in zip(
            chunk["source1_entity_id"], chunk["matched_entity_ids"]
        ):
            s1_id = str(s1_id)
            if isinstance(matched_raw, float) or not matched_raw:
                gt.setdefault(s1_id, set())
            else:
                ids = {tok.strip() for tok in str(matched_raw).split(",") if tok.strip()}
                gt[s1_id] = gt.get(s1_id, set()) | ids
    return gt


def aggregate_predictions(
    scored_df: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, set[str]]:
    """Apply threshold and aggregate matched IDs per S1 entity.

    Parameters
    ----------
    scored_df : DataFrame with source1_entity_id, candidate_entity_id, match_proba
    threshold : accept pairs with match_proba >= threshold

    Returns {s1_id: set(matched_ids)}
    """
    matched = scored_df[scored_df["match_proba"] >= threshold]
    predictions: dict[str, set[str]] = {}
    for s1_id, cid in zip(matched["source1_entity_id"], matched["candidate_entity_id"]):
        predictions.setdefault(str(s1_id), set()).add(str(cid))
    return predictions


def evaluate_on_val(
    val_scored: pd.DataFrame,
    threshold: float,
    gt: dict[str, set[str]],
    val_s1_ids: set[str],
) -> dict[str, Any]:
    """Compute entity-level metrics for the validation split at a given threshold."""
    predictions = aggregate_predictions(val_scored, threshold=threshold)
    # Ensure every val S1 appears (even if no candidates passed threshold)
    for s1_id in val_s1_ids:
        predictions.setdefault(s1_id, set())
    val_gt = {k: gt.get(k, set()) for k in val_s1_ids}
    return entity_level_f05(predictions, val_gt)


def run_phase8(
    val_scored: pd.DataFrame,
    threshold: float,
    *,
    gt_path: Path = TRAIN_GROUND_TRUTH,
) -> dict[str, Any]:
    """Phase 8 entry point: evaluate model on validation set at given threshold."""
    print(f"=== Phase 8: Entity-Level Validation (threshold={threshold}) ===")
    gt = _load_gt_dict(gt_path)
    val_s1_ids = set(val_scored["source1_entity_id"].astype(str).unique())
    result = evaluate_on_val(val_scored, threshold, gt, val_s1_ids)

    print(f"  macro F0.5:      {result['macro_f05']:.4f}")
    print(f"  macro precision: {result['macro_precision']:.4f}")
    print(f"  macro recall:    {result['macro_recall']:.4f}")
    print(f"  n_entities:      {result['n_entities']:,}")
    print(f"  singleton correct/wrong: {result['n_true_singletons_correctly_empty']} / {result['n_true_singletons_incorrectly_matched']}")
    return result


def run_phase10(
    val_scored: pd.DataFrame,
    threshold: float,
    *,
    gt_path: Path = TRAIN_GROUND_TRUTH,
    top_n_fp: int = 20,
    top_n_fn: int = 20,
) -> str:
    """Phase 10: Error analysis report.

    Identifies false positives (predicted match, no ground truth), false
    negatives (in candidates but below threshold despite being true match),
    and singleton false merges.

    Returns the report as a string and saves it to reports/.
    """
    print("=== Phase 10: Error Analysis ===")
    t0 = time.perf_counter()

    gt = _load_gt_dict(gt_path)
    val_s1_ids = set(val_scored["source1_entity_id"].astype(str).unique())

    predictions = aggregate_predictions(val_scored, threshold=threshold)
    for s1_id in val_s1_ids:
        predictions.setdefault(s1_id, set())

    lines = [
        "PHASE 10 — ERROR ANALYSIS",
        f"Threshold: {threshold}",
        "",
    ]

    # False positives (predicted but not in GT)
    fp_examples: list[str] = []
    fn_examples: list[str] = []
    singleton_fp: list[str] = []

    for s1_id in val_s1_ids:
        pred = predictions.get(s1_id, set())
        truth = gt.get(s1_id, set())
        fp = pred - truth
        fn = truth - pred

        if not truth and pred:
            singleton_fp.append(f"  {s1_id} → falsely matched to {sorted(pred)[:3]}")

        for cid in fp:
            if len(fp_examples) < top_n_fp:
                row = val_scored[
                    (val_scored["source1_entity_id"] == s1_id)
                    & (val_scored["candidate_entity_id"] == cid)
                ]
                proba = float(row["match_proba"].values[0]) if len(row) else -1.0
                fp_examples.append(f"  {s1_id} → {cid}  (proba={proba:.3f})")

        for cid in fn:
            if len(fn_examples) < top_n_fn:
                row = val_scored[
                    (val_scored["source1_entity_id"] == s1_id)
                    & (val_scored["candidate_entity_id"] == cid)
                ]
                if len(row):
                    proba = float(row["match_proba"].values[0])
                    fn_examples.append(f"  {s1_id} → {cid}  (proba={proba:.3f})")
                else:
                    fn_examples.append(f"  {s1_id} → {cid}  (NOT IN CANDIDATES — blocking miss)")

    lines += [f"Sample False Positives (top {top_n_fp}):"]
    lines += fp_examples or ["  (none)"]
    lines += ["", f"Sample False Negatives (top {top_n_fn}):"]
    lines += fn_examples or ["  (none)"]
    lines += ["", "Singleton False Merges (predicted match on true singleton):"]
    lines += singleton_fp[:20] or ["  (none)"]

    report = "\n".join(lines) + "\n"
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "error_analysis.txt").write_text(report, encoding="utf-8")
    print(f"  Report saved: {REPORTS_DIR / 'error_analysis.txt'}")
    print(f"  Phase 10 complete in {time.perf_counter()-t0:.1f}s")
    return report
