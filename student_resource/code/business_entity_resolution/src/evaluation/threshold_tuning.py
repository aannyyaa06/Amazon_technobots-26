"""Phase 9: Threshold tuning.

Searches the threshold grid (THRESHOLDS from config) on the validation set
and selects the threshold maximising macro F0.5. Results are saved to
output/artifacts/best_threshold.json and a full sweep report is written to
output/reports/threshold_sweep.txt.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ARTIFACTS_DIR, THRESHOLDS, TRAIN_GROUND_TRUTH
from src.evaluation.validation import evaluate_on_val, _load_gt_dict

REPORTS_DIR = ARTIFACTS_DIR.parent / "reports"
BEST_THRESHOLD_PATH = ARTIFACTS_DIR / "best_threshold.json"


def tune_threshold(
    val_scored: pd.DataFrame,
    *,
    gt_path: Path = TRAIN_GROUND_TRUTH,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Search thresholds and return the best by macro F0.5.

    Parameters
    ----------
    val_scored : validation pairs with 'source1_entity_id', 'candidate_entity_id',
                 'match_proba' columns.
    gt_path    : ground truth path.
    thresholds : list of float thresholds to try. Defaults to config.THRESHOLDS.

    Returns a dict with best_threshold and full sweep results.
    """
    print("=== Phase 9: Threshold Tuning ===")
    if thresholds is None:
        thresholds = THRESHOLDS

    t0 = time.perf_counter()
    gt = _load_gt_dict(gt_path)
    val_s1_ids = set(val_scored["source1_entity_id"].astype(str).unique())

    sweep: list[dict[str, Any]] = []
    best_f05 = -1.0
    best_threshold = thresholds[0]

    header = f"{'threshold':>10}  {'macro_f05':>10}  {'precision':>10}  {'recall':>10}"
    print(header)
    print("-" * len(header))

    for thr in thresholds:
        result = evaluate_on_val(val_scored, thr, gt, val_s1_ids)
        f05 = result["macro_f05"]
        prec = result["macro_precision"]
        rec = result["macro_recall"]
        print(f"  {thr:>8.2f}   {f05:>10.4f}   {prec:>10.4f}   {rec:>10.4f}")
        sweep.append({"threshold": thr, "macro_f05": f05, "macro_precision": prec, "macro_recall": rec})
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thr

    print(f"\n  Best threshold: {best_threshold} → macro F0.5 = {best_f05:.4f}")

    payload = {
        "best_threshold": best_threshold,
        "best_macro_f05": best_f05,
        "sweep": sweep,
        "elapsed_seconds": time.perf_counter() - t0,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_THRESHOLD_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Saved: {BEST_THRESHOLD_PATH}")

    # Human-readable sweep report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["PHASE 9 — THRESHOLD SWEEP", header, "-" * len(header)]
    for row in sweep:
        lines.append(
            f"  {row['threshold']:>8.2f}   {row['macro_f05']:>10.4f}"
            f"   {row['macro_precision']:>10.4f}   {row['macro_recall']:>10.4f}"
        )
    lines.append(f"\nBest threshold: {best_threshold}  →  macro F0.5 = {best_f05:.4f}")
    (REPORTS_DIR / "threshold_sweep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return payload


def load_best_threshold(default: float = 0.50) -> float:
    """Load the best threshold from disk, or return default if not yet tuned."""
    if not BEST_THRESHOLD_PATH.exists():
        print(f"  Warning: best_threshold.json not found. Using default {default}.")
        return default
    payload = json.loads(BEST_THRESHOLD_PATH.read_text(encoding="utf-8"))
    return float(payload["best_threshold"])
