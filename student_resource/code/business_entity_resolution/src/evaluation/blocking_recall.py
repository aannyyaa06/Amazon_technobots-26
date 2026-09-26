"""Phase 4: Blocking recall evaluation.

Measures what fraction of ground-truth positive pairs are recovered by the
candidate generator. This is the hard ceiling for the downstream matcher.

Outputs a report to output/reports/blocking_recall.txt and returns a dict
with per-strategy contributions.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ARTIFACTS_DIR, TRAIN_GROUND_TRUTH
from src.data.loader import iter_ground_truth_chunks

REPORTS_DIR = ARTIFACTS_DIR.parent / "reports"


def load_positive_pairs(gt_path: Path = TRAIN_GROUND_TRUTH) -> pd.DataFrame:
    """Convert ground truth to (source1_entity_id, candidate_entity_id) rows."""
    rows: list[tuple[str, str]] = []
    for chunk in iter_ground_truth_chunks(gt_path):
        for s1_id, matched_raw in zip(
            chunk["source1_entity_id"], chunk["matched_entity_ids"]
        ):
            if not matched_raw or (isinstance(matched_raw, float)):
                continue
            for mid in str(matched_raw).split(","):
                mid = mid.strip()
                if mid:
                    rows.append((str(s1_id), mid))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def evaluate_blocking_recall(
    candidates_df: pd.DataFrame,
    positives_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute blocking recall and per-strategy statistics.

    Parameters
    ----------
    candidates_df : output of generate_candidates()
    positives_df  : output of load_positive_pairs()

    Returns a stats dict with overall recall and per-strategy pair counts.
    """
    # Build a set of candidate (s1, cand) tuples for O(1) lookup
    cand_set: set[tuple[str, str]] = set(
        zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"])
    )

    n_positives = len(positives_df)
    recovered = 0
    missed_examples: list[tuple[str, str]] = []

    for s1_id, cid in zip(
        positives_df["source1_entity_id"], positives_df["candidate_entity_id"]
    ):
        if (s1_id, cid) in cand_set:
            recovered += 1
        elif len(missed_examples) < 10:
            missed_examples.append((s1_id, cid))

    recall = recovered / n_positives if n_positives else 0.0
    missed = n_positives - recovered

    # Per-strategy pair counts
    strategy_counts: Counter[str] = Counter()
    for strats in candidates_df["block_strategies"]:
        for s in str(strats).split(","):
            strategy_counts[s.strip()] += 1

    stats = {
        "n_positive_pairs": n_positives,
        "n_recovered": recovered,
        "n_missed": missed,
        "blocking_recall": round(recall, 6),
        "blocking_recall_pct": round(recall * 100, 2),
        "n_candidate_pairs": len(candidates_df),
        "n_unique_s1_with_candidates": candidates_df["source1_entity_id"].nunique(),
        "per_strategy_counts": dict(strategy_counts),
        "missed_examples": missed_examples,
    }
    return stats


def run_phase4(candidates_df: pd.DataFrame) -> dict[str, Any]:
    """Run Phase 4: blocking recall evaluation."""
    t0 = time.perf_counter()
    print("=== Phase 4: Blocking Recall Evaluation ===")
    print("Loading ground-truth positive pairs …")
    positives = load_positive_pairs()
    print(f"  Total positive pairs: {len(positives):,}")

    stats = evaluate_blocking_recall(candidates_df, positives)

    # Report
    lines = [
        "PHASE 4 — BLOCKING RECALL",
        f"  positive pairs:       {stats['n_positive_pairs']:>10,}",
        f"  recovered in cands:   {stats['n_recovered']:>10,}",
        f"  missed:               {stats['n_missed']:>10,}",
        f"  blocking recall:      {stats['blocking_recall_pct']:>9.2f}%",
        f"  total candidate pairs:{stats['n_candidate_pairs']:>10,}",
        "",
        "Per-strategy pair counts:",
    ]
    for strat, cnt in sorted(stats["per_strategy_counts"].items()):
        lines.append(f"  {strat:<30} {cnt:>10,}")
    lines.append("")
    if stats["missed_examples"]:
        lines.append("Sample missed pairs (s1_id, cand_id):")
        for pair in stats["missed_examples"]:
            lines.append(f"  {pair}")

    report = "\n".join(lines) + "\n"
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "blocking_recall.txt").write_text(report, encoding="utf-8")
    print(f"  Report saved to {REPORTS_DIR / 'blocking_recall.txt'}")

    # Gate
    if stats["blocking_recall"] < 0.80:
        print(
            f"  WARNING: blocking recall {stats['blocking_recall_pct']:.1f}% < 80%. "
            "Consider adding more blocking strategies before proceeding."
        )
    else:
        print(f"  ✓ Blocking recall {stats['blocking_recall_pct']:.1f}% — acceptable.")

    stats["elapsed_seconds"] = time.perf_counter() - t0
    return stats
