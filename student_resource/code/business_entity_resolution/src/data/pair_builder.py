"""Phase 5: Build training and validation pair datasets.

Converts ground truth into labelled (positive + hard-negative) pairwise
records split by Source 1 entity, ensuring no leakage between train/val.

Output Parquet schema
---------------------
source1_entity_id : str
candidate_entity_id : str
label : int8  (1 = match, 0 = non-match)
split : str   ("train" or "val")

Negatives are in-block hard negatives — candidates that appeared in the
blocking step but are NOT listed as true matches. This produces harder
negatives than random unrelated pairs and better calibrates the model.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, NEGATIVES_PER_ENTITY, SEED, TRAIN_GROUND_TRUTH, VAL_FRACTION
from src.data.loader import iter_ground_truth_chunks

INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
PAIR_COLS = ["source1_entity_id", "candidate_entity_id", "label"]


def _build_gt_dict(gt_path: Path) -> dict[str, set[str]]:
    """Load ground truth into {s1_id: set(matched_ids)}."""
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


def build_pairs(
    candidates_df: pd.DataFrame,
    *,
    gt_path: Path = TRAIN_GROUND_TRUTH,
    val_fraction: float = VAL_FRACTION,
    negatives_per_entity: int = NEGATIVES_PER_ENTITY,
    seed: int = SEED,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build labelled train/val pairs from candidates + ground truth.

    Parameters
    ----------
    candidates_df      : candidate pairs (source1_entity_id, candidate_entity_id, …)
    gt_path            : ground-truth TSV path
    val_fraction       : fraction of S1 entities reserved for validation
    negatives_per_entity : max hard negatives to sample per S1 entity
    seed               : random seed for reproducibility
    force              : ignore cache

    Returns (train_df, val_df) with columns [source1_entity_id, candidate_entity_id, label].
    """
    train_path = INTERMEDIATE_DIR / "train_pairs.parquet"
    val_path = INTERMEDIATE_DIR / "val_pairs.parquet"

    if train_path.exists() and val_path.exists() and not force:
        print("  Pair cache exists. Loading.")
        return pd.read_parquet(train_path), pd.read_parquet(val_path)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = random.Random(seed)

    print("  Loading ground truth …")
    gt = _build_gt_dict(gt_path)

    # S1-level train/val split
    all_s1 = sorted(gt.keys())
    rng.shuffle(all_s1)
    n_val = max(1, int(len(all_s1) * val_fraction))
    val_s1 = set(all_s1[:n_val])
    train_s1 = set(all_s1[n_val:])
    print(f"  S1 split — train: {len(train_s1):,} | val: {len(val_s1):,}")

    # Group candidates by S1 entity
    cand_by_s1: dict[str, list[str]] = {}
    for s1_id, cid in zip(
        candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]
    ):
        cand_by_s1.setdefault(str(s1_id), []).append(str(cid))

    train_rows: list[tuple[str, str, int]] = []
    val_rows: list[tuple[str, str, int]] = []

    for s1_id, pos_set in gt.items():
        cands = cand_by_s1.get(s1_id, [])
        cand_set = set(cands)

        # Positives (only those recovered in candidates)
        positives = [cid for cid in pos_set if cid in cand_set]
        # Hard negatives: candidates not in ground truth
        hard_negs = [cid for cid in cands if cid not in pos_set]
        hard_negs_sample = rng.sample(hard_negs, min(len(hard_negs), negatives_per_entity))

        pairs = [(s1_id, cid, 1) for cid in positives] + \
                [(s1_id, cid, 0) for cid in hard_negs_sample]

        if s1_id in val_s1:
            val_rows.extend(pairs)
        else:
            train_rows.extend(pairs)

    train_df = pd.DataFrame(train_rows, columns=PAIR_COLS)
    train_df["label"] = train_df["label"].astype("int8")

    val_df = pd.DataFrame(val_rows, columns=PAIR_COLS)
    val_df["label"] = val_df["label"].astype("int8")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    elapsed = time.perf_counter() - t0
    pos_train = int(train_df["label"].sum())
    neg_train = len(train_df) - pos_train
    pos_val = int(val_df["label"].sum())
    neg_val = len(val_df) - pos_val

    print(f"  Train pairs: {len(train_df):,}  (pos={pos_train:,}, neg={neg_train:,})")
    print(f"  Val pairs:   {len(val_df):,}  (pos={pos_val:,}, neg={neg_val:,})")
    print(f"  Positive rate train: {pos_train/max(1,len(train_df)):.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return train_df, val_df


def run_phase5(candidates_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 5 entry point."""
    print("=== Phase 5: Training Pair Construction ===")
    return build_pairs(candidates_df)
