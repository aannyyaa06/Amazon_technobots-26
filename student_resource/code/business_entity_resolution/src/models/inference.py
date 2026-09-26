"""Phase 11: Test inference.

Full end-to-end pipeline for generating the two required submission files:
  - output/candidate_pairs.tsv
  - output/matching_results.tsv

Steps:
  1. Load normalized test sources (cached by Phase 2).
  2. Generate test candidates using all blocking strategies.
  3. Build feature matrix for test candidates.
  4. Score with the saved LightGBM model.
  5. Apply the best threshold (from Phase 9).
  6. Aggregate predictions per test S1 entity.
  7. Write output TSVs — every test S1 entity appears exactly once.
  8. Verify every match ID appears in candidate_pairs.tsv.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, OUTPUT_DIR
from src.blocking.candidate_generator import generate_candidates, load_candidates
from src.evaluation.threshold_tuning import load_best_threshold
from src.features.feature_builder import build_feature_matrix
from src.features.tfidf_features import load_tfidf_vectorizers
from src.models.lightgbm_model import ChoiceAModel
from src.models.predict import score_pairs
from src.preprocessing.run import load_normalized

OUTPUT_DIR_FINAL = OUTPUT_DIR
CANDIDATE_OUT = OUTPUT_DIR_FINAL / "candidate_pairs.tsv"
MATCHING_OUT = OUTPUT_DIR_FINAL / "matching_results.tsv"


def run_phase11(
    *,
    threshold: float | None = None,
    force_candidates: bool = False,
    force_features: bool = False,
) -> None:
    """Run full test inference and produce submission files."""
    print("=== Phase 11: Test Inference ===")
    t_start = time.perf_counter()

    # --- Load normalized test sources ---
    print("Loading normalized test sources …")
    s1 = load_normalized("test", "source1")
    s2 = load_normalized("test", "source2")
    s3 = load_normalized("test", "source3")
    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}")

    # --- Candidate generation ---
    print("Generating test candidates …")
    candidates = generate_candidates(
        "test", s1, s2, s3, force=force_candidates
    )
    print(f"  Test candidate pairs: {len(candidates):,}")

    # --- Feature matrix ---
    print("Building test feature matrix …")
    name_vec, addr_vec = load_tfidf_vectorizers()
    # Combined S2+S3 for lookup
    cand_all = pd.concat([s2, s3], ignore_index=True)

    feats = build_feature_matrix(
        candidates,
        s1,
        cand_all,
        name_vec,
        addr_vec,
        tag="test",
        force=force_features,
    )

    # --- Score ---
    print("Scoring pairs …")
    model = ChoiceAModel.load()
    scored = score_pairs(feats, model=model)

    # --- Threshold ---
    if threshold is None:
        threshold = load_best_threshold(default=0.50)
    print(f"Applying threshold: {threshold}")

    # --- Aggregate per S1 ---
    matched = scored[scored["match_proba"] >= threshold]
    predictions: dict[str, list[str]] = {}
    for s1_id, cid in zip(matched["source1_entity_id"], matched["candidate_entity_id"]):
        predictions.setdefault(str(s1_id), []).append(str(cid))

    # Deduplicate per S1 (preserve order, no S1 self-match)
    s1_ids_all = s1["entity_id"].astype(str).tolist()
    for s1_id in s1_ids_all:
        predictions.setdefault(s1_id, [])
        # Remove duplicates and any accidental S1 IDs in match list
        seen: set[str] = set()
        clean: list[str] = []
        for cid in predictions[s1_id]:
            if cid not in seen and not cid.startswith("S1-"):
                seen.add(cid)
                clean.append(cid)
        predictions[s1_id] = clean

    # --- Write candidate_pairs.tsv ---
    OUTPUT_DIR_FINAL.mkdir(parents=True, exist_ok=True)

    # Candidate pairs: one row per (S1, candidate)
    cand_rows = [
        (row["source1_entity_id"], row["candidate_entity_id"])
        for _, row in candidates.iterrows()
    ]
    cand_df = pd.DataFrame(cand_rows, columns=["source1_entity_id", "candidate_entity_id"])
    cand_df.to_csv(CANDIDATE_OUT, sep="\t", index=False)
    print(f"  Written: {CANDIDATE_OUT} ({len(cand_df):,} rows)")

    # --- Write matching_results.tsv ---
    # Every test S1 entity must appear exactly once
    result_rows = []
    candidate_set = set(
        zip(candidates["source1_entity_id"].astype(str), candidates["candidate_entity_id"].astype(str))
    )

    for s1_id in s1_ids_all:
        matched_ids = predictions.get(s1_id, [])
        # Final integrity check: every match must be in candidates
        valid_ids = [cid for cid in matched_ids if (s1_id, cid) in candidate_set]
        result_rows.append((s1_id, ",".join(valid_ids)))

    result_df = pd.DataFrame(result_rows, columns=["source1_entity_id", "matched_entity_ids"])
    result_df.to_csv(MATCHING_OUT, sep="\t", index=False)
    print(f"  Written: {MATCHING_OUT} ({len(result_df):,} rows)")

    # --- Sanity checks ---
    n_with_matches = (result_df["matched_entity_ids"] != "").sum()
    n_empty = (result_df["matched_entity_ids"] == "").sum()
    print(f"\n  S1 entities with ≥1 match: {n_with_matches:,}")
    print(f"  S1 entities with 0 matches (singletons): {n_empty:,}")
    print(f"  Expected S1 count: 1,732,544  Got: {len(result_df):,}")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Phase 11 complete in {elapsed:.1f}s")
    print(f"  ✓ candidate_pairs.tsv: {CANDIDATE_OUT}")
    print(f"  ✓ matching_results.tsv: {MATCHING_OUT}")
