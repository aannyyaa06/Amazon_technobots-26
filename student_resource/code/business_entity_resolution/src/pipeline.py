"""End-to-end pipeline entry point — Choice A.

Usage
-----
From code/business_entity_resolution/:

    python -m src.pipeline --phase 1     # Data exploration (read-only)
    python -m src.pipeline --phase 2     # Normalization + Parquet cache
    python -m src.pipeline --phase 3     # Blocking / candidate generation
    python -m src.pipeline --phase 4     # Blocking recall evaluation
    python -m src.pipeline --phase 5     # Training pair construction
    python -m src.pipeline --phase 6     # Feature engineering
    python -m src.pipeline --phase 7     # LightGBM training
    python -m src.pipeline --phase 8     # Entity-level F0.5 validation
    python -m src.pipeline --phase 9     # Threshold tuning
    python -m src.pipeline --phase 10    # Error analysis
    python -m src.pipeline --phase 11    # Test inference → TSV outputs
    python -m src.pipeline --phase 12    # Submission validation
    python -m src.pipeline --phase all   # Run phases 2–12 sequentially

Flags
-----
--force   : Ignore caches and recompute from scratch (Phases 2–11).
--threshold FLOAT : Override the threshold for Phase 8 / 10 / 11
                    (default: load from best_threshold.json or 0.50).
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_train_norm():
    """Load all three normalized train sources."""
    from src.preprocessing.run import load_normalized
    print("Loading normalized train sources …")
    s1 = load_normalized("train", "source1")
    s2 = load_normalized("train", "source2")
    s3 = load_normalized("train", "source3")
    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}")
    return s1, s2, s3


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase1() -> int:
    from src.data.explore import run_phase1 as _run
    _run()
    return 0


def run_phase2(force: bool = False) -> int:
    from src.preprocessing.run import run_phase2 as _run
    _run(force=force)
    return 0


def run_phase3(force: bool = False) -> int:
    from src.blocking.candidate_generator import generate_candidates
    s1, s2, s3 = _get_train_norm()
    generate_candidates("train", s1, s2, s3, force=force)
    return 0


def run_phase4() -> int:
    from src.blocking.candidate_generator import load_candidates
    from src.evaluation.blocking_recall import run_phase4 as _run
    candidates = load_candidates("train")
    _run(candidates)
    return 0


def run_phase5(force: bool = False) -> int:
    from src.blocking.candidate_generator import load_candidates
    from src.data.pair_builder import run_phase5 as _run
    candidates = load_candidates("train")
    _run(candidates)
    return 0


def run_phase6(force: bool = False) -> int:
    """Build feature matrices for train and val splits."""
    import pandas as pd
    from src.blocking.candidate_generator import load_candidates
    from src.config import ARTIFACTS_DIR
    from src.features.feature_builder import build_feature_matrix, FEATURE_COLS
    from src.features.tfidf_features import fit_tfidf_vectorizers

    INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
    s1, s2, s3 = _get_train_norm()
    candidates = load_candidates("train")

    # Load train/val pair splits
    train_pairs = pd.read_parquet(INTERMEDIATE_DIR / "train_pairs.parquet")
    val_pairs = pd.read_parquet(INTERMEDIATE_DIR / "val_pairs.parquet")

    # Fit TF-IDF on all training text (S1+S2+S3 name + address)
    print("Preparing TF-IDF corpora …")
    all_norm = pd.concat([s1, s2, s3], ignore_index=True)
    name_vec, addr_vec = fit_tfidf_vectorizers(
        all_norm["name_norm"], all_norm["address_norm"], force=force
    )

    cand_all = pd.concat([s2, s3], ignore_index=True)

    build_feature_matrix(
        train_pairs, s1, cand_all, name_vec, addr_vec, tag="train", force=force
    )
    build_feature_matrix(
        val_pairs, s1, cand_all, name_vec, addr_vec, tag="val", force=force
    )
    return 0


def run_phase7(force: bool = False) -> int:
    import pandas as pd
    from src.config import ARTIFACTS_DIR
    from src.models.train import run_phase7 as _run

    INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
    train_feats = pd.read_parquet(INTERMEDIATE_DIR / "features_train.parquet")
    val_feats = pd.read_parquet(INTERMEDIATE_DIR / "features_val.parquet")
    _run(train_feats, val_feats, force=force)
    return 0


def run_phase8(threshold: float | None) -> int:
    import pandas as pd
    from src.config import ARTIFACTS_DIR
    from src.evaluation.threshold_tuning import load_best_threshold
    from src.evaluation.validation import run_phase8 as _run
    from src.models.predict import score_pairs

    INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
    val_feats = pd.read_parquet(INTERMEDIATE_DIR / "features_val.parquet")
    val_scored = score_pairs(val_feats)
    thr = threshold if threshold is not None else load_best_threshold(0.50)
    _run(val_scored, thr)
    return 0


def run_phase9() -> int:
    import pandas as pd
    from src.config import ARTIFACTS_DIR
    from src.evaluation.threshold_tuning import tune_threshold
    from src.models.predict import score_pairs

    INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
    val_feats = pd.read_parquet(INTERMEDIATE_DIR / "features_val.parquet")
    val_scored = score_pairs(val_feats)
    tune_threshold(val_scored)
    return 0


def run_phase10(threshold: float | None) -> int:
    import pandas as pd
    from src.config import ARTIFACTS_DIR
    from src.evaluation.threshold_tuning import load_best_threshold
    from src.evaluation.validation import run_phase10 as _run
    from src.models.predict import score_pairs

    INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
    val_feats = pd.read_parquet(INTERMEDIATE_DIR / "features_val.parquet")
    val_scored = score_pairs(val_feats)
    thr = threshold if threshold is not None else load_best_threshold(0.50)
    _run(val_scored, thr)
    return 0


def run_phase11(threshold: float | None, force: bool = False) -> int:
    from src.models.inference import run_phase11 as _run
    _run(threshold=threshold, force_candidates=force, force_features=force)
    return 0


def run_phase12() -> int:
    """Run utils/validate_submission.py as a subprocess."""
    import subprocess
    from src.config import OUTPUT_DIR, TEST_DIR

    result = subprocess.run(
        [
            sys.executable,
            "utils/validate_submission.py",
            "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
            "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
            "--test-dir", str(TEST_DIR),
        ],
        capture_output=False,
    )
    return result.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Business entity resolution pipeline (Choice A)."
    )
    parser.add_argument(
        "--phase",
        default="1",
        help="Pipeline phase to run: 1–12 or 'all' (runs 2–12 sequentially).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore caches and recompute from scratch.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override match threshold for Phases 8, 10, 11.",
    )
    args = parser.parse_args(argv)

    phase = args.phase.strip().lower()
    force = args.force
    threshold = args.threshold

    PHASE_MAP = {
        "1": lambda: run_phase1(),
        "2": lambda: run_phase2(force),
        "3": lambda: run_phase3(force),
        "4": lambda: run_phase4(),
        "5": lambda: run_phase5(force),
        "6": lambda: run_phase6(force),
        "7": lambda: run_phase7(force),
        "8": lambda: run_phase8(threshold),
        "9": lambda: run_phase9(),
        "10": lambda: run_phase10(threshold),
        "11": lambda: run_phase11(threshold, force),
        "12": lambda: run_phase12(),
    }

    if phase == "all":
        for p in [str(i) for i in range(2, 13)]:
            print(f"\n{'='*60}")
            print(f"RUNNING PHASE {p}")
            print(f"{'='*60}\n")
            rc = PHASE_MAP[p]()
            if rc != 0:
                print(f"Phase {p} failed with exit code {rc}.", file=sys.stderr)
                return rc
        return 0

    if phase not in PHASE_MAP:
        print(f"Unknown phase: {args.phase}. Choose 1–12 or 'all'.", file=sys.stderr)
        return 1

    return PHASE_MAP[phase]()


if __name__ == "__main__":
    sys.exit(main())
