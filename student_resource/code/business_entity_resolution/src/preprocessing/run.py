"""Phase 2: Apply normalization to all sources and cache results.

For each train and test source, this phase:
1. Loads records in chunks.
2. Applies add_normalized_columns() (name_norm, address_norm, country_norm).
3. Saves a slim Parquet cache containing only (entity_id, name_norm,
   address_norm, country_norm) to output/artifacts/normalized/.

Original TSVs are never written to. Parquet caches are re-used downstream by
blocking, feature engineering, and inference.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    TEST_SOURCE_FILES,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE_FILES,
)
from src.data.loader import iter_source_chunks, load_ground_truth
from src.preprocessing.normalize import add_normalized_columns

NORM_DIR = ARTIFACTS_DIR / "normalized"
NORM_COLUMNS = ["entity_id", "name_norm", "address_norm", "country_norm"]


def _cache_path(split: str, key: str) -> Path:
    return NORM_DIR / f"{split}_{key}_norm.parquet"


def normalize_source(
    path: Path,
    split: str,
    source_key: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Normalize one source TSV and write a Parquet cache.

    Returns a stats dict.
    """
    out_path = _cache_path(split, source_key)
    if out_path.exists() and not force:
        existing = pd.read_parquet(out_path, columns=["entity_id"])
        print(f"  [{split}/{source_key}] Cache already exists ({len(existing):,} rows). Skipping.")
        return {"cached": True, "n_rows": len(existing), "path": str(out_path)}

    NORM_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    parts: list[pd.DataFrame] = []
    n_rows = 0

    for chunk in iter_source_chunks(path):
        norm = add_normalized_columns(chunk)[NORM_COLUMNS]
        parts.append(norm)
        n_rows += len(chunk)

    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out_path, index=False)
    elapsed = time.perf_counter() - t0

    # Print sample rows for human inspection
    sample = df.head(3).to_dict(orient="records")
    print(f"  [{split}/{source_key}] {n_rows:,} rows → {out_path} ({elapsed:.1f}s)")
    for row in sample:
        print(f"    {row}")

    return {"cached": False, "n_rows": n_rows, "elapsed": elapsed, "path": str(out_path)}


def load_normalized(split: str, source_key: str) -> pd.DataFrame:
    """Load cached normalized Parquet for a source.

    Raises FileNotFoundError if cache does not exist (run Phase 2 first).
    """
    path = _cache_path(split, source_key)
    if not path.exists():
        raise FileNotFoundError(
            f"Normalized cache not found: {path}. Run Phase 2 first."
        )
    return pd.read_parquet(path)


def run_phase2(*, force: bool = False) -> dict[str, Any]:
    """Run normalization for all train and test sources."""
    t0 = time.perf_counter()
    results: dict[str, Any] = {}

    print("=== Phase 2: Normalization ===")
    print("Train sources:")
    for key, path in TRAIN_SOURCE_FILES.items():
        results[f"train_{key}"] = normalize_source(path, "train", key, force=force)

    print("Test sources:")
    for key, path in TEST_SOURCE_FILES.items():
        results[f"test_{key}"] = normalize_source(path, "test", key, force=force)

    # Smoke-test: verify a train S1 row looks reasonable
    print("\nSmoke-test (train source1 sample):")
    s1 = load_normalized("train", "source1").head(5)
    print(s1.to_string(index=False))

    elapsed = time.perf_counter() - t0
    print(f"\nPhase 2 complete in {elapsed:.1f}s")
    results["elapsed_seconds"] = elapsed
    return results
