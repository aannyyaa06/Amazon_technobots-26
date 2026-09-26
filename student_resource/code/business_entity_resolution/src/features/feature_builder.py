"""Phase 6: Feature matrix builder.

Joins candidate pairs with normalized source columns, calls string and
TF-IDF feature modules, and writes a flat feature Parquet file.

Feature matrix columns (17 total for Choice A)
-----------------------------------------------
(string features — 15)  see string_features.py
(TF-IDF features — 2)   see tfidf_features.py

Also propagates source1_entity_id, candidate_entity_id, and label (if
available) for training use.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.features.string_features import compute_string_features_batch
from src.features.tfidf_features import compute_tfidf_cosines

INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"

FEATURE_COLS = [
    # Name
    "name_exact",
    "name_token_jaccard",
    "name_token_overlap",
    "name_lev_sim",
    "name_partial_ratio",
    "name_token_sort_sim",
    # Address
    "addr_exact",
    "addr_token_jaccard",
    "addr_lev_sim",
    # Numeric / country / structure
    "num_shared_numeric",
    "country_same",
    "name_len_ratio",
    "addr_len_ratio",
    "name_tok_count_diff",
    "addr_tok_count_diff",
    # TF-IDF
    "name_tfidf_cosine",
    "addr_tfidf_cosine",
]


def _feature_path(tag: str) -> Path:
    return INTERMEDIATE_DIR / f"features_{tag}.parquet"


def build_feature_matrix(
    pairs_df: pd.DataFrame,
    s1_norm: pd.DataFrame,
    cand_norm: pd.DataFrame,
    name_vec: Any,
    addr_vec: Any,
    *,
    tag: str,
    force: bool = False,
) -> pd.DataFrame:
    """Build feature matrix for the given pairs.

    Parameters
    ----------
    pairs_df    : rows with source1_entity_id, candidate_entity_id [, label]
    s1_norm     : normalized S1 (entity_id, name_norm, address_norm, country_norm)
    cand_norm   : normalized S2+S3 combined
    name_vec    : fitted TF-IDF name vectoriser
    addr_vec    : fitted TF-IDF address vectoriser
    tag         : used in cache filename (e.g. "train", "val", "test")
    force       : overwrite cache

    Returns feature DataFrame with id columns + FEATURE_COLS [+ label].
    """
    cache = _feature_path(tag)
    if cache.exists() and not force:
        print(f"  Feature cache exists ({tag}): {cache}")
        return pd.read_parquet(cache)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n = len(pairs_df)
    print(f"  Building features for {n:,} pairs (tag={tag}) …")

    # Build lookup dicts for fast row joins (no DataFrame merge overhead)
    s1_dict = {
        str(eid): (str(name) if name else "", str(addr) if addr else "", str(ctry) if ctry else "")
        for eid, name, addr, ctry in zip(
            s1_norm["entity_id"], s1_norm["name_norm"],
            s1_norm["address_norm"], s1_norm["country_norm"],
        )
    }
    cand_dict = {
        str(eid): (str(name) if name else "", str(addr) if addr else "", str(ctry) if ctry else "")
        for eid, name, addr, ctry in zip(
            cand_norm["entity_id"], cand_norm["name_norm"],
            cand_norm["address_norm"], cand_norm["country_norm"],
        )
    }

    # Expand joined columns
    expanded_rows = []
    for s1_id, cid in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]):
        n1, a1, c1 = s1_dict.get(str(s1_id), ("", "", ""))
        n2, a2, c2 = cand_dict.get(str(cid), ("", "", ""))
        expanded_rows.append((n1, a1, c1, n2, a2, c2))

    joined = pd.DataFrame(
        expanded_rows,
        columns=["name_norm_s1", "addr_norm_s1", "country_norm_s1",
                 "name_norm_cand", "addr_norm_cand", "country_norm_cand"],
        index=pairs_df.index,
    )

    print("    Computing string features …")
    string_feats = compute_string_features_batch(joined)

    print("    Computing TF-IDF cosine features …")
    tfidf_feats = compute_tfidf_cosines(name_vec, addr_vec, joined)

    # Assemble output
    out = pd.concat(
        [
            pairs_df[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True),
            string_feats.reset_index(drop=True),
            tfidf_feats.reset_index(drop=True),
        ],
        axis=1,
    )

    if "label" in pairs_df.columns:
        out["label"] = pairs_df["label"].values

    # Enforce feature dtypes
    for col in FEATURE_COLS:
        if col in out.columns:
            out[col] = out[col].astype(np.float32)

    out.to_parquet(cache, index=False)
    elapsed = time.perf_counter() - t0
    print(f"  Feature matrix shape: {out.shape} | {elapsed:.1f}s → {cache}")
    assert out[FEATURE_COLS].isna().sum().sum() == 0, "NaN found in feature matrix!"
    return out
