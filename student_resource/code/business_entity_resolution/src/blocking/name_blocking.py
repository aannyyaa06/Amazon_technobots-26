"""Name-based blocking strategies.

Produces inverted indices from normalized name fields. All posting lists
are truncated at MAX_POSTING to avoid high-frequency token collisions
drowning S1 candidates from unrelated businesses.

Strategies
----------
B1 – exact normalized name
    Key: name_norm (verbatim). Catches perfect normalisation matches.
B2 – significant name tokens
    Key: each token from significant_name_tokens(). High recall for
    partial / reordered / abbreviated names.
B3 – name character trigram prefix
    Key: first 3 characters of name_norm. Tolerates prefix variants.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import pandas as pd

from src.config import MAX_POSTING
from src.preprocessing.normalize_names import significant_name_tokens


def _trigram_prefix(name_norm: str, n: int = 3) -> str:
    """Return first n characters of name_norm, or the full string if shorter."""
    return name_norm[:n] if len(name_norm) >= n else name_norm


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_name_exact_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """B1: entity_id lists keyed by exact name_norm."""
    idx: dict[str, list[str]] = defaultdict(list)
    for eid, name in zip(df["entity_id"], df["name_norm"]):
        if name:
            idx[name].append(str(eid))
    # Truncate hot keys
    return {k: v for k, v in idx.items() if len(v) <= MAX_POSTING}


def build_name_token_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """B2: entity_id lists keyed by each significant name token."""
    idx: dict[str, list[str]] = defaultdict(list)
    for eid, name in zip(df["entity_id"], df["name_norm"]):
        for tok in significant_name_tokens(str(name) if name else ""):
            idx[tok].append(str(eid))
    return {k: v for k, v in idx.items() if len(v) <= MAX_POSTING}


def build_name_trigram_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """B3: entity_id lists keyed by 3-character name prefix."""
    idx: dict[str, list[str]] = defaultdict(list)
    for eid, name in zip(df["entity_id"], df["name_norm"]):
        key = _trigram_prefix(str(name) if name else "")
        if key:
            idx[key].append(str(eid))
    return {k: v for k, v in idx.items() if len(v) <= MAX_POSTING}


# ---------------------------------------------------------------------------
# Pair generator
# ---------------------------------------------------------------------------

def name_candidate_pairs(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    *,
    source_tag: str,
) -> Iterator[tuple[str, str, str]]:
    """Yield (s1_id, candidate_id, block_strategy) tuples for name blocks.

    Parameters
    ----------
    s1_df:      normalized S1 records (columns: entity_id, name_norm)
    s2s3_df:    normalized S2 or S3 records (same columns)
    source_tag: "source2" or "source3" (used in reporting only)
    """
    # Build S2/S3 indices once
    exact_idx = build_name_exact_index(s2s3_df)
    token_idx = build_name_token_index(s2s3_df)
    trigram_idx = build_name_trigram_index(s2s3_df)

    for s1_id, s1_name in zip(s1_df["entity_id"], s1_df["name_norm"]):
        s1_name = str(s1_name) if s1_name else ""
        seen: set[str] = set()

        # B1
        for cid in exact_idx.get(s1_name, []):
            if cid not in seen:
                seen.add(cid)
                yield str(s1_id), cid, "B1_name_exact"

        # B2
        for tok in significant_name_tokens(s1_name):
            for cid in token_idx.get(tok, []):
                if cid not in seen:
                    seen.add(cid)
                    yield str(s1_id), cid, "B2_name_token"

        # B3
        prefix = _trigram_prefix(s1_name)
        if prefix:
            for cid in trigram_idx.get(prefix, []):
                if cid not in seen:
                    seen.add(cid)
                    yield str(s1_id), cid, "B3_name_trigram"
