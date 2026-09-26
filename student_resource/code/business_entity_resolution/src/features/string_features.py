"""Phase 6a: String-based similarity features using rapidfuzz.

All features are floats in [0, 1] unless documented otherwise.

Feature names
-------------
name_exact          : int   1 if name_norm is identical
name_token_jaccard  : float token Jaccard similarity
name_token_overlap  : float overlap coefficient (intersection / min)
name_lev_sim        : float rapidfuzz WRatio / 100
name_partial_ratio  : float rapidfuzz partial_ratio / 100
name_token_sort_sim : float rapidfuzz token_sort_ratio / 100
addr_exact          : int   1 if address_norm is identical
addr_token_jaccard  : float token Jaccard on address
addr_lev_sim        : float rapidfuzz WRatio on address / 100
num_shared_numeric  : int   count of shared numeric tokens (≥3 digits)
country_same        : int   1 if country_norm is identical (and non-empty)
name_len_ratio      : float min/max char-length ratio (names)
addr_len_ratio      : float min/max char-length ratio (addresses)
name_tok_count_diff : int   |#name_tokens_s1 − #name_tokens_cand|
addr_tok_count_diff : int   |#addr_tokens_s1 − #addr_tokens_cand|
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.preprocessing.normalize_addresses import numeric_tokens
from src.preprocessing.normalize_names import name_tokens


def _safe_str(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _len_ratio(s1: str, s2: str) -> float:
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


def _shared_numeric(n1: str, a1: str, n2: str, a2: str) -> int:
    nums1 = {n for n in numeric_tokens(n1, a1) if len(n) >= 3}
    nums2 = {n for n in numeric_tokens(n2, a2) if len(n) >= 3}
    return len(nums1 & nums2)


def compute_string_features(
    name1: str,
    addr1: str,
    country1: str,
    name2: str,
    addr2: str,
    country2: str,
) -> dict[str, float | int]:
    """Compute all string features for a single candidate pair."""
    n1, n2 = _safe_str(name1), _safe_str(name2)
    a1, a2 = _safe_str(addr1), _safe_str(addr2)
    c1, c2 = _safe_str(country1), _safe_str(country2)

    n1_toks = set(name_tokens(n1)) if n1 else set()
    n2_toks = set(name_tokens(n2)) if n2 else set()
    a1_toks = set(a1.split()) if a1 else set()
    a2_toks = set(a2.split()) if a2 else set()

    return {
        # Name features
        "name_exact": int(n1 == n2 and bool(n1)),
        "name_token_jaccard": _jaccard(n1_toks, n2_toks),
        "name_token_overlap": _overlap(n1_toks, n2_toks),
        "name_lev_sim": fuzz.WRatio(n1, n2) / 100.0 if n1 or n2 else 0.0,
        "name_partial_ratio": fuzz.partial_ratio(n1, n2) / 100.0 if n1 or n2 else 0.0,
        "name_token_sort_sim": fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 or n2 else 0.0,
        # Address features
        "addr_exact": int(a1 == a2 and bool(a1)),
        "addr_token_jaccard": _jaccard(a1_toks, a2_toks),
        "addr_lev_sim": fuzz.WRatio(a1, a2) / 100.0 if a1 or a2 else 0.0,
        # Numeric
        "num_shared_numeric": _shared_numeric(n1, a1, n2, a2),
        # Country
        "country_same": int(c1 == c2 and bool(c1)),
        # Length / structure
        "name_len_ratio": _len_ratio(n1, n2),
        "addr_len_ratio": _len_ratio(a1, a2),
        "name_tok_count_diff": abs(len(n1_toks) - len(n2_toks)),
        "addr_tok_count_diff": abs(len(a1_toks) - len(a2_toks)),
    }


def compute_string_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute string features for a DataFrame with columns:
    name_norm_s1, addr_norm_s1, country_norm_s1,
    name_norm_cand, addr_norm_cand, country_norm_cand

    Returns a DataFrame of feature columns aligned to df.
    """
    records = df.apply(
        lambda r: compute_string_features(
            r["name_norm_s1"],
            r["addr_norm_s1"],
            r["country_norm_s1"],
            r["name_norm_cand"],
            r["addr_norm_cand"],
            r["country_norm_cand"],
        ),
        axis=1,
        result_type="expand",
    )
    return records
