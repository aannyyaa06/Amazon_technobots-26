"""Address-based blocking strategies.

Strategies
----------
B4 – significant address tokens
    Key: each token from significant_address_tokens(). Helps when the name
    has noisy variants but the address is stable.
B5 – shared numeric tokens
    Key: numeric tokens (building/plot/unit/PIN numbers). Very precise
    for records that share a specific number.
B6 – country + first significant name token (combo block)
    Key: country_norm + "|" + first sig. name token. Reduces cross-country
    false candidates while staying open-set for France.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import pandas as pd

from src.config import MAX_POSTING
from src.preprocessing.normalize_addresses import (
    numeric_tokens,
    significant_address_tokens,
)
from src.preprocessing.normalize_names import significant_name_tokens


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_address_token_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """B4: entity_id lists keyed by each significant address token."""
    idx: dict[str, list[str]] = defaultdict(list)
    for eid, addr in zip(df["entity_id"], df["address_norm"]):
        for tok in significant_address_tokens(str(addr) if addr else ""):
            idx[tok].append(str(eid))
    return {k: v for k, v in idx.items() if len(v) <= MAX_POSTING}


def build_numeric_token_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """B5: entity_id lists keyed by numeric tokens from name + address."""
    idx: dict[str, list[str]] = defaultdict(list)
    for eid, name, addr in zip(
        df["entity_id"], df["name_norm"], df["address_norm"]
    ):
        for num in numeric_tokens(
            str(name) if name else "",
            str(addr) if addr else "",
        ):
            if len(num) >= 3:  # skip very short digits (too common)
                idx[num].append(str(eid))
    return {k: v for k, v in idx.items() if len(v) <= MAX_POSTING}


def build_country_name_combo_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """B6: entity_id lists keyed by country_norm + first significant name token."""
    idx: dict[str, list[str]] = defaultdict(list)
    for eid, country, name in zip(
        df["entity_id"], df["country_norm"], df["name_norm"]
    ):
        country_str = str(country).strip() if country else ""
        toks = significant_name_tokens(str(name) if name else "")
        if toks and country_str:
            key = f"{country_str}|{toks[0]}"
            idx[key].append(str(eid))
    return {k: v for k, v in idx.items() if len(v) <= MAX_POSTING}


# ---------------------------------------------------------------------------
# Pair generators
# ---------------------------------------------------------------------------

def address_candidate_pairs(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    *,
    source_tag: str,
) -> Iterator[tuple[str, str, str]]:
    """Yield (s1_id, candidate_id, block_strategy) from address blocks B4, B5, B6."""
    addr_idx = build_address_token_index(s2s3_df)
    num_idx = build_numeric_token_index(s2s3_df)
    combo_idx = build_country_name_combo_index(s2s3_df)

    for s1_id, s1_addr, s1_name, s1_country in zip(
        s1_df["entity_id"],
        s1_df["address_norm"],
        s1_df["name_norm"],
        s1_df["country_norm"],
    ):
        s1_addr = str(s1_addr) if s1_addr else ""
        s1_name = str(s1_name) if s1_name else ""
        s1_country = str(s1_country).strip() if s1_country else ""
        seen: set[str] = set()

        # B4
        for tok in significant_address_tokens(s1_addr):
            for cid in addr_idx.get(tok, []):
                if cid not in seen:
                    seen.add(cid)
                    yield str(s1_id), cid, "B4_address_token"

        # B5
        for num in numeric_tokens(s1_name, s1_addr):
            if len(num) >= 3:
                for cid in num_idx.get(num, []):
                    if cid not in seen:
                        seen.add(cid)
                        yield str(s1_id), cid, "B5_numeric_token"

        # B6
        s1_toks = significant_name_tokens(s1_name)
        if s1_toks and s1_country:
            key = f"{s1_country}|{s1_toks[0]}"
            for cid in combo_idx.get(key, []):
                if cid not in seen:
                    seen.add(cid)
                    yield str(s1_id), cid, "B6_country_name_combo"
