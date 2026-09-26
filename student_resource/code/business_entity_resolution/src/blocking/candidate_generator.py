"""Candidate generator: union of all blocking strategies.

Runs all 6 blocking strategies (B1–B6) for both S2 and S3, deduplicates,
applies a cheap Jaccard pre-rank, then caps each S1 entity at MAX_CANDIDATES
before writing the candidate Parquet file.

Output columns
--------------
source1_entity_id : str
candidate_entity_id : str
block_strategies : str   (comma-separated strategy tags that generated this pair)
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ARTIFACTS_DIR, MAX_CANDIDATES
from src.blocking.address_blocking import address_candidate_pairs
from src.blocking.name_blocking import name_candidate_pairs
from src.preprocessing.normalize_names import significant_name_tokens
from src.preprocessing.normalize_addresses import significant_address_tokens

INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
CANDIDATE_COLS = ["source1_entity_id", "candidate_entity_id", "block_strategies"]


def _candidate_path(split: str) -> Path:
    return INTERMEDIATE_DIR / f"candidate_pairs_{split}.parquet"


def _jaccard(a_toks: set[str], b_toks: set[str]) -> float:
    if not a_toks and not b_toks:
        return 0.0
    inter = len(a_toks & b_toks)
    union = len(a_toks | b_toks)
    return inter / union if union else 0.0


def _pre_rank_and_cap(
    s1_candidates: dict[str, list[tuple[str, str]]],
    s1_norm: pd.DataFrame,
    cand_norm: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    """For each S1 entity, keep at most MAX_CANDIDATES candidates by Jaccard score."""
    # Build lookup dicts for fast access
    s1_tokens: dict[str, set[str]] = {}
    for eid, name, addr in zip(
        s1_norm["entity_id"], s1_norm["name_norm"], s1_norm["address_norm"]
    ):
        toks = set(significant_name_tokens(str(name) if name else ""))
        toks.update(significant_address_tokens(str(addr) if addr else ""))
        s1_tokens[str(eid)] = toks

    cand_tokens: dict[str, set[str]] = {}
    for eid, name, addr in zip(
        cand_norm["entity_id"], cand_norm["name_norm"], cand_norm["address_norm"]
    ):
        toks = set(significant_name_tokens(str(name) if name else ""))
        toks.update(significant_address_tokens(str(addr) if addr else ""))
        cand_tokens[str(eid)] = toks

    rows: list[tuple[str, str, str]] = []
    for s1_id, candidates in s1_candidates.items():
        s1_tok = s1_tokens.get(s1_id, set())
        if len(candidates) <= MAX_CANDIDATES:
            for cid, strategies in candidates:
                rows.append((s1_id, cid, strategies))
            continue

        # Score and cap
        scored = []
        for cid, strategies in candidates:
            c_tok = cand_tokens.get(cid, set())
            score = _jaccard(s1_tok, c_tok)
            scored.append((score, cid, strategies))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, cid, strategies in scored[:MAX_CANDIDATES]:
            rows.append((s1_id, cid, strategies))

    return rows


def generate_candidates(
    split: str,
    s1_norm: pd.DataFrame,
    s2_norm: pd.DataFrame,
    s3_norm: pd.DataFrame,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Run all blocking strategies and write candidate pairs Parquet.

    Parameters
    ----------
    split:    "train" or "test"
    s1_norm:  normalized Source 1 (entity_id, name_norm, address_norm, country_norm)
    s2_norm:  normalized Source 2
    s3_norm:  normalized Source 3
    force:    overwrite cache if True

    Returns the candidate pairs DataFrame.
    """
    out_path = _candidate_path(split)
    if out_path.exists() and not force:
        print(f"  Candidate cache exists ({split}): {out_path}")
        return pd.read_parquet(out_path)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Per-S1 accumulator: {s1_id: {cand_id: [strategies]}}
    accumulator: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for s2s3_tag, s2s3_df in [("source2", s2_norm), ("source3", s3_norm)]:
        print(f"  Blocking S1 vs {s2s3_tag} …")
        n_pairs = 0

        for s1_id, cid, strategy in name_candidate_pairs(s1_norm, s2s3_df, source_tag=s2s3_tag):
            accumulator[s1_id][cid].append(strategy)
            n_pairs += 1

        for s1_id, cid, strategy in address_candidate_pairs(s1_norm, s2s3_df, source_tag=s2s3_tag):
            accumulator[s1_id][cid].append(strategy)
            n_pairs += 1

        print(f"    Raw pairs (before dedup/cap): {n_pairs:,}")

    # Collapse strategies to comma-separated string
    s1_candidates: dict[str, list[tuple[str, str]]] = {
        s1_id: [(cid, ",".join(sorted(set(strats)))) for cid, strats in cid_map.items()]
        for s1_id, cid_map in accumulator.items()
    }

    total_raw = sum(len(v) for v in s1_candidates.values())
    print(f"  Deduped pairs: {total_raw:,}  (pre-cap)")

    # Pre-rank and cap
    # Build a combined cand lookup from s2+s3
    cand_all = pd.concat([s2_norm, s3_norm], ignore_index=True)
    rows = _pre_rank_and_cap(s1_candidates, s1_norm, cand_all)

    df = pd.DataFrame(rows, columns=CANDIDATE_COLS)
    df.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    print(f"  Final candidate pairs: {len(df):,} | Unique S1 entities: {df['source1_entity_id'].nunique():,} | {elapsed:.1f}s")
    print(f"  Written to: {out_path}")
    return df


def load_candidates(split: str) -> pd.DataFrame:
    """Load cached candidate pairs Parquet."""
    path = _candidate_path(split)
    if not path.exists():
        raise FileNotFoundError(f"Candidate cache not found: {path}. Run Phase 3 first.")
    return pd.read_parquet(path)
