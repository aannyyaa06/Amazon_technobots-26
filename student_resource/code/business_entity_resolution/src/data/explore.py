"""Phase 1: read-only exploration of train and test TSVs.

Computes row counts, nulls, duplicate IDs, country distribution, unique names/
addresses, and ground-truth match statistics. Writes JSON + a text report under
output/exploration/. Does not modify dataset files.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import (
    EXPLORATION_DIR,
    TEST_SOURCE_FILES,
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE_FILES,
)
from src.data.loader import iter_ground_truth_chunks, iter_source_chunks
from src.data.validation import validate_ground_truth_schema, validate_source_schema


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "na"}


def _missing_mask(series) -> "Any":
    import pandas as pd

    s = series.astype("string")
    lowered = s.str.strip().str.lower()
    return s.isna() | lowered.eq("") | lowered.isin(["nan", "none", "null", "na"])


def explore_source(path: Path, source_key: str) -> dict[str, Any]:
    issues = validate_source_schema(path, source_key)
    stats: dict[str, Any] = {
        "path": str(path),
        "source_key": source_key,
        "file_bytes": path.stat().st_size if path.is_file() else None,
        "schema_issues": issues,
        "n_rows": 0,
        "columns": ["entity_id", "business_name", "business_address", "country"],
        "nulls": {
            "entity_id": 0,
            "business_name": 0,
            "business_address": 0,
            "country": 0,
        },
        "duplicate_entity_ids": 0,
        "n_unique_entity_ids": 0,
        "country_counts": {},
        "n_unique_business_names": 0,
        "n_unique_business_addresses": 0,
        "id_prefix_mismatch": 0,
        "sample_rows": [],
    }
    if issues:
        return stats

    expected_prefix = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}[source_key]
    seen_ids: set[str] = set()
    extra_id_occurrences = 0
    duplicate_id_examples: list[str] = []
    countries: Counter[str] = Counter()
    names: set[str] = set()
    addresses: set[str] = set()
    sample_rows: list[dict[str, str]] = []

    n_rows = 0
    prefix_mismatch = 0
    nulls = stats["nulls"]

    for chunk in iter_source_chunks(path):
        n_rows += len(chunk)
        for col in nulls:
            nulls[col] += int(_missing_mask(chunk[col]).sum())

        ids = chunk["entity_id"].fillna("").astype(str)
        prefix_mismatch += int((~ids.str.startswith(expected_prefix)).sum())
        for eid in ids.tolist():
            if eid in seen_ids:
                extra_id_occurrences += 1
                if len(duplicate_id_examples) < 10:
                    duplicate_id_examples.append(eid)
            else:
                seen_ids.add(eid)

        country_series = chunk["country"].mask(_missing_mask(chunk["country"]), other="__MISSING__")
        countries.update(country_series.astype(str).str.strip().tolist())

        names.update(chunk["business_name"].fillna("").astype(str).tolist())
        addresses.update(chunk["business_address"].fillna("").astype(str).tolist())

        if len(sample_rows) < 3:
            for rec in chunk.head(3 - len(sample_rows)).to_dict(orient="records"):
                sample_rows.append({k: ("" if _is_missing(v) else str(v)) for k, v in rec.items()})

    n_unique_ids = len(seen_ids)
    duplicate_ids = extra_id_occurrences

    stats.update(
        {
            "n_rows": n_rows,
            "nulls": nulls,
            "n_unique_entity_ids": n_unique_ids,
            "duplicate_entity_ids": duplicate_ids,
            "duplicate_id_examples": duplicate_id_examples,
            "country_counts": dict(countries.most_common()),
            "n_unique_countries": len([c for c in countries if c != "__MISSING__"]),
            "n_unique_business_names": len(names),
            "n_unique_business_addresses": len(addresses),
            "id_prefix_mismatch": prefix_mismatch,
            "sample_rows": sample_rows,
        }
    )
    # Keep IDs only for Source 1 coverage checks; drop before JSON dump.
    stats["_entity_ids"] = seen_ids if source_key == "source1" else None
    return stats


def explore_ground_truth(path: Path) -> dict[str, Any]:
    issues = validate_ground_truth_schema(path)
    stats: dict[str, Any] = {
        "path": str(path),
        "file_bytes": path.stat().st_size if path.is_file() else None,
        "schema_issues": issues,
        "n_source1_entities": 0,
        "n_duplicate_source1_rows": 0,
        "n_entities_with_zero_matches": 0,
        "n_entities_with_at_least_one_match": 0,
        "n_total_matched_ids": 0,
        "n_unique_matched_ids": 0,
        "match_count_distribution": {},
        "matched_id_prefix_counts": {},
        "n_invalid_matched_prefixes": 0,
        "n_intra_list_duplicate_ids": 0,
    }
    if issues:
        return stats

    s1_counts: Counter[str] = Counter()
    s1_ids: set[str] = set()
    match_len: Counter[int] = Counter()
    prefix_counts: Counter[str] = Counter()
    unique_matched: set[str] = set()
    n_zero = 0
    n_total_matched = 0
    n_invalid_prefix = 0
    n_intra_dup = 0
    n_rows = 0

    for chunk in iter_ground_truth_chunks(path):
        n_rows += len(chunk)
        s1_list = chunk["source1_entity_id"].fillna("").astype(str).tolist()
        s1_counts.update(s1_list)
        s1_ids.update(s1_list)
        for raw in chunk["matched_entity_ids"].tolist():
            if _is_missing(raw):
                ids: list[str] = []
            else:
                ids = [tok.strip() for tok in str(raw).split(",") if tok.strip()]
            if len(ids) != len(set(ids)):
                n_intra_dup += 1
            match_len[len(ids)] += 1
            if not ids:
                n_zero += 1
            n_total_matched += len(ids)
            unique_matched.update(ids)
            for mid in ids:
                if mid.startswith("S2-"):
                    prefix_counts["S2"] += 1
                elif mid.startswith("S3-"):
                    prefix_counts["S3"] += 1
                else:
                    prefix_counts["other"] += 1
                    n_invalid_prefix += 1

    n_unique_s1 = len(s1_counts)
    stats.update(
        {
            "n_rows": n_rows,
            "n_source1_entities": n_unique_s1,
            "n_duplicate_source1_rows": sum(c - 1 for c in s1_counts.values() if c > 1),
            "n_entities_with_zero_matches": n_zero,
            "n_entities_with_at_least_one_match": n_rows - n_zero,
            "n_total_matched_ids": n_total_matched,
            "n_unique_matched_ids": len(unique_matched),
            "mean_matches_per_s1": (n_total_matched / n_rows) if n_rows else 0.0,
            "match_count_distribution": {str(k): v for k, v in sorted(match_len.items())},
            "matched_id_prefix_counts": dict(prefix_counts),
            "n_invalid_matched_prefixes": n_invalid_prefix,
            "n_intra_list_duplicate_ids": n_intra_dup,
        }
    )
    stats["_source1_ids"] = s1_ids
    return stats


def cross_check_train(train_stats: dict[str, Any], gt_stats: dict[str, Any]) -> dict[str, Any]:
    """Coverage of Source 1 IDs vs ground truth, plus Cartesian size we must not materialize."""
    s1_rows = train_stats["source1"]["n_rows"]
    gt_s1 = gt_stats["n_source1_entities"]
    source1_ids = train_stats["source1"].pop("_entity_ids", None) or set()
    gt_ids = gt_stats.pop("_source1_ids", None) or set()
    in_s1_not_gt = source1_ids - gt_ids
    in_gt_not_s1 = gt_ids - source1_ids
    s2 = train_stats["source2"]["n_rows"]
    s3 = train_stats["source3"]["n_rows"]
    return {
        "train_s1_rows": s1_rows,
        "gt_unique_s1": gt_s1,
        "s1_row_count_equals_gt_s1_count": s1_rows == gt_s1,
        "n_s1_ids_missing_from_gt": len(in_s1_not_gt),
        "n_gt_ids_missing_from_s1": len(in_gt_not_s1),
        "example_s1_missing_from_gt": sorted(in_s1_not_gt)[:5],
        "example_gt_missing_from_s1": sorted(in_gt_not_s1)[:5],
        "train_s2_rows": s2,
        "train_s3_rows": s3,
        "cartesian_s1_x_s2s3": s1_rows * (s2 + s3),
    }


def format_report(payload: dict[str, Any]) -> str:
    lines = [
        "PHASE 1 — DATA EXPLORATION (read-only)",
        f"elapsed_seconds: {payload['elapsed_seconds']:.1f}",
        "",
    ]

    def dump_source(split: str, key: str, st: dict[str, Any]) -> None:
        lines.append(f"=== {split} {key} ===")
        lines.append(f"  path: {st['path']}")
        lines.append(f"  file_bytes: {st['file_bytes']}")
        lines.append(f"  schema_issues: {st['schema_issues'] or 'none'}")
        lines.append(f"  n_rows: {st['n_rows']}")
        lines.append(f"  n_unique_entity_ids: {st['n_unique_entity_ids']}")
        lines.append(f"  duplicate_entity_ids: {st['duplicate_entity_ids']}")
        lines.append(f"  id_prefix_mismatch: {st['id_prefix_mismatch']}")
        lines.append(f"  nulls: {st['nulls']}")
        lines.append(f"  n_unique_business_names: {st['n_unique_business_names']}")
        lines.append(f"  n_unique_business_addresses: {st['n_unique_business_addresses']}")
        lines.append(f"  n_unique_countries: {st.get('n_unique_countries')}")
        lines.append(f"  country_counts: {st['country_counts']}")
        lines.append("")

    for key, st in payload["train"]["sources"].items():
        dump_source("train", key, st)

    gt = payload["train"]["ground_truth"]
    lines.append("=== train ground truth ===")
    lines.append(f"  n_rows: {gt.get('n_rows')}")
    lines.append(f"  n_source1_entities: {gt['n_source1_entities']}")
    lines.append(f"  n_duplicate_source1_rows: {gt['n_duplicate_source1_rows']}")
    lines.append(f"  n_entities_with_zero_matches: {gt['n_entities_with_zero_matches']}")
    lines.append(f"  n_entities_with_at_least_one_match: {gt['n_entities_with_at_least_one_match']}")
    lines.append(f"  n_total_matched_ids: {gt['n_total_matched_ids']}")
    lines.append(f"  n_unique_matched_ids: {gt['n_unique_matched_ids']}")
    lines.append(f"  mean_matches_per_s1: {gt.get('mean_matches_per_s1')}")
    lines.append(f"  match_count_distribution: {gt['match_count_distribution']}")
    lines.append(f"  matched_id_prefix_counts: {gt['matched_id_prefix_counts']}")
    lines.append(f"  n_invalid_matched_prefixes: {gt['n_invalid_matched_prefixes']}")
    lines.append(f"  n_intra_list_duplicate_ids: {gt['n_intra_list_duplicate_ids']}")
    lines.append("")
    lines.append(f"=== train cross-check ===")
    lines.append(f"  {payload['train']['cross_check']}")
    lines.append("")

    for key, st in payload["test"]["sources"].items():
        dump_source("test", key, st)

    return "\n".join(lines) + "\n"


def run_phase1() -> dict[str, Any]:
    t0 = time.perf_counter()
    EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)

    train_sources = {key: explore_source(path, key) for key, path in TRAIN_SOURCE_FILES.items()}
    gt = explore_ground_truth(TRAIN_GROUND_TRUTH)
    cross = cross_check_train(train_sources, gt)
    for st in train_sources.values():
        st.pop("_entity_ids", None)
    test_sources = {key: explore_source(path, key) for key, path in TEST_SOURCE_FILES.items()}
    for st in test_sources.values():
        st.pop("_entity_ids", None)

    payload = {
        "phase": 1,
        "elapsed_seconds": time.perf_counter() - t0,
        "train": {
            "sources": train_sources,
            "ground_truth": gt,
            "cross_check": cross,
        },
        "test": {"sources": test_sources},
    }

    json_path = EXPLORATION_DIR / "phase1_stats.json"
    report_path = EXPLORATION_DIR / "phase1_report.txt"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = format_report(payload)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved stats: {json_path}")
    print(f"Saved report: {report_path}")
    return payload


if __name__ == "__main__":
    run_phase1()
