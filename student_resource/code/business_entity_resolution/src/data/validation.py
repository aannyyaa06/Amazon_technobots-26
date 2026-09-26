"""Schema / integrity checks. These never mutate source files."""

from __future__ import annotations

from pathlib import Path

from src.config import EXPECTED_ID_PREFIX, GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS


def validate_header(path: Path | str, expected: tuple[str, ...]) -> list[str]:
    """Return issues if the first line is not a tab-separated expected header."""
    issues: list[str] = []
    path = Path(path)
    if not path.is_file():
        return [f"File not found: {path}"]

    with path.open(encoding="utf-8") as f:
        header = f.readline()

    if not header:
        return [f"{path.name} is empty."]
    if "\t" not in header and "," in header:
        issues.append(
            f"{path.name}: header has no TAB but contains commas — file looks comma-separated."
        )
        return issues

    cols = [c.strip() for c in header.rstrip("\n").split("\t")]
    if tuple(cols) != expected:
        issues.append(f"{path.name}: unexpected header {cols}. Expected {list(expected)}.")
    return issues


def validate_source_schema(path: Path | str, source_key: str) -> list[str]:
    issues = validate_header(path, SOURCE_COLUMNS)
    prefix = EXPECTED_ID_PREFIX.get(source_key)
    if prefix is None:
        issues.append(f"Unknown source key: {source_key}")
    return issues


def validate_ground_truth_schema(path: Path | str) -> list[str]:
    return validate_header(path, GROUND_TRUTH_COLUMNS)
