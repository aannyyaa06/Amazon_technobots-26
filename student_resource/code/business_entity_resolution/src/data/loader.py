"""Load challenge TSVs with an explicit tab separator.

Never infer the delimiter: addresses and ID lists contain commas, so a default
CSV read would collapse each row into a single column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from src.config import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, TSV_CHUNKSIZE

# C engine + explicit tab separator. Do not let pandas sniff commas in addresses.
TSV_READ_KWARGS = dict(
    sep="\t",
    dtype=str,
    encoding="utf-8",
    keep_default_na=True,
    na_values=["", "NA", "NaN", "nan", "None"],
    on_bad_lines="warn",
)


def load_source(
    path: Path | str,
    *,
    usecols: Optional[tuple[str, ...] | list[str]] = None,
) -> pd.DataFrame:
    """Load a source file into a DataFrame. Original columns are not renamed."""
    df = pd.read_csv(path, usecols=list(usecols) if usecols else None, **TSV_READ_KWARGS)
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}. Got {list(df.columns)}")
    return df


def iter_source_chunks(
    path: Path | str,
    *,
    chunksize: int = TSV_CHUNKSIZE,
    usecols: Optional[tuple[str, ...] | list[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks so large sources are not fully materialized."""
    reader = pd.read_csv(
        path,
        usecols=list(usecols) if usecols else None,
        chunksize=chunksize,
        **TSV_READ_KWARGS,
    )
    for chunk in reader:
        missing = [c for c in SOURCE_COLUMNS if c not in chunk.columns]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}. Got {list(chunk.columns)}")
        yield chunk


def load_ground_truth(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, **TSV_READ_KWARGS)
    missing = [c for c in GROUND_TRUTH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}. Got {list(df.columns)}")
    return df


def iter_ground_truth_chunks(
    path: Path | str,
    *,
    chunksize: int = TSV_CHUNKSIZE,
) -> Iterator[pd.DataFrame]:
    reader = pd.read_csv(path, chunksize=chunksize, **TSV_READ_KWARGS)
    for chunk in reader:
        missing = [c for c in GROUND_TRUTH_COLUMNS if c not in chunk.columns]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}. Got {list(chunk.columns)}")
        yield chunk
