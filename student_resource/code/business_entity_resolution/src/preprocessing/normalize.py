"""Add normalized columns without mutating original fields.

Country is lowercased + trimmed only. No US/India/France whitelist — unseen
labels such as France flow through automatically.
"""

from __future__ import annotations

import unicodedata

import pandas as pd

from .normalize_addresses import normalize_address
from .normalize_names import normalize_name


def normalize_country(raw: str | None) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).lower()


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with name_norm / address_norm / country_norm added."""
    out = df.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["address_norm"] = out["business_address"].map(normalize_address)
    out["country_norm"] = out["country"].map(normalize_country)
    return out
