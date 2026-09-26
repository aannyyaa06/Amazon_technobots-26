"""Business-name normalization.

Original `business_name` is never overwritten. These helpers produce `name_norm`
and name tokens. Legal-suffix mapping is canonicalization, not deletion, so
“Acme Ltd” and “Acme Limited” meet without collapsing unrelated names.
"""

from __future__ import annotations

import re
import unicodedata

# Whole-token legal / org-form aliases. Country-agnostic (covers US, India, France).
LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "pvt.": "private",
    "pvtltd": "private limited",
    "ltd": "limited",
    "ltd.": "limited",
    "ltee": "limited",
    "corp": "corporation",
    "corp.": "corporation",
    "inc": "incorporated",
    "inc.": "incorporated",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "co": "company",
    "co.": "company",
    "gmbh": "gmbh",
    "sarl": "sarl",
    "sas": "sas",
    "sa": "sa",
    "spa": "spa",
    "bv": "bv",
    "nv": "nv",
    "pty": "proprietary",
    "lp": "lp",
}

NAME_STOP = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "for",
    "to",
    "at",
    "in",
    "on",
}

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_AND_RE = re.compile(r"\s*&\s*")


def normalize_name(raw: str | None) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _AND_RE.sub(" and ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""
    mapped = [LEGAL_SUFFIX_MAP.get(tok, tok) for tok in text.split(" ")]
    # Suffix map may inject multi-word values.
    text = " ".join(" ".join(mapped).split())
    return text


def name_tokens(name_norm: str) -> list[str]:
    if not name_norm:
        return []
    return name_norm.split(" ")


def significant_name_tokens(name_norm: str, *, min_len: int = 2) -> list[str]:
    """Tokens used as blocking keys: drop tiny tokens and function words."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in name_tokens(name_norm):
        if len(tok) < min_len or tok in NAME_STOP:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out
