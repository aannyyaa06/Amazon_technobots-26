"""Address normalization. Original `business_address` is never overwritten."""

from __future__ import annotations

import re
import unicodedata

# Common street-type abbreviations only. Do not map N/S/E/W or country names.
ADDRESS_ABBREV = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "no": "number",
    "num": "number",
}

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+")


def normalize_address(raw: str | None) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""
    mapped = [ADDRESS_ABBREV.get(tok, tok) for tok in text.split(" ")]
    return " ".join(mapped)


def address_tokens(address_norm: str) -> list[str]:
    if not address_norm:
        return []
    return address_norm.split(" ")


def numeric_tokens(*texts: str) -> list[str]:
    """Preserve numeric tokens from names/addresses (street nos, PIN/ZIP, unit)."""
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for m in _NUM_RE.findall(str(text)):
            if m not in seen:
                seen.add(m)
                found.append(m)
    return found


def significant_address_tokens(address_norm: str, *, min_len: int = 3) -> list[str]:
    skip = {
        "road",
        "street",
        "avenue",
        "boulevard",
        "drive",
        "lane",
        "highway",
        "parkway",
        "suite",
        "apartment",
        "building",
        "floor",
        "number",
        "near",
        "opp",
        "opposite",
        "behind",
        "plot",
        "area",
        "sector",
        "phase",
        "block",
        "and",
        "the",
    }
    out: list[str] = []
    seen: set[str] = set()
    for tok in address_tokens(address_norm):
        if tok.isdigit():
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
            continue
        if len(tok) < min_len or tok in skip:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out
