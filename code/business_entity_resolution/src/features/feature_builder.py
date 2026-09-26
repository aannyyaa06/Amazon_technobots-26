import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from typing import Dict, Any
from src.preprocessing.normalize import extract_numbers


def token_jaccard(tokens1: set, tokens2: set) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return float(intersection) / union if union > 0 else 0.0


def token_overlap(tokens1: set, tokens2: set) -> float:
    """Compute overlap coefficient: |A n B| / min(|A|, |B|)."""
    if not tokens1 or not tokens2:
        return 0.0
    min_len = min(len(tokens1), len(tokens2))
    return float(len(tokens1 & tokens2)) / min_len if min_len > 0 else 0.0


def compute_pair_features(
    s1_name: str,
    cand_name: str,
    s1_addr: str,
    cand_addr: str,
    s1_country: str,
    cand_country: str,
    cand_source: str  # 'S2' or 'S3'
) -> Dict[str, float]:
    """
    Compute tabular similarity features for a candidate pair.
    """
    # 1. Name Features
    s1_name_toks = set(s1_name.split()) if s1_name else set()
    cand_name_toks = set(cand_name.split()) if cand_name else set()

    name_exact = 1.0 if s1_name and s1_name == cand_name else 0.0
    name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_partial_ratio = fuzz.partial_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_token_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0 if s1_name and cand_name else 0.0
    name_jaccard = token_jaccard(s1_name_toks, cand_name_toks)
    name_overlap = token_overlap(s1_name_toks, cand_name_toks)
    name_len_diff = abs(len(s1_name) - len(cand_name))
    name_tok_diff = abs(len(s1_name_toks) - len(cand_name_toks))

    # 2. Address Features
    s1_addr_toks = set(s1_addr.split()) if s1_addr else set()
    cand_addr_toks = set(cand_addr.split()) if cand_addr else set()

    addr_exact = 1.0 if s1_addr and s1_addr == cand_addr else 0.0
    addr_ratio = fuzz.ratio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0 if s1_addr and cand_addr else 0.0
    addr_jaccard = token_jaccard(s1_addr_toks, cand_addr_toks)
    addr_overlap = token_overlap(s1_addr_toks, cand_addr_toks)
    addr_len_diff = abs(len(s1_addr) - len(cand_addr))

    # 3. Numeric Features (House/street/building numbers)
    s1_nums = extract_numbers(s1_addr)
    cand_nums = extract_numbers(cand_addr)
    shared_numbers = len(s1_nums & cand_nums)
    numeric_jaccard = token_jaccard(s1_nums, cand_nums)
    has_same_number = 1.0 if shared_numbers > 0 else 0.0

    # 4. Country & Source Features
    same_country = 1.0 if s1_country == cand_country else 0.0
    is_s2 = 1.0 if cand_source == 'S2' else 0.0
    is_s3 = 1.0 if cand_source == 'S3' else 0.0

    # 5. Combined Interaction Features
    name_addr_mean = 0.5 * (name_ratio + addr_ratio)
    has_high_name = 1.0 if name_token_set >= 0.85 else 0.0
    has_high_addr = 1.0 if addr_token_set >= 0.85 else 0.0

    return {
        'name_exact': name_exact,
        'name_ratio': name_ratio,
        'name_partial_ratio': name_partial_ratio,
        'name_token_sort': name_token_sort,
        'name_token_set': name_token_set,
        'name_jaccard': name_jaccard,
        'name_overlap': name_overlap,
        'name_len_diff': float(name_len_diff),
        'name_tok_diff': float(name_tok_diff),
        'addr_exact': addr_exact,
        'addr_ratio': addr_ratio,
        'addr_token_set': addr_token_set,
        'addr_jaccard': addr_jaccard,
        'addr_overlap': addr_overlap,
        'addr_len_diff': float(addr_len_diff),
        'shared_numbers': float(shared_numbers),
        'numeric_jaccard': numeric_jaccard,
        'has_same_number': has_same_number,
        'same_country': same_country,
        'is_s2': is_s2,
        'is_s3': is_s3,
        'name_addr_mean': name_addr_mean,
        'has_high_name': has_high_name,
        'has_high_addr': has_high_addr
    }
