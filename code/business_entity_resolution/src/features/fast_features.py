import sys
import gc
import re
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz
from tqdm import tqdm

from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_MATCHING, OUTPUT_CANDIDATE,
    RANDOM_STATE, LGBM_PARAMS
)
from src.preprocessing.normalize import normalize_name, normalize_address, extract_numbers
from src.evaluation.metrics import compute_macro_f05

STOPWORDS = {
    'the', 'and', 'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'co', 'company', 'services', 'service', 'group', 'enterprises',
    'enterprise', 'technologies', 'technology', 'india', 'us', 'usa', 'france',
    'rd', 'road', 'st', 'street', 'ave', 'avenue', 'nagar', 'bldg', 'floor',
    'com', 'net', 'org', 'www'
}


def clean_tokens(text: str, min_len: int = 3) -> set:
    if not text:
        return set()
    cleaned = re.sub(r'\.(com|net|org|in|us|fr|co)', ' ', text.lower())
    tokens = re.findall(r'[a-z0-9]{' + str(min_len) + r',}', cleaned)
    return set(t for t in tokens if t not in STOPWORDS)


def compute_fast_features(s1_name, s1_addr, c_name, c_addr, s1_country, c_country, c_src):
    s1_ntoks = clean_tokens(s1_name, 3)
    c_ntoks = clean_tokens(c_name, 3)
    s1_atoks = clean_tokens(s1_addr, 3)
    c_atoks = clean_tokens(c_addr, 3)

    # Jaccard
    n_inter = len(s1_ntoks & c_ntoks)
    n_union = len(s1_ntoks | c_ntoks)
    name_jaccard = (n_inter / n_union) if n_union > 0 else 0.0

    a_inter = len(s1_atoks & c_atoks)
    a_union = len(s1_atoks | c_atoks)
    addr_jaccard = (a_inter / a_union) if a_union > 0 else 0.0

    # Overlap
    n_min = min(len(s1_ntoks), len(c_ntoks))
    name_overlap = (n_inter / n_min) if n_min > 0 else 0.0
    a_min = min(len(s1_atoks), len(c_atoks))
    addr_overlap = (a_inter / a_min) if a_min > 0 else 0.0

    # Fuzzy ratios
    name_ratio = fuzz.ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    name_token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    addr_ratio = fuzz.ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0

    # Numeric
    s1_nums = extract_numbers(s1_addr)
    c_nums = extract_numbers(c_addr)
    shared_nums = len(s1_nums & c_nums)
    has_same_num = 1.0 if shared_nums > 0 else 0.0

    return [
        1.0 if s1_name == c_name and s1_name else 0.0,
        name_ratio,
        name_token_sort,
        name_token_set,
        name_jaccard,
        name_overlap,
        abs(len(s1_name) - len(c_name)),
        abs(len(s1_ntoks) - len(c_ntoks)),
        1.0 if s1_addr == c_addr and s1_addr else 0.0,
        addr_ratio,
        addr_token_set,
        addr_jaccard,
        addr_overlap,
        abs(len(s1_addr) - len(c_addr)),
        float(shared_nums),
        has_same_num,
        1.0 if s1_country == c_country else 0.0,
        1.0 if c_src == 'S2' else 0.0,
        0.5 * (name_ratio + addr_ratio),
        1.0 if name_token_set >= 0.85 else 0.0,
        1.0 if addr_token_set >= 0.85 else 0.0
    ]


FEATURE_NAMES = [
    'name_exact', 'name_ratio', 'name_token_sort', 'name_token_set',
    'name_jaccard', 'name_overlap', 'name_len_diff', 'name_tok_diff',
    'addr_exact', 'addr_ratio', 'addr_token_set', 'addr_jaccard',
    'addr_overlap', 'addr_len_diff', 'shared_numbers', 'has_same_num',
    'same_country', 'is_s2', 'name_addr_mean', 'has_high_name', 'has_high_addr'
]
