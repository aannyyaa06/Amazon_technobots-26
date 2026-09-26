from collections import defaultdict
from typing import Dict, Set, List, Tuple
import pandas as pd
import re
from src.preprocessing.normalize import extract_numbers

STOPWORDS = {
    'the', 'and', 'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited',
    'pvt', 'co', 'company', 'services', 'service', 'group', 'enterprises',
    'enterprise', 'technologies', 'technology', 'india', 'us', 'usa', 'france',
    'rd', 'road', 'st', 'street', 'ave', 'avenue', 'nagar', 'bldg', 'floor',
    'com', 'net', 'org', 'www'
}


def get_name_tokens(norm_name: str) -> List[str]:
    """Extract informative name tokens (length >= 3, not in stopwords)."""
    if not norm_name:
        return []
    # Strip common web domain markers
    cleaned = re.sub(r'\.(com|net|org|in|us|fr|co|io)', ' ', norm_name)
    tokens = re.findall(r'[a-z0-9]{3,}', cleaned)
    return [t for t in tokens if t not in STOPWORDS]


def get_address_tokens(norm_address: str) -> List[str]:
    """Extract informative street / locality tokens from address."""
    if not norm_address:
        return []
    tokens = re.findall(r'[a-z0-9]{4,}', norm_address)
    return [t for t in tokens if t not in STOPWORDS]


class MultiBlocker:
    """
    High-recall multi-strategy candidate generator.
    Blocks on:
      1. Exact normalized name
      2. Domain/stripped compressed name (handles websitedomain.com vs name)
      3. Informative name tokens
      4. Name character prefix (first 4 & 5 chars)
      5. Address street token + number combination
    """

    def __init__(self, max_candidates_per_key: int = 150):
        self.max_candidates_per_key = max_candidates_per_key
        # Partitioned by country for 100% precision constraint
        self.exact_name_idx = defaultdict(lambda: defaultdict(list))
        self.compact_name_idx = defaultdict(lambda: defaultdict(list))
        self.token_idx = defaultdict(lambda: defaultdict(list))
        self.prefix_idx = defaultdict(lambda: defaultdict(list))
        self.addr_combo_idx = defaultdict(lambda: defaultdict(list))

    def fit(self, cand_df: pd.DataFrame):
        """Index candidate records from S2 and S3."""
        for row in cand_df.itertuples(index=False):
            cid = row.entity_id
            country = row.country
            name = row.business_name_norm
            addr = row.business_address_norm

            if not name and not addr:
                continue

            if name:
                # 1. Exact normalized name
                self.exact_name_idx[country][name].append(cid)

                # 2. Compact name without spaces/punctuation (catches "fetech national twin" vs "fetechnationaltwin.com")
                compact = re.sub(r'[^a-z0-9]', '', name)
                compact = re.sub(r'(com|net|org|in|us|fr)$', '', compact)
                if len(compact) >= 5:
                    self.compact_name_idx[country][compact].append(cid)
                    # Prefix of compact
                    self.prefix_idx[country][compact[:4]].append(cid)

                # 3. Informative name tokens
                tokens = get_name_tokens(name)
                for tok in tokens:
                    self.token_idx[country][tok].append(cid)

            # 4. Address block: street token + house number
            if addr:
                nums = extract_numbers(addr)
                addr_toks = get_address_tokens(addr)
                if nums and addr_toks:
                    for num in list(nums)[:2]:
                        for atok in addr_toks[:2]:
                            combo_key = f"{num}_{atok}"
                            self.addr_combo_idx[country][combo_key].append(cid)

    def generate_candidates_for_s1(
        self, 
        s1_row, 
        max_total_candidates: int = 90
    ) -> List[str]:
        """Generate candidates for a single S1 record using union of strategies."""
        country = s1_row.country
        name = s1_row.business_name_norm
        addr = s1_row.business_address_norm

        candidates = set()

        # Strategy 1: Exact Name Match
        if name:
            exact_matches = self.exact_name_idx[country].get(name, [])
            candidates.update(exact_matches[:self.max_candidates_per_key])

            # Strategy 2: Compact/Domain Match
            compact = re.sub(r'[^a-z0-9]', '', name)
            compact = re.sub(r'(com|net|org|in|us|fr)$', '', compact)
            if len(compact) >= 5:
                compact_matches = self.compact_name_idx[country].get(compact, [])
                candidates.update(compact_matches[:self.max_candidates_per_key])

            # Strategy 3: Significant Name Tokens
            tokens = get_name_tokens(name)
            if tokens:
                first_tok = tokens[0]
                tok_matches = self.token_idx[country].get(first_tok, [])
                if len(tok_matches) <= self.max_candidates_per_key:
                    candidates.update(tok_matches)
                else:
                    candidates.update(tok_matches[:self.max_candidates_per_key])

                for tok in tokens[1:3]:
                    if len(candidates) >= max_total_candidates:
                        break
                    tok_matches = self.token_idx[country].get(tok, [])
                    if len(tok_matches) <= 60:
                        candidates.update(tok_matches)

            # Strategy 4: Prefix Match
            if len(candidates) < max_total_candidates and len(compact) >= 4:
                prefix_matches = self.prefix_idx[country].get(compact[:4], [])
                if len(prefix_matches) <= 60:
                    candidates.update(prefix_matches)

        # Strategy 5: Address Combo Match (Street token + Number)
        # This recovers entities with alias/DBA names at the identical address
        if len(candidates) < max_total_candidates and addr:
            nums = extract_numbers(addr)
            addr_toks = get_address_tokens(addr)
            if nums and addr_toks:
                for num in list(nums)[:2]:
                    for atok in addr_toks[:2]:
                        combo_key = f"{num}_{atok}"
                        combo_matches = self.addr_combo_idx[country].get(combo_key, [])
                        if len(combo_matches) <= 30:
                            candidates.update(combo_matches)
                        if len(candidates) >= max_total_candidates:
                            break

        cand_list = list(candidates)
        if len(cand_list) > max_total_candidates:
            cand_list = cand_list[:max_total_candidates]

        return cand_list
