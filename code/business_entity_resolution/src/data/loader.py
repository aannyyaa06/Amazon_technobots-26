import pandas as pd
from typing import Tuple, List, Dict
from pathlib import Path


def load_source_tsv(filepath: Path) -> pd.DataFrame:
    """Load a source TSV file safely."""
    df = pd.read_csv(
        filepath,
        sep='\t',
        dtype={'entity_id': str, 'business_name': str, 'business_address': str, 'country': str},
        keep_default_na=False
    )
    return df


def load_ground_truth(filepath: Path) -> pd.DataFrame:
    """Load train_ground_truth.tsv."""
    df = pd.read_csv(
        filepath,
        sep='\t',
        dtype={'source1_entity_id': str, 'matched_entity_ids': str},
        keep_default_na=False
    )
    return df


def parse_ground_truth_pairs(gt_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Convert ground truth into list of (s1_id, match_id) pairs."""
    pairs = []
    for _, row in gt_df.iterrows():
        s1 = row['source1_entity_id']
        matches = row['matched_entity_ids']
        if matches:
            for m in matches.split(','):
                m = m.strip()
                if m:
                    pairs.append((s1, m))
    return pairs


def get_ground_truth_dict(gt_df: pd.DataFrame) -> Dict[str, set]:
    """Map source1_entity_id to set of matching IDs."""
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1 = row['source1_entity_id']
        matches = row['matched_entity_ids']
        if matches:
            gt_map[s1] = set(m.strip() for m in matches.split(',') if m.strip())
        else:
            gt_map[s1] = set()
    return gt_map
