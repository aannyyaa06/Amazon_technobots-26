"""Shared paths and reproducibility settings.

Project layout (student_resource root):
    dataset/{train,test}/*.tsv
    output/
    code/business_entity_resolution/src/config.py
"""

from pathlib import Path

SEED = 42

# src/config.py -> src -> package -> code -> student_resource
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXPLORATION_DIR = OUTPUT_DIR / "exploration"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"

TRAIN_SOURCE_FILES = {
    "source1": TRAIN_DIR / "train_source1.tsv",
    "source2": TRAIN_DIR / "train_source2.tsv",
    "source3": TRAIN_DIR / "train_source3.tsv",
}
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE_FILES = {
    "source1": TEST_DIR / "test_source1.tsv",
    "source2": TEST_DIR / "test_source2.tsv",
    "source3": TEST_DIR / "test_source3.tsv",
}

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")

CACHE_DIR = OUTPUT_DIR / "cache"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Streaming chunk size for large TSVs (laptop-safe).
TSV_CHUNKSIZE = 50_000

# Blocking: skip keys whose inverted list exceeds this (too common).
MAX_POSTING = 180
# Hard cap after cheap ranking — this is the set fed to LightGBM.
MAX_CANDIDATES = 80
NEGATIVES_PER_ENTITY = 6
VAL_FRACTION = 0.20

# LightGBM
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 250,
    "n_jobs": -1,
    "random_state": SEED,
    "is_unbalance": True,
    "verbose": -1,
}

THRESHOLDS = [round(x * 0.05, 2) for x in range(6, 20)]  # 0.30 .. 0.95

EXPECTED_ID_PREFIX = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}
