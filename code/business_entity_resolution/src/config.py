import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(r"D:\amazon challenge\business-entity-resolution")
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# TSV File Names
TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

OUTPUT_MATCHING = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATE = OUTPUT_DIR / "candidate_pairs.tsv"

# Validation Script
VALIDATE_SCRIPT = Path(r"D:\amazon challenge\6ab10eb3b23ba_student_resource\student_resource\utils\validate_submission.py")

# Pipeline Configuration
RANDOM_STATE = 42
VALIDATION_SPLIT_RATIO = 0.20

# Blocking Settings
MAX_CANDIDATES_PER_S1 = 100
MIN_TOKEN_LEN = 3

# Model Settings
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.08,
    'num_leaves': 63,
    'max_depth': 7,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'n_estimators': 300,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}
