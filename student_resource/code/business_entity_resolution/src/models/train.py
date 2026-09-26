"""Phase 7: Train the LightGBM model.

Loads the feature matrices for train and validation splits, fits the model
with early stopping on the validation set, logs feature importances, and
saves the fitted model to output/artifacts/model_choiceA.pkl.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import ARTIFACTS_DIR
from src.features.feature_builder import FEATURE_COLS
from src.models.lightgbm_model import ChoiceAModel

INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
REPORTS_DIR = ARTIFACTS_DIR.parent / "reports"


def run_phase7(
    train_feats: pd.DataFrame,
    val_feats: pd.DataFrame,
    *,
    force: bool = False,
) -> ChoiceAModel:
    """Train LightGBM on train_feats and evaluate on val_feats.

    Parameters
    ----------
    train_feats : feature matrix with 'label' column (from Phase 6)
    val_feats   : validation feature matrix with 'label' column
    force       : retrain even if a saved model exists

    Returns the fitted ChoiceAModel.
    """
    from src.models.lightgbm_model import MODEL_PATH

    print("=== Phase 7: LightGBM Training ===")

    if Path(MODEL_PATH).exists() and not force:
        print("  Saved model found. Loading (pass force=True to retrain).")
        return ChoiceAModel.load()

    X_train = train_feats[[c for c in FEATURE_COLS if c in train_feats.columns]]
    y_train = train_feats["label"].astype(np.int32)
    X_val = val_feats[[c for c in FEATURE_COLS if c in val_feats.columns]]
    y_val = val_feats["label"].astype(np.int32)

    print(f"  Train: {len(X_train):,} pairs (pos={int(y_train.sum()):,})")
    print(f"  Val:   {len(X_val):,} pairs (pos={int(y_val.sum()):,})")

    t0 = time.perf_counter()
    model = ChoiceAModel()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # Evaluate
    val_proba = model.predict_proba(X_val)
    auc = roc_auc_score(y_val, val_proba)
    print(f"  Validation AUC: {auc:.4f}")

    # Feature importance report
    importances = model.feature_importances()
    print("\n  Feature importances (top 10):")
    print(importances.head(10).to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importances.to_csv(REPORTS_DIR / "feature_importances.csv")

    model.save()
    elapsed = time.perf_counter() - t0
    print(f"\n  Phase 7 complete in {elapsed:.1f}s | AUC: {auc:.4f}")

    return model
