"""LightGBM pairwise classifier for business entity resolution (Choice A).

Wraps LGBMClassifier with:
- Config-driven hyperparameters (LGBM_PARAMS from config.py).
- Serialisation to / deserialisation from a single .pkl file.
- Feature schema validation at inference time to catch column mismatches early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.config import ARTIFACTS_DIR, LGBM_PARAMS, SEED
from src.features.feature_builder import FEATURE_COLS

MODEL_PATH = ARTIFACTS_DIR / "model_choiceA.pkl"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.txt"


class ChoiceAModel:
    """Thin wrapper around LGBMClassifier for the Choice A pipeline."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(LGBM_PARAMS) if params is None else dict(params)
        self._clf: LGBMClassifier | None = None
        self.feature_names: list[str] = list(FEATURE_COLS)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "ChoiceAModel":
        """Fit the LightGBM classifier.

        Early stopping is used when a validation set is provided.
        """
        self.feature_names = [c for c in FEATURE_COLS if c in X_train.columns]
        X_tr = X_train[self.feature_names].astype(np.float32)

        eval_set = None
        callbacks = None
        if X_val is not None and y_val is not None:
            from lightgbm import early_stopping, log_evaluation
            X_vl = X_val[self.feature_names].astype(np.float32)
            eval_set = [(X_vl, y_val)]
            callbacks = [early_stopping(50, verbose=False), log_evaluation(50)]

        self._clf = LGBMClassifier(**self.params)
        self._clf.fit(
            X_tr,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        print(f"  Best iteration: {self._clf.best_iteration_}")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(match) for each row in X."""
        if self._clf is None:
            raise RuntimeError("Model not fitted. Call fit() or load().")
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Feature columns missing: {missing}")
        X_in = X[self.feature_names].astype(np.float32)
        return self._clf.predict_proba(X_in)[:, 1]

    def feature_importances(self) -> pd.Series:
        """Return gain-based feature importances as a Series."""
        if self._clf is None:
            raise RuntimeError("Model not fitted.")
        return pd.Series(
            self._clf.feature_importances_,
            index=self.feature_names,
            name="importance",
        ).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH) -> None:
        """Save model + feature names to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._clf, "feature_names": self.feature_names}, path)
        # Write human-readable feature schema
        schema_path = Path(str(path).replace(".pkl", "_features.txt"))
        schema_path.write_text("\n".join(self.feature_names), encoding="utf-8")
        print(f"  Model saved to: {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ChoiceAModel":
        """Load a saved model from disk."""
        payload = joblib.load(path)
        obj = cls()
        obj._clf = payload["model"]
        obj.feature_names = payload["feature_names"]
        print(f"  Model loaded from: {path}")
        return obj
