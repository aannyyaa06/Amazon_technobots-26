"""Prediction utilities: apply trained model to any feature matrix.

The threshold is NOT applied here; raw probabilities are returned. Thresholding
is the responsibility of the evaluation / inference modules (Phases 9 & 11).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.lightgbm_model import ChoiceAModel


def predict_proba(
    feature_df: pd.DataFrame,
    *,
    model: ChoiceAModel | None = None,
) -> np.ndarray:
    """Return match probability for each row in feature_df.

    Parameters
    ----------
    feature_df : DataFrame containing the feature columns (e.g. from feature_builder).
    model      : fitted ChoiceAModel; if None, loads from default path.

    Returns
    -------
    np.ndarray of shape (n,) with probabilities in [0, 1].
    """
    if model is None:
        model = ChoiceAModel.load()
    return model.predict_proba(feature_df)


def score_pairs(
    feature_df: pd.DataFrame,
    *,
    model: ChoiceAModel | None = None,
) -> pd.DataFrame:
    """Add a 'match_proba' column to the pairs DataFrame.

    Parameters
    ----------
    feature_df : must contain 'source1_entity_id' and 'candidate_entity_id' columns
                 plus feature columns.
    model      : fitted ChoiceAModel; if None, loads from default path.

    Returns the input DataFrame with an additional 'match_proba' column.
    """
    proba = predict_proba(feature_df, model=model)
    out = feature_df.copy()
    out["match_proba"] = proba
    return out
