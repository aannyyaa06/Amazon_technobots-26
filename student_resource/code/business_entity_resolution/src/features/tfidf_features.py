"""Phase 6b: TF-IDF cosine similarity features.

Fits two TF-IDF vectorisers (character n-gram 2–4, sublinear_tf) separately
for business names and addresses, then computes cosine similarity for each
candidate pair using sparse dot products.

Vectorisers are serialised to output/artifacts/tfidf/ and reloaded on
subsequent runs (no re-fitting on test data).

Features produced
-----------------
name_tfidf_cosine : float   character-ngram TF-IDF cosine similarity (names)
addr_tfidf_cosine : float   character-ngram TF-IDF cosine similarity (addresses)
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ARTIFACTS_DIR

TFIDF_DIR = ARTIFACTS_DIR / "tfidf"
NAME_VEC_PATH = TFIDF_DIR / "name_tfidf_vectorizer.pkl"
ADDR_VEC_PATH = TFIDF_DIR / "addr_tfidf_vectorizer.pkl"

_TFIDF_KWARGS = dict(
    analyzer="char_wb",
    ngram_range=(2, 4),
    sublinear_tf=True,
    min_df=3,
    max_features=200_000,
    dtype=np.float32,
)


def fit_tfidf_vectorizers(
    name_corpus: pd.Series,
    addr_corpus: pd.Series,
    *,
    force: bool = False,
) -> tuple["TfidfVectorizer", "TfidfVectorizer"]:
    """Fit or load TF-IDF vectorisers on training corpora.

    Parameters
    ----------
    name_corpus : all name_norm values from training sources (S1+S2+S3)
    addr_corpus : all address_norm values
    force       : re-fit even if saved models exist
    """
    TFIDF_DIR.mkdir(parents=True, exist_ok=True)

    if NAME_VEC_PATH.exists() and ADDR_VEC_PATH.exists() and not force:
        print("  Loading existing TF-IDF vectorisers …")
        name_vec = joblib.load(NAME_VEC_PATH)
        addr_vec = joblib.load(ADDR_VEC_PATH)
        return name_vec, addr_vec

    print("  Fitting TF-IDF name vectoriser …")
    t0 = time.perf_counter()
    name_corpus_clean = name_corpus.fillna("").astype(str)
    name_vec = TfidfVectorizer(**_TFIDF_KWARGS)
    name_vec.fit(name_corpus_clean)
    joblib.dump(name_vec, NAME_VEC_PATH)
    print(f"    name vocab size: {len(name_vec.vocabulary_):,}  ({time.perf_counter()-t0:.1f}s)")

    print("  Fitting TF-IDF address vectoriser …")
    t1 = time.perf_counter()
    addr_corpus_clean = addr_corpus.fillna("").astype(str)
    addr_vec = TfidfVectorizer(**_TFIDF_KWARGS)
    addr_vec.fit(addr_corpus_clean)
    joblib.dump(addr_vec, ADDR_VEC_PATH)
    print(f"    addr vocab size: {len(addr_vec.vocabulary_):,}  ({time.perf_counter()-t1:.1f}s)")

    return name_vec, addr_vec


def load_tfidf_vectorizers() -> tuple["TfidfVectorizer", "TfidfVectorizer"]:
    """Load pre-fitted TF-IDF vectorisers from disk."""
    if not NAME_VEC_PATH.exists() or not ADDR_VEC_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorisers not found in {TFIDF_DIR}. Run Phase 6 first."
        )
    return joblib.load(NAME_VEC_PATH), joblib.load(ADDR_VEC_PATH)


def compute_tfidf_cosines(
    name_vec: "TfidfVectorizer",
    addr_vec: "TfidfVectorizer",
    df: pd.DataFrame,
    *,
    batch_size: int = 5000,
) -> pd.DataFrame:
    """Compute TF-IDF cosine similarities for candidate pairs.

    Expects df to have columns:
        name_norm_s1, addr_norm_s1, name_norm_cand, addr_norm_cand

    Returns a 2-column DataFrame: name_tfidf_cosine, addr_tfidf_cosine
    """
    n = len(df)
    name_cosines = np.zeros(n, dtype=np.float32)
    addr_cosines = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df.iloc[start:end]

        n_s1 = name_vec.transform(batch["name_norm_s1"].fillna("").astype(str))
        n_cd = name_vec.transform(batch["name_norm_cand"].fillna("").astype(str))
        # Row-wise cosine: dot each pair
        name_cosines[start:end] = np.array(
            n_s1.multiply(n_cd).sum(axis=1)
        ).flatten()

        a_s1 = addr_vec.transform(batch["addr_norm_s1"].fillna("").astype(str))
        a_cd = addr_vec.transform(batch["addr_norm_cand"].fillna("").astype(str))
        addr_cosines[start:end] = np.array(
            a_s1.multiply(a_cd).sum(axis=1)
        ).flatten()

    return pd.DataFrame(
        {"name_tfidf_cosine": name_cosines, "addr_tfidf_cosine": addr_cosines},
        index=df.index,
    )
