# scikitplot/datasets/_autoscout24_tasks.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_autoscout24_tasks.py.

Examples for ML/MLOps workflows on an exported subset of the AutoScout24 dataset.

Notes
-----
- The dataset is tabular + text. There are no images, so "CNN" is not a natural fit
  unless you introduce external images or you specifically build a 1D CNN over text.
- These are reference baselines intended to be readable and reproducible, not to
  be the best-performing models out-of-the-box.

Examples
--------
This file assumes you have already created a subset with data_export.py, e.g.
a 10% deterministic subset (~12k rows) and saved it as Parquet:

>>> python -m scikitplot.datasets._data_export.py \
>>>     --input autoscout24_dataset_20251108.csv \
>>>     --output-dir data/subsets \
>>>     --percentages 10 \
>>>     --rounding round \
>>>     --strategy hash \
>>>     --id-col id \
>>>     --required-cols id price \
>>>     --format parquet \
>>>     --write-manifest \
>>>     --write-profile

A more representative 10% subset that preserves key distributions (recommended for most ML baselines):

>>> python -m scikitplot.datasets._data_export.py \
>>>     --input autoscout24_dataset_20251108.csv \
>>>     --output-dir data/subsets \
>>>     --percentages 10 \
>>>     --rounding round \
>>>     --strategy stratified \
>>>     --strata-cols country_code vehicle_type fuel_category is_used seller_is_dealer \
>>>     --allocation proportional \
>>>     --within linspace \
>>>     --required-cols id price \
>>>     --format parquet \
>>>     --write-manifest \
>>>     --write-profile

A balanced (equal-per-stratum) subset is useful when you want uniform class coverage, e.g. for a label in 1..9:

>>> python -m scikitplot.datasets._data_export.py \
>>>     --input dataset.csv \
>>>     --output-dir data/subsets \
>>>     --sizes 12000 \
>>>     --strategy stratified \
>>>     --strata-cols label \
>>>     --allocation equal \
>>>     --within linspace \
>>>     --format parquet \
>>>     --write-manifest


Then run one of the tasks below on the exported subset:

>>> python -m scikitplot.datasets._autoscout24_tasks.py --data data/subsets/subset_hash_11838.parquet eda
>>> python -m scikitplot.datasets._autoscout24_tasks.py --data data/subsets/subset_hash_11838.parquet regression_price
>>> python -m scikitplot.datasets._autoscout24_tasks.py --data data/subsets/subset_hash_11838.parquet classification_dealer
>>> python -m scikitplot.datasets._autoscout24_tasks.py --data data/subsets/subset_hash_11838.parquet ann_price
>>> python -m scikitplot.datasets._autoscout24_tasks.py --data data/subsets/subset_hash_11838.parquet nlp_dealer
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

LOGGER = logging.getLogger("autoscout24_tasks")


def _infer_format(path: str) -> Literal["csv", "parquet"]:
    lower = path.lower()
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return "csv"
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return "parquet"
    raise ValueError("Cannot infer file format. Use .csv/.parquet.")


def load_data(path: str) -> pd.DataFrame:
    fmt = _infer_format(path)
    if fmt == "csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def savefig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved figure: %s", p)


def eda(df: pd.DataFrame, *, out_dir: Path) -> None:
    """
    Produce simple deterministic EDA plots.

    - Missing values per column (top 25)
    - Price distribution (log-scale)
    - Price vs mileage scatter

    Outputs are saved under out_dir.
    """
    ensure_cols(df, ["price"])

    # Missingness
    miss = df.isna().sum().sort_values(ascending=False).head(25)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.barh(miss.index[::-1], miss.values[::-1])
    ax1.set_title("Missing values per column (top 25)")
    ax1.set_xlabel("Missing count")
    savefig(fig1, out_dir, "eda_missing_top25.png")

    # Price distribution
    price = df["price"].dropna().astype(float)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.hist(np.log1p(price), bins=50)
    ax2.set_title("log1p(price) distribution")
    ax2.set_xlabel("log1p(price)")
    ax2.set_ylabel("count")
    savefig(fig2, out_dir, "eda_price_log_hist.png")

    # Price vs mileage
    if "mileage_km_raw" in df.columns:
        d2 = df[["price", "mileage_km_raw"]].dropna()
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.scatter(d2["mileage_km_raw"].astype(float), np.log1p(d2["price"].astype(float)), s=6)
        ax3.set_title("log1p(price) vs mileage_km_raw")
        ax3.set_xlabel("mileage_km_raw")
        ax3.set_ylabel("log1p(price)")
        savefig(fig3, out_dir, "eda_price_vs_mileage.png")


def _split_features_target(df: pd.DataFrame, *, target: str) -> tuple[pd.DataFrame, pd.Series]:
    ensure_cols(df, [target])
    d = df.copy()
    y = d[target]
    X = d.drop(columns=[target])
    return X, y


def _make_tabular_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    # Select by dtype
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return pre, numeric_cols, categorical_cols


def regression_price(df: pd.DataFrame, *, test_size: float, seed: int) -> None:
    """
    Baseline regression: predict price using tabular features.

    Model: Ridge regression with OneHot for categoricals.
    Metric: MAE on holdout split.
    """
    ensure_cols(df, ["price"])
    d = df.dropna(subset=["price"]).copy()
    X, y = _split_features_target(d, target="price")
    y = y.astype(float)

    pre, _, _ = _make_tabular_preprocessor(X)

    model = Ridge(alpha=1.0, random_state=seed)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    mae = mean_absolute_error(y_te, pred)
    LOGGER.info("Regression(price) MAE=%.4f (test_size=%.2f rows=%d)", mae, test_size, len(y_te))


def classification_dealer(df: pd.DataFrame, *, test_size: float, seed: int) -> None:
    """
    Baseline classification: predict seller_is_dealer.

    Model: LogisticRegression with OneHot for categoricals.
    Metrics: ROC-AUC (if possible) + accuracy.
    """
    ensure_cols(df, ["seller_is_dealer"])
    d = df.dropna(subset=["seller_is_dealer"]).copy()
    X, y = _split_features_target(d, target="seller_is_dealer")
    y = y.astype(int)

    pre, _, _ = _make_tabular_preprocessor(X)

    model = LogisticRegression(max_iter=2000, n_jobs=None, random_state=seed)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if y.nunique() > 1 else None
    )
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_te, pred)
    try:
        auc = roc_auc_score(y_te, proba) if y_te.nunique() > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    LOGGER.info("Classification(seller_is_dealer) ACC=%.4f AUC=%.4f (test_size=%.2f rows=%d)", acc, auc, test_size, len(y_te))


def ann_price(df: pd.DataFrame, *, test_size: float, seed: int) -> None:
    """
    ANN example (MLPRegressor) for price.

    Important
    ---------
    scikit-learn MLP does not accept sparse matrices. To keep the feature matrix
    dense and bounded, we use OrdinalEncoder for categoricals (not OneHot).

    Metric: MAE on holdout split.
    """
    ensure_cols(df, ["price"])
    d = df.dropna(subset=["price"]).copy()
    X, y = _split_features_target(d, target="price")
    y = y.astype(float)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=seed,
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    mae = mean_absolute_error(y_te, pred)
    LOGGER.info("ANN Regression(price) MAE=%.4f (test_size=%.2f rows=%d)", mae, test_size, len(y_te))


def nlp_dealer(df: pd.DataFrame, *, test_size: float, seed: int) -> None:
    """
    NLP baseline: use description text to predict seller_is_dealer.

    Model: TF-IDF + LogisticRegression

    Notes
    -----
    - This ignores tabular features on purpose; combine with tabular features if needed.
    """
    ensure_cols(df, ["description", "seller_is_dealer"])
    d = df.dropna(subset=["description", "seller_is_dealer"]).copy()
    X_text = d["description"].astype(str)
    y = d["seller_is_dealer"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_text, y, test_size=test_size, random_state=seed, stratify=y if y.nunique() > 1 else None
    )

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        ngram_range=(1, 2),
        max_features=100_000,
    )
    model = LogisticRegression(max_iter=2000, random_state=seed)

    X_tr_vec = vec.fit_transform(X_tr)
    X_te_vec = vec.transform(X_te)

    model.fit(X_tr_vec, y_tr)
    proba = model.predict_proba(X_te_vec)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_te, pred)
    auc = roc_auc_score(y_te, proba) if y_te.nunique() > 1 else float("nan")

    LOGGER.info("NLP Classification(dealer) ACC=%.4f AUC=%.4f (rows=%d)", acc, auc, len(y_te))


def rag_demo(df: pd.DataFrame, *, query: str, top_k: int = 5) -> None:
    """
    Minimal RAG-style retrieval over listing text.

    This uses sentence-transformers + FAISS if installed.

    Install (example):
        pip install sentence-transformers faiss-cpu

    The retrieval corpus is built from a few columns plus description.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset subset.
    query : str
        Natural language query (e.g., "low mileage electric hatchback").
    top_k : int, default=5
        Number of retrieved rows to show.

    Returns
    -------
    None
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("sentence-transformers is required: pip install sentence-transformers") from e

    try:
        import faiss  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("faiss is required: pip install faiss-cpu") from e

    ensure_cols(df, ["id"])
    cols = [c for c in ["make", "model", "model_version", "price", "mileage_km_raw", "primary_fuel", "description"] if c in df.columns]

    d = df[cols].copy()
    d["__doc__"] = (
        d[cols]
        .astype(str)
        .fillna("")
        .agg(" | ".join, axis=1)
    )

    docs = d["__doc__"].tolist()
    ids = d["id"].astype(str).tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, idx = index.search(q, top_k)

    LOGGER.info("RAG query=%r top_k=%d", query, top_k)
    for rank, (i, s) in enumerate(zip(idx[0], scores[0]), start=1):
        LOGGER.info("#%d score=%.4f id=%s doc=%s", rank, float(s), ids[int(i)], docs[int(i)][:240].replace("\n", " "))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AutoScout24 example ML tasks (on exported subsets).")
    p.add_argument("--data", required=True, help="Path to exported subset (.csv/.parquet).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--out-dir", default="reports/autoscout24", help="Where to save plots.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("eda")
    sub.add_parser("regression_price")
    sub.add_parser("classification_dealer")
    sub.add_parser("ann_price")
    sub.add_parser("nlp_dealer")

    rag = sub.add_parser("rag_demo")
    rag.add_argument("--query", required=True)
    rag.add_argument("--top-k", type=int, default=5)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args(argv)

    df = load_data(str(args.data))
    LOGGER.info("Loaded rows=%d cols=%d from %s", len(df), df.shape[1], args.data)

    out_dir = Path(args.out_dir)

    if args.cmd == "eda":
        eda(df, out_dir=out_dir)
        return 0
    if args.cmd == "regression_price":
        regression_price(df, test_size=float(args.test_size), seed=int(args.seed))
        return 0
    if args.cmd == "classification_dealer":
        classification_dealer(df, test_size=float(args.test_size), seed=int(args.seed))
        return 0
    if args.cmd == "ann_price":
        ann_price(df, test_size=float(args.test_size), seed=int(args.seed))
        return 0
    if args.cmd == "nlp_dealer":
        nlp_dealer(df, test_size=float(args.test_size), seed=int(args.seed))
        return 0
    if args.cmd == "rag_demo":
        rag_demo(df, query=str(args.query), top_k=int(args.top_k))
        return 0

    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
