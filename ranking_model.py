"""
SRAP - Smart Roadside Assistance Platform
Learning-to-Rank model training, evaluation and inference with XGBoost (XGBRanker).

Usage:
    python ranking_model.py            # train + evaluate + save model + demo inference
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "srap_ranking_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "srap_ranking_test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "srap_xgb_ranker.json")

QID_COL = "request_id"
ID_COL = "provider_id"
TARGET_COL = "relevance_label"
FEATURE_COLS = [
    "service_type",
    "user_vehicle_tier",
    "tier_difference",
    "distance_km",
    "rating",
    "quoted_price",
    "price_ratio",
    "mobile_workshop",
]
EXPLOITATION_THRESHOLD = 1.30
NDCG_KS = (1, 3, 5)
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ranking_split(path: str, name: str):
    """Load a ranking CSV, validate schema and qid ordering, return X, y, qid, df."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{name}] File not found: {path}")

    df = pd.read_csv(path)

    required = [QID_COL, ID_COL, TARGET_COL] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError(f"[{name}] Dataset contains null values in required columns.")

    # XGBoost requires rows of the same query group to be contiguous and qid
    # to be non-decreasing. We verify this instead of silently re-sorting/shuffling.
    if not df[QID_COL].is_monotonic_increasing:
        raise ValueError(
            f"[{name}] Rows are NOT sorted ascending by '{QID_COL}'. "
            "Re-generate the dataset or sort it before training."
        )

    X = df[FEATURE_COLS].astype(np.float32)
    y = df[TARGET_COL].astype(np.int32)
    qid = df[QID_COL].astype(np.int64)

    print(
        f"[{name}] rows={len(df):,} | requests={qid.nunique():,} | "
        f"avg candidates/request={len(df) / qid.nunique():.2f} | sorted by {QID_COL}: OK"
    )
    return X, y, qid, df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=np.float64)[:k]
    if rel.size == 0:
        return 0.0
    gains = np.power(2.0, rel) - 1.0  # exponential gain (same as XGBoost rank:ndcg)
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true_sorted_by_pred: np.ndarray, k: int) -> float:
    """NDCG@k for one group. Groups with no relevant items score 1.0 (XGBoost convention)."""
    ideal = _dcg_at_k(np.sort(y_true_sorted_by_pred)[::-1], k)
    if ideal == 0.0:
        return 1.0
    return _dcg_at_k(y_true_sorted_by_pred, k) / ideal


def evaluate_ranking(eval_df: pd.DataFrame, scores: np.ndarray) -> dict:
    """Compute grouped NDCG@k, Top-1 accuracy and anti-exploitation safety rate."""
    df = eval_df[[QID_COL, ID_COL, TARGET_COL, "price_ratio"]].copy()
    df["ai_score"] = scores

    ndcg = {k: [] for k in NDCG_KS}
    top1_is_best = []                # top-1 has label 3 (over all requests)
    top1_is_best_when_possible = []  # same, restricted to requests that contain a label-3 provider
    top1_safe = []                   # top-1 is not exploitative (over all requests)
    top1_safe_when_possible = []     # same, restricted to requests with >=1 fair candidate
    violations = []

    for req_id, group in df.groupby(QID_COL, sort=False):
        # Stable sort so ties are resolved deterministically by original order.
        ranked = group.sort_values("ai_score", ascending=False, kind="mergesort")
        labels = ranked[TARGET_COL].to_numpy()

        for k in NDCG_KS:
            ndcg[k].append(ndcg_at_k(labels, k))

        top = ranked.iloc[0]
        is_best = int(top[TARGET_COL]) == 3
        is_safe = float(top["price_ratio"]) <= EXPLOITATION_THRESHOLD

        top1_is_best.append(is_best)
        if (group[TARGET_COL] == 3).any():
            top1_is_best_when_possible.append(is_best)

        top1_safe.append(is_safe)
        if (group["price_ratio"] <= EXPLOITATION_THRESHOLD).any():
            top1_safe_when_possible.append(is_safe)

        if not is_safe:
            violations.append(
                {
                    QID_COL: req_id,
                    ID_COL: int(top[ID_COL]),
                    "price_ratio": float(top["price_ratio"]),
                    "ai_score": float(top["ai_score"]),
                }
            )

    return {
        **{f"ndcg@{k}": float(np.mean(v)) for k, v in ndcg.items()},
        "top1_accuracy": float(np.mean(top1_is_best)),
        "top1_accuracy_when_possible": float(np.mean(top1_is_best_when_possible))
        if top1_is_best_when_possible else float("nan"),
        "n_requests": len(top1_is_best),
        "n_requests_with_top_pick": len(top1_is_best_when_possible),
        "safety_rate": float(np.mean(top1_safe)),
        "safety_rate_when_possible": float(np.mean(top1_safe_when_possible))
        if top1_safe_when_possible else float("nan"),
        "n_requests_with_fair_option": len(top1_safe_when_possible),
        "violations": pd.DataFrame(violations),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_model() -> xgb.XGBRanker:
    # NOTE: Since XGBoost 2.0, `early_stopping_rounds` and `eval_metric` are
    # constructor arguments (they were removed from `.fit()`). Early stopping
    # monitors the LAST metric (ndcg@5) on the LAST eval_set entry (test set).
    return xgb.XGBRanker(
        tree_method="hist",
        objective="rank:ndcg",
        eval_metric=["ndcg@1", "ndcg@3", "ndcg@5"],
        learning_rate=0.05,
        n_estimators=300,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        early_stopping_rounds=30,
    )


def train(X_train, y_train, qid_train, X_test, y_test, qid_test) -> xgb.XGBRanker:
    model = build_model()
    model.fit(
        X_train,
        y_train,
        qid=qid_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_qid=[qid_train, qid_test],
        verbose=50,
    )
    print(
        f"\nBest iteration: {model.best_iteration} | "
        f"best validation ndcg@5: {model.best_score:.4f}"
    )
    return model


def print_feature_importance(model: xgb.XGBRanker) -> pd.DataFrame:
    booster = model.get_booster()
    rows = []
    for imp_type in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=imp_type)
        for feat in FEATURE_COLS:
            rows.append((feat, imp_type, scores.get(feat, 0.0)))
    imp = (
        pd.DataFrame(rows, columns=["feature", "type", "value"])
        .pivot(index="feature", columns="type", values="value")
    )
    imp["gain_pct"] = 100.0 * imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False)[["gain_pct", "gain", "weight", "cover"]]

    print("\n=== Feature Importance (sorted by total gain share) ===")
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(imp.to_string())
    return imp


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_ranker(model_path: str = MODEL_PATH) -> xgb.XGBRanker:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model first."
        )
    model = xgb.XGBRanker()
    model.load_model(model_path)
    return model


def rank_candidates(request_df: pd.DataFrame, model_path: str = MODEL_PATH) -> pd.DataFrame:
    """
    Score and rank candidate providers for one (or several) breakdown requests.

    Parameters
    ----------
    request_df : DataFrame containing at least FEATURE_COLS (+ optionally
                 provider_id / request_id). Each row is one candidate provider.

    Returns
    -------
    Copy of request_df with an `ai_score` column (and `ai_rank`), sorted from
    highest to lowest ai_score (within each request_id if present).
    """
    if request_df is None or len(request_df) == 0:
        raise ValueError("request_df is empty - no candidates to rank.")

    missing = [c for c in FEATURE_COLS if c not in request_df.columns]
    if missing:
        raise ValueError(f"request_df is missing required feature columns: {missing}")

    model = _load_ranker(model_path)
    X = request_df[FEATURE_COLS].astype(np.float32)

    out = request_df.copy()
    out["ai_score"] = model.predict(X)

    if QID_COL in out.columns:
        out = out.sort_values([QID_COL, "ai_score"], ascending=[True, False], kind="mergesort")
        out["ai_rank"] = out.groupby(QID_COL).cumcount() + 1
    else:
        out = out.sort_values("ai_score", ascending=False, kind="mergesort")
        out["ai_rank"] = np.arange(1, len(out) + 1)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"XGBoost version: {xgb.__version__}\n")

    X_train, y_train, qid_train, train_df = load_ranking_split(TRAIN_PATH, "train")
    X_test, y_test, qid_test, test_df = load_ranking_split(TEST_PATH, "test")

    overlap = set(qid_train.unique()) & set(qid_test.unique())
    if overlap:
        raise ValueError(f"Data leakage: {len(overlap)} request_ids appear in both splits.")

    print("\n=== Training XGBRanker ===")
    model = train(X_train, y_train, qid_train, X_test, y_test, qid_test)

    # Predictions use the best iteration automatically when early stopping is enabled.
    test_scores = model.predict(X_test)
    metrics = evaluate_ranking(test_df, test_scores)

    print("\n=== Test Set Evaluation ===")
    for k in NDCG_KS:
        print(f"Mean NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
    print(
        f"Top-1 Match Accuracy (label==3): {metrics['top1_accuracy']:.2%} "
        f"over {metrics['n_requests']} requests"
    )
    print(
        f"Top-1 Match Accuracy (only requests that contain a Top Pick): "
        f"{metrics['top1_accuracy_when_possible']:.2%} "
        f"over {metrics['n_requests_with_top_pick']} requests"
    )
    print(
        f"Anti-Exploitation Safety Rate (#1 has price_ratio <= {EXPLOITATION_THRESHOLD}): "
        f"{metrics['safety_rate']:.2%} over {metrics['n_requests']} requests"
    )
    print(
        f"Anti-Exploitation Safety Rate (only requests with a fair option): "
        f"{metrics['safety_rate_when_possible']:.2%} "
        f"over {metrics['n_requests_with_fair_option']} requests"
    )
    violations = metrics["violations"]
    if violations.empty:
        print("PASS: No exploitative provider was ranked #1.")
    else:
        print(f"WARNING: {len(violations)} request(s) ranked an exploitative provider #1:")
        print(violations.to_string(index=False))

    print_feature_importance(model)

    model.save_model(MODEL_PATH)
    _load_ranker.cache_clear()
    print(f"\nModel saved to: {MODEL_PATH}")

    # Demo: rank candidates of the first test request using the saved model.
    demo_req = test_df[QID_COL].iloc[0]
    demo_df = test_df[test_df[QID_COL] == demo_req].drop(columns=[TARGET_COL])
    ranked = rank_candidates(demo_df)
    print(f"\n=== Demo inference for request_id={demo_req} ===")
    cols = [ID_COL, "ai_rank", "ai_score", "distance_km", "rating", "price_ratio",
            "tier_difference", "mobile_workshop"]
    print(ranked[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

