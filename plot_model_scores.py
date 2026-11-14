"""
SRAP - Model Score & Evaluation Visualization
Generates a 6-panel dashboard (`srap_model_scores.png`) covering:
  1. Training vs. Test NDCG@1, @3, @5 learning curves
  2. Test set evaluation metrics summary
  3. Feature importance (gain %)
  4. Predicted `ai_score` distribution by true `relevance_label`
  5. Predicted `ai_score` vs. `distance_km`
  6. Predicted `ai_score` vs. `price_ratio` (with anti-exploitation threshold)

Usage:
    python plot_model_scores.py
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from ranking_model import (
    BASE_DIR,
    EXPLOITATION_THRESHOLD,
    FEATURE_COLS,
    MODEL_PATH,
    QID_COL,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    build_model,
    evaluate_ranking,
    load_ranking_split,
)

PLOT_PATH = os.path.join(BASE_DIR, "srap_model_scores.png")

LABEL_NAMES = {
    0: "0: Exploitative / Unsuitable",
    1: "1: Acceptable Backup",
    2: "2: Strong Alternative",
    3: "3: Top Pick",
}
LABEL_COLORS = {
    0: "#d62728",
    1: "#ff7f0e",
    2: "#1f77b4",
    3: "#2ca02c",
}


def compute_learning_curves(X_train, y_train, qid_train, X_test, y_test, qid_test):
    """Fit XGBRanker silently with the exact config to capture per-iteration NDCG curves."""
    model = build_model()
    model.fit(
        X_train,
        y_train,
        qid=qid_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_qid=[qid_train, qid_test],
        verbose=False,
    )
    return model.evals_result(), int(model.best_iteration)


def plot_dashboard() -> str:
    X_train, y_train, qid_train, _ = load_ranking_split(TRAIN_PATH, "train")
    X_test, y_test, qid_test, test_df = load_ranking_split(TEST_PATH, "test")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}. Run ranking_model.py first.")

    model = xgb.XGBRanker()
    model.load_model(MODEL_PATH)

    test_scores = model.predict(X_test)
    metrics = evaluate_ranking(test_df, test_scores)

    plot_df = test_df.copy()
    plot_df["ai_score"] = test_scores
    plot_df["is_top1"] = (
        plot_df.groupby(QID_COL)["ai_score"]
        .rank(ascending=False, method="first")
        .eq(1)
    )

    evals_result, best_iter = compute_learning_curves(
        X_train, y_train, qid_train, X_test, y_test, qid_test
    )

    # Feature importance (gain %)
    gain_scores = model.get_booster().get_score(importance_type="gain")
    gains = pd.Series({f: gain_scores.get(f, 0.0) for f in FEATURE_COLS})
    gain_pct = (100.0 * gains / gains.sum()).sort_values(ascending=True)

    # ------------------------------------------------------------------
    # Build 2x3 figure
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle(
        "SRAP XGBRanker — Model Scores, Ranking Metrics & Feature Analysis",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # 1. Learning curves (Train vs Test NDCG@1, @3, @5)
    ax = axes[0, 0]
    ndcg_colors = {"ndcg@1": "#1f77b4", "ndcg@3": "#ff7f0e", "ndcg@5": "#2ca02c"}
    iterations = np.arange(len(evals_result["validation_0"]["ndcg@1"]))
    for metric_name, color in ndcg_colors.items():
        ax.plot(
            iterations,
            evals_result["validation_0"][metric_name],
            linestyle="--",
            alpha=0.65,
            color=color,
            label=f"Train {metric_name.upper()}",
        )
        ax.plot(
            iterations,
            evals_result["validation_1"][metric_name],
            linestyle="-",
            linewidth=2.0,
            color=color,
            label=f"Test {metric_name.upper()}",
        )
    ax.axvline(best_iter, color="#333333", linestyle=":", linewidth=1.6, label=f"Best iter ({best_iter})")
    ax.set_title("Learning Curves (Train vs. Test NDCG)", fontweight="bold")
    ax.set_xlabel("Boosting Iteration")
    ax.set_ylabel("NDCG Score")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=8, ncol=2, loc="lower right")

    # 2. Test Evaluation Metrics Bar Chart
    ax = axes[0, 1]
    metric_labels = [
        "NDCG@1",
        "NDCG@3",
        "NDCG@5",
        "Top-1 Match\nAccuracy",
        "Anti-Exploitation\nSafety Rate",
    ]
    metric_values = [
        metrics["ndcg@1"],
        metrics["ndcg@3"],
        metrics["ndcg@5"],
        metrics["top1_accuracy"],
        metrics["safety_rate"],
    ]
    bar_colors = ["#4c72b0", "#4c72b0", "#4c72b0", "#dd8452", "#55a868"]
    bars = ax.bar(metric_labels, metric_values, color=bar_colors, width=0.58, edgecolor="#222222", linewidth=0.7)
    for label, bar, val in zip(metric_labels, bars, metric_values):
        txt = f"{val:.2%}" if ("Accuracy" in label or "Safety" in label) else f"{val:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.015,
            txt,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_title("Test Set Evaluation Summary (200 Requests)", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.12)
    ax.tick_params(axis="x", labelsize=8.5)

    # 3. Feature Importance (Gain Share %)
    ax = axes[0, 2]
    ax.barh(gain_pct.index, gain_pct.values, color="#3b75af", edgecolor="#222222", linewidth=0.6)
    for idx, val in enumerate(gain_pct.values):
        ax.text(val + 0.6, idx, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Feature Importance (Share of Total Gain)", fontweight="bold")
    ax.set_xlabel("Gain Share (%)")
    ax.set_xlim(0, gain_pct.max() * 1.18)

    # 4. Model `ai_score` Distribution by True Relevance Label
    ax = axes[1, 0]
    labels_order = [0, 1, 2, 3]
    grouped_scores = [plot_df.loc[plot_df[TARGET_COL] == lbl, "ai_score"].to_numpy() for lbl in labels_order]
    bp = ax.boxplot(
        grouped_scores,
        positions=labels_order,
        widths=0.52,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.35},
        medianprops={"color": "black", "linewidth": 1.8},
    )
    for patch, lbl in zip(bp["boxes"], labels_order):
        patch.set_facecolor(LABEL_COLORS[lbl])
        patch.set_alpha(0.72)
    for lbl, scores in zip(labels_order, grouped_scores):
        med = np.median(scores)
        ax.text(lbl, med + 0.08, f"med={med:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title("Predicted `ai_score` by Relevance Label (Test Set)", fontweight="bold")
    ax.set_xticks(labels_order)
    ax.set_xticklabels(["0\n(Unsuitable)", "1\n(Backup)", "2\n(Strong)", "3\n(Top Pick)"], fontsize=8.5)
    ax.set_ylabel("Model `ai_score`")

    # 5. Model `ai_score` vs. Distance (km)
    ax = axes[1, 1]
    for lbl in labels_order:
        sub = plot_df[plot_df[TARGET_COL] == lbl]
        ax.scatter(
            sub["distance_km"],
            sub["ai_score"],
            s=14,
            alpha=0.50,
            color=LABEL_COLORS[lbl],
            label=LABEL_NAMES[lbl],
            edgecolors="none",
        )
    ax.set_title("Predicted `ai_score` vs. Distance (km)", fontweight="bold")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Model `ai_score`")
    ax.legend(fontsize=7.8, loc="upper right", markerscale=1.4)

    # 6. Model `ai_score` vs. Price Ratio + Anti-Exploitation Cutoff
    ax = axes[1, 2]
    for lbl in labels_order:
        sub = plot_df[plot_df[TARGET_COL] == lbl]
        ax.scatter(
            sub["price_ratio"],
            sub["ai_score"],
            s=14,
            alpha=0.45,
            color=LABEL_COLORS[lbl],
            label=LABEL_NAMES[lbl],
            edgecolors="none",
        )
    top1_df = plot_df[plot_df["is_top1"]]
    ax.scatter(
        top1_df["price_ratio"],
        top1_df["ai_score"],
        s=32,
        facecolors="none",
        edgecolors="black",
        linewidths=0.9,
        label="Model #1 Pick",
    )
    ax.axvline(
        EXPLOITATION_THRESHOLD,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Exploitation cutoff ({EXPLOITATION_THRESHOLD:.2f})",
    )
    ax.set_title("Predicted `ai_score` vs. Price Ratio", fontweight="bold")
    ax.set_xlabel("Price Ratio (quoted / fair market)")
    ax.set_ylabel("Model `ai_score`")
    ax.legend(fontsize=7.5, loc="upper right", markerscale=1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved model score plots to: {PLOT_PATH}")
    return PLOT_PATH


if __name__ == "__main__":
    plot_dashboard()
