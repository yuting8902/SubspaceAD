"""Offline score-distribution analysis for SubspaceAD H0/H1 experiments.

This script reads only CSV outputs from a completed experiment. It never loads
DINOv2, PCA weights, images, or GPU resources.

Required files in --run_dir:
    category_results.csv
    validation_results.csv
    image_results.csv

Outputs per category:
    score_histogram.png
    score_ecdf.png
    score_strip.png
    roc_curve.png       (only when test has both normal and anomaly)
    pr_curve.png        (only when test has both normal and anomaly)

Also outputs:
    all_categories_relative_score_histogram.png

The relative score used only for cross-category visualization is:
    anomaly_score / category_threshold
so the decision boundary is always 1.0. Raw model scores are not modified.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return name or "category"


def _finite(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _ecdf(values):
    x = np.sort(_finite(values))
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def _threshold_for_category(category_df, category, validation_df, test_df):
    rows = category_df[category_df["category"] == category]
    if not rows.empty:
        vals = _finite(rows["threshold"])
        if vals.size:
            return float(vals[0])

    for df in (validation_df, test_df):
        rows = df[df["category"] == category]
        if not rows.empty and "threshold" in rows.columns:
            vals = _finite(rows["threshold"])
            if vals.size:
                return float(vals[0])
    return np.nan


def _save_histogram(out_path, category, threshold, val_normal, test_normal, test_anomaly):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for values, label in (
        (val_normal, "Validation normal"),
        (test_normal, "Test normal"),
        (test_anomaly, "Test anomaly"),
    ):
        values = _finite(values)
        if values.size:
            ax.hist(values, bins="auto", alpha=0.45, density=True, label=label)
            plotted = True
    if np.isfinite(threshold):
        ax.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.6g}")
    ax.set_title(f"{category} - anomaly score distribution")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    if plotted or np.isfinite(threshold):
        ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_ecdf(out_path, category, threshold, val_normal, test_normal, test_anomaly):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for values, label in (
        (val_normal, "Validation normal"),
        (test_normal, "Test normal"),
        (test_anomaly, "Test anomaly"),
    ):
        x, y = _ecdf(values)
        if x.size:
            ax.step(x, y, where="post", label=label)
            plotted = True
    if np.isfinite(threshold):
        ax.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.6g}")
    ax.set_title(f"{category} - ECDF of anomaly scores")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Cumulative proportion")
    ax.set_ylim(0.0, 1.02)
    if plotted or np.isfinite(threshold):
        ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_strip(out_path, category, threshold, val_normal, test_normal, test_anomaly):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    rng = np.random.default_rng(0)
    groups = [
        (_finite(val_normal), 0, "Validation normal"),
        (_finite(test_normal), 1, "Test normal"),
        (_finite(test_anomaly), 2, "Test anomaly"),
    ]
    for values, y, label in groups:
        if values.size:
            jitter = rng.normal(0.0, 0.055, size=values.size)
            ax.scatter(values, np.full(values.size, y) + jitter, alpha=0.7, label=label)
    if np.isfinite(threshold):
        ax.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.6g}")
    ax.set_yticks([0, 1, 2], ["Validation normal", "Test normal", "Test anomaly"])
    ax.set_title(f"{category} - per-image anomaly scores")
    ax.set_xlabel("Anomaly score")
    ax.set_ylim(-0.5, 2.5)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_roc(out_path, category, y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return False
    fpr, tpr, _ = roc_curve(y_true, y_score)
    score = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ax.plot(fpr, tpr, label=f"AUROC = {score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_title(f"{category} - ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_pr(out_path, category, y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return False
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    ax.plot(recall, precision, label=f"AUPR = {ap:.4f}")
    prevalence = float(np.mean(y_true == 1))
    ax.axhline(prevalence, linestyle="--", label=f"Positive prevalence = {prevalence:.4f}")
    ax.set_title(f"{category} - precision-recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _append_relative(rows, source, category, scores, labels, threshold):
    if not np.isfinite(threshold) or threshold == 0:
        return
    scores = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy(dtype=float)
    labels = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(scores) & np.isfinite(labels)
    for score, label in zip(scores[keep], labels[keep]):
        rows.append(
            {
                "source": source,
                "category": category,
                "label": int(label),
                "relative_score": float(score / threshold),
            }
        )


def _save_all_category_relative_histogram(out_path, relative_df):
    if relative_df.empty:
        return False
    fig, ax = plt.subplots(figsize=(9, 5.5))
    groups = [
        (relative_df[(relative_df["source"] == "validation") & (relative_df["label"] == 0)], "Validation normal"),
        (relative_df[(relative_df["source"] == "test") & (relative_df["label"] == 0)], "Test normal"),
        (relative_df[(relative_df["source"] == "test") & (relative_df["label"] == 1)], "Test anomaly"),
    ]
    plotted = False
    for group, label in groups:
        values = _finite(group["relative_score"])
        if values.size:
            ax.hist(values, bins="auto", alpha=0.45, density=True, label=label)
            plotted = True
    ax.axvline(1.0, linestyle="--", linewidth=2, label="Normalized threshold = 1")
    ax.set_title("All categories - threshold-normalized anomaly score distribution")
    ax.set_xlabel("Relative score = anomaly_score / category_threshold")
    ax.set_ylabel("Density")
    if plotted:
        ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _read_csv(path: Path, required_columns):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Offline visualization of validation/test SubspaceAD anomaly scores."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Experiment output directory containing the three CSV files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Plot output directory. Default: <run_dir>/score_analysis",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else run_dir / "score_analysis"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    category_df = _read_csv(
        run_dir / "category_results.csv", ["category", "threshold"]
    )
    validation_df = _read_csv(
        run_dir / "validation_results.csv",
        ["category", "gt_label", "anomaly_score", "threshold"],
    )
    test_df = _read_csv(
        run_dir / "image_results.csv",
        ["category", "gt_label", "anomaly_score", "threshold"],
    )

    categories = sorted(
        set(validation_df["category"].dropna().astype(str))
        | set(test_df["category"].dropna().astype(str))
    )
    relative_rows = []

    for category in categories:
        category_outdir = outdir / _safe_name(category)
        category_outdir.mkdir(parents=True, exist_ok=True)

        val_cat = validation_df[validation_df["category"].astype(str) == category]
        test_cat = test_df[test_df["category"].astype(str) == category]
        threshold = _threshold_for_category(
            category_df, category, validation_df, test_df
        )

        val_normal = val_cat.loc[val_cat["gt_label"] == 0, "anomaly_score"]
        test_normal = test_cat.loc[test_cat["gt_label"] == 0, "anomaly_score"]
        test_anomaly = test_cat.loc[test_cat["gt_label"] == 1, "anomaly_score"]

        _save_histogram(
            category_outdir / "score_histogram.png",
            category,
            threshold,
            val_normal,
            test_normal,
            test_anomaly,
        )
        _save_ecdf(
            category_outdir / "score_ecdf.png",
            category,
            threshold,
            val_normal,
            test_normal,
            test_anomaly,
        )
        _save_strip(
            category_outdir / "score_strip.png",
            category,
            threshold,
            val_normal,
            test_normal,
            test_anomaly,
        )

        y_true = pd.to_numeric(test_cat["gt_label"], errors="coerce").to_numpy(dtype=float)
        y_score = pd.to_numeric(test_cat["anomaly_score"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y_true) & np.isfinite(y_score)
        y_true = y_true[keep].astype(int)
        y_score = y_score[keep]
        if y_true.size and len(np.unique(y_true)) > 1:
            _save_roc(category_outdir / "roc_curve.png", category, y_true, y_score)
            _save_pr(category_outdir / "pr_curve.png", category, y_true, y_score)
        else:
            print(f"[{category}] ROC/PR skipped: test set has only one class.")

        _append_relative(
            relative_rows,
            "validation",
            category,
            val_cat["anomaly_score"],
            val_cat["gt_label"],
            threshold,
        )
        _append_relative(
            relative_rows,
            "test",
            category,
            test_cat["anomaly_score"],
            test_cat["gt_label"],
            threshold,
        )

        print(
            f"[{category}] threshold={threshold:.8g} | "
            f"val_normal={len(val_normal)} | test_normal={len(test_normal)} | "
            f"test_anomaly={len(test_anomaly)}"
            if np.isfinite(threshold)
            else f"[{category}] threshold=N/A"
        )

    relative_df = pd.DataFrame(relative_rows)
    _save_all_category_relative_histogram(
        outdir / "all_categories_relative_score_histogram.png", relative_df
    )

    print(f"Score analysis saved to: {outdir}")


if __name__ == "__main__":
    main()