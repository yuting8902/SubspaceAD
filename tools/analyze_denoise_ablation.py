#!/usr/bin/env python3
"""
Minimal D0 / D1 / D2 / D1+D2 ablation analysis.

Each run directory must contain:
    image_results.csv
    validation_results.csv

Example:
    python tools/analyze_denoise_ablation.py \
        --d0 /path/to/D0_run \
        --d1 /path/to/D1_run \
        --d2 /path/to/D2_run \
        --d1d2 /path/to/D1_D2_run \
        --outdir /path/to/ablation_analysis

Outputs:
    ablation_summary.csv
    ablation_per_image.csv

Important:
- Test images are grouped by the D0 (baseline) result_type.
  Therefore a D0 FP stays in the "FP" analysis group even if D2 fixes it.
- D1/D2 predictions are NOT used to redefine groups.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ("D0", "D1", "D2", "D1D2")

IMAGE_REQUIRED = {
    "category",
    "image_path",
    "defect_type",
    "gt_label",
    "pred_label",
    "result_type",
    "anomaly_score",
    "threshold",
}

VAL_REQUIRED = {
    "category",
    "image_path",
    "gt_label",
    "pred_label",
    "result_type",
    "anomaly_score",
    "threshold",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare D0, D1, D2 and D1+D2 image scores using the same images."
        )
    )
    parser.add_argument(
        "--d0",
        required=True,
        help="D0 baseline run directory containing image_results.csv and validation_results.csv.",
    )
    parser.add_argument(
        "--d1",
        required=True,
        help="D1 run directory containing image_results.csv and validation_results.csv.",
    )
    parser.add_argument(
        "--d2",
        required=True,
        help="D2 run directory containing image_results.csv and validation_results.csv.",
    )
    parser.add_argument(
        "--d1d2",
        required=True,
        help="D1+D2 run directory containing image_results.csv and validation_results.csv.",
    )
    parser.add_argument(
        "--outdir",
        default="./denoise_ablation_analysis",
        help="Directory for analysis CSV files.",
    )
    return parser.parse_args()


def _read_csv(run_dir, filename, required_columns):
    path = Path(run_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    # These are the stable identifiers written by main.py.
    df["category"] = df["category"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    return df


def _safe_ratio(num, den):
    num = pd.to_numeric(num, errors="coerce").astype(float)
    den = pd.to_numeric(den, errors="coerce").astype(float)
    out = np.full(len(num), np.nan, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[valid] = num[valid] / den[valid]
    return out


def _prepare_variant(df, prefix, is_test):
    keep = [
        "category",
        "image_path",
        "gt_label",
        "pred_label",
        "result_type",
        "anomaly_score",
        "threshold",
    ]
    if is_test:
        keep.insert(2, "defect_type")

    df = df[keep].copy()

    rename = {
        "gt_label": f"{prefix}_gt_label",
        "pred_label": f"{prefix}_pred_label",
        "result_type": f"{prefix}_result_type",
        "anomaly_score": f"{prefix}_score",
        "threshold": f"{prefix}_threshold",
    }
    if is_test:
        rename["defect_type"] = f"{prefix}_defect_type"

    return df.rename(columns=rename)


def _merge_variants(frames, is_test):
    key = ["category", "image_path"]

    merged = _prepare_variant(frames["D0"], "d0", is_test)

    for variant, prefix in (("D1", "d1"), ("D2", "d2"), ("D1D2", "d1d2")):
        current = _prepare_variant(frames[variant], prefix, is_test)
        merged = merged.merge(
            current,
            on=key,
            how="inner",
            validate="one_to_one",
        )

    # Verify labels refer to the same physical samples.
    for prefix in ("d1", "d2", "d1d2"):
        mismatch = merged[f"{prefix}_gt_label"] != merged["d0_gt_label"]
        if mismatch.any():
            bad = merged.loc[mismatch, key].head(5).to_dict("records")
            raise ValueError(
                f"GT label mismatch between D0 and {prefix.upper()}: {bad}"
            )

    if is_test:
        merged["defect_type"] = merged["d0_defect_type"].astype(str)
        merged["baseline_group"] = merged["d0_result_type"].astype(str)
    else:
        merged["defect_type"] = ""
        gt = pd.to_numeric(merged["d0_gt_label"], errors="coerce")
        merged["baseline_group"] = np.where(
            gt == 0,
            "VAL_NORMAL",
            "VAL_ANOMALY",
        )

    merged["split"] = "test" if is_test else "validation"

    # Per-variant relative score = image score / that run's threshold.
    for prefix in ("d0", "d1", "d2", "d1d2"):
        merged[f"{prefix}_relative"] = _safe_ratio(
            merged[f"{prefix}_score"],
            merged[f"{prefix}_threshold"],
        )

    # Ratios to D0 tell us whether suppression is selective.
    for prefix in ("d1", "d2", "d1d2"):
        merged[f"{prefix}_score_ratio"] = _safe_ratio(
            merged[f"{prefix}_score"],
            merged["d0_score"],
        )
        merged[f"{prefix}_relative_ratio"] = _safe_ratio(
            merged[f"{prefix}_relative"],
            merged["d0_relative"],
        )
        merged[f"{prefix}_threshold_ratio"] = _safe_ratio(
            merged[f"{prefix}_threshold"],
            merged["d0_threshold"],
        )

    return merged


def _as_float_array(values):
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _mean_finite(values):
    x = _as_float_array(values)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def _median_finite(values):
    x = _as_float_array(values)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def _predicted_anomaly_rate(values):
    x = _as_float_array(values)
    x = x[np.isfinite(x)]
    return float(np.mean(x == 1)) if x.size else np.nan


def _summary_rows_for_subset(subset, category, split, baseline_group):
    rows = []

    for variant, prefix in (
        ("D0", "d0"),
        ("D1", "d1"),
        ("D2", "d2"),
        ("D1D2", "d1d2"),
    ):
        if variant == "D0":
            score_ratio = np.ones(len(subset), dtype=np.float64)
            relative_ratio = np.ones(len(subset), dtype=np.float64)
            threshold_ratio = np.ones(len(subset), dtype=np.float64)
        else:
            score_ratio = subset[f"{prefix}_score_ratio"]
            relative_ratio = subset[f"{prefix}_relative_ratio"]
            threshold_ratio = subset[f"{prefix}_threshold_ratio"]

        rows.append(
            {
                "category": category,
                "split": split,
                "baseline_group": baseline_group,
                "variant": variant,
                "n_images": int(len(subset)),
                "mean_score": _mean_finite(subset[f"{prefix}_score"]),
                "median_score": _median_finite(subset[f"{prefix}_score"]),
                "mean_score_ratio_to_D0": _mean_finite(score_ratio),
                "median_score_ratio_to_D0": _median_finite(score_ratio),
                "mean_relative_score": _mean_finite(
                    subset[f"{prefix}_relative"]
                ),
                "median_relative_score": _median_finite(
                    subset[f"{prefix}_relative"]
                ),
                "mean_relative_ratio_to_D0": _mean_finite(relative_ratio),
                "median_relative_ratio_to_D0": _median_finite(relative_ratio),
                "predicted_anomaly_rate": _predicted_anomaly_rate(
                    subset[f"{prefix}_pred_label"]
                ),
                "mean_threshold": _mean_finite(
                    subset[f"{prefix}_threshold"]
                ),
                "mean_threshold_ratio_to_D0": _mean_finite(threshold_ratio),
            }
        )

    return rows


def _make_summary(test_df, val_df):
    rows = []

    # Per-category rows are the safest because thresholds are category-specific.
    for category in sorted(test_df["category"].unique()):
        cat_df = test_df[test_df["category"] == category]
        for group in ("TN", "FP", "TP", "FN"):
            subset = cat_df[cat_df["baseline_group"] == group]
            if len(subset):
                rows.extend(
                    _summary_rows_for_subset(
                        subset,
                        category=category,
                        split="test",
                        baseline_group=group,
                    )
                )

    for category in sorted(val_df["category"].unique()):
        cat_df = val_df[val_df["category"] == category]
        for group in ("VAL_NORMAL", "VAL_ANOMALY"):
            subset = cat_df[cat_df["baseline_group"] == group]
            if len(subset):
                rows.extend(
                    _summary_rows_for_subset(
                        subset,
                        category=category,
                        split="validation",
                        baseline_group=group,
                    )
                )

    # Overall rows are useful for a fast first look.
    for group in ("TN", "FP", "TP", "FN"):
        subset = test_df[test_df["baseline_group"] == group]
        if len(subset):
            rows.extend(
                _summary_rows_for_subset(
                    subset,
                    category="ALL",
                    split="test",
                    baseline_group=group,
                )
            )

    for group in ("VAL_NORMAL", "VAL_ANOMALY"):
        subset = val_df[val_df["baseline_group"] == group]
        if len(subset):
            rows.extend(
                _summary_rows_for_subset(
                    subset,
                    category="ALL",
                    split="validation",
                    baseline_group=group,
                )
            )

    return pd.DataFrame(rows)


def _make_per_image(test_df):
    cols = [
        "category",
        "image_path",
        "defect_type",
        "d0_gt_label",
        "baseline_group",
    ]

    for prefix in ("d0", "d1", "d2", "d1d2"):
        cols.extend(
            [
                f"{prefix}_pred_label",
                f"{prefix}_result_type",
                f"{prefix}_score",
                f"{prefix}_threshold",
                f"{prefix}_relative",
            ]
        )
        if prefix != "d0":
            cols.extend(
                [
                    f"{prefix}_score_ratio",
                    f"{prefix}_relative_ratio",
                    f"{prefix}_threshold_ratio",
                ]
            )

    out = test_df[cols].copy()

    # Put the most diagnostically useful D0 groups first.
    order = {"FP": 0, "TP": 1, "TN": 2, "FN": 3}
    out["_group_order"] = out["baseline_group"].map(order).fillna(9)

    # Within each group, strongest D2 relative change first.
    out = out.sort_values(
        ["_group_order", "category", "d2_relative_ratio", "image_path"],
        ascending=[True, True, True, True],
        na_position="last",
    ).drop(columns="_group_order")

    return out


def _print_quick_read(summary):
    print("\n=== Quick read (category=ALL where available) ===")

    overall = summary[summary["category"] == "ALL"].copy()
    wanted_groups = ["VAL_NORMAL", "FP", "TP", "TN"]

    for group in wanted_groups:
        block = overall[overall["baseline_group"] == group]
        if block.empty:
            continue

        print(f"\n[{group}]")
        show = block[
            [
                "variant",
                "n_images",
                "mean_score_ratio_to_D0",
                "mean_relative_ratio_to_D0",
                "predicted_anomaly_rate",
                "mean_threshold_ratio_to_D0",
            ]
        ]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(
        "\nInterpretation:"
        "\n  score_ratio < 1      : absolute anomaly score was suppressed."
        "\n  relative_ratio < 1   : score moved toward NORMAL even after the new threshold moved."
        "\n  FP group              : lower relative_ratio and lower predicted_anomaly_rate are better."
        "\n  TP group              : ratios near 1 and predicted_anomaly_rate near 1 are safer."
        "\n  VAL_NORMAL            : shows how strongly the threshold-driving validation normals were compressed."
    )


def main():
    args = parse_args()
    run_dirs = {
        "D0": args.d0,
        "D1": args.d1,
        "D2": args.d2,
        "D1D2": args.d1d2,
    }

    test_frames = {}
    val_frames = {}

    for variant, run_dir in run_dirs.items():
        test_frames[variant] = _read_csv(
            run_dir,
            "image_results.csv",
            IMAGE_REQUIRED,
        )
        val_frames[variant] = _read_csv(
            run_dir,
            "validation_results.csv",
            VAL_REQUIRED,
        )

    test_df = _merge_variants(test_frames, is_test=True)
    val_df = _merge_variants(val_frames, is_test=False)

    # Report matching coverage so missing/mismatched runs are not silently ignored.
    print("Matched images across all four runs:")
    print(f"  test       : {len(test_df)}")
    print(f"  validation : {len(val_df)}")

    summary = _make_summary(test_df, val_df)
    per_image = _make_per_image(test_df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "ablation_summary.csv"
    per_image_path = outdir / "ablation_per_image.csv"

    summary.to_csv(summary_path, index=False, float_format="%.6f")
    per_image.to_csv(per_image_path, index=False, float_format="%.6f")

    _print_quick_read(summary)

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {per_image_path}")


if __name__ == "__main__":
    main()
