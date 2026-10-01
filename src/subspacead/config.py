import argparse
import os


def parse_layer_indices(arg_str: str):
    """Parse a comma-separated list of integer layer indices."""
    return [int(x.strip()) for x in arg_str.split(",")]


def parse_grouped_layers(arg_str: str):
    """Parse grouped layers such as '-1,-2:-3,-4'."""
    if not arg_str:
        return []
    return [parse_layer_indices(group) for group in arg_str.split(":")]


def _validate_wafer_roots(parser: argparse.ArgumentParser, args):
    """Validate arguments and map custom wafer roots for main.py compatibility."""
    if not 0.0 <= args.target_img_fpr <= 1.0:
        parser.error("--target_img_fpr must be in [0, 1].")

    if args.dataset_name != "wafer":
        if not args.dataset_path:
            parser.error("--dataset_path is required for non-wafer datasets.")
        return args

    if not args.train_root:
        parser.error("--train_root is required when --dataset_name wafer.")
    if not args.test_root:
        parser.error("--test_root is required when --dataset_name wafer.")

    # main.py discovers categories from args.dataset_path and passes that value to
    # get_dataset_handler(). Point it at train_root for compatibility.
    args.dataset_path = args.train_root

    # get_dataset_handler() receives one root path. Store the additional wafer
    # split roots in process-local environment variables for datasets.py.
    os.environ["SUBSPACEAD_WAFER_TRAIN_ROOT"] = os.path.abspath(args.train_root)
    os.environ["SUBSPACEAD_WAFER_TEST_ROOT"] = os.path.abspath(args.test_root)
    if args.val_root:
        os.environ["SUBSPACEAD_WAFER_VAL_ROOT"] = os.path.abspath(args.val_root)
    else:
        os.environ.pop("SUBSPACEAD_WAFER_VAL_ROOT", None)

    return args


def get_args():
    parser = argparse.ArgumentParser(
        description="SubspaceAD anomaly detection benchmark"
    )

    data_group = parser.add_argument_group("Dataset Arguments")
    model_group = parser.add_argument_group("Model & Feature Extraction Arguments")
    aug_group = parser.add_argument_group("Augmentation Arguments (for k-shot)")
    pca_group = parser.add_argument_group("Anomaly Detection (PCA) Arguments")
    score_group = parser.add_argument_group("Scoring & Evaluation Arguments")
    mask_group = parser.add_argument_group("Background Removal (Saliency) Arguments")
    specular_group = parser.add_argument_group("Specular Reflection Filter Arguments")
    log_group = parser.add_argument_group("Logistics")

    # ---------------- Dataset ----------------
    data_group.add_argument("--seed", type=int, default=42)
    data_group.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["mvtec_ad", "mvtec_loco", "mvtec_ad2", "visa", "wafer"],
    )
    data_group.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Root path for original datasets. Not needed for --dataset_name wafer.",
    )
    data_group.add_argument(
        "--train_root",
        type=str,
        default=None,
        help="Wafer training root: <train_root>/<category>/train/good/.",
    )
    data_group.add_argument(
        "--val_root",
        type=str,
        default=None,
        help=(
            "Wafer validation root: <val_root>/<category>/val/<type>/. "
            "Folders named 'good' or 'normal' are normal; all other type folders "
            "are treated as anomaly. Optional."
        ),
    )
    data_group.add_argument(
        "--test_root",
        type=str,
        default=None,
        help="Wafer test root: <test_root>/<category>/test/<type>/.",
    )
    data_group.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Categories to run. If omitted, categories are discovered from "
            "train_root/dataset_path."
        ),
    )

    # ---------------- Backbone / features ----------------
    model_group.add_argument(
        "--feature_source",
        type=str,
        default="dinov2",
        choices=["dinov2", "anomalyvfm"],
        help=(
            "Feature extractor source. 'dinov2' keeps the completed H0 Hugging "
            "Face extractor; 'anomalyvfm' uses the Meta/AnomalyVFM extractor."
        ),
    )
    model_group.add_argument(
        "--anomalyvfm_variant",
        type=str,
        default="a1_adapted_final",
        choices=[
            "a0_original_final",
            "a1_adapted_final",
            "a2_adapted_middle",
        ],
        help=(
            "Ablation used only when --feature_source anomalyvfm. "
            "a0_original_final: Meta DINOv2-L/14-Reg original final normalized "
            "patch tokens, no DoRA; "
            "a1_adapted_final: H1, AnomalyVFM DoRA-adapted final "
            "x_norm_patchtokens; "
            "a2_adapted_middle: AnomalyVFM DoRA-adapted selected middle layers, "
            "normalized per layer and mean-aggregated."
        ),
    )
    model_group.add_argument(
        "--model_ckpt",
        type=str,
        default="facebook/dinov2-with-registers-large",
        help=(
            "Hugging Face model directory/checkpoint. For offline H0, pass the "
            "local DINOv2-L/14-Reg folder."
        ),
    )
    model_group.add_argument(
        "--anomalyvfm_root",
        type=str,
        default="/workspace/Venessa/AnomalyVFM_",
        help=(
            "Local AnomalyVFM_ root. It must contain "
            "models/dinov2_offline.py and peft_local/."
        ),
    )
    model_group.add_argument(
        "--anomalyvfm_ckpt",
        type=str,
        default="/workspace/Venessa/AnomalyVFM_/pretrained_models/anomalyvfm_dinov2.pkl",
        help="Local AnomalyVFM DINOv2 trained checkpoint used by A1/A2.",
    )
    model_group.add_argument(
        "--dino_repo_path",
        type=str,
        default="/workspace/Venessa/dinov2",
        help="Local official facebookresearch/dinov2 repository used by A0/A1/A2.",
    )
    model_group.add_argument(
        "--dino_weight_path",
        type=str,
        default="/workspace/model_weight/dinov2_vitl14_reg4_pretrain.pth",
        help="Local official DINOv2 ViT-L/14-register pretrained .pth.",
    )
    model_group.add_argument("--image_res", type=int, default=256)
    model_group.add_argument("--patch_size", type=int, default=None)
    model_group.add_argument("--patch_overlap", type=float, default=0.0)
    model_group.add_argument("--batch_size", type=int, default=1)
    model_group.add_argument(
        "--k_shot",
        type=int,
        default=None,
        help=(
            "Number of normal training images to use. Omit to use all training "
            "images."
        ),
    )
    model_group.add_argument(
        "--agg_method",
        type=str,
        default="mean",
        choices=["concat", "mean", "group"],
    )
    model_group.add_argument(
        "--layers",
        type=str,
        default="-12,-13,-14,-15,-16,-17,-18",
    )
    model_group.add_argument("--grouped_layers", type=str, default=None)
    model_group.add_argument("--docrop", action="store_true")
    model_group.add_argument("--use_clahe", action="store_true")

    # ---------------- Augmentation ----------------
    aug_group.add_argument("--aug_count", type=int, default=0)
    aug_group.add_argument(
        "--aug_list",
        type=str,
        nargs="+",
        default=["rotate"],
        help=(
            "Choices supported by transforms.py include hflip, vflip, rotate, "
            "color_jitter, affine."
        ),
    )
    aug_group.add_argument(
        "--no_aug_categories",
        type=str,
        nargs="+",
        default=["transistor"],
    )

    # ---------------- PCA ----------------
    pca_group.add_argument("--pca_dim", type=int, default=None)
    pca_group.add_argument("--pca_ev", type=float, default=0.99)
    pca_group.add_argument("--whiten", action="store_true")
    pca_group.add_argument("--use_kernel_pca", action="store_true")
    pca_group.add_argument(
        "--kernel_pca_kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly", "sigmoid", "cosine"],
    )
    pca_group.add_argument("--kernel_pca_gamma", type=float, default=None)

    # ---------------- Score / evaluation ----------------
    score_group.add_argument(
        "--score_method",
        type=str,
        default="reconstruction",
        choices=["reconstruction", "mahalanobis", "cosine", "euclidean"],
    )
    score_group.add_argument("--drop_k", type=int, default=0)
    score_group.add_argument(
        "--img_score_agg",
        type=str,
        default="mtop1p",
        choices=["max", "mean", "p99", "mtop5", "mtop1p"],
    )
    score_group.add_argument("--pro_integration_limit", type=float, default=0.3)
    score_group.add_argument(
        "--threshold_policy",
        type=str,
        default="best_f1",
        choices=["normal_quantile", "best_f1", "fpr_constrained"],
        help=(
            "Image-level threshold selection policy. 'normal_quantile' uses only "
            "normal validation images and selects the (1-target_img_fpr) normal-score "
            "quantile. 'best_f1' maximizes validation F1 on mixed normal+anomaly "
            "validation. 'fpr_constrained' maximizes anomaly recall subject to "
            "validation FPR <= --target_img_fpr."
        ),
    )
    score_group.add_argument(
        "--target_img_fpr",
        type=float,
        default=0.05,
        help=(
            "Target image-level validation FPR. Used by 'normal_quantile' to choose "
            "the normal-score quantile, by 'fpr_constrained' as the FPR constraint, "
            "and as the normal-only fallback target."
        ),
    )
    score_group.add_argument(
        "--target_px_fpr",
        type=float,
        default=0.05,
        help=(
            "Pixel fallback FPR. Not meaningful for the wafer dataset because "
            "no GT masks are supplied."
        ),
    )
    score_group.add_argument(
        "--use_d1_denoise",
        action="store_true",
        help=(
            "Enable D1 local-contrast denoising on the raw patch anomaly map "
            "before the existing SubspaceAD post_process_map()."
        ),
    )
    score_group.add_argument(
        "--d1_sigma",
        type=float,
        default=4.0,
        help=(
            "D1 Gaussian sigma on the raw patch anomaly map. Larger values "
            "estimate a broader low-frequency illumination background."
        ),
    )
    score_group.add_argument(
        "--d1_alpha",
        type=float,
        default=0.5,
        help=(
            "D1 background subtraction strength in [0,1]. 0 disables "
            "subtraction; 1 subtracts the full estimated background."
        ),
    )
    score_group.add_argument(
        "--d1_lambda",
        type=float,
        default=0.5,
        help=(
            "D1 blend strength in [0,1]. 0 keeps the original anomaly map; "
            "1 uses only the local-contrast residual."
        ),
    )
    score_group.add_argument(
        "--use_d2_denoise",
        action="store_true",
        help=(
            "Enable D2 spatial-coherence soft suppression on the raw patch "
            "anomaly map. If D1 is also enabled, D2 runs after D1."
        ),
    )
    score_group.add_argument(
        "--d2_percentile",
        type=float,
        default=95.0,
        help=(
            "D2 high-score candidate percentile over positive patch scores. "
            "95 analyzes roughly the strongest 5 percent of positive scores."
        ),
    )
    score_group.add_argument(
        "--d2_grid_size",
        type=int,
        default=6,
        help=(
            "D2 spatial-entropy grid size per axis. For a 48x48 H1 map, "
            "6 creates a 6x6 analysis grid."
        ),
    )
    score_group.add_argument(
        "--d2_lambda",
        type=float,
        default=0.5,
        help=(
            "D2 suppression strength in [0,1]. Larger values more strongly "
            "reduce spatially dispersed high-score regions."
        ),
    )
    score_group.add_argument(
        "--d2_min_weight",
        type=float,
        default=0.5,
        help=(
            "D2 safety floor in [0,1]. Secondary candidate regions are never "
            "multiplied by less than this weight; the strongest region is "
            "protected even more."
        ),
    )

    # ---------------- Background mask ----------------
    mask_group.add_argument(
        "--bg_mask_method",
        type=str,
        default=None,
        choices=["dino_saliency", "pca_normality"],
    )
    mask_group.add_argument(
        "--mask_threshold_method",
        type=str,
        default="percentile",
        choices=["percentile", "otsu"],
    )
    mask_group.add_argument("--percentile_threshold", type=float, default=0.15)
    mask_group.add_argument("--dino_saliency_layer", type=int, default=6)

    # ---------------- Specular filter ----------------
    specular_group.add_argument("--use_specular_filter", action="store_true")
    specular_group.add_argument("--specular_tau", type=float, default=0.6)
    specular_group.add_argument(
        "--specular_size_threshold_factor", type=float, default=1.5
    )

    # ---------------- Output / debug ----------------
    log_group.add_argument("--outdir", type=str, default="./results_full_shot")
    log_group.add_argument("--vis_count", type=int, default=0)
    log_group.add_argument("--save_intro_overlays", action="store_true")
    log_group.add_argument(
        "--fixed_heatmap_scale",
        action="store_true",
        help=(
            "Use one fixed anomaly-score-to-color scale for all saved heatmaps "
            "instead of per-image min-max normalization."
        ),
    )
    log_group.add_argument(
        "--fixed_heatmap_vmin",
        type=float,
        default=0.0,
        help=(
            "Raw anomaly-map value mapped to heatmap value 0 (blue) when "
            "--fixed_heatmap_scale is enabled."
        ),
    )
    log_group.add_argument(
        "--fixed_heatmap_vmax",
        type=float,
        default=None,
        help=(
            "Raw anomaly-map value mapped to heatmap value 1 (red) when "
            "--fixed_heatmap_scale is enabled. Use the same value for "
            "Baseline, D1 and D2 when comparing heatmaps."
        ),
    )
    log_group.add_argument("--no_log_file", action="store_true")
    log_group.add_argument("--debug_limit", type=int, default=None)
    log_group.add_argument("--batched_zero_shot", action="store_true")

    args = parser.parse_args()
    return _validate_wafer_roots(parser, args)
