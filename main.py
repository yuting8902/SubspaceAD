import logging
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from src.subspacead.config import get_args, parse_grouped_layers, parse_layer_indices
from src.subspacead.core.extractor import FeatureExtractor
from src.subspacead.core.anomalyvfm_extractor import AnomalyVFMFeatureExtractor
from src.subspacead.core.patching import get_patch_coords, process_image_patched
from src.subspacead.core.pca import KernelPCAModel, PCAModel
from src.subspacead.data.datasets import get_dataset_handler
from src.subspacead.data.transforms import get_augmentation_transform
from src.subspacead.post_process.scoring import calculate_anomaly_scores, post_process_map
from src.subspacead.post_process.denoise import (
    local_contrast_denoise,
    spatial_coherence_suppress,
)
from src.subspacead.post_process.specular import filter_specular_anomalies, specular_mask_torch
from src.subspacead.utils.common import min_max_norm, save_config, setup_logging
from src.subspacead.utils.viz import save_overlay_for_intro, save_visualization


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


CATEGORY_RESULT_COLUMNS = [
    "category",
    "pca_dim",
    "threshold",
    "TP",
    "TN",
    "FP",
    "FN",
    "fpr",
    "fnr",
    "Image_AUROC",
    "Image_AUPR",
    "Image_F1",
    "Avg_Inference_time",
]

DATASET_COUNT_COLUMNS = [
    "category",
    "train_good_count",
    "val_good_count",
    "val_bad_count",
    "val_total_count",
    "test_good_count",
    "test_bad_count",
    "test_total_count",
]

IMAGE_RESULT_COLUMNS = [
    "category",
    "image_path",
    "defect_type",
    "gt_label",
    "pred_label",
    "result_type",
    "anomaly_score",
    "threshold",
    "inference_time",
    "heatmap_path",
]

VALIDATION_RESULT_COLUMNS = [
    "category",
    "image_path",
    "pca_dim",
    "gt_label",
    "anomaly_score",
    "threshold",
    "pred_label",
    "result_type",
]


def _best_f1_threshold_from_scores(y_true, y_score):
    """Return threshold maximizing positive-class F1 on validation scores."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or y_score.size == 0 or y_true.max() == y_true.min():
        return None, 0.0
    p, r, t = precision_recall_curve(y_true, y_score)
    if t.size == 0:
        return None, 0.0
    f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])


def _quantile_threshold_from_negatives(y_true, y_score, target_fpr=0.05):
    """For normal-only validation, use the (1-target_fpr) normal-score quantile."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    neg = y_score[y_true == 0]
    if neg.size == 0:
        return None
    q = np.clip(1.0 - float(target_fpr), 0.0, 1.0)
    # method= is supported by modern NumPy; interpolation= keeps compatibility with older versions.
    try:
        return float(np.quantile(neg, q, method="linear"))
    except TypeError:
        return float(np.quantile(neg, q, interpolation="linear"))


def _classification_stats_from_scores(y_true, y_score, threshold):
    """Return image-level classification statistics at a fixed threshold."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or y_score.size == 0 or threshold is None:
        return {}
    if y_true.size != y_score.size:
        raise ValueError("y_true and y_score must have the same length.")

    y_pred = y_score >= float(threshold)
    pos = y_true == 1
    neg = y_true == 0

    tp = int(np.sum(pos & y_pred))
    tn = int(np.sum(neg & ~y_pred))
    fp = int(np.sum(neg & y_pred))
    fn = int(np.sum(pos & ~y_pred))

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    fpr = _safe_ratio(fp, fp + tn)
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if np.isfinite(precision)
        and np.isfinite(recall)
        and (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
    }


def _fpr_constrained_threshold_from_scores(y_true, y_score, target_fpr=0.05):
    """
    Maximize anomaly recall subject to empirical validation FPR <= target_fpr.

    Ties are resolved by preferring, in order:
      1) lower FPR,
      2) higher precision,
      3) higher (more conservative) threshold.
    """
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.size == 0 or y_score.size == 0:
        return None, {}
    if y_true.size != y_score.size:
        raise ValueError("y_true and y_score must have the same length.")
    if not np.any(y_true == 0) or not np.any(y_true == 1):
        return None, {}

    target_fpr = float(np.clip(target_fpr, 0.0, 1.0))

    # Predictions only change when the threshold crosses an observed score.
    # Include one value above the maximum so FPR=0 is always a feasible candidate.
    candidates = np.unique(y_score)
    candidates = np.append(candidates, np.nextafter(np.max(y_score), np.inf))

    best_threshold = None
    best_stats = {}
    best_key = None

    for threshold in candidates:
        stats = _classification_stats_from_scores(y_true, y_score, threshold)
        fpr = stats["fpr"]
        recall = stats["recall"]
        precision = stats["precision"]

        if not np.isfinite(fpr) or not np.isfinite(recall):
            continue
        if fpr > target_fpr + 1e-12:
            continue

        precision_key = precision if np.isfinite(precision) else -np.inf
        key = (
            recall,
            -fpr,
            precision_key,
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_stats = stats

    return best_threshold, best_stats


def _pick_threshold_with_fallback(
    y_true,
    y_score,
    target_fpr,
    threshold_policy="best_f1",
):
    """Select an image threshold, with normal-only quantile as the fallback."""
    y_true = np.asarray(y_true).astype(np.uint8)
    has_normal = np.any(y_true == 0)
    has_anomaly = np.any(y_true == 1)

    if has_normal and has_anomaly:
        if threshold_policy == "best_f1":
            threshold, _ = _best_f1_threshold_from_scores(y_true, y_score)
            if threshold is not None:
                return threshold, "best_f1"
        elif threshold_policy == "fpr_constrained":
            threshold, _ = _fpr_constrained_threshold_from_scores(
                y_true, y_score, target_fpr
            )
            if threshold is not None:
                return threshold, "fpr_constrained"
        else:
            raise ValueError(f"Unknown threshold_policy: {threshold_policy}")

    # Preserve the original normal-only behavior.
    if has_normal:
        threshold = _quantile_threshold_from_negatives(y_true, y_score, target_fpr)
        if threshold is not None:
            return threshold, "quantile"

    return None, "none"


def _topk_mean(arr, frac=0.01):
    flat = np.asarray(arr).ravel()
    k = max(1, int(len(flat) * frac))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def _aggregate_image_score(anomaly_map: np.ndarray, method: str) -> float:
    if method == "max":
        return float(np.max(anomaly_map))
    if method == "p99":
        return float(np.percentile(anomaly_map, 99))
    if method == "mtop5":
        return float(np.mean(np.sort(anomaly_map.flatten())[-5:]))
    if method == "mtop1p":
        return _topk_mean(anomaly_map, frac=0.01)
    return float(np.mean(anomaly_map))


def _safe_ratio(num, den):
    return float(num / den) if den > 0 else np.nan


def _nanmean_or_nan(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    return float(np.mean(arr[finite])) if finite.any() else np.nan


def _normalize_anomaly_map_for_viz(anomaly_map: np.ndarray, args) -> np.ndarray:
    """
    Visualization-only normalization.

    Default behavior is unchanged: per-image min-max normalization.

    With --fixed_heatmap_scale, every image uses the same raw anomaly-score
    interval [fixed_heatmap_vmin, fixed_heatmap_vmax]. Values below vmin are
    clipped to 0 and values above vmax are clipped to 1.

    This function affects saved heatmaps only. It does not modify image scores,
    thresholds, predictions, CSV outputs, D1, D2, PCA, or the anomaly map used
    for evaluation.
    """
    if not args.fixed_heatmap_scale:
        return min_max_norm(anomaly_map)

    vmin = float(args.fixed_heatmap_vmin)
    vmax = float(args.fixed_heatmap_vmax)
    normalized = (np.asarray(anomaly_map, dtype=np.float32) - vmin) / (vmax - vmin)
    return np.clip(normalized, 0.0, 1.0)


def _result_type(gt_label: int, pred_label: int) -> str:
    if gt_label == 1 and pred_label == 1:
        return "TP"
    if gt_label == 0 and pred_label == 0:
        return "TN"
    if gt_label == 0 and pred_label == 1:
        return "FP"
    return "FN"


def _get_pca_dim(pca_params: dict) -> float:
    """Return the actual fitted PCA subspace dimensionality."""
    k = pca_params.get("k")
    if k is not None:
        return int(k)
    if "components" in pca_params:
        return int(np.asarray(pca_params["components"]).shape[1])
    kpca = pca_params.get("kpca")
    if kpca is not None:
        for attr in ("eigenvalues_", "lambdas_"):
            values = getattr(kpca, attr, None)
            if values is not None:
                return int(len(values))
    return np.nan


def _build_run_name(args):
    if args.feature_source == "anomalyvfm":
        variant_labels = {
            "a0_original_final": "A0-original-final",
            "a1_adapted_final": "A1-adapted-final",
            "a2_adapted_middle": "A2-adapted-middle",
        }
        variant_label = variant_labels[args.anomalyvfm_variant]
        if args.anomalyvfm_variant == "a2_adapted_middle":
            run_name = (
                f"{args.dataset_name}_{variant_label}"
                f"_mean_layers{''.join(args.layers.split(','))}"
                f"_res{args.image_res}_docrop{int(args.docrop)}"
            )
        else:
            run_name = (
                f"{args.dataset_name}_{variant_label}"
                f"_res{args.image_res}_docrop{int(args.docrop)}"
            )
    else:
        run_name = (
            f"{args.dataset_name}_{args.agg_method}_layers{''.join(args.layers.split(','))}"
            f"_res{args.image_res}_docrop{int(args.docrop)}"
        )
    if args.patch_size:
        run_name += f"_patch{args.patch_size}"
    if args.use_kernel_pca:
        run_name += f"_kpca-{args.kernel_pca_kernel}"
    if args.use_specular_filter:
        run_name += "_spec-filt"
    if args.bg_mask_method:
        run_name += f"_mask-{args.bg_mask_method}_thr-{args.mask_threshold_method}"
        if args.mask_threshold_method == "percentile":
            run_name += f"{args.percentile_threshold}"
        if args.bg_mask_method == "dino_saliency":
            run_name += f"_L{args.dino_saliency_layer}"
    run_name += f"_score-{args.score_method}"
    run_name += f"_clahe{int(args.use_clahe)}"
    run_name += f"_dropk{args.drop_k}"
    if args.feature_source == "anomalyvfm":
        if args.anomalyvfm_variant == "a0_original_final":
            run_name += f"_model-{Path(args.dino_weight_path).stem}"
        else:
            run_name += f"_model-{Path(args.anomalyvfm_ckpt).stem}"
    else:
        run_name += f"_model-{Path(args.model_ckpt).name}"
    run_name += f"_pca_ev{args.pca_ev}" if args.pca_ev is not None else f"_pca_dim{args.pca_dim}"
    run_name += f"_i-score{args.img_score_agg}"
    run_name += f"_thr-{args.threshold_policy}"
    if args.threshold_policy == "fpr_constrained":
        run_name += f"-fpr{args.target_img_fpr:g}"
    if args.use_d1_denoise:
        run_name += (
            f"_D1-s{args.d1_sigma:g}"
            f"-a{args.d1_alpha:g}"
            f"-l{args.d1_lambda:g}"
        )
    if args.use_d2_denoise:
        run_name += (
            f"_D2-p{args.d2_percentile:g}"
            f"-g{args.d2_grid_size}"
            f"-l{args.d2_lambda:g}"
            f"-w{args.d2_min_weight:g}"
        )
    if args.fixed_heatmap_scale:
        run_name += (
            f"_vis-fixed-{args.fixed_heatmap_vmin:g}"
            f"-{args.fixed_heatmap_vmax:g}"
        )
    if args.k_shot is not None:
        run_name += f"_k{args.k_shot}"
        if args.aug_count > 0 and args.aug_list:
            aug_str = "".join(sorted([a[0] for a in args.aug_list]))
            run_name += f"_aug{args.aug_count}x{aug_str}"
    run_name += f"_seed{args.seed}"
    return run_name


def _build_feature_extractor(args):
    """Build H0 or Meta/AnomalyVFM extractor; downstream SubspaceAD stays unchanged."""
    if args.feature_source == "dinov2":
        return FeatureExtractor(args.model_ckpt)

    if args.feature_source != "anomalyvfm":
        raise ValueError(f"Unknown feature_source: {args.feature_source}")

    if args.image_res != 672:
        raise ValueError(
            "Meta/AnomalyVFM DINOv2 variants require --image_res 672."
        )

    if args.docrop:
        raise ValueError(
            "Meta/AnomalyVFM DINOv2 variants do not support --docrop because "
            "they use the native direct Resize(672,672) preprocessing."
        )

    if args.bg_mask_method == "dino_saliency":
        raise ValueError(
            "Meta/AnomalyVFM variants do not implement SubspaceAD "
            "DINO-attention saliency. Use --bg_mask_method pca_normality "
            "or omit --bg_mask_method."
        )

    if args.patch_size and args.bg_mask_method is not None:
        raise ValueError(
            "Meta/AnomalyVFM patching mode currently requires "
            "--bg_mask_method to be omitted. Official SubspaceAD patching "
            "routes enabled masking through DINO saliency, which this "
            "extractor intentionally does not implement."
        )

    if (
        args.anomalyvfm_variant == "a2_adapted_middle"
        and args.agg_method != "mean"
    ):
        raise ValueError(
            "A2 is defined as adapted middle layers with mean aggregation. "
            "Please use --agg_method mean."
        )

    logging.info(
        "Meta/AnomalyVFM variant selected: %s",
        args.anomalyvfm_variant,
    )

    if args.anomalyvfm_variant == "a0_original_final":
        logging.info(
            "A0: original Meta DINOv2 final normalized patch tokens; "
            "no DoRA / AnomalyVFM checkpoint."
        )
    elif args.anomalyvfm_variant == "a1_adapted_final":
        logging.info(
            "A1/H1: AnomalyVFM DoRA-adapted final x_norm_patchtokens."
        )
    elif args.anomalyvfm_variant == "a2_adapted_middle":
        logging.info(
            "A2: AnomalyVFM DoRA-adapted middle layers=%s, "
            "per-layer norm=True, mean aggregation.",
            args.layers,
        )

    return AnomalyVFMFeatureExtractor(
        anomalyvfm_root=args.anomalyvfm_root,
        anomalyvfm_ckpt=args.anomalyvfm_ckpt,
        dino_repo_path=args.dino_repo_path,
        dino_weight_path=args.dino_weight_path,
        variant=args.anomalyvfm_variant,
    )


def _apply_dino_train_mask(tokens_flat, saliency_masks_batch, args):
    """Keep the official DINO-saliency training mask behavior."""
    masks_flat = saliency_masks_batch.reshape(-1)
    try:
        if args.mask_threshold_method == "percentile":
            threshold = np.percentile(masks_flat, args.percentile_threshold * 100)
            foreground_tokens = tokens_flat[masks_flat >= threshold]
        else:
            norm_mask = cv2.normalize(
                masks_flat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            _, binary_mask = cv2.threshold(
                norm_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            foreground_tokens = tokens_flat[binary_mask.flatten() > 0]
        if foreground_tokens.shape[0] > 0:
            return foreground_tokens
        logging.warning("No foreground tokens found. Using all tokens.")
    except Exception as exc:
        logging.warning("Training mask failed: %s. Using all tokens.", exc)
    return tokens_flat


def _prepare_pca_training(
    train_paths,
    extractor,
    args,
    layers,
    grouped_layers,
    aug_transform,
):
    """Build the official-style streaming feature generator and metadata."""
    if args.patch_size:
        if args.bg_mask_method == "pca_normality":
            raise ValueError("Cannot use pca_normality mask with --patch_size.")

        temp_img = Image.open(train_paths[0]).convert("RGB")
        temp_patch = temp_img.crop((0, 0, args.patch_size, args.patch_size))
        temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
            [temp_patch],
            args.image_res,
            layers,
            args.agg_method,
            grouped_layers,
            args.docrop,
            use_clahe=args.use_clahe,
            dino_saliency_layer=args.dino_saliency_layer,
        )
        feature_dim = temp_tokens.shape[-1]
        tokens_per_patch = h_p * w_p
        aug_mult = (1 + args.aug_count) if aug_transform else 1
        total_patches = 0
        num_batches = 0
        for path in train_paths:
            img = Image.open(path).convert("RGB")
            coords = get_patch_coords(
                img.height, img.width, args.patch_size, args.patch_overlap
            )
            total_patches += len(coords) * aug_mult
            num_batches += math.ceil(len(coords) / args.batch_size) * aug_mult
        total_tokens = total_patches * tokens_per_patch

        def feature_generator():
            for path in train_paths:
                pil_img = Image.open(path).convert("RGB")
                images = [pil_img]
                if aug_transform:
                    images.extend(aug_transform(pil_img) for _ in range(args.aug_count))
                for img in images:
                    coords = get_patch_coords(
                        img.height, img.width, args.patch_size, args.patch_overlap
                    )
                    for i in range(0, len(coords), args.batch_size):
                        patch_batch = [img.crop(c) for c in coords[i : i + args.batch_size]]
                        tokens_batch, _, saliency = extractor.extract_tokens(
                            patch_batch,
                            args.image_res,
                            layers,
                            args.agg_method,
                            grouped_layers,
                            args.docrop,
                            use_clahe=args.use_clahe,
                            dino_saliency_layer=args.dino_saliency_layer,
                        )
                        tokens_flat = tokens_batch.reshape(-1, feature_dim)
                        if args.bg_mask_method == "dino_saliency":
                            tokens_flat = _apply_dino_train_mask(tokens_flat, saliency, args)
                        yield tokens_flat

        return feature_generator, feature_dim, total_tokens, num_batches, h_p, w_p

    temp_img = Image.open(train_paths[0]).convert("RGB")
    temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
        [temp_img],
        args.image_res,
        layers,
        args.agg_method,
        grouped_layers,
        args.docrop,
        use_clahe=args.use_clahe,
        dino_saliency_layer=args.dino_saliency_layer,
    )
    feature_dim = temp_tokens.shape[-1]
    aug_mult = (1 + args.aug_count) if aug_transform else 1
    total_train_images = len(train_paths) * aug_mult
    total_tokens = total_train_images * h_p * w_p
    num_batches = math.ceil(total_train_images / args.batch_size)

    def feature_generator():
        all_imgs = []
        for path in train_paths:
            pil_img = Image.open(path).convert("RGB")
            all_imgs.append(pil_img)
            if aug_transform:
                all_imgs.extend(aug_transform(pil_img) for _ in range(args.aug_count))
        for i in range(0, len(all_imgs), args.batch_size):
            img_batch = all_imgs[i : i + args.batch_size]
            tokens_batch, _, saliency = extractor.extract_tokens(
                img_batch,
                args.image_res,
                layers,
                args.agg_method,
                grouped_layers,
                args.docrop,
                use_clahe=args.use_clahe,
                dino_saliency_layer=args.dino_saliency_layer,
            )
            tokens_flat = tokens_batch.reshape(-1, feature_dim)
            if args.bg_mask_method == "dino_saliency":
                tokens_flat = _apply_dino_train_mask(tokens_flat, saliency, args)
            yield tokens_flat

    return feature_generator, feature_dim, total_tokens, num_batches, h_p, w_p


def _infer_batch_maps(
    pil_imgs,
    extractor,
    pca_params,
    args,
    layers,
    grouped_layers,
    h_p,
    w_p,
    feature_dim,
):
    """Run the official H0 feature -> PCA residual -> post-process path for one batch."""
    if args.patch_size:
        anomaly_maps_batch, saliency_maps_batch = process_image_patched(
            pil_imgs, extractor, pca_params, args, DEVICE, h_p, w_p, feature_dim
        )
        final_maps = []
        for j, anomaly_map in enumerate(anomaly_maps_batch):
            anomaly_map_final = anomaly_map
            if args.use_specular_filter:
                img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                _, _, conf = specular_mask_torch(img_tensor, tau=args.specular_tau)
                conf = torch.nn.functional.interpolate(
                    conf,
                    size=anomaly_map_final.shape,
                    mode="bilinear",
                    align_corners=False,
                )
                conf_map = conf.squeeze().cpu().numpy()
                anomaly_map_final = (
                    filter_specular_anomalies(anomaly_map_final, conf_map).cpu().numpy()
                )
            final_maps.append(anomaly_map_final)
        saliency_for_viz = (
            None if args.feature_source == "anomalyvfm" else saliency_maps_batch
        )
        return final_maps, saliency_for_viz

    tokens, (batch_h_p, batch_w_p), saliency_masks_batch = extractor.extract_tokens(
        pil_imgs,
        args.image_res,
        layers,
        args.agg_method,
        grouped_layers,
        args.docrop,
        use_clahe=args.use_clahe,
        dino_saliency_layer=args.dino_saliency_layer,
    )
    b, _, _, c = tokens.shape
    scores = calculate_anomaly_scores(
        tokens.reshape(b * batch_h_p * batch_w_p, c),
        pca_params,
        args.score_method,
        args.drop_k,
    )
    anomaly_maps = scores.reshape(b, batch_h_p, batch_w_p)

    mask_for_viz = None
    background_mask = np.zeros_like(anomaly_maps, dtype=bool)
    if args.bg_mask_method == "dino_saliency":
        mask_for_viz = saliency_masks_batch
        for j in range(b):
            saliency_map = saliency_masks_batch[j]
            try:
                if args.mask_threshold_method == "percentile":
                    threshold = np.percentile(
                        saliency_map, args.percentile_threshold * 100
                    )
                    background_mask[j] = saliency_map < threshold
                else:
                    norm_mask = cv2.normalize(
                        saliency_map,
                        None,
                        0,
                        255,
                        cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U,
                    )
                    _, binary_mask = cv2.threshold(
                        norm_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    background_mask[j] = binary_mask == 0
            except Exception as exc:
                logging.warning("DINO saliency mask failed: %s", exc)

    elif args.bg_mask_method == "pca_normality":
        threshold = 10.0
        kernel_size = 3
        border = 0.2
        grid_size = (batch_h_p, batch_w_p)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_for_viz = np.zeros_like(anomaly_maps)
        for j in range(b):
            img_features = tokens[j].reshape(-1, c)
            try:
                pca = PCA(n_components=1, svd_solver="randomized")
                first_pc = pca.fit_transform(img_features.astype(np.float32))
                mask = first_pc > threshold
                mask_2d = mask.reshape(grid_size)
                h_start, h_end = int(grid_size[0] * border), int(grid_size[0] * (1 - border))
                w_start, w_end = int(grid_size[1] * border), int(grid_size[1] * (1 - border))
                center = mask_2d[h_start:h_end, w_start:w_end]
                if center.sum() <= center.size * 0.35:
                    mask_2d = (-first_pc > threshold).reshape(grid_size)
                mask_processed = cv2.dilate(mask_2d.astype(np.uint8), kernel).astype(bool)
                mask_processed = cv2.morphologyEx(
                    mask_processed.astype(np.uint8), cv2.MORPH_CLOSE, kernel
                ).astype(bool)
                background_mask[j] = ~mask_processed
                mask_for_viz[j] = mask_processed.astype(np.float32)
            except Exception as exc:
                logging.warning("PCA normality mask failed: %s", exc)

    anomaly_maps[background_mask] = 0.0

    if args.use_d1_denoise:
        anomaly_maps = np.stack(
            [
                local_contrast_denoise(
                    anomaly_map,
                    sigma=args.d1_sigma,
                    alpha=args.d1_alpha,
                    blend=args.d1_lambda,
                )
                for anomaly_map in anomaly_maps
            ],
            axis=0,
        )

    if args.use_d2_denoise:
        anomaly_maps = np.stack(
            [
                spatial_coherence_suppress(
                    anomaly_map,
                    percentile=args.d2_percentile,
                    grid_size=args.d2_grid_size,
                    strength=args.d2_lambda,
                    min_weight=args.d2_min_weight,
                )
                for anomaly_map in anomaly_maps
            ],
            axis=0,
        )

    final_maps = []
    for j in range(b):
        anomaly_map_final = post_process_map(anomaly_maps[j], args.image_res)
        if args.use_specular_filter:
            img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
            _, _, conf = specular_mask_torch(img_tensor, tau=args.specular_tau)
            conf = torch.nn.functional.interpolate(
                conf,
                size=anomaly_map_final.shape,
                mode="bilinear",
                align_corners=False,
            )
            conf_map = conf.squeeze().cpu().numpy()
            anomaly_map_final = (
                filter_specular_anomalies(anomaly_map_final, conf_map).cpu().numpy()
            )
        final_maps.append(anomaly_map_final)
    return final_maps, mask_for_viz


def _write_csvs(
    outdir,
    category_results,
    dataset_counts,
    image_results,
    validation_results,
    final=False,
):
    """Persist progress after every category; append summary rows only at the end."""
    category_df = pd.DataFrame(category_results, columns=CATEGORY_RESULT_COLUMNS)
    counts_df = pd.DataFrame(dataset_counts, columns=DATASET_COUNT_COLUMNS)
    image_df = pd.DataFrame(image_results, columns=IMAGE_RESULT_COLUMNS)
    validation_df = pd.DataFrame(
        validation_results, columns=VALIDATION_RESULT_COLUMNS
    )

    if final and not category_df.empty:
        test_counts = counts_df.set_index("category")["test_total_count"].to_dict()
        weights = np.array(
            [float(test_counts.get(cat, 0)) for cat in category_df["category"]],
            dtype=np.float64,
        )
        times = category_df["Avg_Inference_time"].to_numpy(dtype=np.float64)
        valid_time = np.isfinite(times) & (weights > 0)
        weighted_time = (
            float(np.sum(times[valid_time] * weights[valid_time]) / np.sum(weights[valid_time]))
            if valid_time.any()
            else np.nan
        )

        summary = {
            "category": "MacroAvg",
            "pca_dim": _nanmean_or_nan(category_df["pca_dim"].to_numpy(dtype=float)),
            "threshold": _nanmean_or_nan(category_df["threshold"].to_numpy(dtype=float)),
            "TP": int(category_df["TP"].sum()),
            "TN": int(category_df["TN"].sum()),
            "FP": int(category_df["FP"].sum()),
            "FN": int(category_df["FN"].sum()),
            "fpr": _nanmean_or_nan(category_df["fpr"].to_numpy(dtype=float)),
            "fnr": _nanmean_or_nan(category_df["fnr"].to_numpy(dtype=float)),
            "Image_AUROC": _nanmean_or_nan(category_df["Image_AUROC"].to_numpy(dtype=float)),
            "Image_AUPR": _nanmean_or_nan(category_df["Image_AUPR"].to_numpy(dtype=float)),
            "Image_F1": _nanmean_or_nan(category_df["Image_F1"].to_numpy(dtype=float)),
            "Avg_Inference_time": weighted_time,
        }
        category_df = pd.concat([category_df, pd.DataFrame([summary])], ignore_index=True)

    if final and not counts_df.empty:
        counts_summary = {"category": "ALL"}
        for col in DATASET_COUNT_COLUMNS[1:]:
            counts_summary[col] = int(counts_df[col].sum())
        counts_df = pd.concat([counts_df, pd.DataFrame([counts_summary])], ignore_index=True)

    category_df.to_csv(
        os.path.join(outdir, "category_results.csv"), index=False, float_format="%.8f"
    )
    counts_df.to_csv(os.path.join(outdir, "dataset_counts.csv"), index=False)
    image_df.to_csv(
        os.path.join(outdir, "image_results.csv"), index=False, float_format="%.8f"
    )
    validation_df.to_csv(
        os.path.join(outdir, "validation_results.csv"),
        index=False,
        float_format="%.8f",
    )
    return category_df, counts_df, image_df, validation_df


def main():
    args = get_args()

    if args.fixed_heatmap_scale:
        if args.fixed_heatmap_vmax is None:
            raise ValueError(
                "--fixed_heatmap_vmax is required when "
                "--fixed_heatmap_scale is enabled."
            )
        if not np.isfinite(args.fixed_heatmap_vmin):
            raise ValueError("--fixed_heatmap_vmin must be finite.")
        if not np.isfinite(args.fixed_heatmap_vmax):
            raise ValueError("--fixed_heatmap_vmax must be finite.")
        if args.fixed_heatmap_vmax <= args.fixed_heatmap_vmin:
            raise ValueError(
                "--fixed_heatmap_vmax must be greater than "
                "--fixed_heatmap_vmin."
            )

    if args.use_d1_denoise:
        if args.patch_size:
            raise ValueError(
                "D1 local-contrast denoising is defined on the native raw patch "
                "anomaly map and currently does not support --patch_size."
            )
        if args.d1_sigma <= 0:
            raise ValueError("--d1_sigma must be > 0.")
        if not 0.0 <= args.d1_alpha <= 1.0:
            raise ValueError("--d1_alpha must be in [0, 1].")
        if not 0.0 <= args.d1_lambda <= 1.0:
            raise ValueError("--d1_lambda must be in [0, 1].")

    if args.use_d2_denoise:
        if args.patch_size:
            raise ValueError(
                "D2 spatial-coherence suppression is defined on the native raw "
                "patch anomaly map and currently does not support --patch_size."
            )
        if not 0.0 <= args.d2_percentile < 100.0:
            raise ValueError("--d2_percentile must be in [0, 100).")
        if args.d2_grid_size < 2:
            raise ValueError("--d2_grid_size must be >= 2.")
        if not 0.0 <= args.d2_lambda <= 1.0:
            raise ValueError("--d2_lambda must be in [0, 1].")
        if not 0.0 <= args.d2_min_weight <= 1.0:
            raise ValueError("--d2_min_weight must be in [0, 1].")

    if args.seed is None:
        print("No seed specified; aborting for reproducibility.")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.outdir = os.path.join(args.outdir, _build_run_name(args))
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir, not args.no_log_file)
    save_config(args)

    layers = parse_layer_indices(args.layers)
    grouped_layers = (
        parse_grouped_layers(args.grouped_layers) if args.agg_method == "group" else []
    )
    extractor = _build_feature_extractor(args)

    if args.categories:
        categories = [c for c in args.categories if c != ".ipynb_checkpoints"]
    else:
        categories = sorted(
            f.name
            for f in Path(args.dataset_path).iterdir()
            if f.is_dir()
            and f.name not in {"split_csv", ".ipynb_checkpoints"}
        )

    category_results = []
    dataset_counts = []
    image_results = []
    validation_results = []

    for category in categories:
        logging.info("--- Processing Category: %s ---", category)
        handler = get_dataset_handler(args.dataset_name, args.dataset_path, category)

        train_paths_all = handler.get_train_paths()
        val_paths_all = handler.get_validation_paths()
        test_paths_all = handler.get_test_paths()

        val_labels_all = [handler.get_image_label(p) for p in val_paths_all]
        test_labels_all = [handler.get_image_label(p) for p in test_paths_all]
        dataset_counts.append(
            {
                "category": category,
                "train_good_count": len(train_paths_all),
                "val_good_count": sum(label == 0 for label in val_labels_all),
                "val_bad_count": sum(label == 1 for label in val_labels_all),
                "val_total_count": len(val_paths_all),
                "test_good_count": sum(label == 0 for label in test_labels_all),
                "test_bad_count": sum(label == 1 for label in test_labels_all),
                "test_total_count": len(test_paths_all),
            }
        )

        train_paths = list(train_paths_all)
        val_paths = list(val_paths_all)
        test_paths = list(test_paths_all)

        if args.debug_limit is not None:
            logging.warning(
                "DEBUG MODE: limiting validation/test to %d images.", args.debug_limit
            )
            val_paths = val_paths[: args.debug_limit]
            test_paths = test_paths[: args.debug_limit]

        if not train_paths:
            logging.warning("No training images found for %s. Skipping.", category)
            _write_csvs(
                args.outdir,
                category_results,
                dataset_counts,
                image_results,
                validation_results,
            )
            continue

        if args.batched_zero_shot:
            logging.info("Batched zero-shot: fitting PCA on test images.")
            train_paths = test_paths.copy()
            val_paths = []

        if args.k_shot is not None:
            if args.k_shot < len(train_paths):
                random.shuffle(train_paths)
                train_paths = train_paths[: args.k_shot]
            elif args.k_shot > len(train_paths):
                logging.warning(
                    "Requested k_shot=%d but only %d images exist; using all.",
                    args.k_shot,
                    len(train_paths),
                )
            for idx, path in enumerate(train_paths, 1):
                logging.info("K-shot image %d/%d: %s", idx, len(train_paths), Path(path).name)

        aug_transform = None
        if args.k_shot is not None and args.aug_count > 0 and args.aug_list:
            aug_transform = get_augmentation_transform(args.aug_list, args.image_res)
            if not aug_transform.transforms:
                aug_transform = None
        if category in args.no_aug_categories:
            logging.warning("Disabling augmentation for category %s", category)
            aug_transform = None

        (
            feature_generator,
            feature_dim,
            total_tokens,
            num_batches,
            h_p,
            w_p,
        ) = _prepare_pca_training(
            train_paths, extractor, args, layers, grouped_layers, aug_transform
        )
        logging.info(
            "Feature dim=%d, total PCA tokens=%d, PCA batches=%d",
            feature_dim,
            total_tokens,
            num_batches,
        )

        if args.use_kernel_pca:
            all_train_tokens = np.concatenate(
                list(tqdm(feature_generator(), desc="Feature Collection", total=num_batches))
            )
            pca_model = KernelPCAModel(
                k=args.pca_dim,
                kernel=args.kernel_pca_kernel,
                gamma=args.kernel_pca_gamma,
            )
            pca_params = pca_model.fit(all_train_tokens)
        else:
            pca_model = PCAModel(k=args.pca_dim, ev=args.pca_ev, whiten=args.whiten)
            pca_params = pca_model.fit(
                feature_generator, feature_dim, total_tokens, num_batches
            )

        actual_pca_dim = _get_pca_dim(pca_params)
        logging.info("Actual fitted PCA dimension for %s: %s", category, actual_pca_dim)

        # Validation threshold. Mixed normal+anomaly validation can use either
        # Best-F1 or FPR-constrained selection. Normal-only validation preserves
        # the original negative-quantile fallback controlled by target_img_fpr.
        # Raw validation scores are saved so score distributions can be analyzed
        # later without re-running DINOv2/PCA inference.
        thr_img = None
        if val_paths:
            val_scores, val_labels, val_score_paths = [], [], []
            for i in tqdm(range(0, len(val_paths), args.batch_size), desc=f"Validating {category}"):
                path_batch = val_paths[i : i + args.batch_size]
                pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
                maps, _ = _infer_batch_maps(
                    pil_imgs,
                    extractor,
                    pca_params,
                    args,
                    layers,
                    grouped_layers,
                    h_p,
                    w_p,
                    feature_dim,
                )
                for path, anomaly_map in zip(path_batch, maps):
                    val_score_paths.append(path)
                    val_scores.append(_aggregate_image_score(anomaly_map, args.img_score_agg))
                    val_labels.append(handler.get_image_label(path))

            thr_img, how_img = _pick_threshold_with_fallback(
                val_labels,
                val_scores,
                args.target_img_fpr,
                args.threshold_policy,
            )
            logging.info(
                "Chosen image threshold for %s: %s (%s)",
                category,
                f"{thr_img:.8f}" if thr_img is not None else "N/A",
                how_img,
            )
            if thr_img is not None:
                val_stats = _classification_stats_from_scores(
                    val_labels, val_scores, thr_img
                )
                logging.info(
                    "%s validation operating point | policy=%s | target_FPR=%.6f | "
                    "FPR=%s | Recall=%s | Precision=%s | F1=%s | "
                    "TP=%d TN=%d FP=%d FN=%d",
                    category,
                    how_img,
                    args.target_img_fpr,
                    (
                        f"{val_stats['fpr']:.6f}"
                        if np.isfinite(val_stats.get("fpr", np.nan))
                        else "N/A"
                    ),
                    (
                        f"{val_stats['recall']:.6f}"
                        if np.isfinite(val_stats.get("recall", np.nan))
                        else "N/A"
                    ),
                    (
                        f"{val_stats['precision']:.6f}"
                        if np.isfinite(val_stats.get("precision", np.nan))
                        else "N/A"
                    ),
                    (
                        f"{val_stats['f1']:.6f}"
                        if np.isfinite(val_stats.get("f1", np.nan))
                        else "N/A"
                    ),
                    int(val_stats.get("tp", 0)),
                    int(val_stats.get("tn", 0)),
                    int(val_stats.get("fp", 0)),
                    int(val_stats.get("fn", 0)),
                )

            for path, gt_label, val_score in zip(
                val_score_paths, val_labels, val_scores
            ):
                pred_label = (
                    int(val_score >= thr_img) if thr_img is not None else None
                )
                result_type = (
                    _result_type(int(gt_label), pred_label)
                    if pred_label is not None
                    else "N/A"
                )
                validation_results.append(
                    {
                        "category": category,
                        "image_path": str(Path(path).resolve()),
                        "pca_dim": actual_pca_dim,
                        "gt_label": int(gt_label),
                        "anomaly_score": float(val_score),
                        "threshold": (
                            float(thr_img) if thr_img is not None else np.nan
                        ),
                        "pred_label": (
                            pred_label if pred_label is not None else np.nan
                        ),
                        "result_type": result_type,
                    }
                )
        else:
            logging.warning("No validation images for %s; predictions/F1 unavailable.", category)

        # GPU warm-up is outside timing.
        if test_paths:
            try:
                dummy_img = [Image.open(test_paths[0]).convert("RGB")]
                maps, _ = _infer_batch_maps(
                    dummy_img,
                    extractor,
                    pca_params,
                    args,
                    layers,
                    grouped_layers,
                    h_p,
                    w_p,
                    feature_dim,
                )
                _ = _aggregate_image_score(maps[0], args.img_score_agg)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(DEVICE)
            except Exception as exc:
                logging.warning("Warm-up failed: %s", exc)

        img_true = []
        img_scores = []
        img_preds = []
        category_times = []
        vis_saved_count = 0

        for i in tqdm(range(0, len(test_paths), args.batch_size), desc=f"Testing {category}"):
            path_batch = test_paths[i : i + args.batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]

            if torch.cuda.is_available():
                torch.cuda.synchronize(DEVICE)
            start_time = time.perf_counter()

            maps, saliency_maps = _infer_batch_maps(
                pil_imgs,
                extractor,
                pca_params,
                args,
                layers,
                grouped_layers,
                h_p,
                w_p,
                feature_dim,
            )
            batch_scores = [
                _aggregate_image_score(anomaly_map, args.img_score_agg)
                for anomaly_map in maps
            ]

            if torch.cuda.is_available():
                torch.cuda.synchronize(DEVICE)
            batch_elapsed = time.perf_counter() - start_time
            per_image_time = batch_elapsed / max(1, len(path_batch))

            for j, (path, pil_img, anomaly_map, img_score) in enumerate(
                zip(path_batch, pil_imgs, maps, batch_scores)
            ):
                gt_label = int(handler.get_image_label(path))
                pred_label = int(img_score >= thr_img) if thr_img is not None else None
                result_type = (
                    _result_type(gt_label, pred_label) if pred_label is not None else "N/A"
                )

                img_true.append(gt_label)
                img_scores.append(float(img_score))
                if pred_label is not None:
                    img_preds.append(pred_label)
                category_times.append(per_image_time)

                anomaly_map_normalized = _normalize_anomaly_map_for_viz(
                    anomaly_map,
                    args,
                )
                heatmap_path = ""
                if args.save_intro_overlays:
                    heatmap_path = save_overlay_for_intro(
                        path=path,
                        img=pil_img,
                        anom_map=anomaly_map_normalized,
                        outdir=args.outdir,
                        category=category,
                        gt_label=gt_label,
                        pred_label=pred_label,
                        result_type=result_type,
                        anomaly_score=float(img_score),
                        threshold=thr_img,
                    )

                if gt_label == 1 and vis_saved_count < args.vis_count:
                    gt_mask = handler.get_ground_truth_mask(path, pil_img.size)
                    saliency_for_viz = None
                    if saliency_maps is not None:
                        try:
                            saliency_for_viz = saliency_maps[j]
                        except Exception:
                            saliency_for_viz = None
                    save_visualization(
                        path,
                        pil_img,
                        gt_mask,
                        anomaly_map_normalized,
                        args.outdir,
                        category,
                        vis_saved_count,
                        saliency_mask=saliency_for_viz,
                    )
                    vis_saved_count += 1

                image_results.append(
                    {
                        "category": category,
                        "image_path": str(Path(path).resolve()),
                        "defect_type": handler.get_defect_type(path),
                        "gt_label": gt_label,
                        "pred_label": pred_label if pred_label is not None else np.nan,
                        "result_type": result_type,
                        "anomaly_score": float(img_score),
                        "threshold": float(thr_img) if thr_img is not None else np.nan,
                        "inference_time": float(per_image_time),
                        "heatmap_path": heatmap_path,
                    }
                )

        y_true = np.asarray(img_true, dtype=np.int64)
        y_score = np.asarray(img_scores, dtype=np.float64)
        if thr_img is not None:
            y_pred = np.asarray(img_preds, dtype=np.int64)
        else:
            y_pred = np.array([], dtype=np.int64)

        if thr_img is not None and len(y_true) == len(y_pred):
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        else:
            tp = tn = fp = fn = 0

        fpr = _safe_ratio(fp, fp + tn)
        fnr = _safe_ratio(fn, fn + tp)
        img_auroc = (
            float(roc_auc_score(y_true, y_score))
            if y_true.size > 0 and len(np.unique(y_true)) > 1
            else np.nan
        )
        img_aupr = (
            float(average_precision_score(y_true, y_score))
            if y_true.size > 0 and len(np.unique(y_true)) > 1
            else np.nan
        )
        img_f1 = (
            float(f1_score(y_true, y_pred, zero_division=0))
            if thr_img is not None and (y_true == 1).any()
            else np.nan
        )
        avg_inference_time = (
            float(np.mean(category_times)) if category_times else np.nan
        )

        category_results.append(
            {
                "category": category,
                "pca_dim": actual_pca_dim,
                "threshold": float(thr_img) if thr_img is not None else np.nan,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "fpr": fpr,
                "fnr": fnr,
                "Image_AUROC": img_auroc,
                "Image_AUPR": img_aupr,
                "Image_F1": img_f1,
                "Avg_Inference_time": avg_inference_time,
            }
        )

        logging.info(
            "%s | PCA=%s | thr=%s | TP=%d TN=%d FP=%d FN=%d | "
            "FPR=%s FNR=%s | AUROC=%s AUPR=%s F1=%s | avg=%.6fs",
            category,
            actual_pca_dim,
            f"{thr_img:.6f}" if thr_img is not None else "N/A",
            tp,
            tn,
            fp,
            fn,
            f"{fpr:.4f}" if np.isfinite(fpr) else "N/A",
            f"{fnr:.4f}" if np.isfinite(fnr) else "N/A",
            f"{img_auroc:.4f}" if np.isfinite(img_auroc) else "N/A",
            f"{img_aupr:.4f}" if np.isfinite(img_aupr) else "N/A",
            f"{img_f1:.4f}" if np.isfinite(img_f1) else "N/A",
            avg_inference_time if np.isfinite(avg_inference_time) else float("nan"),
        )

        _write_csvs(
            args.outdir,
            category_results,
            dataset_counts,
            image_results,
            validation_results,
        )

    category_df, counts_df, _, validation_df = _write_csvs(
        args.outdir,
        category_results,
        dataset_counts,
        image_results,
        validation_results,
        final=True,
    )
    logging.info("\n--- Final Category Results ---\n%s", category_df.to_string(index=False, na_rep="N/A"))
    logging.info("\n--- Dataset Counts ---\n%s", counts_df.to_string(index=False))
    logging.info(
        "Saved %d validation image scores to validation_results.csv",
        len(validation_df),
    )
    logging.info("Results saved under: %s", args.outdir)


if __name__ == "__main__":
    main()
