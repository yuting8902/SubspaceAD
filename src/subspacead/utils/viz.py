from pathlib import Path
from typing import Optional
import logging

import cv2
import numpy as np
from PIL import Image


def _ensure_rgb(img_np: np.ndarray) -> np.ndarray:
    """Ensure a numpy image is 3-channel RGB."""
    if img_np.ndim == 2:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    if img_np.shape[-1] == 4:
        return cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    return img_np


def _create_heatmap(anom_map_norm_float: np.ndarray) -> np.ndarray:
    """
    Convert a normalized anomaly map to an RGB JET heatmap.

    cv2.applyColorMap() returns BGR. Converting to RGB before PIL saving makes
    high anomaly values appear red and low values appear blue as intended.
    """
    anom_map = np.clip(anom_map_norm_float, 0.0, 1.0)
    anom_map_u8 = np.round(anom_map * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(anom_map_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def _draw_lines(
    img_rgb: np.ndarray,
    lines: list[str],
    origin=(12, 28),
    font_scale: float = 0.62,
    thickness: int = 2,
    line_gap: int = 27,
) -> np.ndarray:
    """Draw readable white text with a dark background box."""
    out = img_rgb.copy()
    if not lines:
        return out

    font = cv2.FONT_HERSHEY_SIMPLEX
    x0, y0 = origin
    widths, heights = [], []
    for text in lines:
        (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        widths.append(w)
        heights.append(h + baseline)

    box_w = min(out.shape[1] - x0 - 4, max(widths, default=0) + 18)
    box_h = min(out.shape[0] - 4, len(lines) * line_gap + 14)
    overlay = out.copy()
    cv2.rectangle(overlay, (x0 - 7, 5), (x0 - 7 + box_w, 5 + box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.62, out, 0.38, 0)

    for idx, text in enumerate(lines):
        y = y0 + idx * line_gap
        cv2.putText(
            out,
            text,
            (x0, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out


def _add_panel_title(img_rgb: np.ndarray, title: str) -> np.ndarray:
    return _draw_lines(img_rgb, [title], origin=(12, 28), font_scale=0.68, thickness=2)


def save_overlay_for_intro(
    path: str,
    img: Image.Image,
    anom_map: np.ndarray,
    outdir: str,
    category: str,
    gt_label: Optional[int] = None,
    pred_label: Optional[int] = None,
    result_type: Optional[str] = None,
    anomaly_score: Optional[float] = None,
    threshold: Optional[float] = None,
    kernel_size: int = 5,
    overlay_intensity: float = 0.4,
) -> str:
    """
    Save one horizontal three-panel image: Original | Overlay | Heatmap.

    The overlay preserves the author's Otsu + morphology idea, while the heatmap
    is converted from OpenCV BGR to RGB before saving. Returns the saved path.
    """
    img_h, img_w = anom_map.shape
    img_np = np.array(img.resize((img_w, img_h)))
    img_rgb = _ensure_rgb(img_np)

    anom_map_clipped = np.clip(anom_map, 0.0, 1.0)
    anom_map_u8 = np.round(anom_map_clipped * 255.0).astype(np.uint8)
    heatmap_rgb = _create_heatmap(anom_map_clipped)

    try:
        _, binary_mask = cv2.threshold(
            anom_map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    except cv2.error:
        binary_mask = np.zeros_like(anom_map_u8)

    kernel_size = max(1, int(kernel_size))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    denoised_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    denoised_mask = cv2.dilate(denoised_mask, kernel, iterations=1)

    blended = cv2.addWeighted(
        img_rgb, 1.0 - overlay_intensity, heatmap_rgb, overlay_intensity, 0
    )
    mask_3d = np.repeat((denoised_mask > 0)[..., None], 3, axis=2)
    overlay_rgb = np.where(mask_3d, blended, img_rgb)

    gt_text = "N/A" if gt_label is None else ("Anomaly" if int(gt_label) == 1 else "Normal")
    pred_text = "N/A" if pred_label is None else ("Anomaly" if int(pred_label) == 1 else "Normal")
    score_text = "N/A" if anomaly_score is None else f"{float(anomaly_score):.6f}"
    thr_text = "N/A" if threshold is None or not np.isfinite(threshold) else f"{float(threshold):.6f}"

    original_panel = _add_panel_title(img_rgb, "Original")
    overlay_panel = _draw_lines(
        overlay_rgb,
        [
            "Overlay",
            f"GT: {gt_text}",
            f"Pred: {pred_text}",
            f"Result: {result_type or 'N/A'}",
            f"Score: {score_text}",
            f"Threshold: {thr_text}",
        ],
        origin=(12, 28),
        font_scale=0.58,
        thickness=2,
        line_gap=26,
    )
    heatmap_panel = _add_panel_title(heatmap_rgb, "Heatmap")
    combined = np.hstack([original_panel, overlay_panel, heatmap_panel])

    result_dir = result_type if result_type else "unclassified"
    vis_dir = Path(outdir) / "intro_overlays" / category / result_dir
    vis_dir.mkdir(parents=True, exist_ok=True)

    p = Path(path)
    unique_filename = f"{p.parent.name}_{p.name}"
    out_path = vis_dir / unique_filename
    Image.fromarray(combined).save(out_path)
    return str(out_path.resolve())


def save_visualization(
    path: str,
    img: Image.Image,
    gt_mask: np.ndarray,
    anom_map: np.ndarray,
    outdir: str,
    category: str,
    vis_idx: int,
    saliency_mask: Optional[np.ndarray] = None,
):
    """Keep the author's 2x2 debug visualization, with correct RGB heatmap colors."""
    target_shape = (anom_map.shape[1], anom_map.shape[0])
    target_shape_hw = (anom_map.shape[0], anom_map.shape[1])
    img_np = np.array(img.resize(target_shape))
    img_np_rgb = _ensure_rgb(img_np)

    heatmap = _create_heatmap(anom_map)
    if gt_mask.shape != target_shape_hw:
        logging.warning(
            f"GT shape {gt_mask.shape} != Anom map shape {target_shape_hw}. Resizing GT."
        )
        gt_mask = cv2.resize(
            gt_mask.astype(np.uint8), target_shape, interpolation=cv2.INTER_NEAREST
        )
    gt_mask_vis = _ensure_rgb((gt_mask * 255).astype(np.uint8))

    panel1 = _add_panel_title(img_np_rgb, "Original")
    panel2 = _add_panel_title(gt_mask_vis, "Ground Truth")
    panel3 = _add_panel_title(heatmap, "Anomaly Map")

    if saliency_mask is not None:
        saliency_mask_u8 = np.clip(saliency_mask, 0.0, 1.0)
        saliency_mask_u8 = np.round(saliency_mask_u8 * 255.0).astype(np.uint8)
        saliency_mask_vis = _ensure_rgb(saliency_mask_u8)
        panel4 = _add_panel_title(saliency_mask_vis, "Saliency Mask (FG)")
    else:
        overlay = cv2.addWeighted(img_np_rgb, 0.6, heatmap, 0.4, 0)
        panel4 = _add_panel_title(overlay, "Overlay")

    combined_img = np.vstack([np.hstack([panel1, panel2]), np.hstack([panel3, panel4])])
    vis_dir = Path(outdir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_path = vis_dir / f"{category}_example_{vis_idx}.png"
    Image.fromarray(combined_img).save(out_path)
    return str(out_path.resolve())