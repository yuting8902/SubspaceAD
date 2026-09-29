import cv2
import numpy as np


def local_contrast_denoise(
    anomaly_map: np.ndarray,
    sigma: float = 4.0,
    alpha: float = 0.5,
    blend: float = 0.5,
) -> np.ndarray:
    """
    D1 local-contrast anomaly-map denoising.

    Intended insertion point:
        raw patch anomaly map (e.g. 48x48)
            -> local_contrast_denoise()
            -> existing SubspaceAD post_process_map()

    The method estimates a low-frequency anomaly background with a Gaussian blur,
    subtracts part of that background, and softly blends the residual back with
    the original anomaly map.

    Equations
    ---------
        background = GaussianBlur(A, sigma)
        residual   = max(A - alpha * background, 0)
        refined    = (1 - blend) * A + blend * residual

    Parameters
    ----------
    anomaly_map:
        2-D raw patch-level anomaly-score map.
    sigma:
        Spatial scale used to estimate the slowly varying background.
        Larger values treat broader structures as background.
    alpha:
        Background subtraction strength. Must be in [0, 1].
        0 means no background subtraction; 1 subtracts the full estimated
        background before clipping at zero.
    blend:
        Strength of D1 in the final output. Must be in [0, 1].
        0 returns the original map; 1 returns only the local-contrast residual.

    Returns
    -------
    np.ndarray
        Refined anomaly map with the same shape as the input.

    Notes
    -----
    This function intentionally does not normalize the map. Validation and test
    therefore remain on the same score definition, and SubspaceAD can continue
    to determine the image threshold from validation scores.
    """
    anomaly_map = np.asarray(anomaly_map)

    if anomaly_map.ndim != 2:
        raise ValueError(
            f"D1 expects a 2-D anomaly map, got shape {anomaly_map.shape}."
        )
    if not np.isfinite(anomaly_map).all():
        raise FloatingPointError("D1 received NaN or Inf in the anomaly map.")
    if sigma <= 0:
        raise ValueError(f"D1 sigma must be > 0, got {sigma}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"D1 alpha must be in [0, 1], got {alpha}.")
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"D1 blend/lambda must be in [0, 1], got {blend}.")

    # Work in float32 and never modify the caller's array in-place.
    raw = anomaly_map.astype(np.float32, copy=True)

    # ksize=(0, 0) lets OpenCV choose a kernel large enough for the requested
    # Gaussian sigma. BORDER_REFLECT_101 avoids introducing artificial zeros
    # along the anomaly-map boundary.
    background = cv2.GaussianBlur(
        raw,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )

    residual = np.maximum(raw - float(alpha) * background, 0.0)
    refined = (1.0 - float(blend)) * raw + float(blend) * residual

    return refined.astype(np.float32, copy=False)


def _normalized_spatial_entropy(
    candidate_energy: np.ndarray,
    grid_size: int,
) -> float:
    """
    Measure how widely candidate anomaly energy is spread over the map.

    0.0 -> concentrated in very few spatial cells.
    1.0 -> broadly distributed across many spatial cells.
    """
    h, w = candidate_energy.shape
    if grid_size < 2:
        raise ValueError(f"D2 grid_size must be >= 2, got {grid_size}.")
    if grid_size > min(h, w):
        raise ValueError(
            f"D2 grid_size={grid_size} is larger than anomaly-map shape "
            f"{candidate_energy.shape}."
        )

    row_edges = np.linspace(0, h, grid_size + 1, dtype=int)
    col_edges = np.linspace(0, w, grid_size + 1, dtype=int)

    cell_energy = []
    for r in range(grid_size):
        for c in range(grid_size):
            block = candidate_energy[
                row_edges[r] : row_edges[r + 1],
                col_edges[c] : col_edges[c + 1],
            ]
            cell_energy.append(float(block.sum()))

    cell_energy = np.asarray(cell_energy, dtype=np.float64)
    total = float(cell_energy.sum())
    if total <= 0.0:
        return 0.0

    probs = cell_energy / total
    probs = probs[probs > 0.0]
    entropy = -float(np.sum(probs * np.log(probs)))

    max_entropy = float(np.log(grid_size * grid_size))
    if max_entropy <= 0.0:
        return 0.0

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def spatial_coherence_suppress(
    anomaly_map: np.ndarray,
    percentile: float = 95.0,
    grid_size: int = 6,
    strength: float = 0.5,
    min_weight: float = 0.5,
) -> np.ndarray:
    """
    D2 spatial-coherence soft suppression.

    Intended insertion point:
        raw patch anomaly map
            -> optional D1 local_contrast_denoise()
            -> spatial_coherence_suppress()
            -> existing SubspaceAD post_process_map()

    D2 is designed for cases where false alarms appear as several spatially
    separated bright regions. It does NOT delete small components by area.

    Steps
    -----
    1. Use a high percentile of positive anomaly scores to define candidate
       high-score regions.
    2. Find 8-connected candidate components.
    3. Measure:
         - concentration: how much candidate anomaly mass is contained in the
           strongest connected component.
         - spatial entropy: how broadly candidate anomaly mass is distributed.
    4. Convert those measurements into a scatter score:
           scatter = (1 - concentration) * spatial_entropy
    5. When scatter is high, softly reduce candidate-region scores.
       The strongest component receives only half of the suppression applied
       to secondary components as a safety measure for a dominant true defect.

    Parameters
    ----------
    anomaly_map:
        2-D raw/refined patch-level anomaly-score map.
    percentile:
        Anomaly-map percentile used to define candidate high-score regions.
        95 means roughly the strongest 5% of patch scores are analyzed.
    grid_size:
        Number of spatial bins per axis used for the entropy measurement.
        For a 48x48 H1 map, 6 means a 6x6 spatial grid.
    strength:
        D2 suppression strength in [0, 1]. Larger values suppress dispersed
        candidate regions more strongly.
    min_weight:
        Safety floor in [0, 1]. No secondary candidate region is multiplied by
        less than this value. The strongest component is protected even more.

    Returns
    -------
    np.ndarray
        Refined anomaly map with the same shape as the input.

    Notes
    -----
    - A single connected high-score region is returned unchanged.
    - No component is hard-deleted.
    - Pixels outside the high-score candidate regions are unchanged.
    - The function does not normalize the map.
    """
    anomaly_map = np.asarray(anomaly_map)

    if anomaly_map.ndim != 2:
        raise ValueError(
            f"D2 expects a 2-D anomaly map, got shape {anomaly_map.shape}."
        )
    if not np.isfinite(anomaly_map).all():
        raise FloatingPointError("D2 received NaN or Inf in the anomaly map.")
    if not 0.0 <= percentile < 100.0:
        raise ValueError(
            f"D2 percentile must be in [0, 100), got {percentile}."
        )
    if grid_size < 2:
        raise ValueError(f"D2 grid_size must be >= 2, got {grid_size}.")
    if grid_size > min(anomaly_map.shape):
        raise ValueError(
            f"D2 grid_size={grid_size} is larger than anomaly-map shape "
            f"{anomaly_map.shape}."
        )
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"D2 strength/lambda must be in [0, 1], got {strength}.")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError(
            f"D2 min_weight must be in [0, 1], got {min_weight}."
        )

    raw = anomaly_map.astype(np.float32, copy=True)

    # Use the full map for the percentile so zero/low-score background remains
    # part of the reference distribution. A strict ">" avoids turning a flat
    # background plateau into one giant connected component.
    if raw.size < 2:
        return raw

    candidate_threshold = float(np.percentile(raw, percentile))
    candidate_mask = (raw > candidate_threshold) & (raw > 0.0)

    if int(candidate_mask.sum()) < 2:
        return raw

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8),
        connectivity=8,
    )
    num_components = num_labels - 1

    # One coherent region is exactly what D2 is intended to preserve.
    if num_components <= 1:
        return raw

    component_masses = np.asarray(
        [
            float(raw[labels == label_id].sum())
            for label_id in range(1, num_labels)
        ],
        dtype=np.float64,
    )

    total_mass = float(component_masses.sum())
    if total_mass <= 0.0:
        return raw

    largest_index = int(np.argmax(component_masses))
    largest_label = largest_index + 1
    largest_mass = float(component_masses[largest_index])

    # 1.0 means one region dominates all candidate anomaly evidence.
    concentration = float(np.clip(largest_mass / total_mass, 0.0, 1.0))

    candidate_energy = np.where(candidate_mask, raw, 0.0).astype(np.float32)
    spatial_entropy = _normalized_spatial_entropy(
        candidate_energy,
        grid_size=grid_size,
    )

    # High only when anomaly evidence is both non-dominant and spatially spread.
    scatter = float(
        np.clip((1.0 - concentration) * spatial_entropy, 0.0, 1.0)
    )

    if scatter <= 0.0:
        return raw

    # Global weight for secondary candidate regions.
    secondary_weight = max(
        float(min_weight),
        1.0 - float(strength) * scatter,
    )

    # Protect the strongest region: it receives only half as much suppression.
    primary_weight = 1.0 - 0.5 * (1.0 - secondary_weight)

    refined = raw.copy()
    for label_id in range(1, num_labels):
        weight = primary_weight if label_id == largest_label else secondary_weight
        refined[labels == label_id] *= float(weight)

    return refined.astype(np.float32, copy=False)

