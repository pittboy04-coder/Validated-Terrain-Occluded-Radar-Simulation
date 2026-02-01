"""Extract geometric and statistical features from PPI data for validation."""
import math
from typing import List, Tuple, Dict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def detect_blobs(ppi_array, threshold: float = 0.05) -> List[Dict]:
    """Detect blobs (connected components) in a 360 x N PPI array.

    Simple flood-fill connected component detection on the polar grid.

    Returns:
        List of dicts with keys: bearing_center, range_center, area, peak_intensity.
    """
    if not HAS_NUMPY:
        return []

    num_bearings, num_bins = ppi_array.shape
    visited = np.zeros_like(ppi_array, dtype=bool)
    blobs = []

    for b in range(num_bearings):
        for r in range(num_bins):
            if visited[b, r] or ppi_array[b, r] < threshold:
                continue

            # Flood fill
            stack = [(b, r)]
            component = []
            while stack:
                cb, cr = stack.pop()
                if visited[cb % num_bearings, cr]:
                    continue
                if cr < 0 or cr >= num_bins:
                    continue
                if ppi_array[cb % num_bearings, cr] < threshold:
                    continue
                visited[cb % num_bearings, cr] = True
                component.append((cb % num_bearings, cr))
                # 4-connected neighbors
                for db, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (cb + db) % num_bearings
                    nr = cr + dr
                    if 0 <= nr < num_bins and not visited[nb, nr]:
                        stack.append((nb, nr))

            if len(component) >= 2:
                bearings = [c[0] for c in component]
                ranges = [c[1] for c in component]
                peak = max(ppi_array[c[0], c[1]] for c in component)
                blobs.append({
                    'bearing_center': sum(bearings) / len(bearings),
                    'range_center': sum(ranges) / len(ranges),
                    'area': len(component),
                    'peak_intensity': float(peak),
                })

    return blobs


def intensity_histogram(ppi_array, bins: int = 50) -> Tuple:
    """Compute intensity histogram of non-zero PPI values.

    Returns:
        (hist_counts, bin_edges) as numpy arrays.
    """
    if not HAS_NUMPY:
        return ([], [])

    values = ppi_array[ppi_array > 0.001].flatten()
    if len(values) == 0:
        values = np.array([0.0])
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return hist.astype(float), edges


def clutter_statistics(ppi_array) -> Dict:
    """Compute clutter statistics from PPI array.

    Returns:
        Dict with mean, std, median, sparsity (fraction of near-zero bins).
    """
    if not HAS_NUMPY:
        return {'mean': 0, 'std': 0, 'median': 0, 'sparsity': 1.0}

    flat = ppi_array.flatten()
    nonzero = flat[flat > 0.001]

    sparsity = 1.0 - (len(nonzero) / max(1, len(flat)))
    if len(nonzero) == 0:
        return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'sparsity': sparsity}

    return {
        'mean': float(np.mean(nonzero)),
        'std': float(np.std(nonzero)),
        'median': float(np.median(nonzero)),
        'sparsity': sparsity,
    }


def estimate_noise_floor(ppi_array) -> float:
    """Estimate noise floor as the mode of low-intensity values.

    Returns:
        Estimated noise floor level (0-1).
    """
    if not HAS_NUMPY:
        return 0.0

    flat = ppi_array.flatten()
    # Look at the bottom 50% of non-zero values
    nonzero = flat[flat > 0.0001]
    if len(nonzero) == 0:
        return 0.0
    sorted_vals = np.sort(nonzero)
    # Noise floor ~ median of bottom quartile
    q25 = sorted_vals[:max(1, len(sorted_vals) // 4)]
    return float(np.median(q25))
