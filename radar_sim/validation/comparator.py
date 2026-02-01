"""Compare real vs synthetic PPI data and produce a validation report."""
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .geometry_extractor import (
    detect_blobs, intensity_histogram, clutter_statistics, estimate_noise_floor
)


@dataclass
class ValidationReport:
    """Result of comparing real vs synthetic radar data."""
    overall_score: float = 0.0
    blob_score: float = 0.0
    intensity_score: float = 0.0
    clutter_score: float = 0.0
    noise_score: float = 0.0
    sparsity_score: float = 0.0
    details: Dict = field(default_factory=dict)


def compare(real_ppi, synthetic_ppi) -> ValidationReport:
    """Compare real and synthetic PPI arrays.

    Both arrays should be (360, num_bins) numpy float arrays.

    Returns:
        ValidationReport with scores 0-1 (higher = more similar).
    """
    report = ValidationReport()
    if not HAS_NUMPY:
        report.details['error'] = 'numpy not available'
        return report

    # --- Blob size distribution (Earth Mover's Distance approximation) ---
    real_blobs = detect_blobs(real_ppi)
    synth_blobs = detect_blobs(synthetic_ppi)

    real_sizes = sorted([b['area'] for b in real_blobs]) if real_blobs else [0]
    synth_sizes = sorted([b['area'] for b in synth_blobs]) if synth_blobs else [0]

    report.blob_score = _blob_distribution_score(real_sizes, synth_sizes)
    report.details['real_blob_count'] = len(real_blobs)
    report.details['synth_blob_count'] = len(synth_blobs)

    # --- Intensity histogram (KL divergence) ---
    real_hist, real_edges = intensity_histogram(real_ppi)
    synth_hist, synth_edges = intensity_histogram(synthetic_ppi)

    report.intensity_score = _histogram_similarity(real_hist, synth_hist)

    # --- Clutter statistics ---
    real_clutter = clutter_statistics(real_ppi)
    synth_clutter = clutter_statistics(synthetic_ppi)

    mean_diff = abs(real_clutter['mean'] - synth_clutter['mean'])
    std_diff = abs(real_clutter['std'] - synth_clutter['std'])
    report.clutter_score = max(0.0, 1.0 - (mean_diff + std_diff) * 5.0)
    report.details['real_clutter'] = real_clutter
    report.details['synth_clutter'] = synth_clutter

    # --- Noise floor ---
    real_noise = estimate_noise_floor(real_ppi)
    synth_noise = estimate_noise_floor(synthetic_ppi)
    noise_diff = abs(real_noise - synth_noise)
    report.noise_score = max(0.0, 1.0 - noise_diff * 20.0)
    report.details['real_noise_floor'] = real_noise
    report.details['synth_noise_floor'] = synth_noise

    # --- Sparsity ---
    sparsity_diff = abs(real_clutter['sparsity'] - synth_clutter['sparsity'])
    report.sparsity_score = max(0.0, 1.0 - sparsity_diff * 5.0)

    # --- Overall ---
    report.overall_score = (
        0.25 * report.blob_score +
        0.25 * report.intensity_score +
        0.20 * report.clutter_score +
        0.15 * report.noise_score +
        0.15 * report.sparsity_score
    )

    return report


def _blob_distribution_score(real_sizes, synth_sizes) -> float:
    """Score similarity of two blob size distributions using EMD approximation."""
    if not real_sizes and not synth_sizes:
        return 1.0
    if not real_sizes or not synth_sizes:
        return 0.0

    # Pad shorter list with zeros
    max_len = max(len(real_sizes), len(synth_sizes))
    r = real_sizes + [0] * (max_len - len(real_sizes))
    s = synth_sizes + [0] * (max_len - len(synth_sizes))

    # Simple EMD: sort both, sum absolute differences, normalize
    r.sort()
    s.sort()
    total_diff = sum(abs(a - b) for a, b in zip(r, s))
    max_possible = sum(max(a, b) for a, b in zip(r, s))

    if max_possible == 0:
        return 1.0
    return max(0.0, 1.0 - total_diff / max_possible)


def _histogram_similarity(hist_a, hist_b) -> float:
    """Score histogram similarity using symmetric KL divergence."""
    if not HAS_NUMPY:
        return 0.0

    a = np.array(hist_a, dtype=float)
    b = np.array(hist_b, dtype=float)

    # Normalize to distributions
    sa = a.sum()
    sb = b.sum()
    if sa == 0 or sb == 0:
        return 0.5 if sa == 0 and sb == 0 else 0.0

    a = a / sa + 1e-10
    b = b / sb + 1e-10

    # Symmetric KL divergence
    kl_ab = float(np.sum(a * np.log(a / b)))
    kl_ba = float(np.sum(b * np.log(b / a)))
    sym_kl = (kl_ab + kl_ba) / 2.0

    # Convert to 0-1 score (KL=0 → 1.0, KL=5+ → ~0)
    return max(0.0, 1.0 - sym_kl / 5.0)
