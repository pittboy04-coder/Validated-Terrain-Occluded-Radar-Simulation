"""Signal processing functions: log compression, quantization, STC."""
import math
from typing import List


def log_compress(sweep_linear: List[float], dynamic_range_db: float = 80.0) -> List[float]:
    """Log-compress a sweep from linear power to normalized 0-1 dB domain.

    Args:
        sweep_linear: Sweep data in linear power domain.
        dynamic_range_db: Display dynamic range in dB.

    Returns:
        Normalized 0-1 values in dB domain.
    """
    floor = 10.0 ** (-dynamic_range_db / 10.0)
    result = []
    for val in sweep_linear:
        clamped = max(val, floor)
        db = 10.0 * math.log10(clamped)
        # Normalize: 0 dB -> 1.0, -dynamic_range_db -> 0.0
        normalized = (db + dynamic_range_db) / dynamic_range_db
        result.append(max(0.0, min(1.0, normalized)))
    return result


def quantize(sweep: List[float], bits: int = 8) -> List[float]:
    """Quantize sweep to discrete levels matching hardware output.

    Args:
        sweep: Normalized 0-1 sweep data.
        bits: Number of output bits.

    Returns:
        Quantized sweep values still in 0-1 range.
    """
    levels = (1 << bits) - 1
    inv = 1.0 / levels
    return [round(max(0.0, min(1.0, val)) * levels) * inv for val in sweep]


def apply_stc(sweep_db: List[float], stc_curve: List[float],
              max_range_m: float) -> List[float]:
    """Apply Sensitivity Time Control (near-range attenuation).

    Args:
        sweep_db: Sweep data (linear power domain, pre-log-compression).
        stc_curve: Attenuation curve (0-1) per range bin, 1 = full suppression.
        max_range_m: Maximum range (unused, kept for interface consistency).

    Returns:
        Sweep with STC applied.
    """
    n = len(sweep_db)
    nc = len(stc_curve)
    result = []
    for i in range(n):
        if i < nc:
            atten = 1.0 - stc_curve[i]
            result.append(sweep_db[i] * atten)
        else:
            result.append(sweep_db[i])
    return result
