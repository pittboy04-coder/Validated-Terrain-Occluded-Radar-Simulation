"""Validation module for radar simulator.

Provides comprehensive comparison between real radar captures and simulator output.
"""

from .validator import (
    RadarValidator,
    ValidationResult,
    BatchResult,
    ValidationVisualizer,
    write_summary_report,
    write_batch_report,
)
from .comparator import compare, ValidationReport

__all__ = [
    'RadarValidator',
    'ValidationResult',
    'BatchResult',
    'ValidationVisualizer',
    'write_summary_report',
    'write_batch_report',
    'compare',
    'ValidationReport',
]
