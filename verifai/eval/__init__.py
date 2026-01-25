"""
Evaluation Module
==================

Tools for evaluating detection performance:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Calibration metrics (ECE, reliability diagrams)
- Benchmark harness for robustness testing
"""

from verifai.eval.metrics import (
    compute_metrics,
    compute_binary_metrics,
    compute_calibration_error,
    MetricsResult,
)

__all__ = [
    "compute_metrics",
    "compute_binary_metrics",
    "compute_calibration_error",
    "MetricsResult",
]
