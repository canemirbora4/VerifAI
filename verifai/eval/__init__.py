"""
Evaluation Module
==================

Tools for evaluating detection performance:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Calibration metrics (ECE, reliability diagrams)
- Corruption harness for robustness testing (image + video)
- Benchmark runner with auto-generated reports
"""

from verifai.eval.metrics import (
    compute_metrics,
    compute_binary_metrics,
    compute_calibration_error,
    find_optimal_threshold,
    MetricsResult,
)
from verifai.eval.corruptions import (
    # Image corruptions
    ImageCorruptor,
    CorruptionType,
    CorruptionConfig,
    CorruptionResult,
    apply_jpeg_compression,
    apply_resize,
    apply_blur,
    # Video corruptions
    VideoCorruptor,
    VideoCorruptionType,
    VideoCorruptionConfig,
    VideoCorruptionResult,
)
from verifai.eval.benchmark import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    RobustnessResult,
    run_quick_benchmark,
)

__all__ = [
    # Metrics
    "compute_metrics",
    "compute_binary_metrics",
    "compute_calibration_error",
    "find_optimal_threshold",
    "MetricsResult",
    # Image Corruptions
    "ImageCorruptor",
    "CorruptionType",
    "CorruptionConfig",
    "CorruptionResult",
    "apply_jpeg_compression",
    "apply_resize",
    "apply_blur",
    # Video Corruptions
    "VideoCorruptor",
    "VideoCorruptionType",
    "VideoCorruptionConfig",
    "VideoCorruptionResult",
    # Benchmark
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "RobustnessResult",
    "run_quick_benchmark",
]
