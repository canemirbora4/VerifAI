"""
Fusion Module - Ensemble & Calibration
=======================================

Combines outputs from multiple detectors:
- Ensemble methods (average, weighted, learned)
- Probability calibration (isotonic, Platt, temperature)
- Explainability (heatmaps, attention visualization)

This module produces the final calibrated confidence scores.
"""

from verifai.fusion.ensemble import (
    Ensemble,
    EnsembleConfig,
    EnsembleOutput,
    FusionMethod,
    create_metadata_detector_output,
)
from verifai.fusion.calibration import (
    Calibrator,
    CalibrationResult,
    calibrate_scores,
)
from verifai.fusion.explainer import (
    Explainer,
    ExplanationResult,
    GradCAM,
    generate_heatmap,
)

__all__ = [
    # Ensemble
    "Ensemble",
    "EnsembleConfig",
    "EnsembleOutput",
    "FusionMethod",
    "create_metadata_detector_output",
    # Calibration
    "Calibrator",
    "CalibrationResult",
    "calibrate_scores",
    # Explainer
    "Explainer",
    "ExplanationResult",
    "GradCAM",
    "generate_heatmap",
]
