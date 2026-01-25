"""
VerifAI - AI-Generated Media Detector
======================================

A comprehensive toolkit for detecting AI-generated images and videos with:
- Calibrated confidence scores
- Localized evidence (heatmaps, suspicious frames)
- Provenance verification
- Robustness evaluation under real-world conditions

Usage:
    >>> from verifai import VerifAI
    >>> detector = VerifAI()
    >>> result = detector.detect("image.jpg")
    >>> print(result.confidence, result.label)

CLI:
    $ verifai detect image.jpg
    $ verifai detect video.mp4 --output results.json
"""

__version__ = "0.1.0"
__author__ = "VerifAI Team"

from verifai.pipeline import VerifAI, DetectionResult

__all__ = [
    "VerifAI",
    "DetectionResult",
    "__version__",
]
