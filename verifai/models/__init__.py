"""
Models Module - Detection Models
=================================

Contains model implementations for AI-generated media detection:
- Neural detector (ViT/CNN based classifiers)
- Base classes for consistent interfaces
"""

from verifai.models.base import BaseDetector, DetectorOutput
from verifai.models.neural_detector import NeuralDetector

__all__ = [
    "BaseDetector",
    "DetectorOutput",
    "NeuralDetector",
]
