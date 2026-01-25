"""
Models Module - Detection Models
=================================

Contains model implementations for AI-generated media detection:
- Neural detector (ViT/CNN based classifiers)
- Frequency detector (FFT/DCT based classifier)
- Base classes for consistent interfaces
"""

from verifai.models.base import BaseDetector, DetectorOutput, Label
from verifai.models.neural_detector import NeuralDetector
from verifai.models.frequency_classifier import FrequencyDetector, FrequencyMLP

__all__ = [
    "BaseDetector",
    "DetectorOutput",
    "Label",
    "NeuralDetector",
    "FrequencyDetector",
    "FrequencyMLP",
]
