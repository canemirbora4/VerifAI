"""
Models Module - Detection Models
=================================

Contains model implementations for AI-generated media detection:
- Fusion detector (CLIP + Frequency ensemble) - RECOMMENDED
- CLIP detector (CLIP ViT-L/14 frozen backbone + trainable head)
- Neural detector (ViT/CNN based classifiers)
- Frequency detector (FFT/DCT based classifier)
- Base classes for consistent interfaces
"""

from verifai.models.base import BaseDetector, DetectorOutput, Label
from verifai.models.neural_detector import NeuralDetector
from verifai.models.clip_detector import CLIPDetector, ClassificationHead
from verifai.models.frequency_classifier import FrequencyDetector, FrequencyMLP
from verifai.models.fusion_detector import FusionDetector, create_fusion_detector

__all__ = [
    # Base
    "BaseDetector",
    "DetectorOutput",
    "Label",
    # Fusion (recommended - best accuracy)
    "FusionDetector",
    "create_fusion_detector",
    # CLIP-based
    "CLIPDetector",
    "ClassificationHead",
    # Legacy neural detector
    "NeuralDetector",
    # Frequency
    "FrequencyDetector",
    "FrequencyMLP",
]
