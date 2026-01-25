"""
Base Detector Interface
========================

Defines the abstract interface for all detectors in the VerifAI pipeline.
All detector implementations should inherit from BaseDetector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, Any
from enum import Enum

import torch
import numpy as np
from loguru import logger


class Label(Enum):
    """Classification labels for AI detection."""
    REAL = "real"
    AI_GENERATED = "ai_generated"
    UNCERTAIN = "uncertain"


@dataclass
class DetectorOutput:
    """
    Standardized output from any detector.
    
    Attributes:
        raw_score: Raw model output score (before calibration)
        confidence: Calibrated confidence score [0, 1]
        label: Predicted label
        probabilities: Class probabilities {label: probability}
        features: Optional extracted features (for ensemble/analysis)
        evidence: Optional evidence data (heatmaps, attention, etc.)
        metadata: Additional detector-specific metadata
    """
    
    # Core predictions
    raw_score: float
    confidence: float
    label: Label
    
    # Detailed probabilities
    probabilities: dict[str, float] = field(default_factory=dict)
    
    # Optional extracted features (embeddings)
    features: Optional[np.ndarray] = None
    
    # Evidence for explainability
    evidence: dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set defaults."""
        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Set default probabilities if not provided
        if not self.probabilities:
            self.probabilities = {
                Label.REAL.value: 1.0 - self.confidence,
                Label.AI_GENERATED.value: self.confidence,
            }
    
    @property
    def is_ai_generated(self) -> bool:
        """Check if the prediction is AI-generated."""
        return self.label == Label.AI_GENERATED
    
    @property
    def is_confident(self, threshold: float = 0.7) -> bool:
        """Check if the prediction is confident enough."""
        return abs(self.confidence - 0.5) > (threshold - 0.5)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "raw_score": self.raw_score,
            "confidence": self.confidence,
            "label": self.label.value,
            "probabilities": self.probabilities,
            "metadata": self.metadata,
        }
        
        # Include evidence keys (but not full data)
        if self.evidence:
            result["evidence_keys"] = list(self.evidence.keys())
        
        return result


class BaseDetector(ABC):
    """
    Abstract base class for all detectors.
    
    All detector implementations must:
    1. Implement the detect() method
    2. Return DetectorOutput objects
    3. Support both single and batch processing
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        threshold: float = 0.5,
        name: str = "base_detector",
    ):
        """
        Initialize the detector.
        
        Args:
            device: Device to run inference on ("cuda", "mps", "cpu", or None for auto)
            threshold: Classification threshold
            name: Identifier for this detector
        """
        self.name = name
        self.threshold = threshold
        self._device = self._resolve_device(device)
        self._is_loaded = False
        
        logger.debug(f"Initialized {self.name} on device: {self._device}")
    
    @property
    def device(self) -> torch.device:
        """Get the device for inference."""
        return self._device
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._is_loaded
    
    def _resolve_device(self, device: Optional[str]) -> torch.device:
        """Resolve the device string to a torch.device."""
        if device is None or device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model weights and prepare for inference.
        
        This method should be called before detect() is used.
        Implementations should set self._is_loaded = True when done.
        """
        pass
    
    @abstractmethod
    def detect(
        self,
        input_tensor: torch.Tensor,
        return_features: bool = False,
        return_evidence: bool = False,
    ) -> DetectorOutput:
        """
        Run detection on a preprocessed input tensor.
        
        Args:
            input_tensor: Preprocessed input tensor (C, H, W) or (N, C, H, W)
            return_features: Whether to return extracted features
            return_evidence: Whether to return evidence (heatmaps, etc.)
            
        Returns:
            DetectorOutput with predictions
        """
        pass
    
    def detect_batch(
        self,
        input_tensors: torch.Tensor,
        return_features: bool = False,
        return_evidence: bool = False,
    ) -> list[DetectorOutput]:
        """
        Run detection on a batch of inputs.
        
        Args:
            input_tensors: Batch of preprocessed tensors (N, C, H, W)
            return_features: Whether to return extracted features
            return_evidence: Whether to return evidence
            
        Returns:
            List of DetectorOutput, one per input
        """
        # Default implementation: process one at a time
        # Subclasses can override for more efficient batch processing
        results = []
        for i in range(input_tensors.size(0)):
            result = self.detect(
                input_tensors[i],
                return_features=return_features,
                return_evidence=return_evidence,
            )
            results.append(result)
        return results
    
    def _score_to_label(self, score: float) -> Label:
        """Convert a score to a label based on threshold."""
        if score >= self.threshold:
            return Label.AI_GENERATED
        else:
            return Label.REAL
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', device='{self._device}')"
