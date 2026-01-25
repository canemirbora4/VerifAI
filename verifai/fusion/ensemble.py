"""
Ensemble Fusion
================

Combines outputs from multiple detectors into a single prediction.

Fusion strategies:
1. Simple average - Equal weight to all detectors
2. Weighted average - Learned or manual weights
3. Stacking - Train a meta-classifier on detector outputs
4. Confidence-weighted - Weight by each detector's confidence

The ensemble helps because different detectors catch different artifacts:
- Neural: Texture patterns, semantic inconsistencies
- Frequency: FFT/DCT artifacts from generators
- Metadata: Provenance signals

Combined, they're more robust than any single detector.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum

import numpy as np
from loguru import logger

from verifai.models.base import DetectorOutput, Label


class FusionMethod(Enum):
    """Available fusion methods."""
    AVERAGE = "average"
    WEIGHTED = "weighted"
    MAX = "max"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble fusion.
    
    Attributes:
        method: Fusion method to use
        weights: Detector weights (for weighted fusion)
        detectors: List of detector names to include
        threshold: Classification threshold
    """
    method: FusionMethod = FusionMethod.WEIGHTED
    weights: dict = field(default_factory=lambda: {
        "neural": 0.6,
        "frequency": 0.25,
        "metadata": 0.15,
    })
    detectors: list = field(default_factory=lambda: ["neural", "frequency", "metadata"])
    threshold: float = 0.5


@dataclass
class EnsembleOutput:
    """
    Output from ensemble fusion.
    
    Attributes:
        final_score: Combined confidence score
        final_label: Final classification label
        detector_scores: Individual detector scores
        detector_weights: Weights used for each detector
        fusion_method: Method used for fusion
    """
    final_score: float
    final_label: Label
    detector_scores: dict
    detector_weights: dict
    fusion_method: str
    
    @property
    def is_ai_generated(self) -> bool:
        return self.final_label == Label.AI_GENERATED
    
    def to_dict(self) -> dict:
        return {
            "final_score": self.final_score,
            "final_label": self.final_label.value,
            "detector_scores": self.detector_scores,
            "detector_weights": self.detector_weights,
            "fusion_method": self.fusion_method,
        }


class Ensemble:
    """
    Combines multiple detector outputs into a single prediction.
    
    Usage:
        ensemble = Ensemble(config)
        
        outputs = {
            "neural": neural_detector.detect(image),
            "frequency": frequency_detector.detect(image),
            "metadata": metadata_output,
        }
        
        result = ensemble.fuse(outputs)
        print(result.final_score, result.final_label)
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize the ensemble.
        
        Args:
            config: Ensemble configuration (uses defaults if None)
        """
        self.config = config or EnsembleConfig()
        
        # Normalize weights
        self._normalize_weights()
        
        logger.info(
            f"Ensemble initialized: method={self.config.method.value}, "
            f"detectors={self.config.detectors}"
        )
    
    def _normalize_weights(self) -> None:
        """Ensure weights sum to 1.0."""
        weights = self.config.weights
        active_detectors = self.config.detectors
        
        # Filter to active detectors
        active_weights = {k: v for k, v in weights.items() if k in active_detectors}
        
        # Normalize
        total = sum(active_weights.values())
        if total > 0:
            self.config.weights = {k: v / total for k, v in active_weights.items()}
        else:
            # Equal weights if none specified
            n = len(active_detectors)
            self.config.weights = {d: 1.0 / n for d in active_detectors}
    
    def fuse(
        self,
        detector_outputs: dict[str, DetectorOutput],
    ) -> EnsembleOutput:
        """
        Fuse multiple detector outputs.
        
        Args:
            detector_outputs: Dict mapping detector names to their outputs
            
        Returns:
            EnsembleOutput with combined prediction
        """
        # Extract scores
        scores = {}
        for name, output in detector_outputs.items():
            if name in self.config.detectors:
                scores[name] = output.confidence
        
        # Check we have at least one detector
        if not scores:
            logger.warning("No detector outputs to fuse!")
            return EnsembleOutput(
                final_score=0.5,
                final_label=Label.UNCERTAIN,
                detector_scores={},
                detector_weights={},
                fusion_method=self.config.method.value,
            )
        
        # Apply fusion method
        if self.config.method == FusionMethod.AVERAGE:
            final_score = self._fuse_average(scores)
        elif self.config.method == FusionMethod.WEIGHTED:
            final_score = self._fuse_weighted(scores)
        elif self.config.method == FusionMethod.MAX:
            final_score = self._fuse_max(scores)
        elif self.config.method == FusionMethod.CONFIDENCE_WEIGHTED:
            final_score = self._fuse_confidence_weighted(scores, detector_outputs)
        else:
            final_score = self._fuse_weighted(scores)
        
        # Determine label
        if final_score >= self.config.threshold:
            final_label = Label.AI_GENERATED
        else:
            final_label = Label.REAL
        
        # Get weights used
        weights_used = {k: self.config.weights.get(k, 0.0) for k in scores.keys()}
        
        return EnsembleOutput(
            final_score=final_score,
            final_label=final_label,
            detector_scores=scores,
            detector_weights=weights_used,
            fusion_method=self.config.method.value,
        )
    
    def _fuse_average(self, scores: dict[str, float]) -> float:
        """Simple average of all scores."""
        return sum(scores.values()) / len(scores)
    
    def _fuse_weighted(self, scores: dict[str, float]) -> float:
        """Weighted average using configured weights."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, score in scores.items():
            weight = self.config.weights.get(name, 0.0)
            weighted_sum += weight * score
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return self._fuse_average(scores)
    
    def _fuse_max(self, scores: dict[str, float]) -> float:
        """Take the maximum score (most suspicious detector wins)."""
        return max(scores.values())
    
    def _fuse_confidence_weighted(
        self,
        scores: dict[str, float],
        outputs: dict[str, DetectorOutput],
    ) -> float:
        """
        Weight by detector confidence (how far from 0.5).
        
        Detectors that are more certain get more weight.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, score in scores.items():
            # Confidence weight = distance from 0.5
            conf_weight = abs(score - 0.5) * 2  # Scale to [0, 1]
            base_weight = self.config.weights.get(name, 1.0)
            weight = base_weight * (0.5 + conf_weight)  # Combine
            
            weighted_sum += weight * score
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return self._fuse_average(scores)
    
    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update detector weights.
        
        Args:
            new_weights: New weight dictionary
        """
        self.config.weights.update(new_weights)
        self._normalize_weights()
        logger.info(f"Updated weights: {self.config.weights}")
    
    def add_detector(self, name: str, weight: float = 0.1) -> None:
        """
        Add a new detector to the ensemble.
        
        Args:
            name: Detector name
            weight: Initial weight
        """
        if name not in self.config.detectors:
            self.config.detectors.append(name)
            self.config.weights[name] = weight
            self._normalize_weights()
            logger.info(f"Added detector '{name}' with weight {weight}")
    
    def remove_detector(self, name: str) -> None:
        """
        Remove a detector from the ensemble.
        
        Args:
            name: Detector name to remove
        """
        if name in self.config.detectors:
            self.config.detectors.remove(name)
            self.config.weights.pop(name, None)
            self._normalize_weights()
            logger.info(f"Removed detector '{name}'")


def create_metadata_detector_output(
    metadata_features,  # MetadataFeatures
) -> DetectorOutput:
    """
    Convert MetadataFeatures to DetectorOutput for ensemble compatibility.
    
    Args:
        metadata_features: MetadataFeatures from parser
        
    Returns:
        DetectorOutput compatible with ensemble
    """
    # Invert confidence_real to get AI probability
    # confidence_real = 0.8 means 80% chance real, so 20% AI
    ai_prob = 1.0 - metadata_features.confidence_real
    
    if ai_prob >= 0.5:
        label = Label.AI_GENERATED
    else:
        label = Label.REAL
    
    return DetectorOutput(
        raw_score=ai_prob,
        confidence=ai_prob,
        label=label,
        probabilities={
            Label.REAL.value: metadata_features.confidence_real,
            Label.AI_GENERATED.value: ai_prob,
        },
        metadata={
            "is_suspicious": metadata_features.is_suspicious,
            "suspicion_reasons": metadata_features.suspicion_reasons,
        }
    )
