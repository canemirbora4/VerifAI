"""
Fusion Detector - CLIP + Frequency Ensemble
============================================

Combines CLIP semantic features with frequency-domain analysis
for improved AI-generated image detection.

Optimal weights determined through testing:
- CLIP: 0.80 (primary signal)
- Frequency: 0.20 (supporting signal for edge cases)
"""

from typing import Optional, Tuple
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from PIL import Image
from loguru import logger

from verifai.models.base import BaseDetector, DetectorOutput, Label
from verifai.features.frequency import FrequencyExtractor


class CLIPClassificationHead(nn.Module):
    """Classification head for CLIP features."""
    
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class FusionDetector(BaseDetector):
    """
    Fusion detector combining CLIP and Frequency classifiers.
    
    This detector uses weighted ensemble of:
    1. CLIP-based semantic features (primary, weight=0.80)
    2. Frequency-domain features (secondary, weight=0.20)
    
    The fusion approach helps catch edge cases where one model
    might miss but the other catches.
    
    Attributes:
        clip_weight: Weight for CLIP predictions (default: 0.80)
        freq_weight: Weight for frequency predictions (default: 0.20)
    """
    
    # Default optimal weights from testing
    DEFAULT_CLIP_WEIGHT = 0.80
    DEFAULT_FREQ_WEIGHT = 0.20
    
    def __init__(
        self,
        device: Optional[str] = None,
        threshold: float = 0.5,
        clip_weight: float = DEFAULT_CLIP_WEIGHT,
        freq_weight: float = DEFAULT_FREQ_WEIGHT,
        clip_model_path: Optional[str] = None,
        freq_model_path: Optional[str] = None,
    ):
        """
        Initialize the Fusion Detector.
        
        Args:
            device: Device for inference ('cuda', 'mps', 'cpu')
            threshold: Classification threshold (default: 0.5)
            clip_weight: Weight for CLIP predictions (default: 0.80)
            freq_weight: Weight for frequency predictions (default: 0.20)
            clip_model_path: Path to CLIP classification head weights
            freq_model_path: Path to frequency classifier weights
        """
        super().__init__(
            device=device,
            threshold=threshold,
            name="fusion_detector",
        )
        
        self.clip_weight = clip_weight
        self.freq_weight = freq_weight
        
        # Normalize weights to sum to 1
        total_weight = self.clip_weight + self.freq_weight
        if abs(total_weight - 1.0) > 0.01:
            self.clip_weight = self.clip_weight / total_weight
            self.freq_weight = self.freq_weight / total_weight
            logger.warning(
                f"Weights normalized to sum to 1: "
                f"clip={self.clip_weight:.2f}, freq={self.freq_weight:.2f}"
            )
        
        # Model paths - use defaults if not provided
        models_dir = Path(__file__).parent.parent.parent / "models"
        self.clip_model_path = clip_model_path or str(models_dir / "modern_ai_detector.pt")
        self.freq_model_path = freq_model_path or str(models_dir / "frequency_classifier.joblib")
        
        # Models (loaded lazily)
        self.clip_model = None
        self.clip_head = None
        self.clip_processor = None
        self.freq_classifier = None
        self.freq_extractor = None
        
        logger.info(
            f"FusionDetector initialized: "
            f"clip_weight={self.clip_weight:.2f}, freq_weight={self.freq_weight:.2f}"
        )
    
    def load(self) -> None:
        """Load all models."""
        if self._is_loaded:
            return
        
        logger.info("Loading FusionDetector models...")
        
        # Load CLIP
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            logger.info("Loading CLIP backbone...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model.to(self._device)
            self.clip_model.eval()
            
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
            logger.info(f"Loading CLIP head from {self.clip_model_path}...")
            self.clip_head = CLIPClassificationHead()
            self.clip_head.load_state_dict(
                torch.load(self.clip_model_path, map_location=self._device)
            )
            self.clip_head.to(self._device)
            self.clip_head.eval()
            
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise
        
        # Load Frequency classifier
        try:
            logger.info(f"Loading frequency classifier from {self.freq_model_path}...")
            self.freq_classifier = joblib.load(self.freq_model_path)
            
            self.freq_extractor = FrequencyExtractor(
                image_size=(256, 256),
                patch_size=64,
                num_azimuthal_bins=64,
                compute_patches=True,
                normalize=True,
            )
            
        except Exception as e:
            logger.error(f"Failed to load frequency classifier: {e}")
            raise
        
        self._is_loaded = True
        logger.info(f"FusionDetector loaded on {self._device}")
    
    def _predict_clip(self, image: Image.Image) -> float:
        """Get CLIP AI probability."""
        inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        
        with torch.no_grad():
            features = self.clip_model.get_image_features(pixel_values=pixel_values)
            logits = self.clip_head(features)
            probs = F.softmax(logits, dim=1)
            ai_prob = probs[0, 1].item()
        
        return ai_prob
    
    def _predict_freq(self, image: Image.Image) -> float:
        """Get frequency AI probability."""
        freq_features = self.freq_extractor.extract(image)
        feature_vector = freq_features.feature_vector.reshape(1, -1)
        
        if hasattr(self.freq_classifier, "predict_proba"):
            probs = self.freq_classifier.predict_proba(feature_vector)
            ai_prob = probs[0, 1]
        else:
            pred = self.freq_classifier.predict(feature_vector)[0]
            ai_prob = float(pred)
        
        return ai_prob
    
    def detect(
        self,
        input_data: torch.Tensor | Image.Image,
        return_features: bool = False,
        return_evidence: bool = False,
    ) -> DetectorOutput:
        """
        Run fusion detection on an image.
        
        Args:
            input_data: Input image (PIL Image or tensor)
            return_features: Include individual model scores
            return_evidence: Include detailed analysis
            
        Returns:
            DetectorOutput with fusion prediction
        """
        if not self._is_loaded:
            self.load()
        
        # Convert tensor to PIL if needed
        if isinstance(input_data, torch.Tensor):
            if input_data.dim() == 4:
                input_data = input_data[0]
            # Convert CHW to HWC and to PIL
            arr = input_data.cpu().numpy()
            if arr.shape[0] in [1, 3]:
                arr = arr.transpose(1, 2, 0)
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            image = Image.fromarray(arr).convert("RGB")
        else:
            image = input_data.convert("RGB")
        
        # Get predictions from both models
        clip_prob = self._predict_clip(image)
        freq_prob = self._predict_freq(image)
        
        # Weighted fusion
        fusion_prob = self.clip_weight * clip_prob + self.freq_weight * freq_prob
        
        # Determine label
        label = self._score_to_label(fusion_prob)
        
        # Build result
        result = DetectorOutput(
            raw_score=fusion_prob,
            confidence=fusion_prob,
            label=label,
            probabilities={
                Label.REAL.value: 1 - fusion_prob,
                Label.AI_GENERATED.value: fusion_prob,
            },
            metadata={
                "clip_weight": self.clip_weight,
                "freq_weight": self.freq_weight,
                "clip_prob": clip_prob,
                "freq_prob": freq_prob,
            }
        )
        
        # Add evidence if requested
        if return_evidence:
            result.evidence = {
                "clip_score": clip_prob,
                "clip_prediction": "ai" if clip_prob > 0.5 else "real",
                "frequency_score": freq_prob,
                "frequency_prediction": "ai" if freq_prob > 0.5 else "real",
                "fusion_score": fusion_prob,
                "models_agree": (clip_prob > 0.5) == (freq_prob > 0.5),
            }
        
        return result
    
    def detect_batch(
        self,
        input_tensors: torch.Tensor,
        return_features: bool = False,
        return_evidence: bool = False,
    ) -> list[DetectorOutput]:
        """Run detection on a batch of images."""
        results = []
        for i in range(input_tensors.size(0)):
            result = self.detect(
                input_tensors[i],
                return_features=return_features,
                return_evidence=return_evidence,
            )
            results.append(result)
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the fusion model."""
        return {
            "name": self.name,
            "loaded": self._is_loaded,
            "device": str(self._device),
            "clip_weight": self.clip_weight,
            "freq_weight": self.freq_weight,
            "clip_model_path": self.clip_model_path,
            "freq_model_path": self.freq_model_path,
            "threshold": self.threshold,
        }


def create_fusion_detector(
    device: Optional[str] = None,
    **kwargs,
) -> FusionDetector:
    """
    Factory function to create a FusionDetector.
    
    Args:
        device: Device for inference
        **kwargs: Additional arguments for FusionDetector
        
    Returns:
        Configured FusionDetector instance
    """
    detector = FusionDetector(device=device, **kwargs)
    return detector
