"""
Frequency-Based Classifier
===========================

A lightweight classifier that uses frequency-domain features
to detect AI-generated images.

This is a secondary detector that:
1. Extracts FFT/DCT features from images
2. Passes them through a small MLP
3. Outputs a probability score

It's designed to complement the neural detector in the ensemble.
"""

from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from verifai.models.base import BaseDetector, DetectorOutput, Label
from verifai.features.frequency import FrequencyExtractor, FrequencyFeatures


class FrequencyMLP(nn.Module):
    """
    Simple MLP for frequency-based classification.
    
    Architecture:
        Input -> FC1 -> ReLU -> Dropout -> FC2 -> ReLU -> Dropout -> FC3 -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)


class FrequencyDetector(BaseDetector):
    """
    Frequency-domain based detector for AI-generated images.
    
    This detector uses FFT/DCT analysis to identify artifacts
    that are common in AI-generated images but rare in real photos.
    
    Features:
    - Lightweight (small MLP, ~100K parameters)
    - Fast inference
    - Complementary to neural detector
    - Works on different signal than pixel-based models
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        threshold: float = 0.5,
        image_size: Tuple[int, int] = (256, 256),
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the Frequency Detector.
        
        Args:
            device: Device for inference
            threshold: Classification threshold
            image_size: Size for frequency analysis
            hidden_dims: Hidden layer dimensions for MLP
            dropout: Dropout rate
            model_path: Path to pretrained weights (optional)
        """
        super().__init__(
            device=device,
            threshold=threshold,
            name="frequency_detector",
        )
        
        self.image_size = image_size
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.model_path = model_path
        
        # Initialize feature extractor
        self._extractor = FrequencyExtractor(
            image_size=image_size,
            patch_size=64,
            num_azimuthal_bins=64,
            compute_patches=True,
            normalize=True,
        )
        
        # Get feature dimension
        self._feature_dim = self._extractor.get_feature_dim()
        
        # Initialize model (will be created in load())
        self.model: Optional[FrequencyMLP] = None
        
        logger.info(
            f"FrequencyDetector initialized: feature_dim={self._feature_dim}, "
            f"hidden_dims={hidden_dims}"
        )
    
    def load(self) -> None:
        """Load or initialize the model."""
        if self._is_loaded:
            return
        
        logger.info("Loading FrequencyDetector model...")
        
        # Create model
        self.model = FrequencyMLP(
            input_dim=self._feature_dim,
            hidden_dims=self.hidden_dims,
            num_classes=2,
            dropout=self.dropout,
        )
        
        # Load pretrained weights if available
        if self.model_path and Path(self.model_path).exists():
            logger.info(f"Loading weights from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        else:
            logger.warning(
                "No pretrained weights loaded. "
                "FrequencyDetector will output random predictions until trained."
            )
        
        # Move to device
        self.model.to(self._device)
        self.model.eval()
        
        self._is_loaded = True
        logger.info(f"FrequencyDetector loaded on {self._device}")
    
    def detect(
        self,
        input_tensor: torch.Tensor,
        return_features: bool = False,
        return_evidence: bool = False,
    ) -> DetectorOutput:
        """
        Run detection using frequency features.
        
        Args:
            input_tensor: Input image tensor (C, H, W) or (N, C, H, W)
            return_features: Include frequency features in output
            return_evidence: Include frequency spectrum as evidence
            
        Returns:
            DetectorOutput with predictions
        """
        if not self._is_loaded:
            self.load()
        
        # Handle batch dimension
        if input_tensor.dim() == 4:
            # Batch - process first image only for now
            input_tensor = input_tensor[0]
        
        # Extract frequency features
        freq_features = self._extractor.extract(
            input_tensor,
            return_spectra=return_evidence,
        )
        
        # Convert to tensor
        feature_tensor = torch.from_numpy(freq_features.feature_vector)
        feature_tensor = feature_tensor.unsqueeze(0).to(self._device)
        
        # Run classifier
        with torch.no_grad():
            logits = self.model(feature_tensor)
            probs = F.softmax(logits, dim=-1)
        
        # Get probabilities
        real_prob = probs[0, 0].item()
        ai_prob = probs[0, 1].item()
        
        # Determine label
        label = self._score_to_label(ai_prob)
        
        # Build output
        result = DetectorOutput(
            raw_score=ai_prob,
            confidence=ai_prob,
            label=label,
            probabilities={
                Label.REAL.value: real_prob,
                Label.AI_GENERATED.value: ai_prob,
            },
            metadata={
                "feature_dim": self._feature_dim,
            }
        )
        
        # Add features if requested
        if return_features:
            result.features = freq_features.feature_vector
        
        # Add evidence if requested
        if return_evidence:
            result.evidence = {
                "fft_stats": freq_features.fft_stats,
                "dct_stats": freq_features.dct_stats,
                "azimuthal_profile": freq_features.azimuthal_profile,
            }
            if freq_features.fft_magnitude is not None:
                # Downsample for storage
                mag = freq_features.fft_magnitude
                result.evidence["fft_magnitude_downsampled"] = mag[::4, ::4]
        
        return result
    
    def detect_batch(
        self,
        input_tensors: torch.Tensor,
        return_features: bool = False,
        return_evidence: bool = False,
    ) -> list[DetectorOutput]:
        """
        Run detection on a batch of images.
        
        Args:
            input_tensors: Batch of tensors (N, C, H, W)
            return_features: Include features
            return_evidence: Include evidence
            
        Returns:
            List of DetectorOutput
        """
        if not self._is_loaded:
            self.load()
        
        batch_size = input_tensors.size(0)
        
        # Extract features for all images
        feature_list = []
        features_objects = []
        
        for i in range(batch_size):
            freq_features = self._extractor.extract(
                input_tensors[i],
                return_spectra=return_evidence,
            )
            feature_list.append(freq_features.feature_vector)
            features_objects.append(freq_features)
        
        # Stack features
        feature_batch = np.stack(feature_list)
        feature_tensor = torch.from_numpy(feature_batch).to(self._device)
        
        # Run classifier
        with torch.no_grad():
            logits = self.model(feature_tensor)
            probs = F.softmax(logits, dim=-1)
        
        # Build results
        results = []
        for i in range(batch_size):
            real_prob = probs[i, 0].item()
            ai_prob = probs[i, 1].item()
            label = self._score_to_label(ai_prob)
            
            result = DetectorOutput(
                raw_score=ai_prob,
                confidence=ai_prob,
                label=label,
                probabilities={
                    Label.REAL.value: real_prob,
                    Label.AI_GENERATED.value: ai_prob,
                },
                metadata={"batch_index": i},
            )
            
            if return_features:
                result.features = features_objects[i].feature_vector
            
            results.append(result)
        
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        param_count = 0
        if self.model is not None:
            param_count = sum(p.numel() for p in self.model.parameters())
        
        return {
            "name": self.name,
            "loaded": self._is_loaded,
            "device": str(self._device),
            "feature_dim": self._feature_dim,
            "hidden_dims": self.hidden_dims,
            "total_parameters": param_count,
            "has_pretrained_weights": bool(self.model_path),
        }
    
    def save_weights(self, path: str) -> None:
        """Save model weights."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved weights to {path}")
    
    def get_feature_extractor(self) -> FrequencyExtractor:
        """Get the frequency feature extractor for external use."""
        return self._extractor
