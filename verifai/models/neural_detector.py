"""
Neural Detector (LEGACY)
=========================

⚠️ LEGACY MODULE - Use FusionDetector instead!
================================================

This module is kept for backward compatibility only.
For new projects, use FusionDetector which provides better accuracy (97.0% vs ~80%).

Recommended usage:
    from verifai.models import FusionDetector
    detector = FusionDetector()

Or via pipeline:
    from verifai import VerifAI
    detector = VerifAI()  # Uses FusionDetector by default

---

Original description:
Vision Transformer (ViT) based detector for AI-generated media.
Uses HuggingFace transformers for model loading and inference.
"""

from typing import Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoConfig,
    ViTForImageClassification,
    ViTImageProcessor,
)
from loguru import logger

from verifai.models.base import BaseDetector, DetectorOutput, Label


class NeuralDetector(BaseDetector):
    """
    Neural network-based detector using Vision Transformers.
    
    This detector uses a ViT model (or similar architecture) to classify
    images as real or AI-generated. It can be used with:
    - Pre-trained models from HuggingFace Hub
    - Custom fine-tuned models
    - Various architectures (ViT, ConvNeXt, Swin, etc.)
    
    Features:
    - Automatic mixed-precision inference (FP16)
    - Feature extraction for downstream tasks
    - GradCAM-style attention visualization
    - Batch processing support
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device: Optional[str] = None,
        threshold: float = 0.5,
        num_classes: int = 2,
        fp16: bool = True,
        class_labels: Optional[dict[int, str]] = None,
    ):
        """
        Initialize the Neural Detector.
        
        Args:
            model_name: HuggingFace model ID or path to local model
            device: Device for inference ("cuda", "mps", "cpu", or None for auto)
            threshold: Classification threshold for AI-generated
            num_classes: Number of output classes
            fp16: Whether to use FP16 inference (faster on GPU)
            class_labels: Mapping of class indices to labels
        """
        super().__init__(device=device, threshold=threshold, name="neural_detector")
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.fp16 = fp16 and self._device.type == "cuda"
        
        # Class labels
        self.class_labels = class_labels or {
            0: Label.REAL.value,
            1: Label.AI_GENERATED.value,
        }
        self.label_to_idx = {v: k for k, v in self.class_labels.items()}
        
        # Model components (loaded lazily)
        self.model: Optional[nn.Module] = None
        self.processor: Optional[AutoImageProcessor] = None
        self._feature_extractor_hook = None
        self._features = None
        
        logger.info(
            f"NeuralDetector initialized: model={model_name}, "
            f"device={self._device}, fp16={self.fp16}"
        )
    
    def load(self) -> None:
        """Load the model and processor from HuggingFace or local path."""
        if self._is_loaded:
            logger.debug("Model already loaded")
            return
        
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            # Check if it's a local path or HuggingFace model
            if Path(self.model_name).exists():
                self._load_local_model()
            else:
                self._load_hf_model()
            
            # Move model to device
            self.model.to(self._device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Enable FP16 if requested
            if self.fp16:
                self.model.half()
                logger.debug("Enabled FP16 inference")
            
            # Register hook for feature extraction
            self._register_feature_hook()
            
            self._is_loaded = True
            logger.info(f"Model loaded successfully on {self._device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_hf_model(self) -> None:
        """Load model from HuggingFace Hub."""
        # Load configuration
        config = AutoConfig.from_pretrained(self.model_name)
        
        # Check if model is already a classifier or needs modification
        if hasattr(config, 'num_labels') and config.num_labels == self.num_classes:
            # Load as-is
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                config=config,
            )
            
            # Auto-detect label mapping from model config
            if hasattr(config, 'id2label') and config.id2label:
                self._update_label_mapping(config.id2label)
        else:
            # Need to modify the classifier head
            logger.info(
                f"Adapting model from {getattr(config, 'num_labels', 1000)} "
                f"classes to {self.num_classes}"
            )
            
            # Load base model
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
            )
        
        # Load processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
    
    def _update_label_mapping(self, id2label: dict) -> None:
        """
        Update label mapping based on model's config.
        
        Handles different naming conventions:
        - artificial/human (umm-maybe/AI-image-detector)
        - ai/real
        - generated/authentic
        - fake/real
        """
        # Normalize labels to lowercase
        id2label_lower = {k: v.lower() for k, v in id2label.items()}
        
        # Find AI label index (various names used)
        ai_names = {'artificial', 'ai', 'generated', 'fake', 'synthetic', 'ai_generated'}
        real_names = {'human', 'real', 'authentic', 'natural', 'photo'}
        
        ai_idx = None
        real_idx = None
        
        for idx, label in id2label_lower.items():
            if label in ai_names:
                ai_idx = int(idx)
            elif label in real_names:
                real_idx = int(idx)
        
        # Update mapping if found
        if ai_idx is not None and real_idx is not None:
            self.class_labels = {
                real_idx: Label.REAL.value,
                ai_idx: Label.AI_GENERATED.value,
            }
            self.label_to_idx = {v: k for k, v in self.class_labels.items()}
            logger.info(
                f"Label mapping updated: real={real_idx}, ai={ai_idx} "
                f"(from model config: {id2label})"
            )
    
    def _load_local_model(self) -> None:
        """Load model from local path."""
        model_path = Path(self.model_name)
        
        if model_path.suffix in {".pt", ".pth", ".safetensors"}:
            # Load weights into a fresh model
            # First, try to load config from parent directory
            config_path = model_path.parent
            
            try:
                config = AutoConfig.from_pretrained(config_path)
                self.model = AutoModelForImageClassification.from_config(config)
            except Exception:
                # Fallback to default ViT
                logger.warning("Could not load config, using default ViT architecture")
                self.model = ViTForImageClassification.from_pretrained(
                    "google/vit-base-patch16-224",
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                )
            
            # Load weights
            if model_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location="cpu")
            
            self.model.load_state_dict(state_dict, strict=False)
            
            # Use default ViT processor
            self.processor = ViTImageProcessor.from_pretrained(
                "google/vit-base-patch16-224"
            )
        else:
            # Assume it's a directory with full model
            self.model = AutoModelForImageClassification.from_pretrained(
                model_path,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
            )
            
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_path)
            except Exception:
                self.processor = ViTImageProcessor.from_pretrained(
                    "google/vit-base-patch16-224"
                )
    
    def _register_feature_hook(self) -> None:
        """Register forward hook to extract features from the model."""
        # Find the layer before the classifier
        # This is architecture-dependent
        
        if hasattr(self.model, 'vit'):
            # ViT architecture
            target_layer = self.model.vit.layernorm
        elif hasattr(self.model, 'convnext'):
            # ConvNeXt architecture
            target_layer = self.model.convnext.layernorm
        elif hasattr(self.model, 'swin'):
            # Swin architecture  
            target_layer = self.model.swin.layernorm
        else:
            # Try to find a suitable layer
            logger.warning("Unknown architecture, feature extraction may not work")
            return
        
        def hook_fn(module, input, output):
            # Store features for later use
            if isinstance(output, tuple):
                self._features = output[0].detach()
            else:
                self._features = output.detach()
        
        self._feature_extractor_hook = target_layer.register_forward_hook(hook_fn)
    
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
            return_evidence: Whether to return attention evidence
            
        Returns:
            DetectorOutput with predictions
        """
        if not self._is_loaded:
            self.load()
        
        # Ensure batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Move to device
        input_tensor = input_tensor.to(self._device)
        
        # Apply FP16 if enabled
        if self.fp16:
            input_tensor = input_tensor.half()
        
        # Clear previous features
        self._features = None
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs.logits
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # For binary classification, use probability of AI-generated class
        ai_prob = probs[0, self.label_to_idx.get(Label.AI_GENERATED.value, 1)].item()
        real_prob = probs[0, self.label_to_idx.get(Label.REAL.value, 0)].item()
        
        # Determine label
        label = self._score_to_label(ai_prob)
        
        # Build output
        result = DetectorOutput(
            raw_score=ai_prob,
            confidence=ai_prob,  # Will be calibrated later in pipeline
            label=label,
            probabilities={
                Label.REAL.value: real_prob,
                Label.AI_GENERATED.value: ai_prob,
            },
            metadata={
                "model": self.model_name,
                "threshold": self.threshold,
            }
        )
        
        # Add features if requested
        if return_features and self._features is not None:
            # Pool features if needed
            features = self._features
            if features.dim() > 2:
                # Global average pooling
                features = features.mean(dim=tuple(range(1, features.dim() - 1)))
            result.features = features[0].cpu().float().numpy()
        
        # Add attention evidence if requested
        if return_evidence:
            evidence = self._extract_attention_evidence(input_tensor)
            result.evidence = evidence
        
        return result
    
    def detect_batch(
        self,
        input_tensors: torch.Tensor,
        return_features: bool = False,
        return_evidence: bool = False,
    ) -> list[DetectorOutput]:
        """
        Run detection on a batch of inputs (optimized).
        
        Args:
            input_tensors: Batch of tensors (N, C, H, W)
            return_features: Whether to return features
            return_evidence: Whether to return evidence
            
        Returns:
            List of DetectorOutput, one per input
        """
        if not self._is_loaded:
            self.load()
        
        batch_size = input_tensors.size(0)
        
        # Move to device
        input_tensors = input_tensors.to(self._device)
        
        if self.fp16:
            input_tensors = input_tensors.half()
        
        # Clear features
        self._features = None
        
        # Run batch inference
        with torch.no_grad():
            outputs = self.model(input_tensors)
            logits = outputs.logits
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Build results
        results = []
        for i in range(batch_size):
            ai_idx = self.label_to_idx.get(Label.AI_GENERATED.value, 1)
            real_idx = self.label_to_idx.get(Label.REAL.value, 0)
            
            ai_prob = probs[i, ai_idx].item()
            real_prob = probs[i, real_idx].item()
            
            label = self._score_to_label(ai_prob)
            
            result = DetectorOutput(
                raw_score=ai_prob,
                confidence=ai_prob,
                label=label,
                probabilities={
                    Label.REAL.value: real_prob,
                    Label.AI_GENERATED.value: ai_prob,
                },
                metadata={
                    "model": self.model_name,
                    "batch_index": i,
                }
            )
            
            # Add features if requested
            if return_features and self._features is not None:
                features = self._features
                if features.dim() > 2:
                    features = features.mean(dim=tuple(range(1, features.dim() - 1)))
                result.features = features[i].cpu().float().numpy()
            
            results.append(result)
        
        return results
    
    def _extract_attention_evidence(
        self,
        input_tensor: torch.Tensor,
    ) -> dict:
        """
        Extract attention-based evidence from the model.
        
        Args:
            input_tensor: Input tensor
            
        Returns:
            Dictionary containing attention maps and other evidence
        """
        evidence = {}
        
        try:
            # Run forward pass with attention output
            with torch.no_grad():
                outputs = self.model(
                    input_tensor,
                    output_attentions=True,
                )
            
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # Get attention from last layer
                last_attention = outputs.attentions[-1]  # (B, heads, seq, seq)
                
                # Average across heads
                attention = last_attention.mean(dim=1)  # (B, seq, seq)
                
                # Get CLS token attention to patches
                cls_attention = attention[0, 0, 1:]  # (num_patches,)
                
                # Reshape to spatial grid
                num_patches = cls_attention.size(0)
                grid_size = int(num_patches ** 0.5)
                
                if grid_size * grid_size == num_patches:
                    attention_map = cls_attention.reshape(grid_size, grid_size)
                    attention_map = attention_map.cpu().float().numpy()
                    
                    evidence["attention_map"] = attention_map
                    evidence["attention_shape"] = (grid_size, grid_size)
        
        except Exception as e:
            logger.debug(f"Could not extract attention evidence: {e}")
        
        return evidence
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._is_loaded:
            return {
                "model_name": self.model_name,
                "loaded": False,
            }
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            "model_name": self.model_name,
            "loaded": True,
            "device": str(self._device),
            "fp16": self.fp16,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": self.num_classes,
            "class_labels": self.class_labels,
        }
    
    def __del__(self):
        """Cleanup hook on deletion."""
        if self._feature_extractor_hook is not None:
            self._feature_extractor_hook.remove()
