"""
CLIP-based Detector
====================

Uses CLIP ViT-L/14 as a frozen backbone with a trainable classification head.
This approach provides better cross-generator generalization compared to
models trained on specific AI generators.

Key benefits:
- CLIP is trained on 400M+ diverse internet images
- Captures both semantic and low-level texture information
- Generator-agnostic: doesn't overfit to specific generator fingerprints
- Better robustness to JPEG compression, resize, and other corruptions
"""

from typing import Optional, Union, Literal
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from loguru import logger

from verifai.models.base import BaseDetector, DetectorOutput, Label


class ClassificationHead(nn.Module):
    """
    Trainable classification head for CLIP embeddings.
    
    Architecture:
        Linear(768, 512) -> ReLU -> Dropout -> 
        Linear(512, 256) -> ReLU -> Dropout ->
        Linear(256, 2)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: tuple[int, ...] = (512, 256),
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
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CLIPDetector(BaseDetector):
    """
    CLIP ViT-L/14 based detector with frozen backbone and trainable head.
    
    The CLIP vision encoder is kept frozen to preserve its rich, 
    distribution-aware representations. Only the classification head
    is trained/fine-tuned.
    
    This approach provides:
    - Better cross-generator generalization
    - More stable calibration
    - Lower degradation under JPEG/resize corruptions
    - Generator-agnostic detection
    
    Usage:
        >>> detector = CLIPDetector()
        >>> detector.load()
        >>> result = detector.detect(image_tensor)
        >>> print(result.confidence, result.label)
    """
    
    # CLIP ViT-L/14 embedding dimension
    EMBEDDING_DIM = 768
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        threshold: float = 0.5,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
        head_weights_path: Optional[str] = None,
    ):
        """
        Initialize the CLIP Detector.
        
        Args:
            model_name: CLIP model identifier from HuggingFace
            device: Device for inference ("cuda", "mps", "cpu", or None for auto)
            threshold: Classification threshold for AI-generated
            hidden_dims: Hidden layer dimensions for classification head
            dropout: Dropout rate for classification head
            head_weights_path: Path to pre-trained head weights (optional)
        """
        super().__init__(device=device, threshold=threshold, name="clip_detector")
        
        self.model_name = model_name
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.head_weights_path = head_weights_path
        
        # Model components (loaded lazily)
        self.clip_model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.classification_head: Optional[ClassificationHead] = None
        
        # Stored embeddings for feature extraction
        self._last_embeddings: Optional[torch.Tensor] = None
        
        # Class labels
        self.class_labels = {
            0: Label.REAL.value,
            1: Label.AI_GENERATED.value,
        }
        self.label_to_idx = {v: k for k, v in self.class_labels.items()}
        
        logger.info(
            f"CLIPDetector initialized: model={model_name}, "
            f"device={self._device}, head_dims={hidden_dims}"
        )
    
    def load(self) -> None:
        """Load CLIP model (frozen) and classification head."""
        if self._is_loaded:
            logger.debug("CLIPDetector already loaded")
            return
        
        logger.info(f"Loading CLIP model: {self.model_name}")
        
        try:
            # Load CLIP model
            self.clip_model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move to device
            self.clip_model.to(self._device)
            
            # FREEZE the CLIP model - this is critical!
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            logger.info("CLIP backbone frozen (no gradient updates)")
            
            # Initialize classification head
            self.classification_head = ClassificationHead(
                input_dim=self.EMBEDDING_DIM,
                hidden_dims=self.hidden_dims,
                num_classes=2,
                dropout=self.dropout,
            )
            self.classification_head.to(self._device)
            
            # Load pre-trained head weights if available
            if self.head_weights_path and Path(self.head_weights_path).exists():
                self._load_head_weights(self.head_weights_path)
                logger.info(f"Loaded head weights from {self.head_weights_path}")
            else:
                logger.warning(
                    "No pre-trained head weights loaded. "
                    "Classification head has random weights - train before use!"
                )
            
            self._is_loaded = True
            
            # Log model info
            clip_params = sum(p.numel() for p in self.clip_model.parameters())
            head_params = sum(p.numel() for p in self.classification_head.parameters())
            logger.info(
                f"CLIPDetector loaded: "
                f"CLIP params={clip_params/1e6:.1f}M (frozen), "
                f"Head params={head_params/1e3:.1f}K (trainable)"
            )
            
        except Exception as e:
            logger.error(f"Failed to load CLIPDetector: {e}")
            raise
    
    def _load_head_weights(self, path: str) -> None:
        """Load pre-trained classification head weights."""
        state_dict = torch.load(path, map_location=self._device)
        self.classification_head.load_state_dict(state_dict)
    
    def save_head_weights(self, path: str) -> None:
        """Save classification head weights for later use."""
        if self.classification_head is None:
            raise RuntimeError("Classification head not initialized")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.classification_head.state_dict(), path)
        logger.info(f"Saved head weights to {path}")
    
    def extract_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP image embeddings.
        
        Args:
            pixel_values: Preprocessed image tensor (N, C, H, W)
            
        Returns:
            Image embeddings (N, 768)
        """
        if not self._is_loaded:
            self.load()
        
        pixel_values = pixel_values.to(self._device)
        
        with torch.no_grad():
            # Get image features from CLIP
            image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
            
            # Normalize (CLIP convention)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        self._last_embeddings = image_features
        return image_features
    
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
            return_features: Whether to return CLIP embeddings
            return_evidence: Whether to return attention evidence
            
        Returns:
            DetectorOutput with predictions
        """
        if not self._is_loaded:
            self.load()
        
        # Ensure batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self._device)
        
        # Extract CLIP embeddings (frozen forward pass)
        embeddings = self.extract_embeddings(input_tensor)
        
        # Run through classification head
        self.classification_head.eval()
        with torch.no_grad():
            logits = self.classification_head(embeddings)
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        ai_idx = self.label_to_idx[Label.AI_GENERATED.value]
        real_idx = self.label_to_idx[Label.REAL.value]
        
        ai_prob = probs[0, ai_idx].item()
        real_prob = probs[0, real_idx].item()
        
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
                "model": f"CLIP ({self.model_name})",
                "threshold": self.threshold,
                "embedding_dim": self.EMBEDDING_DIM,
            }
        )
        
        # Add features if requested
        if return_features:
            result.features = embeddings[0].cpu().float().numpy()
        
        # Add evidence if requested
        if return_evidence:
            result.evidence = self._extract_evidence(input_tensor)
        
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
        input_tensors = input_tensors.to(self._device)
        
        # Extract embeddings for entire batch
        embeddings = self.extract_embeddings(input_tensors)
        
        # Run through classification head
        self.classification_head.eval()
        with torch.no_grad():
            logits = self.classification_head(embeddings)
        
        probs = F.softmax(logits, dim=-1)
        
        ai_idx = self.label_to_idx[Label.AI_GENERATED.value]
        real_idx = self.label_to_idx[Label.REAL.value]
        
        # Build results
        results = []
        for i in range(batch_size):
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
                    "model": f"CLIP ({self.model_name})",
                    "batch_index": i,
                }
            )
            
            if return_features:
                result.features = embeddings[i].cpu().float().numpy()
            
            results.append(result)
        
        return results
    
    def _extract_evidence(self, input_tensor: torch.Tensor) -> dict:
        """Extract evidence from CLIP attention (if available)."""
        evidence = {}
        
        try:
            # CLIP ViT has attention weights we can extract
            with torch.no_grad():
                outputs = self.clip_model.vision_model(
                    pixel_values=input_tensor,
                    output_attentions=True,
                )
            
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # Get last layer attention
                last_attention = outputs.attentions[-1]  # (B, heads, seq, seq)
                
                # Average across heads
                attention = last_attention.mean(dim=1)  # (B, seq, seq)
                
                # Get CLS token attention
                cls_attention = attention[0, 0, 1:]  # (num_patches,)
                
                # Reshape to grid
                num_patches = cls_attention.size(0)
                grid_size = int(num_patches ** 0.5)
                
                if grid_size * grid_size == num_patches:
                    attention_map = cls_attention.reshape(grid_size, grid_size)
                    evidence["attention_map"] = attention_map.cpu().float().numpy()
                    evidence["attention_shape"] = (grid_size, grid_size)
        
        except Exception as e:
            logger.debug(f"Could not extract CLIP attention: {e}")
        
        return evidence
    
    def train_head(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Train the classification head on a dataset.
        
        The CLIP backbone remains frozen - only the head is trained.
        
        Args:
            train_loader: DataLoader with (images, labels)
            val_loader: Optional validation DataLoader
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization
            save_path: Path to save best model weights
            
        Returns:
            Training history dictionary
        """
        if not self._is_loaded:
            self.load()
        
        logger.info(f"Training classification head for {epochs} epochs")
        
        # Setup optimizer (only head parameters)
        optimizer = torch.optim.AdamW(
            self.classification_head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training
            self.classification_head.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self._device)
                labels = labels.to(self._device)
                
                # Extract embeddings (no grad for CLIP)
                embeddings = self.extract_embeddings(images)
                
                # Forward through head
                optimizer.zero_grad()
                logits = self.classification_head(embeddings)
                loss = criterion(logits, labels)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            
            # Validation
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                # Save best model
                if val_acc > best_val_acc and save_path:
                    best_val_acc = val_acc
                    self.save_head_weights(save_path)
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                )
            
            scheduler.step()
        
        # Load best model if saved
        if save_path and Path(save_path).exists():
            self._load_head_weights(save_path)
            logger.info(f"Loaded best model (val_acc={best_val_acc:.4f})")
        
        return history
    
    def _evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Evaluate on a dataset."""
        self.classification_head.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                
                embeddings = self.extract_embeddings(images)
                logits = self.classification_head(embeddings)
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(data_loader), correct / total
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._is_loaded:
            return {
                "model_name": self.model_name,
                "loaded": False,
            }
        
        clip_params = sum(p.numel() for p in self.clip_model.parameters())
        clip_trainable = sum(
            p.numel() for p in self.clip_model.parameters() if p.requires_grad
        )
        head_params = sum(p.numel() for p in self.classification_head.parameters())
        head_trainable = sum(
            p.numel() for p in self.classification_head.parameters() if p.requires_grad
        )
        
        return {
            "model_name": self.model_name,
            "loaded": True,
            "device": str(self._device),
            "clip_parameters": clip_params,
            "clip_trainable": clip_trainable,  # Should be 0 (frozen)
            "head_parameters": head_params,
            "head_trainable": head_trainable,
            "embedding_dim": self.EMBEDDING_DIM,
            "head_architecture": self.hidden_dims,
        }
