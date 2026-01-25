"""
VerifAI Pipeline
=================

Main orchestrator for the AI-generated media detection pipeline.
Combines all components: ingestion, detection, and output formatting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Any
import json
import time

import torch
import numpy as np
from PIL import Image
from loguru import logger

from verifai.ingest import ImageLoader, validate_file_path, get_media_type, MediaType
from verifai.models import NeuralDetector, DetectorOutput
from verifai.models.base import Label


@dataclass
class DetectionResult:
    """
    Complete detection result for a media file.
    
    This is the primary output of the VerifAI pipeline, containing:
    - Classification result (label + confidence)
    - Evidence for explainability
    - Metadata about the input and processing
    """
    
    # Classification result
    label: str
    confidence: float
    is_ai_generated: bool
    
    # Detailed scores from each detector
    detector_scores: dict[str, float] = field(default_factory=dict)
    
    # Evidence
    evidence: dict[str, Any] = field(default_factory=dict)
    
    # Input metadata
    input_path: Optional[str] = None
    input_type: str = "unknown"
    input_size: Optional[tuple[int, int]] = None
    
    # Processing metadata
    processing_time_ms: float = 0.0
    timestamp: str = ""
    version: str = "0.1.0"
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "result": {
                "label": self.label,
                "confidence": round(self.confidence, 4),
                "is_ai_generated": self.is_ai_generated,
            },
            "detector_scores": {
                k: round(v, 4) for k, v in self.detector_scores.items()
            },
            "evidence": {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in self.evidence.items()
            },
            "input": {
                "path": self.input_path,
                "type": self.input_type,
                "size": self.input_size,
            },
            "processing": {
                "time_ms": round(self.processing_time_ms, 2),
                "timestamp": self.timestamp,
                "version": self.version,
            },
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        label_emoji = "🤖" if self.is_ai_generated else "📷"
        confidence_bar = "█" * int(self.confidence * 10) + "░" * (10 - int(self.confidence * 10))
        
        lines = [
            "─" * 50,
            f"  VerifAI Detection Result",
            "─" * 50,
            f"  {label_emoji} Verdict: {self.label.upper()}",
            f"  Confidence: [{confidence_bar}] {self.confidence:.1%}",
            "─" * 50,
        ]
        
        if self.input_path:
            lines.append(f"  File: {Path(self.input_path).name}")
        if self.input_size:
            lines.append(f"  Size: {self.input_size[0]}×{self.input_size[1]}")
        if self.processing_time_ms > 0:
            lines.append(f"  Time: {self.processing_time_ms:.0f}ms")
        
        lines.append("─" * 50)
        
        return "\n".join(lines)


class VerifAI:
    """
    Main VerifAI pipeline for AI-generated media detection.
    
    Usage:
        >>> detector = VerifAI()
        >>> result = detector.detect("image.jpg")
        >>> print(result.confidence, result.label)
        
    Or with custom configuration:
        >>> detector = VerifAI(
        ...     model_name="google/vit-large-patch16-224",
        ...     device="cuda",
        ...     threshold=0.6,
        ... )
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device: Optional[str] = None,
        threshold: float = 0.5,
        fp16: bool = True,
        auto_load: bool = True,
    ):
        """
        Initialize the VerifAI pipeline.
        
        Args:
            model_name: HuggingFace model ID or local path
            device: Device for inference ("cuda", "mps", "cpu", or None for auto)
            threshold: Classification threshold
            fp16: Use FP16 inference (faster on GPU)
            auto_load: Automatically load model on first detection
        """
        self.model_name = model_name
        self.threshold = threshold
        self.auto_load = auto_load
        
        # Initialize components
        self._image_loader = ImageLoader()
        self._neural_detector = NeuralDetector(
            model_name=model_name,
            device=device,
            threshold=threshold,
            fp16=fp16,
        )
        
        self._is_loaded = False
        
        logger.info(f"VerifAI initialized with model: {model_name}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded
    
    def load(self) -> None:
        """Load all models and prepare for inference."""
        if self._is_loaded:
            return
        
        logger.info("Loading VerifAI models...")
        self._neural_detector.load()
        self._is_loaded = True
        logger.info("VerifAI ready for detection")
    
    def detect(
        self,
        source: Union[str, Path, bytes, Image.Image],
        return_evidence: bool = False,
    ) -> DetectionResult:
        """
        Detect if media is AI-generated.
        
        Args:
            source: Input source - file path, bytes, or PIL Image
            return_evidence: Include evidence (attention maps, etc.)
            
        Returns:
            DetectionResult with classification and metadata
        """
        start_time = time.perf_counter()
        
        # Auto-load if needed
        if self.auto_load and not self._is_loaded:
            self.load()
        
        # Determine input type and validate
        input_path = None
        input_type = "unknown"
        
        if isinstance(source, (str, Path)):
            path = validate_file_path(source)
            input_path = str(path)
            media_type = get_media_type(path)
            
            if media_type == MediaType.IMAGE:
                input_type = "image"
            elif media_type == MediaType.VIDEO:
                input_type = "video"
                raise NotImplementedError(
                    "Video detection not yet implemented. Coming in Phase 4!"
                )
            else:
                raise ValueError(f"Unsupported media type: {path.suffix}")
        else:
            input_type = "image"  # Assume image for bytes/PIL
        
        # Process based on type
        if input_type == "image":
            result = self._detect_image(
                source,
                return_evidence=return_evidence,
            )
        else:
            raise NotImplementedError(f"Detection for {input_type} not implemented")
        
        # Add input metadata
        result.input_path = input_path
        result.input_type = input_type
        
        # Calculate processing time
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def _detect_image(
        self,
        source: Union[str, Path, bytes, Image.Image],
        return_evidence: bool = False,
    ) -> DetectionResult:
        """
        Run detection on an image.
        
        Args:
            source: Image source
            return_evidence: Include attention evidence
            
        Returns:
            DetectionResult
        """
        # Load and preprocess image
        image_data = self._image_loader.load(source, preprocess=True)
        
        # Run neural detector
        detector_output = self._neural_detector.detect(
            image_data.tensor,
            return_features=False,
            return_evidence=return_evidence,
        )
        
        # Determine final label and confidence
        # In Phase 1, we only have neural detector
        # Future phases will combine multiple detectors
        final_confidence = detector_output.confidence
        final_label = detector_output.label
        
        # Build result
        result = DetectionResult(
            label=final_label.value,
            confidence=final_confidence,
            is_ai_generated=final_label == Label.AI_GENERATED,
            detector_scores={
                "neural": detector_output.raw_score,
            },
            input_size=(image_data.width, image_data.height),
            metadata={
                "exif_present": bool(image_data.exif),
            }
        )
        
        # Add evidence if requested
        if return_evidence and detector_output.evidence:
            result.evidence = detector_output.evidence
        
        return result
    
    def detect_batch(
        self,
        sources: list[Union[str, Path, bytes, Image.Image]],
        return_evidence: bool = False,
    ) -> list[DetectionResult]:
        """
        Run detection on multiple inputs.
        
        Args:
            sources: List of input sources
            return_evidence: Include evidence for each
            
        Returns:
            List of DetectionResult
        """
        # Auto-load if needed
        if self.auto_load and not self._is_loaded:
            self.load()
        
        results = []
        for source in sources:
            try:
                result = self.detect(source, return_evidence=return_evidence)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {source}: {e}")
                # Create error result
                error_result = DetectionResult(
                    label="error",
                    confidence=0.0,
                    is_ai_generated=False,
                    input_path=str(source) if isinstance(source, (str, Path)) else None,
                    metadata={"error": str(e)},
                )
                results.append(error_result)
        
        return results
    
    def get_info(self) -> dict:
        """Get information about the pipeline configuration."""
        return {
            "version": "0.1.0",
            "model": self.model_name,
            "threshold": self.threshold,
            "is_loaded": self._is_loaded,
            "neural_detector": self._neural_detector.get_model_info(),
        }
