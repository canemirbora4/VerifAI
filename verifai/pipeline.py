"""
VerifAI Pipeline
=================

Main orchestrator for the AI-generated media detection pipeline.
Combines all components: ingestion, detection, ensemble, and output formatting.
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

from verifai.ingest import (
    ImageLoader,
    VideoLoader,
    VideoData,
    validate_file_path,
    get_media_type,
    MediaType,
)
from verifai.models import NeuralDetector, FrequencyDetector, DetectorOutput
from verifai.models.base import Label
from verifai.features import MetadataParser, parse_metadata, TemporalAnalyzer, TemporalFeatures
from verifai.fusion import (
    Ensemble,
    EnsembleConfig,
    FusionMethod,
    Calibrator,
    Explainer,
    create_metadata_detector_output,
)


@dataclass
class FrameScore:
    """Score for a single video frame."""
    frame_number: int
    timestamp: float
    score: float
    is_suspicious: bool = False
    label: str = "unknown"


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
    
    # Calibration info
    raw_score: Optional[float] = None
    calibrated: bool = False
    
    # Evidence
    evidence: dict[str, Any] = field(default_factory=dict)
    heatmap: Optional[np.ndarray] = None
    
    # Input metadata
    input_path: Optional[str] = None
    input_type: str = "unknown"
    input_size: Optional[tuple[int, int]] = None
    
    # Processing metadata
    processing_time_ms: float = 0.0
    timestamp: str = ""
    version: str = "0.1.0"
    
    # Ensemble info
    fusion_method: Optional[str] = None
    detector_weights: dict[str, float] = field(default_factory=dict)
    
    # Video-specific fields
    frame_scores: list[FrameScore] = field(default_factory=list)
    suspicious_frames: list[int] = field(default_factory=list)
    temporal_consistency: Optional[float] = None
    video_duration: Optional[float] = None
    num_frames_analyzed: int = 0
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "result": {
                "label": self.label,
                "confidence": round(self.confidence, 4),
                "is_ai_generated": self.is_ai_generated,
            },
            "detector_scores": {
                k: round(v, 4) for k, v in self.detector_scores.items()
            },
            "calibration": {
                "raw_score": round(self.raw_score, 4) if self.raw_score else None,
                "calibrated": self.calibrated,
            },
            "evidence": {
                k: v if not isinstance(v, np.ndarray) else "array"
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
            "ensemble": {
                "fusion_method": self.fusion_method,
                "detector_weights": {
                    k: round(v, 3) for k, v in self.detector_weights.items()
                } if self.detector_weights else None,
            },
            "metadata": self.metadata,
        }
        
        # Include heatmap info if present
        if self.heatmap is not None:
            result["has_heatmap"] = True
        
        # Include video-specific info
        if self.input_type == "video":
            result["video"] = {
                "duration": self.video_duration,
                "frames_analyzed": self.num_frames_analyzed,
                "suspicious_frames": self.suspicious_frames,
                "temporal_consistency": round(self.temporal_consistency, 4) if self.temporal_consistency else None,
            }
            if self.frame_scores:
                result["video"]["frame_scores"] = [
                    {
                        "frame": fs.frame_number,
                        "timestamp": round(fs.timestamp, 2),
                        "score": round(fs.score, 4),
                        "suspicious": fs.is_suspicious,
                    }
                    for fs in self.frame_scores
                ]
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        label_emoji = "🤖" if self.is_ai_generated else "📷"
        media_emoji = "🎬" if self.input_type == "video" else "🖼️"
        confidence_bar = "█" * int(self.confidence * 10) + "░" * (10 - int(self.confidence * 10))
        
        lines = [
            "─" * 50,
            f"  VerifAI Detection Result {media_emoji}",
            "─" * 50,
            f"  {label_emoji} Verdict: {self.label.upper()}",
            f"  Confidence: [{confidence_bar}] {self.confidence:.1%}",
        ]
        
        if self.calibrated:
            lines.append(f"  (calibrated from {self.raw_score:.1%})")
        
        lines.append("─" * 50)
        
        if self.input_path:
            lines.append(f"  File: {Path(self.input_path).name}")
        if self.input_size:
            lines.append(f"  Size: {self.input_size[0]}×{self.input_size[1]}")
        if self.processing_time_ms > 0:
            lines.append(f"  Time: {self.processing_time_ms:.0f}ms")
        
        # Video-specific info
        if self.input_type == "video":
            lines.append("─" * 50)
            lines.append("  Video Analysis:")
            if self.video_duration:
                lines.append(f"    Duration: {self.video_duration:.1f}s")
            if self.num_frames_analyzed:
                lines.append(f"    Frames analyzed: {self.num_frames_analyzed}")
            if self.temporal_consistency is not None:
                tc_bar = "█" * int(self.temporal_consistency * 10) + "░" * (10 - int(self.temporal_consistency * 10))
                lines.append(f"    Temporal consistency: [{tc_bar}] {self.temporal_consistency:.1%}")
            if self.suspicious_frames:
                lines.append(f"    Suspicious frames: {len(self.suspicious_frames)}")
                # Show first few suspicious frame numbers
                frame_list = ", ".join(str(f) for f in self.suspicious_frames[:5])
                if len(self.suspicious_frames) > 5:
                    frame_list += f", ... (+{len(self.suspicious_frames) - 5} more)"
                lines.append(f"      Frames: {frame_list}")
        
        # Detector breakdown
        if self.detector_scores:
            lines.append("─" * 50)
            lines.append("  Detector Scores:")
            for name, score in self.detector_scores.items():
                weight = self.detector_weights.get(name, 0)
                lines.append(f"    {name}: {score:.1%} (weight: {weight:.0%})")
        
        lines.append("─" * 50)
        
        return "\n".join(lines)


class VerifAI:
    """
    Main VerifAI pipeline for AI-generated media detection.
    
    This pipeline combines multiple detection signals:
    - Neural detector (ViT-based)
    - Frequency detector (FFT/DCT-based)
    - Metadata analysis (EXIF/provenance)
    
    The outputs are combined via ensemble fusion and optionally calibrated.
    
    Usage:
        >>> detector = VerifAI()
        >>> result = detector.detect("image.jpg")
        >>> print(result.confidence, result.label)
        
    Or with custom configuration:
        >>> detector = VerifAI(
        ...     model_name="google/vit-large-patch16-224",
        ...     use_frequency=True,
        ...     use_metadata=True,
        ...     calibration_method="isotonic",
        ... )
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device: Optional[str] = None,
        threshold: float = 0.5,
        fp16: bool = True,
        auto_load: bool = True,
        # Ensemble settings
        use_frequency: bool = True,
        use_metadata: bool = True,
        fusion_method: str = "weighted",
        detector_weights: Optional[dict[str, float]] = None,
        # Calibration
        calibration_method: Optional[str] = None,
        calibration_path: Optional[str] = None,
        # Explainability
        generate_heatmaps: bool = False,
    ):
        """
        Initialize the VerifAI pipeline.
        
        Args:
            model_name: HuggingFace model ID or local path for neural detector
            device: Device for inference ("cuda", "mps", "cpu", or None for auto)
            threshold: Classification threshold
            fp16: Use FP16 inference (faster on GPU)
            auto_load: Automatically load models on first detection
            use_frequency: Enable frequency-based detection
            use_metadata: Enable metadata analysis
            fusion_method: Ensemble fusion method ("average", "weighted", "max")
            detector_weights: Custom weights for detectors
            calibration_method: Calibration method ("isotonic", "platt", None)
            calibration_path: Path to fitted calibrator
            generate_heatmaps: Generate explanation heatmaps
        """
        self.model_name = model_name
        self.threshold = threshold
        self.auto_load = auto_load
        self.use_frequency = use_frequency
        self.use_metadata = use_metadata
        self.generate_heatmaps = generate_heatmaps
        
        # Initialize components
        self._image_loader = ImageLoader()
        self._video_loader = VideoLoader()
        self._temporal_analyzer = TemporalAnalyzer()
        
        # Neural detector
        self._neural_detector = NeuralDetector(
            model_name=model_name,
            device=device,
            threshold=threshold,
            fp16=fp16,
        )
        
        # Frequency detector (optional)
        self._frequency_detector = None
        if use_frequency:
            self._frequency_detector = FrequencyDetector(
                device=device,
                threshold=threshold,
            )
        
        # Metadata parser (optional)
        self._metadata_parser = None
        if use_metadata:
            self._metadata_parser = MetadataParser()
        
        # Ensemble configuration
        active_detectors = ["neural"]
        default_weights = {"neural": 0.6}
        
        if use_frequency:
            active_detectors.append("frequency")
            default_weights["frequency"] = 0.25
        
        if use_metadata:
            active_detectors.append("metadata")
            default_weights["metadata"] = 0.15
        
        # Use custom weights if provided
        if detector_weights:
            default_weights.update(detector_weights)
        
        self._ensemble = Ensemble(EnsembleConfig(
            method=FusionMethod(fusion_method),
            weights=default_weights,
            detectors=active_detectors,
            threshold=threshold,
        ))
        
        # Calibrator (optional)
        self._calibrator = None
        if calibration_method:
            if calibration_path and Path(calibration_path).exists():
                self._calibrator = Calibrator.load(calibration_path)
            else:
                self._calibrator = Calibrator(method=calibration_method)
                logger.warning(
                    "Calibrator created but not fitted. "
                    "Call fit_calibrator() with validation data."
                )
        
        # Explainer
        self._explainer = Explainer() if generate_heatmaps else None
        
        self._is_loaded = False
        
        logger.info(
            f"VerifAI initialized: model={model_name}, "
            f"detectors={active_detectors}, fusion={fusion_method}"
        )
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded
    
    def load(self) -> None:
        """Load all models and prepare for inference."""
        if self._is_loaded:
            return
        
        logger.info("Loading VerifAI models...")
        
        # Load neural detector
        self._neural_detector.load()
        
        # Load frequency detector
        if self._frequency_detector:
            self._frequency_detector.load()
        
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
            return_evidence: Include detailed evidence in output
            
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
        elif input_type == "video":
            result = self._detect_video(
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
        Run detection on an image using the full ensemble.
        
        Args:
            source: Image source
            return_evidence: Include detailed evidence
            
        Returns:
            DetectionResult
        """
        # Load and preprocess image
        image_data = self._image_loader.load(source, preprocess=True)
        
        # Collect detector outputs
        detector_outputs = {}
        
        # 1. Neural detector
        neural_output = self._neural_detector.detect(
            image_data.tensor,
            return_features=False,
            return_evidence=return_evidence and self.generate_heatmaps,
        )
        detector_outputs["neural"] = neural_output
        
        # 2. Frequency detector
        if self._frequency_detector:
            freq_output = self._frequency_detector.detect(
                image_data.tensor,
                return_features=False,
                return_evidence=return_evidence,
            )
            detector_outputs["frequency"] = freq_output
        
        # 3. Metadata analysis
        metadata_features = None
        if self._metadata_parser and image_data.source_path:
            metadata_features = self._metadata_parser.parse(image_data.source_path)
            metadata_output = create_metadata_detector_output(metadata_features)
            detector_outputs["metadata"] = metadata_output
        
        # Ensemble fusion
        ensemble_result = self._ensemble.fuse(detector_outputs)
        
        # Get raw score before calibration
        raw_score = ensemble_result.final_score
        calibrated = False
        final_confidence = raw_score
        
        # Apply calibration if available and fitted
        if self._calibrator and self._calibrator.is_fitted:
            final_confidence = self._calibrator.calibrate(raw_score)
            calibrated = True
        
        # Determine final label based on calibrated score
        if final_confidence >= self.threshold:
            final_label = Label.AI_GENERATED
        else:
            final_label = Label.REAL
        
        # Build result
        result = DetectionResult(
            label=final_label.value,
            confidence=final_confidence,
            is_ai_generated=final_label == Label.AI_GENERATED,
            detector_scores=ensemble_result.detector_scores,
            raw_score=raw_score,
            calibrated=calibrated,
            input_size=(image_data.width, image_data.height),
            fusion_method=ensemble_result.fusion_method,
            detector_weights=ensemble_result.detector_weights,
            metadata={
                "exif_present": bool(image_data.exif),
            }
        )
        
        # Add evidence if requested
        if return_evidence:
            evidence = {}
            
            # Neural detector evidence
            if neural_output.evidence:
                evidence["neural"] = neural_output.evidence
            
            # Frequency detector evidence
            if "frequency" in detector_outputs and detector_outputs["frequency"].evidence:
                evidence["frequency"] = detector_outputs["frequency"].evidence
            
            # Metadata evidence
            if metadata_features:
                evidence["metadata"] = {
                    "has_camera_info": metadata_features.has_camera_info,
                    "camera_make": metadata_features.camera_make,
                    "camera_model": metadata_features.camera_model,
                    "is_suspicious": metadata_features.is_suspicious,
                    "suspicion_reasons": metadata_features.suspicion_reasons,
                }
            
            result.evidence = evidence
        
        # Generate heatmap if requested
        if self.generate_heatmaps and self._explainer:
            try:
                explanation = self._explainer.explain(
                    image_data.original,
                    model=self._neural_detector.model,
                    input_tensor=image_data.tensor,
                    method="gradcam",
                )
                result.heatmap = explanation.heatmap
                result.evidence["suspicious_regions"] = explanation.suspicious_regions
            except Exception as e:
                logger.warning(f"Heatmap generation failed: {e}")
        
        return result
    
    def _detect_video(
        self,
        source: Union[str, Path],
        num_frames: int = 16,
        return_evidence: bool = False,
    ) -> DetectionResult:
        """
        Run detection on a video using frame-level analysis and temporal aggregation.
        
        Args:
            source: Video file path
            num_frames: Number of frames to sample
            return_evidence: Include detailed evidence
            
        Returns:
            DetectionResult with video-specific data
        """
        # Load video and extract frames
        video_data = self._video_loader.load(
            source,
            num_frames=num_frames,
            strategy="uniform",
            preprocess=True,
        )
        
        logger.info(
            f"Analyzing video: {video_data.metadata.duration:.1f}s, "
            f"{len(video_data.frames)} frames"
        )
        
        # Collect per-frame scores
        frame_scores_list = []
        all_detector_scores = {"neural": [], "frequency": [], "metadata": []}
        
        for frame in video_data.frames:
            # Run detectors on each frame
            frame_detector_outputs = {}
            
            # Neural detector
            neural_output = self._neural_detector.detect(
                frame.tensor,
                return_features=False,
                return_evidence=False,
            )
            frame_detector_outputs["neural"] = neural_output
            all_detector_scores["neural"].append(neural_output.ai_probability)
            
            # Frequency detector
            if self._frequency_detector:
                freq_output = self._frequency_detector.detect(
                    frame.tensor,
                    return_features=False,
                    return_evidence=False,
                )
                frame_detector_outputs["frequency"] = freq_output
                all_detector_scores["frequency"].append(freq_output.ai_probability)
            
            # Ensemble frame-level scores
            frame_ensemble = self._ensemble.fuse(frame_detector_outputs)
            frame_score = frame_ensemble.final_score
            
            # Determine if frame is suspicious
            is_suspicious = frame_score >= self.threshold
            frame_label = "ai_generated" if is_suspicious else "real"
            
            frame_scores_list.append(FrameScore(
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                score=frame_score,
                is_suspicious=is_suspicious,
                label=frame_label,
            ))
        
        # Temporal analysis on frame images
        temporal_features = self._temporal_analyzer.analyze(video_data.images)
        
        # Aggregate scores across frames
        frame_score_values = [fs.score for fs in frame_scores_list]
        
        # Different aggregation strategies
        mean_score = float(np.mean(frame_score_values))
        max_score = float(np.max(frame_score_values))
        
        # Weighted aggregation: higher weight to suspicious frames
        weights = [1.5 if fs.is_suspicious else 1.0 for fs in frame_scores_list]
        weighted_score = float(np.average(frame_score_values, weights=weights))
        
        # Final score considers both frame scores and temporal consistency
        # Lower temporal consistency increases suspicion
        temporal_penalty = (1.0 - temporal_features.consistency_score) * 0.1
        final_score = min(1.0, weighted_score + temporal_penalty)
        
        # Apply calibration if available
        raw_score = final_score
        calibrated = False
        if self._calibrator and self._calibrator.is_fitted:
            final_score = self._calibrator.calibrate(final_score)
            calibrated = True
        
        # Determine final label
        if final_score >= self.threshold:
            final_label = Label.AI_GENERATED
        else:
            final_label = Label.REAL
        
        # Find suspicious frames
        suspicious_frames = [
            fs.frame_number for fs in frame_scores_list
            if fs.is_suspicious
        ]
        
        # Also add temporally suspicious frames
        for tf_idx in temporal_features.suspicious_frames:
            if tf_idx < len(frame_scores_list):
                if frame_scores_list[tf_idx].frame_number not in suspicious_frames:
                    suspicious_frames.append(frame_scores_list[tf_idx].frame_number)
        
        suspicious_frames = sorted(set(suspicious_frames))
        
        # Build detector scores (averaged across frames)
        avg_detector_scores = {}
        for detector_name, scores in all_detector_scores.items():
            if scores:
                avg_detector_scores[detector_name] = float(np.mean(scores))
        
        # Add temporal score as a virtual detector
        avg_detector_scores["temporal"] = 1.0 - temporal_features.consistency_score
        
        # Build result
        result = DetectionResult(
            label=final_label.value,
            confidence=final_score,
            is_ai_generated=final_label == Label.AI_GENERATED,
            detector_scores=avg_detector_scores,
            raw_score=raw_score,
            calibrated=calibrated,
            input_size=video_data.metadata.resolution,
            fusion_method="temporal_weighted",
            detector_weights=self._ensemble.config.weights,
            frame_scores=frame_scores_list,
            suspicious_frames=suspicious_frames,
            temporal_consistency=temporal_features.consistency_score,
            video_duration=video_data.metadata.duration,
            num_frames_analyzed=len(video_data.frames),
            metadata={
                "fps": video_data.metadata.fps,
                "total_frames": video_data.metadata.total_frames,
                "codec": video_data.metadata.codec,
                "sampling_strategy": video_data.sampling_strategy,
            },
        )
        
        # Add evidence if requested
        if return_evidence:
            evidence = {
                "temporal": temporal_features.to_dict(),
                "frame_analysis": {
                    "mean_score": mean_score,
                    "max_score": max_score,
                    "weighted_score": weighted_score,
                    "score_variance": float(np.var(frame_score_values)),
                    "suspicious_ratio": len(suspicious_frames) / len(frame_scores_list),
                },
                "aggregation": {
                    "method": "temporal_weighted",
                    "temporal_penalty": temporal_penalty,
                },
            }
            result.evidence = evidence
        
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
    
    def fit_calibrator(
        self,
        images: list[Union[str, Path]],
        labels: list[int],
    ) -> None:
        """
        Fit the calibrator on validation data.
        
        Args:
            images: List of image paths
            labels: Ground truth labels (0=real, 1=AI)
        """
        if self._calibrator is None:
            logger.warning("No calibrator configured")
            return
        
        logger.info(f"Fitting calibrator on {len(images)} samples...")
        
        # Get scores for all images
        scores = []
        for img_path in images:
            try:
                result = self.detect(img_path)
                scores.append(result.raw_score or result.confidence)
            except Exception as e:
                logger.warning(f"Skipping {img_path}: {e}")
                continue
        
        if len(scores) != len(labels):
            logger.error("Mismatch between scores and labels")
            return
        
        # Fit calibrator
        import numpy as np
        self._calibrator.fit(np.array(scores), np.array(labels))
        logger.info("Calibrator fitted successfully")
    
    def save_calibrator(self, path: str) -> None:
        """Save fitted calibrator to file."""
        if self._calibrator:
            self._calibrator.save(path)
    
    def get_info(self) -> dict:
        """Get information about the pipeline configuration."""
        info = {
            "version": "0.1.0",
            "model": self.model_name,
            "threshold": self.threshold,
            "is_loaded": self._is_loaded,
            "detectors": {
                "neural": True,
                "frequency": self._frequency_detector is not None,
                "metadata": self._metadata_parser is not None,
            },
            "ensemble": {
                "method": self._ensemble.config.method.value,
                "weights": self._ensemble.config.weights,
            },
            "calibration": {
                "enabled": self._calibrator is not None,
                "method": self._calibrator.method if self._calibrator else None,
                "fitted": self._calibrator.is_fitted if self._calibrator else False,
            },
            "heatmaps": self.generate_heatmaps,
        }
        
        if self._is_loaded:
            info["neural_detector"] = self._neural_detector.get_model_info()
            if self._frequency_detector:
                info["frequency_detector"] = self._frequency_detector.get_model_info()
        
        return info