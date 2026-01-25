"""
Temporal Features
==================

Analyzes temporal consistency in videos to detect AI-generated content.

AI-generated videos often have:
- Inconsistent noise patterns across frames
- Temporal flicker (sudden brightness/color changes)
- Unnatural motion patterns
- Frame-to-frame artifacts that don't match camera physics

This module extracts features that capture these temporal inconsistencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from PIL import Image
from loguru import logger


@dataclass
class TemporalFeatures:
    """
    Container for temporal analysis results.
    
    Attributes:
        flicker_score: Measure of frame-to-frame brightness variation
        consistency_score: Overall temporal consistency (1.0 = consistent)
        noise_variance: Variance in noise patterns across frames
        motion_smoothness: Smoothness of inter-frame motion
        suspicious_frames: Indices of frames with anomalies
        frame_scores: Per-frame anomaly scores
        feature_vector: Combined feature vector
    """
    
    # Core metrics
    flicker_score: float = 0.0
    consistency_score: float = 1.0
    noise_variance: float = 0.0
    motion_smoothness: float = 1.0
    
    # Per-frame analysis
    suspicious_frames: list[int] = field(default_factory=list)
    frame_scores: list[float] = field(default_factory=list)
    
    # Detailed metrics
    brightness_stability: float = 1.0
    color_stability: float = 1.0
    edge_consistency: float = 1.0
    
    # Combined features
    feature_vector: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        return {
            "flicker_score": round(self.flicker_score, 4),
            "consistency_score": round(self.consistency_score, 4),
            "noise_variance": round(self.noise_variance, 4),
            "motion_smoothness": round(self.motion_smoothness, 4),
            "suspicious_frames": self.suspicious_frames,
            "num_suspicious": len(self.suspicious_frames),
            "brightness_stability": round(self.brightness_stability, 4),
            "color_stability": round(self.color_stability, 4),
            "edge_consistency": round(self.edge_consistency, 4),
        }
    
    @property
    def is_suspicious(self) -> bool:
        """Check if video shows signs of being AI-generated."""
        return (
            self.flicker_score > 0.3 or
            self.consistency_score < 0.7 or
            len(self.suspicious_frames) > 3
        )


class TemporalAnalyzer:
    """
    Analyzes temporal consistency in video frames.
    
    Usage:
        analyzer = TemporalAnalyzer()
        frames = [frame1, frame2, frame3, ...]  # PIL Images or numpy arrays
        features = analyzer.analyze(frames)
        
        if features.is_suspicious:
            print("Video may be AI-generated")
    """
    
    def __init__(
        self,
        flicker_threshold: float = 0.1,
        consistency_window: int = 5,
    ):
        """
        Initialize the temporal analyzer.
        
        Args:
            flicker_threshold: Threshold for detecting flicker
            consistency_window: Number of frames for rolling consistency
        """
        self.flicker_threshold = flicker_threshold
        self.consistency_window = consistency_window
    
    def analyze(
        self,
        frames: list[Union[Image.Image, np.ndarray]],
    ) -> TemporalFeatures:
        """
        Analyze temporal consistency across frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            TemporalFeatures with analysis results
        """
        if len(frames) < 2:
            logger.warning("Need at least 2 frames for temporal analysis")
            return TemporalFeatures()
        
        # Convert to numpy arrays
        arrays = self._to_arrays(frames)
        
        # Compute individual metrics
        flicker_score, brightness_diffs = self._compute_flicker(arrays)
        brightness_stability = self._compute_brightness_stability(arrays)
        color_stability = self._compute_color_stability(arrays)
        edge_consistency = self._compute_edge_consistency(arrays)
        noise_variance = self._compute_noise_variance(arrays)
        motion_smoothness = self._compute_motion_smoothness(arrays)
        
        # Find suspicious frames
        suspicious_frames, frame_scores = self._find_suspicious_frames(
            arrays, brightness_diffs
        )
        
        # Overall consistency score
        consistency_score = self._compute_overall_consistency(
            flicker_score,
            brightness_stability,
            color_stability,
            edge_consistency,
            noise_variance,
        )
        
        # Build feature vector
        feature_vector = np.array([
            flicker_score,
            consistency_score,
            noise_variance,
            motion_smoothness,
            brightness_stability,
            color_stability,
            edge_consistency,
            len(suspicious_frames) / len(frames),  # Suspicious ratio
        ], dtype=np.float32)
        
        return TemporalFeatures(
            flicker_score=flicker_score,
            consistency_score=consistency_score,
            noise_variance=noise_variance,
            motion_smoothness=motion_smoothness,
            suspicious_frames=suspicious_frames,
            frame_scores=frame_scores,
            brightness_stability=brightness_stability,
            color_stability=color_stability,
            edge_consistency=edge_consistency,
            feature_vector=feature_vector,
        )
    
    def _to_arrays(
        self,
        frames: list[Union[Image.Image, np.ndarray]],
    ) -> list[np.ndarray]:
        """Convert frames to numpy arrays."""
        arrays = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                arr = np.array(frame.convert("RGB"))
            else:
                arr = frame
            arrays.append(arr.astype(np.float32))
        return arrays
    
    def _compute_flicker(
        self,
        frames: list[np.ndarray],
    ) -> tuple[float, list[float]]:
        """
        Compute flicker score (brightness variation between frames).
        
        Flicker indicates unnatural frame-to-frame changes that
        don't match how real cameras capture video.
        """
        brightness_diffs = []
        
        for i in range(1, len(frames)):
            # Mean brightness of each frame
            prev_brightness = np.mean(frames[i-1])
            curr_brightness = np.mean(frames[i])
            
            # Relative difference
            diff = abs(curr_brightness - prev_brightness) / (prev_brightness + 1e-6)
            brightness_diffs.append(diff)
        
        if not brightness_diffs:
            return 0.0, []
        
        # Flicker score = std of brightness changes
        # (consistent video has low std, flickering video has high std)
        flicker_score = float(np.std(brightness_diffs))
        
        return flicker_score, brightness_diffs
    
    def _compute_brightness_stability(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """Compute brightness stability across frames."""
        brightnesses = [np.mean(f) for f in frames]
        
        if not brightnesses:
            return 1.0
        
        # Coefficient of variation (std / mean)
        mean_b = np.mean(brightnesses)
        std_b = np.std(brightnesses)
        
        cv = std_b / (mean_b + 1e-6)
        
        # Convert to stability score (lower CV = higher stability)
        stability = max(0.0, 1.0 - cv * 5)
        return float(stability)
    
    def _compute_color_stability(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """Compute color distribution stability."""
        color_means = []
        
        for frame in frames:
            if frame.ndim == 3 and frame.shape[2] >= 3:
                # Mean of each channel
                r_mean = np.mean(frame[:, :, 0])
                g_mean = np.mean(frame[:, :, 1])
                b_mean = np.mean(frame[:, :, 2])
                color_means.append([r_mean, g_mean, b_mean])
        
        if len(color_means) < 2:
            return 1.0
        
        color_means = np.array(color_means)
        
        # Variance of color means across frames
        variance = np.mean(np.var(color_means, axis=0))
        
        # Normalize to [0, 1] stability score
        stability = max(0.0, 1.0 - variance / 1000)
        return float(stability)
    
    def _compute_edge_consistency(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """
        Compute edge consistency across frames.
        
        AI videos often have inconsistent edge sharpness/artifacts.
        """
        edge_strengths = []
        
        for frame in frames:
            # Convert to grayscale if needed
            if frame.ndim == 3:
                gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
            else:
                gray = frame
            
            # Simple edge detection (Sobel-like)
            dx = np.abs(np.diff(gray, axis=1))
            dy = np.abs(np.diff(gray, axis=0))
            
            edge_strength = np.mean(dx) + np.mean(dy)
            edge_strengths.append(edge_strength)
        
        if len(edge_strengths) < 2:
            return 1.0
        
        # Consistency = inverse of variance
        variance = np.var(edge_strengths)
        mean_strength = np.mean(edge_strengths)
        
        cv = variance / (mean_strength + 1e-6)
        consistency = max(0.0, 1.0 - cv)
        
        return float(consistency)
    
    def _compute_noise_variance(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """
        Estimate noise pattern variance across frames.
        
        Real videos have consistent sensor noise; AI videos may have
        varying or absent noise patterns.
        """
        noise_levels = []
        
        for frame in frames:
            # Estimate noise using local variance
            if frame.ndim == 3:
                gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
            else:
                gray = frame
            
            # High-pass filter to isolate noise
            # (subtract smoothed version)
            from scipy.ndimage import uniform_filter
            smoothed = uniform_filter(gray, size=5)
            noise = gray - smoothed
            
            noise_level = np.std(noise)
            noise_levels.append(noise_level)
        
        if len(noise_levels) < 2:
            return 0.0
        
        # Variance of noise levels
        return float(np.var(noise_levels))
    
    def _compute_motion_smoothness(
        self,
        frames: list[np.ndarray],
    ) -> float:
        """
        Compute motion smoothness between frames.
        
        Real videos have smooth, physically plausible motion.
        AI videos may have jerky or inconsistent motion.
        """
        if len(frames) < 3:
            return 1.0
        
        # Simple frame difference magnitude
        diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i] - frames[i-1]))
            diffs.append(diff)
        
        if len(diffs) < 2:
            return 1.0
        
        # Second derivative of motion (acceleration)
        # Smooth motion = low acceleration variance
        accelerations = np.diff(diffs)
        acc_variance = np.var(accelerations)
        
        # Normalize
        smoothness = max(0.0, 1.0 - acc_variance / 100)
        return float(smoothness)
    
    def _find_suspicious_frames(
        self,
        frames: list[np.ndarray],
        brightness_diffs: list[float],
    ) -> tuple[list[int], list[float]]:
        """Find frames with anomalous characteristics."""
        suspicious = []
        scores = [0.0]  # First frame has no predecessor
        
        for i, diff in enumerate(brightness_diffs):
            frame_idx = i + 1  # Offset by 1
            
            score = 0.0
            
            # High brightness change
            if diff > self.flicker_threshold:
                score += diff * 2
            
            # Check for sudden changes
            if i > 0 and i < len(brightness_diffs) - 1:
                prev_diff = brightness_diffs[i - 1]
                next_diff = brightness_diffs[i + 1] if i + 1 < len(brightness_diffs) else diff
                
                # Spike detection
                if diff > 2 * prev_diff and diff > 2 * next_diff:
                    score += 0.5
            
            scores.append(score)
            
            if score > 0.3:
                suspicious.append(frame_idx)
        
        return suspicious, scores
    
    def _compute_overall_consistency(
        self,
        flicker_score: float,
        brightness_stability: float,
        color_stability: float,
        edge_consistency: float,
        noise_variance: float,
    ) -> float:
        """Compute overall temporal consistency score."""
        # Weighted combination
        weights = {
            "flicker": 0.25,
            "brightness": 0.2,
            "color": 0.2,
            "edge": 0.2,
            "noise": 0.15,
        }
        
        # Convert metrics to consistency scores
        flicker_consistency = max(0.0, 1.0 - flicker_score * 3)
        noise_consistency = max(0.0, 1.0 - noise_variance / 10)
        
        consistency = (
            weights["flicker"] * flicker_consistency +
            weights["brightness"] * brightness_stability +
            weights["color"] * color_stability +
            weights["edge"] * edge_consistency +
            weights["noise"] * noise_consistency
        )
        
        return float(consistency)


def analyze_temporal(
    frames: list[Union[Image.Image, np.ndarray]],
) -> TemporalFeatures:
    """
    Convenience function to analyze temporal features.
    
    Args:
        frames: List of video frames
        
    Returns:
        TemporalFeatures object
    """
    analyzer = TemporalAnalyzer()
    return analyzer.analyze(frames)
