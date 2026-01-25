"""
Tests for Video Pipeline
=========================

Tests for video loading, temporal analysis, and video detection.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from verifai.features.temporal import (
    TemporalAnalyzer,
    TemporalFeatures,
    analyze_temporal,
)


# =============================================================================
# Temporal Analyzer Tests
# =============================================================================

class TestTemporalAnalyzer:
    """Tests for TemporalAnalyzer class."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = TemporalAnalyzer()
        assert analyzer.flicker_threshold == 0.1
        assert analyzer.consistency_window == 5
    
    def test_analyze_requires_multiple_frames(self):
        """Test that analyzer requires at least 2 frames."""
        analyzer = TemporalAnalyzer()
        
        # Single frame
        single_frame = [np.zeros((100, 100, 3), dtype=np.float32)]
        features = analyzer.analyze(single_frame)
        
        # Should return default features
        assert features.flicker_score == 0.0
        assert features.consistency_score == 1.0
    
    def test_analyze_consistent_frames(self):
        """Test analysis of consistent (identical) frames."""
        analyzer = TemporalAnalyzer()
        
        # Create 10 identical frames
        frames = [np.ones((100, 100, 3), dtype=np.float32) * 128] * 10
        
        features = analyzer.analyze(frames)
        
        # Identical frames should have:
        # - Low flicker score
        # - High consistency score
        # - No suspicious frames
        assert features.flicker_score < 0.1
        assert features.consistency_score > 0.8
        assert len(features.suspicious_frames) == 0
    
    def test_analyze_flickering_frames(self):
        """Test analysis of flickering (inconsistent brightness) frames."""
        analyzer = TemporalAnalyzer()
        
        # Create alternating bright/dark frames
        frames = []
        for i in range(10):
            brightness = 255 if i % 2 == 0 else 50
            frame = np.ones((100, 100, 3), dtype=np.float32) * brightness
            frames.append(frame)
        
        features = analyzer.analyze(frames)
        
        # Flickering frames should have:
        # - High flicker score
        # - Lower consistency score
        assert features.flicker_score > 0.1
        assert features.brightness_stability < 0.8
        assert features.is_suspicious
    
    def test_analyze_pil_images(self):
        """Test analysis with PIL Images."""
        analyzer = TemporalAnalyzer()
        
        # Create frames as PIL Images
        frames = []
        for _ in range(5):
            img = Image.new("RGB", (100, 100), color=(128, 128, 128))
            frames.append(img)
        
        features = analyzer.analyze(frames)
        
        # Should process successfully
        assert features.consistency_score > 0.8
        assert features.feature_vector is not None
        assert len(features.feature_vector) == 8
    
    def test_feature_vector(self):
        """Test that feature vector is computed correctly."""
        analyzer = TemporalAnalyzer()
        
        frames = [np.ones((100, 100, 3), dtype=np.float32) * 100] * 5
        features = analyzer.analyze(frames)
        
        # Feature vector should contain all metrics
        assert features.feature_vector is not None
        assert len(features.feature_vector) == 8
        assert features.feature_vector.dtype == np.float32
    
    def test_temporal_features_to_dict(self):
        """Test TemporalFeatures.to_dict()."""
        features = TemporalFeatures(
            flicker_score=0.15,
            consistency_score=0.85,
            noise_variance=0.02,
            motion_smoothness=0.95,
            suspicious_frames=[3, 7],
            brightness_stability=0.9,
            color_stability=0.88,
            edge_consistency=0.92,
        )
        
        d = features.to_dict()
        
        assert d["flicker_score"] == 0.15
        assert d["consistency_score"] == 0.85
        assert d["num_suspicious"] == 2
        assert "brightness_stability" in d
    
    def test_temporal_features_is_suspicious(self):
        """Test TemporalFeatures.is_suspicious property."""
        # High flicker = suspicious
        features1 = TemporalFeatures(flicker_score=0.4, consistency_score=0.8)
        assert features1.is_suspicious
        
        # Low consistency = suspicious
        features2 = TemporalFeatures(flicker_score=0.1, consistency_score=0.5)
        assert features2.is_suspicious
        
        # Many suspicious frames = suspicious
        features3 = TemporalFeatures(
            flicker_score=0.1,
            consistency_score=0.8,
            suspicious_frames=[1, 2, 3, 4],
        )
        assert features3.is_suspicious
        
        # Normal = not suspicious
        features4 = TemporalFeatures(
            flicker_score=0.1,
            consistency_score=0.9,
            suspicious_frames=[1],
        )
        assert not features4.is_suspicious


class TestAnalyzeTemporalFunction:
    """Tests for convenience function."""
    
    def test_analyze_temporal(self):
        """Test analyze_temporal convenience function."""
        frames = [np.ones((50, 50, 3), dtype=np.float32) * 100] * 5
        
        features = analyze_temporal(frames)
        
        assert isinstance(features, TemporalFeatures)
        assert features.consistency_score > 0


# =============================================================================
# Video Loader Tests (with mocking)
# =============================================================================

class TestVideoLoaderMocked:
    """Tests for VideoLoader with mocked dependencies."""
    
    @pytest.fixture
    def mock_imageio(self):
        """Mock imageio for testing without actual video files."""
        with patch.dict('sys.modules', {
            'imageio': MagicMock(),
            'imageio.v3': MagicMock(),
            'imageio_ffmpeg': MagicMock(),
        }):
            yield
    
    def test_video_metadata_dataclass(self):
        """Test VideoMetadata dataclass."""
        from verifai.ingest.video_loader import VideoMetadata
        
        metadata = VideoMetadata(
            path=Path("/test/video.mp4"),
            duration=10.5,
            fps=30.0,
            total_frames=315,
            width=1920,
            height=1080,
            codec="h264",
        )
        
        assert metadata.resolution == (1920, 1080)
        assert metadata.aspect_ratio == pytest.approx(1920 / 1080, rel=0.01)
        
        d = metadata.to_dict()
        assert d["duration"] == 10.5
        assert d["fps"] == 30.0
    
    def test_video_frame_dataclass(self):
        """Test VideoFrame dataclass."""
        from verifai.ingest.video_loader import VideoFrame
        import torch
        
        image = Image.new("RGB", (224, 224), color="red")
        tensor = torch.randn(3, 224, 224)
        
        frame = VideoFrame(
            image=image,
            tensor=tensor,
            frame_number=42,
            timestamp=1.4,
        )
        
        assert frame.frame_number == 42
        assert frame.timestamp == 1.4
        assert frame.numpy.shape == (224, 224, 3)
    
    def test_video_data_container(self):
        """Test VideoData container."""
        from verifai.ingest.video_loader import VideoData, VideoFrame, VideoMetadata
        import torch
        
        frames = [
            VideoFrame(
                image=Image.new("RGB", (224, 224)),
                tensor=torch.randn(3, 224, 224),
                frame_number=i,
                timestamp=i * 0.5,
            )
            for i in range(5)
        ]
        
        video_data = VideoData(
            frames=frames,
            metadata=VideoMetadata(fps=30.0, duration=5.0),
            sampling_strategy="uniform",
            frame_indices=[0, 30, 60, 90, 120],
        )
        
        assert len(video_data) == 5
        assert video_data[0].frame_number == 0
        assert len(video_data.images) == 5
        assert video_data.tensors.shape == (5, 3, 224, 224)
        assert video_data.timestamps == [0.0, 0.5, 1.0, 1.5, 2.0]


# =============================================================================
# Video Corruption Tests
# =============================================================================

class TestVideoCorruptor:
    """Tests for VideoCorruptor class."""
    
    def test_video_corruption_types(self):
        """Test VideoCorruptionType enum."""
        from verifai.eval.corruptions import VideoCorruptionType
        
        assert VideoCorruptionType.BITRATE.value == "bitrate"
        assert VideoCorruptionType.RESOLUTION.value == "resolution"
        assert VideoCorruptionType.FPS.value == "fps"
        assert VideoCorruptionType.CODEC.value == "codec"
        assert VideoCorruptionType.PLATFORM.value == "platform"
        assert VideoCorruptionType.CRF.value == "crf"
    
    def test_video_corruption_config(self):
        """Test VideoCorruptionConfig dataclass."""
        from verifai.eval.corruptions import VideoCorruptionConfig, VideoCorruptionType
        
        config = VideoCorruptionConfig(
            corruption_type=VideoCorruptionType.BITRATE,
            params={"bitrate": "1M"},
        )
        
        assert config.corruption_type == VideoCorruptionType.BITRATE
        assert config.params["bitrate"] == "1M"
    
    def test_video_corruption_result(self):
        """Test VideoCorruptionResult dataclass."""
        from verifai.eval.corruptions import VideoCorruptionResult
        
        result = VideoCorruptionResult(
            output_path="/path/to/output.mp4",
            corruption_type="bitrate",
            params_used={"bitrate": "500k"},
            original_metadata={"duration": 10.0},
            corrupted_metadata={"duration": 10.0, "bitrate": 500000},
        )
        
        assert result.output_path == "/path/to/output.mp4"
        assert result.params_used["bitrate"] == "500k"


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================

class TestVideoDetectionMocked:
    """Integration tests for video detection with mocked components."""
    
    def test_frame_score_dataclass(self):
        """Test FrameScore dataclass."""
        from verifai.pipeline import FrameScore
        
        fs = FrameScore(
            frame_number=10,
            timestamp=0.33,
            score=0.75,
            is_suspicious=True,
            label="ai_generated",
        )
        
        assert fs.frame_number == 10
        assert fs.timestamp == 0.33
        assert fs.score == 0.75
        assert fs.is_suspicious
        assert fs.label == "ai_generated"
    
    def test_detection_result_video_fields(self):
        """Test DetectionResult with video-specific fields."""
        from verifai.pipeline import DetectionResult, FrameScore
        
        result = DetectionResult(
            label="ai_generated",
            confidence=0.85,
            is_ai_generated=True,
            input_type="video",
            frame_scores=[
                FrameScore(0, 0.0, 0.8, True, "ai_generated"),
                FrameScore(30, 1.0, 0.6, False, "real"),
            ],
            suspicious_frames=[0, 60],
            temporal_consistency=0.75,
            video_duration=10.5,
            num_frames_analyzed=16,
        )
        
        # Test to_dict includes video info
        d = result.to_dict()
        assert "video" in d
        assert d["video"]["duration"] == 10.5
        assert d["video"]["frames_analyzed"] == 16
        assert d["video"]["temporal_consistency"] == 0.75
        assert len(d["video"]["frame_scores"]) == 2
    
    def test_detection_result_video_summary(self):
        """Test DetectionResult.summary() for video."""
        from verifai.pipeline import DetectionResult, FrameScore
        
        result = DetectionResult(
            label="ai_generated",
            confidence=0.85,
            is_ai_generated=True,
            input_type="video",
            video_duration=30.0,
            num_frames_analyzed=16,
            temporal_consistency=0.72,
            suspicious_frames=[5, 10, 15, 20, 25, 30],
        )
        
        summary = result.summary()
        
        assert "🎬" in summary
        assert "Video Analysis" in summary
        assert "Duration: 30.0s" in summary
        assert "Frames analyzed: 16" in summary
        assert "Temporal consistency" in summary
        assert "Suspicious frames: 6" in summary


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in video processing."""
    
    def test_empty_frame_list(self):
        """Test temporal analyzer with empty list."""
        analyzer = TemporalAnalyzer()
        
        features = analyzer.analyze([])
        
        # Should return defaults
        assert features.flicker_score == 0.0
        assert features.consistency_score == 1.0
    
    def test_grayscale_frames(self):
        """Test temporal analyzer with grayscale frames."""
        analyzer = TemporalAnalyzer()
        
        # Grayscale frames (H, W) shape
        frames = [np.ones((100, 100), dtype=np.float32) * 128] * 5
        
        features = analyzer.analyze(frames)
        
        # Should handle gracefully
        assert features.consistency_score > 0
    
    def test_very_small_frames(self):
        """Test with very small frames."""
        analyzer = TemporalAnalyzer()
        
        # Tiny 10x10 frames
        frames = [np.ones((10, 10, 3), dtype=np.float32) * 128] * 5
        
        features = analyzer.analyze(frames)
        
        # Should still work
        assert features.consistency_score > 0
    
    def test_single_pixel_change(self):
        """Test detection of subtle changes."""
        analyzer = TemporalAnalyzer()
        
        frames = []
        for i in range(5):
            frame = np.ones((100, 100, 3), dtype=np.float32) * 128
            # Add a small change in center pixel
            frame[50, 50, :] = 128 + i * 2
            frames.append(frame)
        
        features = analyzer.analyze(frames)
        
        # Should detect as mostly consistent
        assert features.consistency_score > 0.7
