"""
Tests for the main VerifAI pipeline.
"""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from verifai import VerifAI, DetectionResult


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic DetectionResult creation."""
        result = DetectionResult(
            label="ai_generated",
            confidence=0.85,
            is_ai_generated=True,
        )
        
        assert result.label == "ai_generated"
        assert result.confidence == 0.85
        assert result.is_ai_generated is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = DetectionResult(
            label="real",
            confidence=0.2,
            is_ai_generated=False,
            detector_scores={"neural": 0.2},
            input_size=(1920, 1080),
        )
        
        d = result.to_dict()
        
        assert d["result"]["label"] == "real"
        assert d["result"]["confidence"] == 0.2
        assert d["detector_scores"]["neural"] == 0.2
        assert d["input"]["size"] == (1920, 1080)
    
    def test_to_json(self):
        """Test JSON serialization."""
        result = DetectionResult(
            label="ai_generated",
            confidence=0.9,
            is_ai_generated=True,
        )
        
        json_str = result.to_json()
        
        assert '"label": "ai_generated"' in json_str
        assert '"confidence": 0.9' in json_str
    
    def test_summary(self):
        """Test human-readable summary."""
        result = DetectionResult(
            label="ai_generated",
            confidence=0.75,
            is_ai_generated=True,
            input_path="/path/to/image.jpg",
            input_size=(1024, 768),
            processing_time_ms=150.5,
        )
        
        summary = result.summary()
        
        assert "AI-GENERATED" in summary or "🤖" in summary
        assert "75" in summary  # 75% confidence


class TestVerifAIPipeline:
    """Tests for the main VerifAI pipeline."""
    
    @pytest.fixture
    def sample_image(self) -> Image.Image:
        """Create a sample test image."""
        return Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
    
    @pytest.fixture
    def sample_image_file(self, tmp_path, sample_image) -> Path:
        """Save sample image to a file."""
        path = tmp_path / "test_image.jpg"
        sample_image.save(path)
        return path
    
    @pytest.mark.slow
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = VerifAI(auto_load=False)
        
        assert not pipeline.is_loaded
        assert pipeline.threshold == 0.5
    
    @pytest.mark.slow
    def test_pipeline_load(self):
        """Test loading the pipeline."""
        pipeline = VerifAI(auto_load=False)
        
        pipeline.load()
        
        assert pipeline.is_loaded
    
    @pytest.mark.slow
    def test_pipeline_detect_pil(self, sample_image):
        """Test detection from PIL Image."""
        pipeline = VerifAI()
        
        result = pipeline.detect(sample_image)
        
        assert isinstance(result, DetectionResult)
        assert result.label in ["real", "ai_generated"]
        assert 0 <= result.confidence <= 1
        assert result.input_type == "image"
    
    @pytest.mark.slow
    def test_pipeline_detect_file(self, sample_image_file):
        """Test detection from file path."""
        pipeline = VerifAI()
        
        result = pipeline.detect(sample_image_file)
        
        assert isinstance(result, DetectionResult)
        assert result.input_path == str(sample_image_file)
        assert result.input_type == "image"
        assert result.input_size is not None
    
    @pytest.mark.slow
    def test_pipeline_detect_with_evidence(self, sample_image):
        """Test detection with evidence."""
        pipeline = VerifAI()
        
        result = pipeline.detect(sample_image, return_evidence=True)
        
        assert isinstance(result, DetectionResult)
        # Evidence may or may not be present depending on model support
    
    @pytest.mark.slow
    def test_pipeline_batch_detection(self, tmp_path):
        """Test batch detection."""
        # Create multiple test images
        images = []
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            path = tmp_path / f"test_{i}.jpg"
            img.save(path)
            images.append(path)
        
        pipeline = VerifAI()
        results = pipeline.detect_batch(images)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, DetectionResult)
    
    @pytest.mark.slow
    def test_pipeline_get_info(self):
        """Test getting pipeline information."""
        pipeline = VerifAI(auto_load=False)
        
        info = pipeline.get_info()
        
        assert "version" in info
        assert "model" in info
        assert "threshold" in info
        assert "is_loaded" in info
    
    def test_pipeline_video_not_implemented(self, tmp_path):
        """Test that video detection raises NotImplementedError."""
        # Create a dummy video file
        video_path = tmp_path / "test.mp4"
        video_path.touch()  # Just create empty file
        
        pipeline = VerifAI(auto_load=False)
        
        with pytest.raises(NotImplementedError) as exc_info:
            pipeline.detect(video_path)
        
        assert "Video detection not yet implemented" in str(exc_info.value)
    
    def test_pipeline_unsupported_format(self, tmp_path):
        """Test that unsupported formats raise error."""
        # Create a text file
        text_path = tmp_path / "test.txt"
        text_path.write_text("not an image")
        
        pipeline = VerifAI(auto_load=False)
        
        with pytest.raises(ValueError) as exc_info:
            pipeline.detect(text_path)
        
        assert "Unsupported media type" in str(exc_info.value)
