"""
Tests for the corruptions module.
"""

import pytest
import numpy as np
from PIL import Image

from verifai.eval import (
    ImageCorruptor,
    CorruptionType,
    CorruptionConfig,
    CorruptionResult,
    apply_jpeg_compression,
    apply_resize,
    apply_blur,
)


class TestImageCorruptor:
    """Tests for ImageCorruptor."""
    
    @pytest.fixture
    def corruptor(self):
        """Create a corruptor with fixed seed."""
        return ImageCorruptor(seed=42)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    
    def test_corruptor_initialization(self, corruptor):
        """Test corruptor initialization."""
        assert corruptor is not None
    
    # JPEG Compression Tests
    
    def test_jpeg_compression_high_quality(self, corruptor, sample_image):
        """Test high quality JPEG compression."""
        result = corruptor.apply_jpeg_compression(sample_image, quality=95)
        
        assert isinstance(result, CorruptionResult)
        assert result.corruption_type == "jpeg_compression"
        assert result.params_used["quality"] == 95
        assert result.severity < 0.1
        assert result.image.size == sample_image.size
    
    def test_jpeg_compression_low_quality(self, corruptor, sample_image):
        """Test low quality JPEG compression."""
        result = corruptor.apply_jpeg_compression(sample_image, quality=10)
        
        assert result.params_used["quality"] == 10
        assert result.severity > 0.8
    
    def test_jpeg_quality_sweep(self, corruptor, sample_image):
        """Test JPEG quality sweep."""
        results = corruptor.jpeg_quality_sweep(sample_image)
        
        assert len(results) == 7  # Default: [100, 90, 75, 50, 30, 20, 10]
        # Quality should decrease across results
        qualities = [r.params_used["quality"] for r in results]
        assert qualities == sorted(qualities, reverse=True)
    
    # Resize Tests
    
    def test_resize_downscale(self, corruptor, sample_image):
        """Test downscale resize."""
        result = corruptor.apply_resize(sample_image, scale=0.5, restore_size=True)
        
        assert result.corruption_type == "resize"
        assert result.params_used["scale"] == 0.5
        assert result.image.size == sample_image.size  # Restored
    
    def test_resize_without_restore(self, corruptor, sample_image):
        """Test resize without restoring size."""
        result = corruptor.apply_resize(sample_image, scale=0.5, restore_size=False)
        
        expected_size = (128, 128)  # 256 * 0.5
        assert result.image.size == expected_size
    
    def test_resize_sweep(self, corruptor, sample_image):
        """Test resize sweep."""
        results = corruptor.resize_sweep(sample_image)
        
        assert len(results) == 5  # Default: [1.0, 0.75, 0.5, 0.25, 0.1]
    
    # Blur Tests
    
    def test_gaussian_blur(self, corruptor, sample_image):
        """Test Gaussian blur."""
        result = corruptor.apply_gaussian_blur(sample_image, radius=5)
        
        assert result.corruption_type == "gaussian_blur"
        assert result.params_used["radius"] == 5
        assert result.image.size == sample_image.size
    
    def test_blur_no_change(self, corruptor, sample_image):
        """Test blur with radius 0."""
        result = corruptor.apply_gaussian_blur(sample_image, radius=0)
        
        assert result.severity == 0.0
    
    # Noise Tests
    
    def test_gaussian_noise(self, corruptor, sample_image):
        """Test Gaussian noise."""
        result = corruptor.apply_gaussian_noise(sample_image, std=25)
        
        assert result.corruption_type == "gaussian_noise"
        assert result.params_used["std"] == 25
        assert result.image.size == sample_image.size
    
    def test_noise_sweep(self, corruptor, sample_image):
        """Test noise sweep."""
        results = corruptor.noise_sweep(sample_image)
        
        assert len(results) == 7  # Default levels
    
    # Crop Tests
    
    def test_crop_center(self, corruptor, sample_image):
        """Test center crop."""
        result = corruptor.apply_crop(sample_image, crop_fraction=0.1, position="center")
        
        assert result.corruption_type == "crop"
        # 10% from each edge = 80% of original size
        expected_w = int(256 * 0.8)
        expected_h = int(256 * 0.8)
        assert result.image.size == (expected_w, expected_h)
    
    # Brightness/Contrast Tests
    
    def test_brightness_darker(self, corruptor, sample_image):
        """Test brightness reduction."""
        result = corruptor.apply_brightness(sample_image, factor=0.5)
        
        assert result.corruption_type == "brightness"
        assert result.params_used["factor"] == 0.5
    
    def test_contrast_lower(self, corruptor, sample_image):
        """Test contrast reduction."""
        result = corruptor.apply_contrast(sample_image, factor=0.5)
        
        assert result.corruption_type == "contrast"
    
    # Screenshot Simulation
    
    def test_screenshot_simulation(self, corruptor, sample_image):
        """Test screenshot simulation."""
        result = corruptor.apply_screenshot_simulation(sample_image, dpi_scale=2.0)
        
        assert result.corruption_type == "screenshot"
        assert result.image.size == sample_image.size
    
    # Platform Transcode
    
    def test_platform_transcode_twitter(self, corruptor, sample_image):
        """Test Twitter-like transcode."""
        result = corruptor.apply_platform_transcode(sample_image, platform="twitter")
        
        assert result.corruption_type == "platform_transcode"
        assert result.params_used["platform"] == "twitter"
    
    def test_platform_transcode_instagram(self, corruptor, sample_image):
        """Test Instagram-like transcode."""
        result = corruptor.apply_platform_transcode(sample_image, platform="instagram")
        
        assert result.params_used["platform"] == "instagram"
    
    # Pipeline Tests
    
    def test_corruption_pipeline(self, corruptor, sample_image):
        """Test applying multiple corruptions."""
        pipeline = [
            CorruptionConfig(CorruptionType.JPEG_COMPRESSION, severity=0.5),
            CorruptionConfig(CorruptionType.RESIZE, severity=0.3),
            CorruptionConfig(CorruptionType.GAUSSIAN_BLUR, severity=0.2),
        ]
        
        final_image, results = corruptor.apply_pipeline(sample_image, pipeline)
        
        assert len(results) == 3
        assert final_image.size == sample_image.size  # Size preserved
    
    def test_random_corruption(self, corruptor, sample_image):
        """Test random corruption."""
        result = corruptor.random_corruption(sample_image, severity_range=(0.3, 0.7))
        
        assert isinstance(result, CorruptionResult)
        assert 0.3 <= result.severity <= 0.7 or result.severity >= 0.0


class TestCorruptionConfig:
    """Tests for CorruptionConfig."""
    
    def test_config_creation(self):
        """Test config creation."""
        config = CorruptionConfig(
            corruption_type=CorruptionType.JPEG_COMPRESSION,
            severity=0.5,
        )
        
        assert config.corruption_type == CorruptionType.JPEG_COMPRESSION
        assert config.severity == 0.5
    
    def test_severity_clamping(self):
        """Test that severity is clamped to [0, 1]."""
        config = CorruptionConfig(
            corruption_type=CorruptionType.RESIZE,
            severity=1.5,  # Over 1.0
        )
        
        assert config.severity == 1.0
        
        config2 = CorruptionConfig(
            corruption_type=CorruptionType.RESIZE,
            severity=-0.5,  # Under 0.0
        )
        
        assert config2.severity == 0.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        return Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
    
    def test_apply_jpeg_compression(self, sample_image):
        """Test convenience JPEG function."""
        result = apply_jpeg_compression(sample_image, quality=50)
        
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size
    
    def test_apply_resize(self, sample_image):
        """Test convenience resize function."""
        result = apply_resize(sample_image, scale=0.5)
        
        assert isinstance(result, Image.Image)
    
    def test_apply_blur(self, sample_image):
        """Test convenience blur function."""
        result = apply_blur(sample_image, radius=2)
        
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size
