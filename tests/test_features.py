"""
Tests for the features module.
"""

import pytest
import numpy as np
from PIL import Image

from verifai.features import (
    FrequencyExtractor,
    FrequencyFeatures,
    extract_frequency_features,
    MetadataParser,
    MetadataFeatures,
    parse_metadata,
)


class TestFrequencyExtractor:
    """Tests for frequency feature extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create a frequency extractor."""
        return FrequencyExtractor(
            image_size=(256, 256),
            patch_size=64,
            num_azimuthal_bins=32,
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        return Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.image_size == (256, 256)
        assert extractor.patch_size == 64
        assert extractor.num_azimuthal_bins == 32
    
    def test_extract_from_pil(self, extractor, sample_image):
        """Test extraction from PIL Image."""
        features = extractor.extract(sample_image)
        
        assert isinstance(features, FrequencyFeatures)
        assert features.feature_vector is not None
        assert len(features.feature_vector) > 0
    
    def test_extract_from_numpy(self, extractor):
        """Test extraction from numpy array."""
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        features = extractor.extract(arr)
        
        assert features.feature_vector is not None
    
    def test_extract_from_grayscale(self, extractor):
        """Test extraction from grayscale image."""
        gray = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        features = extractor.extract(gray)
        
        assert features.feature_vector is not None
    
    def test_fft_stats(self, extractor, sample_image):
        """Test that FFT statistics are computed."""
        features = extractor.extract(sample_image)
        
        assert "mean" in features.fft_stats
        assert "std" in features.fft_stats
        assert "band_low_mean" in features.fft_stats
        assert "band_high_mean" in features.fft_stats
    
    def test_dct_stats(self, extractor, sample_image):
        """Test that DCT statistics are computed."""
        features = extractor.extract(sample_image)
        
        assert "dct_mean" in features.dct_stats
        assert "dct_std" in features.dct_stats
        assert "dct_energy" in features.dct_stats
    
    def test_azimuthal_profile(self, extractor, sample_image):
        """Test azimuthal profile computation."""
        features = extractor.extract(sample_image)
        
        assert features.azimuthal_profile is not None
        assert len(features.azimuthal_profile) == extractor.num_azimuthal_bins
    
    def test_patch_features(self, extractor, sample_image):
        """Test patch-based features."""
        features = extractor.extract(sample_image)
        
        assert features.patch_features is not None
        assert len(features.patch_features) > 0
    
    def test_feature_dimension(self, extractor, sample_image):
        """Test feature vector dimension is consistent."""
        dim1 = extractor.get_feature_dim()
        features = extractor.extract(sample_image)
        dim2 = features.feature_dim
        
        assert dim1 == dim2
    
    def test_return_spectra(self, extractor, sample_image):
        """Test returning full spectra."""
        features = extractor.extract(sample_image, return_spectra=True)
        
        assert features.fft_magnitude is not None
        assert features.fft_phase is not None
        assert features.dct_coeffs is not None
    
    def test_convenience_function(self, sample_image):
        """Test extract_frequency_features convenience function."""
        features = extract_frequency_features(sample_image)
        
        assert isinstance(features, FrequencyFeatures)
        assert features.feature_vector is not None


class TestMetadataParser:
    """Tests for metadata parsing."""
    
    @pytest.fixture
    def parser(self):
        """Create a metadata parser."""
        return MetadataParser()
    
    @pytest.fixture
    def image_with_exif(self, tmp_path):
        """Create an image with EXIF data."""
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        # Create image
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        
        # Add basic EXIF
        # Note: PIL has limited EXIF writing capabilities
        # In real tests, we'd use a test image with actual EXIF
        
        path = tmp_path / "test_with_exif.jpg"
        img.save(path, exif=img.getexif())
        return path
    
    @pytest.fixture
    def image_without_exif(self, tmp_path):
        """Create an image without EXIF data."""
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        path = tmp_path / "test_no_exif.png"
        img.save(path)
        return path
    
    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser is not None
    
    def test_parse_image_without_exif(self, parser, image_without_exif):
        """Test parsing image without EXIF."""
        features = parser.parse(image_without_exif)
        
        assert isinstance(features, MetadataFeatures)
        # PNG typically has no EXIF
        assert features.is_suspicious  # No EXIF is suspicious
    
    def test_parse_pil_image(self, parser):
        """Test parsing from PIL Image directly."""
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        features = parser.parse(img)
        
        assert isinstance(features, MetadataFeatures)
    
    def test_feature_vector(self, parser, image_without_exif):
        """Test feature vector generation."""
        features = parser.parse(image_without_exif)
        
        assert features.feature_vector is not None
        assert len(features.feature_vector) == 9  # Number of boolean/numeric features
    
    def test_confidence_range(self, parser, image_without_exif):
        """Test confidence is in valid range."""
        features = parser.parse(image_without_exif)
        
        assert 0.0 <= features.confidence_real <= 1.0
    
    def test_to_dict(self, parser, image_without_exif):
        """Test conversion to dictionary."""
        features = parser.parse(image_without_exif)
        d = features.to_dict()
        
        assert "has_exif" in d
        assert "is_suspicious" in d
        assert "confidence_real" in d
    
    def test_convenience_function(self, image_without_exif):
        """Test parse_metadata convenience function."""
        features = parse_metadata(image_without_exif)
        
        assert isinstance(features, MetadataFeatures)
