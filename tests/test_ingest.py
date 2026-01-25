"""
Tests for the ingest module.
"""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
import io

from verifai.ingest import (
    ImageLoader,
    load_image,
    validate_file_path,
    get_media_type,
    MediaType,
)
from verifai.ingest.utils import (
    FileValidationError,
    UnsupportedFormatError,
    format_file_size,
    get_file_info,
)


class TestMediaTypeDetection:
    """Tests for media type detection."""
    
    def test_image_extensions(self):
        """Test that image extensions are correctly identified."""
        image_files = [
            "photo.jpg",
            "image.JPEG",
            "picture.png",
            "graphic.webp",
            "bitmap.bmp",
        ]
        for filename in image_files:
            assert get_media_type(filename) == MediaType.IMAGE
    
    def test_video_extensions(self):
        """Test that video extensions are correctly identified."""
        video_files = [
            "clip.mp4",
            "movie.avi",
            "recording.mov",
            "video.mkv",
            "stream.webm",
        ]
        for filename in video_files:
            assert get_media_type(filename) == MediaType.VIDEO
    
    def test_unknown_extension(self):
        """Test that unknown extensions return UNKNOWN."""
        assert get_media_type("document.pdf") == MediaType.UNKNOWN
        assert get_media_type("archive.zip") == MediaType.UNKNOWN
        assert get_media_type("file.txt") == MediaType.UNKNOWN


class TestFileValidation:
    """Tests for file path validation."""
    
    def test_nonexistent_file(self):
        """Test that nonexistent files raise error."""
        with pytest.raises(FileValidationError):
            validate_file_path("/nonexistent/path/to/file.jpg")
    
    def test_valid_file(self, tmp_path):
        """Test that valid files are accepted."""
        # Create a temporary file
        test_file = tmp_path / "test.jpg"
        test_file.touch()
        
        result = validate_file_path(test_file)
        assert result.exists()
        assert result.is_file()
    
    def test_directory_rejected(self, tmp_path):
        """Test that directories are rejected when expecting file."""
        with pytest.raises(FileValidationError):
            validate_file_path(tmp_path)  # tmp_path is a directory


class TestFormatFileSize:
    """Tests for file size formatting."""
    
    def test_bytes(self):
        assert format_file_size(100) == "100.0 B"
        assert format_file_size(512) == "512.0 B"
    
    def test_kilobytes(self):
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(2048) == "2.0 KB"
    
    def test_megabytes(self):
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(5 * 1024 * 1024) == "5.0 MB"
    
    def test_gigabytes(self):
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"


class TestImageLoader:
    """Tests for the ImageLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create an ImageLoader instance."""
        return ImageLoader(target_size=(224, 224))
    
    @pytest.fixture
    def sample_image(self) -> Image.Image:
        """Create a sample RGB image."""
        return Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
    
    @pytest.fixture
    def sample_image_file(self, tmp_path, sample_image) -> Path:
        """Save sample image to a temporary file."""
        path = tmp_path / "test_image.jpg"
        sample_image.save(path)
        return path
    
    def test_load_from_pil(self, loader, sample_image):
        """Test loading from PIL Image."""
        data = loader.load(sample_image)
        
        assert data.original is not None
        assert data.tensor is not None
        assert data.tensor.shape == (3, 224, 224)
    
    def test_load_from_path(self, loader, sample_image_file):
        """Test loading from file path."""
        data = loader.load(sample_image_file)
        
        assert data.original is not None
        assert data.tensor is not None
        assert data.source_path == sample_image_file
    
    def test_load_from_bytes(self, loader, sample_image):
        """Test loading from bytes."""
        # Convert image to bytes
        buffer = io.BytesIO()
        sample_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        data = loader.load(image_bytes)
        
        assert data.original is not None
        assert data.tensor is not None
    
    def test_grayscale_conversion(self, loader, tmp_path):
        """Test that grayscale images are converted to RGB."""
        # Create grayscale image
        gray = Image.fromarray(
            np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        )
        path = tmp_path / "gray.png"
        gray.save(path)
        
        data = loader.load(path)
        
        # Should be converted to RGB
        assert data.original.mode == "RGB"
        assert data.tensor.shape[0] == 3  # 3 channels
    
    def test_rgba_conversion(self, loader, tmp_path):
        """Test that RGBA images are converted to RGB."""
        # Create RGBA image
        rgba = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8),
            mode="RGBA"
        )
        path = tmp_path / "rgba.png"
        rgba.save(path)
        
        data = loader.load(path)
        
        assert data.original.mode == "RGB"
        assert data.tensor.shape[0] == 3
    
    def test_large_image_resize(self, loader):
        """Test that large images are resized."""
        # Create a very large image
        large = Image.fromarray(
            np.random.randint(0, 255, (5000, 5000, 3), dtype=np.uint8)
        )
        
        loader.max_dimension = 1000
        data = loader.load(large)
        
        # Original should be resized
        assert max(data.original.size) <= 1000
    
    def test_preprocessing_disabled(self, loader, sample_image):
        """Test loading without preprocessing."""
        data = loader.load(sample_image, preprocess=False)
        
        assert data.original is not None
        assert data.tensor is None  # No preprocessing
    
    def test_to_numpy(self, loader, sample_image):
        """Test conversion to numpy array."""
        data = loader.load(sample_image)
        arr = data.to_numpy()
        
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3  # H, W, C
    
    def test_image_data_properties(self, loader, sample_image):
        """Test ImageData properties."""
        data = loader.load(sample_image)
        
        assert data.width == 256  # Original size
        assert data.height == 256
        assert data.aspect_ratio == 1.0


class TestLoadImageFunction:
    """Tests for the load_image convenience function."""
    
    def test_load_image_basic(self, tmp_path):
        """Test basic load_image usage."""
        # Create test image
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        path = tmp_path / "test.jpg"
        img.save(path)
        
        data = load_image(path)
        
        assert data.original is not None
        assert data.tensor is not None
        assert data.tensor.shape == (3, 224, 224)
    
    def test_load_image_custom_size(self, tmp_path):
        """Test load_image with custom target size."""
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        path = tmp_path / "test.jpg"
        img.save(path)
        
        data = load_image(path, target_size=(384, 384))
        
        assert data.tensor.shape == (3, 384, 384)
