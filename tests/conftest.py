"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Create a sample RGB image for testing."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """Create a sample grayscale image for testing."""
    arr = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def sample_rgba_image() -> Image.Image:
    """Create a sample RGBA image for testing."""
    arr = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def temp_image_file(tmp_path, sample_rgb_image) -> Path:
    """Create a temporary image file."""
    path = tmp_path / "test_image.jpg"
    sample_rgb_image.save(path)
    return path


@pytest.fixture
def temp_dataset_dir(tmp_path, sample_rgb_image) -> Path:
    """Create a temporary dataset directory structure."""
    # Create directory structure
    real_dir = tmp_path / "real"
    ai_dir = tmp_path / "ai_generated"
    real_dir.mkdir()
    ai_dir.mkdir()
    
    # Create sample images in each directory
    for i in range(3):
        # Real images
        real_img = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        )
        real_img.save(real_dir / f"real_{i}.jpg")
        
        # AI images
        ai_img = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        )
        ai_img.save(ai_dir / f"ai_{i}.jpg")
    
    return tmp_path
