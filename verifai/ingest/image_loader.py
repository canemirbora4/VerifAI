"""
Image Loader
=============

Handles loading and preprocessing of images for the detection pipeline.
Supports various formats and provides consistent preprocessing.
"""

from pathlib import Path
from typing import Union, Optional, Tuple
from dataclasses import dataclass, field
import io

import numpy as np
from PIL import Image, ImageOps, ExifTags
import torch
from torchvision import transforms
from loguru import logger

from verifai.ingest.utils import (
    validate_file_path,
    get_media_type,
    MediaType,
    UnsupportedFormatError,
)


@dataclass
class ImageData:
    """Container for loaded image data and metadata."""
    
    # Original image as PIL Image
    original: Image.Image
    
    # Preprocessed tensor ready for model input
    tensor: Optional[torch.Tensor] = None
    
    # Source file path
    source_path: Optional[Path] = None
    
    # Image dimensions (width, height)
    original_size: Tuple[int, int] = field(default=(0, 0))
    
    # EXIF metadata
    exif: dict = field(default_factory=dict)
    
    # Additional metadata
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.original_size == (0, 0):
            self.original_size = self.original.size
    
    @property
    def width(self) -> int:
        return self.original_size[0]
    
    @property
    def height(self) -> int:
        return self.original_size[1]
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0
    
    def to_numpy(self) -> np.ndarray:
        """Convert original image to numpy array (H, W, C)."""
        return np.array(self.original)
    
    def to_tensor_chw(self) -> torch.Tensor:
        """Convert original image to tensor (C, H, W) normalized to [0, 1]."""
        arr = np.array(self.original)
        if arr.ndim == 2:  # Grayscale
            arr = np.stack([arr, arr, arr], axis=-1)
        # HWC -> CHW
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return tensor


class ImageLoader:
    """
    Loads and preprocesses images for the detection pipeline.
    
    Features:
    - Supports multiple image formats (JPEG, PNG, WebP, etc.)
    - Extracts EXIF metadata
    - Handles image orientation correction
    - Provides preprocessing for model input
    - Memory-efficient handling of large images
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        max_dimension: int = 4096,
        auto_orient: bool = True,
    ):
        """
        Initialize the ImageLoader.
        
        Args:
            target_size: Target size for model input (height, width)
            normalize_mean: Mean values for normalization (ImageNet default)
            normalize_std: Std values for normalization (ImageNet default)
            max_dimension: Maximum dimension before initial resize
            auto_orient: Whether to auto-orient based on EXIF
        """
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.max_dimension = max_dimension
        self.auto_orient = auto_orient
        
        # Build preprocessing transform
        self._transform = self._build_transform()
        
        logger.debug(
            f"ImageLoader initialized: target_size={target_size}, "
            f"max_dimension={max_dimension}"
        )
    
    def _build_transform(self) -> transforms.Compose:
        """Build the preprocessing transform pipeline."""
        return transforms.Compose([
            transforms.Resize(
                self.target_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalize_mean,
                std=self.normalize_std,
            ),
        ])
    
    def load(
        self,
        source: Union[str, Path, bytes, Image.Image],
        preprocess: bool = True,
    ) -> ImageData:
        """
        Load an image from various sources.
        
        Args:
            source: Image source - file path, bytes, or PIL Image
            preprocess: Whether to apply preprocessing for model input
            
        Returns:
            ImageData object containing the loaded image and metadata
            
        Raises:
            UnsupportedFormatError: If the format is not supported
            FileValidationError: If file doesn't exist or can't be read
        """
        source_path = None
        
        # Handle different source types
        if isinstance(source, (str, Path)):
            source_path = validate_file_path(
                source,
                must_exist=True,
                allowed_types={MediaType.IMAGE},
            )
            image = self._load_from_path(source_path)
            
        elif isinstance(source, bytes):
            image = self._load_from_bytes(source)
            
        elif isinstance(source, Image.Image):
            image = source.copy()
            
        else:
            raise TypeError(
                f"Unsupported source type: {type(source)}. "
                "Expected str, Path, bytes, or PIL.Image"
            )
        
        # Ensure RGB mode
        image = self._ensure_rgb(image)
        
        # Auto-orient based on EXIF
        if self.auto_orient:
            image = ImageOps.exif_transpose(image)
        
        # Extract EXIF metadata
        exif = self._extract_exif(image)
        
        # Resize if too large
        image = self._limit_size(image)
        
        # Create ImageData object
        data = ImageData(
            original=image,
            source_path=source_path,
            exif=exif,
            metadata={
                "format": image.format or "unknown",
                "mode": image.mode,
                "has_exif": bool(exif),
            }
        )
        
        # Apply preprocessing if requested
        if preprocess:
            data.tensor = self._preprocess(image)
        
        logger.debug(
            f"Loaded image: {data.width}x{data.height}, "
            f"preprocessed={preprocess}"
        )
        
        return data
    
    def _load_from_path(self, path: Path) -> Image.Image:
        """Load image from file path."""
        try:
            image = Image.open(path)
            image.load()  # Force load to catch errors early
            return image
        except Exception as e:
            raise UnsupportedFormatError(
                f"Failed to load image from {path}: {e}"
            ) from e
    
    def _load_from_bytes(self, data: bytes) -> Image.Image:
        """Load image from bytes."""
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            return image
        except Exception as e:
            raise UnsupportedFormatError(
                f"Failed to load image from bytes: {e}"
            ) from e
    
    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB mode if necessary."""
        if image.mode == "RGB":
            return image
        
        if image.mode == "RGBA":
            # Create white background for transparency
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            return background
        
        if image.mode in ("L", "LA", "P"):
            return image.convert("RGB")
        
        # For other modes, try direct conversion
        try:
            return image.convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to convert image mode {image.mode} to RGB: {e}")
            return image.convert("RGB")
    
    def _limit_size(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds max_dimension."""
        width, height = image.size
        max_dim = max(width, height)
        
        if max_dim <= self.max_dimension:
            return image
        
        scale = self.max_dimension / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        logger.debug(
            f"Resizing large image from {width}x{height} to {new_width}x{new_height}"
        )
        
        return image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )
    
    def _extract_exif(self, image: Image.Image) -> dict:
        """Extract EXIF metadata from image."""
        exif_data = {}
        
        try:
            exif = image.getexif()
            if not exif:
                return exif_data
            
            # Map EXIF tag IDs to names
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                
                # Convert bytes to string if needed
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="replace")
                    except Exception:
                        value = str(value)
                
                exif_data[tag_name] = value
            
            # Also try to get IFD data (nested EXIF)
            for ifd_id in ExifTags.IFD:
                try:
                    ifd = exif.get_ifd(ifd_id)
                    if ifd:
                        ifd_name = ifd_id.name
                        exif_data[ifd_name] = {}
                        for tag_id, value in ifd.items():
                            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode("utf-8", errors="replace")
                                except Exception:
                                    value = str(value)
                            exif_data[ifd_name][tag_name] = value
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"Failed to extract EXIF: {e}")
        
        return exif_data
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Apply preprocessing transform to image."""
        tensor = self._transform(image)
        return tensor
    
    def preprocess_batch(
        self,
        images: list[Image.Image],
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Batched tensor of shape (N, C, H, W)
        """
        tensors = [self._preprocess(img) for img in images]
        return torch.stack(tensors)


def load_image(
    source: Union[str, Path, bytes, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    preprocess: bool = True,
) -> ImageData:
    """
    Convenience function to load an image with default settings.
    
    Args:
        source: Image source - file path, bytes, or PIL Image
        target_size: Target size for preprocessing
        preprocess: Whether to apply preprocessing
        
    Returns:
        ImageData object
    """
    loader = ImageLoader(target_size=target_size)
    return loader.load(source, preprocess=preprocess)
