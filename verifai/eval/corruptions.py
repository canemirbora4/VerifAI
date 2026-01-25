"""
Image Corruption Transforms
============================

Simulates real-world image transformations that occur during:
- Social media sharing (compression, resize)
- Screenshots
- Re-encoding
- General image processing

These corruptions test detector robustness - a good detector should
maintain performance even when images are degraded.

Supported corruptions:
- JPEG compression (quality 10-100)
- Resize (downscale/upscale)
- Gaussian blur
- Gaussian noise
- Crop and pad
- Screenshot simulation
- Contrast/brightness adjustment
- Combined corruptions (pipeline)
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Literal
from enum import Enum
import io
import random

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from loguru import logger


class CorruptionType(Enum):
    """Types of image corruptions."""
    JPEG_COMPRESSION = "jpeg_compression"
    RESIZE = "resize"
    GAUSSIAN_BLUR = "gaussian_blur"
    GAUSSIAN_NOISE = "gaussian_noise"
    CROP = "crop"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SCREENSHOT = "screenshot"
    PLATFORM_TRANSCODE = "platform_transcode"


@dataclass
class CorruptionConfig:
    """
    Configuration for a corruption transform.
    
    Attributes:
        corruption_type: Type of corruption
        severity: Severity level (0.0 = none, 1.0 = maximum)
        params: Additional parameters specific to corruption type
    """
    corruption_type: CorruptionType
    severity: float = 0.5
    params: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.severity = max(0.0, min(1.0, self.severity))


@dataclass
class CorruptionResult:
    """
    Result of applying a corruption.
    
    Attributes:
        image: Corrupted image
        corruption_type: Type applied
        severity: Severity level
        params_used: Actual parameters used
    """
    image: Image.Image
    corruption_type: str
    severity: float
    params_used: dict


class ImageCorruptor:
    """
    Applies realistic image corruptions for robustness testing.
    
    Usage:
        corruptor = ImageCorruptor()
        
        # Single corruption
        result = corruptor.apply_jpeg_compression(image, quality=50)
        
        # Sweep over severities
        results = corruptor.jpeg_quality_sweep(image, qualities=[100, 75, 50, 25])
        
        # Random corruption
        result = corruptor.random_corruption(image)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the corruptor.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
    
    # =========================================================================
    # JPEG Compression
    # =========================================================================
    
    def apply_jpeg_compression(
        self,
        image: Image.Image,
        quality: int = 75,
    ) -> CorruptionResult:
        """
        Apply JPEG compression.
        
        Args:
            image: Input image
            quality: JPEG quality (1-100, lower = more compression)
            
        Returns:
            CorruptionResult with compressed image
        """
        quality = max(1, min(100, quality))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Compress via bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        compressed.load()  # Force load
        
        # Convert back to RGB (JPEG opens as RGB anyway)
        compressed = compressed.convert("RGB")
        
        severity = 1.0 - (quality / 100.0)
        
        return CorruptionResult(
            image=compressed,
            corruption_type=CorruptionType.JPEG_COMPRESSION.value,
            severity=severity,
            params_used={"quality": quality},
        )
    
    def jpeg_quality_sweep(
        self,
        image: Image.Image,
        qualities: Optional[list[int]] = None,
    ) -> list[CorruptionResult]:
        """
        Apply JPEG compression at multiple quality levels.
        
        Args:
            image: Input image
            qualities: List of quality values (default: [100, 90, 75, 50, 30, 20, 10])
            
        Returns:
            List of CorruptionResult for each quality
        """
        if qualities is None:
            qualities = [100, 90, 75, 50, 30, 20, 10]
        
        results = []
        for q in qualities:
            result = self.apply_jpeg_compression(image, quality=q)
            results.append(result)
        
        return results
    
    # =========================================================================
    # Resize
    # =========================================================================
    
    def apply_resize(
        self,
        image: Image.Image,
        scale: float = 0.5,
        restore_size: bool = True,
    ) -> CorruptionResult:
        """
        Apply resize (downscale then optionally upscale back).
        
        Args:
            image: Input image
            scale: Scale factor (0.1 to 2.0)
            restore_size: Whether to resize back to original dimensions
            
        Returns:
            CorruptionResult with resized image
        """
        scale = max(0.1, min(2.0, scale))
        
        original_size = image.size
        new_width = int(original_size[0] * scale)
        new_height = int(original_size[1] * scale)
        
        # Prevent zero dimensions
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Downscale
        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )
        
        # Optionally restore to original size
        if restore_size and scale < 1.0:
            resized = resized.resize(
                original_size,
                Image.Resampling.LANCZOS,
            )
        
        severity = abs(1.0 - scale)
        
        return CorruptionResult(
            image=resized,
            corruption_type=CorruptionType.RESIZE.value,
            severity=severity,
            params_used={
                "scale": scale,
                "restore_size": restore_size,
                "intermediate_size": (new_width, new_height),
            },
        )
    
    def resize_sweep(
        self,
        image: Image.Image,
        scales: Optional[list[float]] = None,
        restore_size: bool = True,
    ) -> list[CorruptionResult]:
        """
        Apply resize at multiple scales.
        
        Args:
            image: Input image
            scales: List of scale factors
            restore_size: Restore to original size
            
        Returns:
            List of CorruptionResult
        """
        if scales is None:
            scales = [1.0, 0.75, 0.5, 0.25, 0.1]
        
        results = []
        for s in scales:
            result = self.apply_resize(image, scale=s, restore_size=restore_size)
            results.append(result)
        
        return results
    
    # =========================================================================
    # Gaussian Blur
    # =========================================================================
    
    def apply_gaussian_blur(
        self,
        image: Image.Image,
        radius: float = 2.0,
    ) -> CorruptionResult:
        """
        Apply Gaussian blur.
        
        Args:
            image: Input image
            radius: Blur radius (0 = no blur, higher = more blur)
            
        Returns:
            CorruptionResult with blurred image
        """
        radius = max(0, radius)
        
        if radius > 0:
            blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            blurred = image.copy()
        
        # Normalize severity (radius of 10 = severity 1.0)
        severity = min(1.0, radius / 10.0)
        
        return CorruptionResult(
            image=blurred,
            corruption_type=CorruptionType.GAUSSIAN_BLUR.value,
            severity=severity,
            params_used={"radius": radius},
        )
    
    def blur_sweep(
        self,
        image: Image.Image,
        radii: Optional[list[float]] = None,
    ) -> list[CorruptionResult]:
        """Apply blur at multiple radii."""
        if radii is None:
            radii = [0, 1, 2, 3, 5, 7, 10]
        
        return [self.apply_gaussian_blur(image, r) for r in radii]
    
    # =========================================================================
    # Gaussian Noise
    # =========================================================================
    
    def apply_gaussian_noise(
        self,
        image: Image.Image,
        std: float = 25.0,
    ) -> CorruptionResult:
        """
        Apply Gaussian noise.
        
        Args:
            image: Input image
            std: Standard deviation of noise (0-255 scale)
            
        Returns:
            CorruptionResult with noisy image
        """
        std = max(0, std)
        
        arr = np.array(image, dtype=np.float32)
        noise = self.np_rng.normal(0, std, arr.shape)
        noisy = arr + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        noisy_image = Image.fromarray(noisy)
        
        # Normalize severity (std of 50 = severity 1.0)
        severity = min(1.0, std / 50.0)
        
        return CorruptionResult(
            image=noisy_image,
            corruption_type=CorruptionType.GAUSSIAN_NOISE.value,
            severity=severity,
            params_used={"std": std},
        )
    
    def noise_sweep(
        self,
        image: Image.Image,
        stds: Optional[list[float]] = None,
    ) -> list[CorruptionResult]:
        """Apply noise at multiple levels."""
        if stds is None:
            stds = [0, 5, 10, 15, 25, 35, 50]
        
        return [self.apply_gaussian_noise(image, s) for s in stds]
    
    # =========================================================================
    # Crop
    # =========================================================================
    
    def apply_crop(
        self,
        image: Image.Image,
        crop_fraction: float = 0.1,
        position: str = "center",
    ) -> CorruptionResult:
        """
        Apply crop (remove edges).
        
        Args:
            image: Input image
            crop_fraction: Fraction to crop from each edge (0.0 to 0.4)
            position: Crop position ("center", "random")
            
        Returns:
            CorruptionResult with cropped image
        """
        crop_fraction = max(0.0, min(0.4, crop_fraction))
        
        width, height = image.size
        crop_w = int(width * crop_fraction)
        crop_h = int(height * crop_fraction)
        
        if position == "center":
            left = crop_w
            top = crop_h
            right = width - crop_w
            bottom = height - crop_h
        elif position == "random":
            max_left = crop_w * 2
            max_top = crop_h * 2
            left = self.rng.randint(0, max_left) if max_left > 0 else 0
            top = self.rng.randint(0, max_top) if max_top > 0 else 0
            right = width - (crop_w * 2 - left)
            bottom = height - (crop_h * 2 - top)
        else:
            left, top = crop_w, crop_h
            right, bottom = width - crop_w, height - crop_h
        
        # Ensure valid crop
        right = max(left + 1, right)
        bottom = max(top + 1, bottom)
        
        cropped = image.crop((left, top, right, bottom))
        
        return CorruptionResult(
            image=cropped,
            corruption_type=CorruptionType.CROP.value,
            severity=crop_fraction * 2.5,  # 0.4 crop = severity 1.0
            params_used={
                "crop_fraction": crop_fraction,
                "position": position,
                "crop_box": (left, top, right, bottom),
            },
        )
    
    # =========================================================================
    # Brightness / Contrast
    # =========================================================================
    
    def apply_brightness(
        self,
        image: Image.Image,
        factor: float = 1.0,
    ) -> CorruptionResult:
        """
        Adjust brightness.
        
        Args:
            image: Input image
            factor: Brightness factor (0.5 = darker, 1.0 = unchanged, 1.5 = brighter)
            
        Returns:
            CorruptionResult
        """
        factor = max(0.1, min(2.0, factor))
        
        enhancer = ImageEnhance.Brightness(image)
        adjusted = enhancer.enhance(factor)
        
        severity = abs(1.0 - factor)
        
        return CorruptionResult(
            image=adjusted,
            corruption_type=CorruptionType.BRIGHTNESS.value,
            severity=severity,
            params_used={"factor": factor},
        )
    
    def apply_contrast(
        self,
        image: Image.Image,
        factor: float = 1.0,
    ) -> CorruptionResult:
        """
        Adjust contrast.
        
        Args:
            image: Input image
            factor: Contrast factor (0.5 = less contrast, 1.5 = more contrast)
            
        Returns:
            CorruptionResult
        """
        factor = max(0.1, min(2.0, factor))
        
        enhancer = ImageEnhance.Contrast(image)
        adjusted = enhancer.enhance(factor)
        
        severity = abs(1.0 - factor)
        
        return CorruptionResult(
            image=adjusted,
            corruption_type=CorruptionType.CONTRAST.value,
            severity=severity,
            params_used={"factor": factor},
        )
    
    # =========================================================================
    # Screenshot Simulation
    # =========================================================================
    
    def apply_screenshot_simulation(
        self,
        image: Image.Image,
        dpi_scale: float = 2.0,
    ) -> CorruptionResult:
        """
        Simulate screenshot (downscale, slight blur, re-encode).
        
        This mimics what happens when someone screenshots an image
        on a retina display.
        
        Args:
            image: Input image
            dpi_scale: DPI scaling factor
            
        Returns:
            CorruptionResult
        """
        original_size = image.size
        
        # Downscale (simulating display rendering)
        display_size = (
            int(original_size[0] / dpi_scale),
            int(original_size[1] / dpi_scale),
        )
        displayed = image.resize(display_size, Image.Resampling.LANCZOS)
        
        # Slight blur (screen rendering)
        displayed = displayed.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Scale back up (screenshot capture)
        screenshot = displayed.resize(original_size, Image.Resampling.LANCZOS)
        
        # PNG compression (typical screenshot format)
        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        buffer.seek(0)
        screenshot = Image.open(buffer)
        screenshot.load()
        
        return CorruptionResult(
            image=screenshot,
            corruption_type=CorruptionType.SCREENSHOT.value,
            severity=0.3 + (dpi_scale - 1) * 0.2,
            params_used={"dpi_scale": dpi_scale},
        )
    
    # =========================================================================
    # Platform-like Transcoding
    # =========================================================================
    
    def apply_platform_transcode(
        self,
        image: Image.Image,
        platform: Literal["twitter", "instagram", "facebook", "whatsapp"] = "twitter",
    ) -> CorruptionResult:
        """
        Simulate platform-specific image processing.
        
        Different platforms apply different compression and resizing.
        
        Args:
            image: Input image
            platform: Platform to simulate
            
        Returns:
            CorruptionResult
        """
        # Platform-specific settings (approximate)
        platform_configs = {
            "twitter": {"max_size": 4096, "jpeg_quality": 85, "resize_large": True},
            "instagram": {"max_size": 1080, "jpeg_quality": 70, "resize_large": True},
            "facebook": {"max_size": 2048, "jpeg_quality": 75, "resize_large": True},
            "whatsapp": {"max_size": 1600, "jpeg_quality": 60, "resize_large": True},
        }
        
        config = platform_configs.get(platform, platform_configs["twitter"])
        
        processed = image.copy()
        
        # Convert to RGB
        if processed.mode != "RGB":
            processed = processed.convert("RGB")
        
        # Resize if too large
        if config["resize_large"]:
            max_dim = max(processed.size)
            if max_dim > config["max_size"]:
                scale = config["max_size"] / max_dim
                new_size = (
                    int(processed.size[0] * scale),
                    int(processed.size[1] * scale),
                )
                processed = processed.resize(new_size, Image.Resampling.LANCZOS)
        
        # JPEG compression
        buffer = io.BytesIO()
        processed.save(buffer, format="JPEG", quality=config["jpeg_quality"])
        buffer.seek(0)
        processed = Image.open(buffer)
        processed.load()
        
        severity = 1.0 - (config["jpeg_quality"] / 100.0)
        
        return CorruptionResult(
            image=processed,
            corruption_type=CorruptionType.PLATFORM_TRANSCODE.value,
            severity=severity,
            params_used={
                "platform": platform,
                **config,
            },
        )
    
    # =========================================================================
    # Combined / Random
    # =========================================================================
    
    def apply_pipeline(
        self,
        image: Image.Image,
        corruptions: list[CorruptionConfig],
    ) -> tuple[Image.Image, list[CorruptionResult]]:
        """
        Apply a sequence of corruptions.
        
        Args:
            image: Input image
            corruptions: List of corruption configs to apply in order
            
        Returns:
            Tuple of (final_image, list_of_results)
        """
        current = image.copy()
        results = []
        
        for config in corruptions:
            result = self.apply_corruption(current, config)
            current = result.image
            results.append(result)
        
        return current, results
    
    def apply_corruption(
        self,
        image: Image.Image,
        config: CorruptionConfig,
    ) -> CorruptionResult:
        """
        Apply a single corruption based on config.
        
        Args:
            image: Input image
            config: Corruption configuration
            
        Returns:
            CorruptionResult
        """
        ctype = config.corruption_type
        severity = config.severity
        params = config.params
        
        if ctype == CorruptionType.JPEG_COMPRESSION:
            quality = params.get("quality", int(100 - severity * 90))
            return self.apply_jpeg_compression(image, quality)
        
        elif ctype == CorruptionType.RESIZE:
            scale = params.get("scale", 1.0 - severity * 0.9)
            return self.apply_resize(image, scale)
        
        elif ctype == CorruptionType.GAUSSIAN_BLUR:
            radius = params.get("radius", severity * 10)
            return self.apply_gaussian_blur(image, radius)
        
        elif ctype == CorruptionType.GAUSSIAN_NOISE:
            std = params.get("std", severity * 50)
            return self.apply_gaussian_noise(image, std)
        
        elif ctype == CorruptionType.CROP:
            crop_fraction = params.get("crop_fraction", severity * 0.4)
            return self.apply_crop(image, crop_fraction)
        
        elif ctype == CorruptionType.BRIGHTNESS:
            # Map severity to factor: 0.5 = darker, 1.5 = brighter
            direction = params.get("direction", "darker")
            if direction == "darker":
                factor = 1.0 - severity * 0.5
            else:
                factor = 1.0 + severity * 0.5
            return self.apply_brightness(image, factor)
        
        elif ctype == CorruptionType.CONTRAST:
            direction = params.get("direction", "lower")
            if direction == "lower":
                factor = 1.0 - severity * 0.5
            else:
                factor = 1.0 + severity * 0.5
            return self.apply_contrast(image, factor)
        
        elif ctype == CorruptionType.SCREENSHOT:
            dpi_scale = params.get("dpi_scale", 1.0 + severity)
            return self.apply_screenshot_simulation(image, dpi_scale)
        
        elif ctype == CorruptionType.PLATFORM_TRANSCODE:
            platform = params.get("platform", "twitter")
            return self.apply_platform_transcode(image, platform)
        
        else:
            # Unknown type - return unchanged
            return CorruptionResult(
                image=image.copy(),
                corruption_type="none",
                severity=0.0,
                params_used={},
            )
    
    def random_corruption(
        self,
        image: Image.Image,
        severity_range: tuple[float, float] = (0.2, 0.8),
    ) -> CorruptionResult:
        """
        Apply a random corruption at random severity.
        
        Args:
            image: Input image
            severity_range: (min, max) severity
            
        Returns:
            CorruptionResult
        """
        corruption_types = [
            CorruptionType.JPEG_COMPRESSION,
            CorruptionType.RESIZE,
            CorruptionType.GAUSSIAN_BLUR,
            CorruptionType.GAUSSIAN_NOISE,
            CorruptionType.BRIGHTNESS,
            CorruptionType.CONTRAST,
        ]
        
        ctype = self.rng.choice(corruption_types)
        severity = self.rng.uniform(*severity_range)
        
        config = CorruptionConfig(corruption_type=ctype, severity=severity)
        return self.apply_corruption(image, config)


# Convenience functions
def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression and return image."""
    return ImageCorruptor().apply_jpeg_compression(image, quality).image


def apply_resize(image: Image.Image, scale: float) -> Image.Image:
    """Apply resize and return image."""
    return ImageCorruptor().apply_resize(image, scale).image


def apply_blur(image: Image.Image, radius: float) -> Image.Image:
    """Apply blur and return image."""
    return ImageCorruptor().apply_gaussian_blur(image, radius).image
