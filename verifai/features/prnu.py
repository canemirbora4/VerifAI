"""
PRNU (Photo-Response Non-Uniformity) Analysis
==============================================

PRNU is a unique fingerprint for each camera sensor, caused by manufacturing
imperfections in the silicon. Every photo taken by a real camera contains
this noise pattern, while AI-generated images do not.

This module provides:
- PRNU noise extraction from images
- Reference fingerprint estimation from multiple images
- Fingerprint comparison for source verification
- AI detection based on PRNU presence/absence

Key Concepts:
- Real photos: Have consistent PRNU pattern from their source camera
- AI images: No PRNU or random/inconsistent noise patterns
- Edited/synthetic regions: PRNU discontinuities indicate manipulation

References:
- Lukáš, Fridrich, Goljan: "Digital Camera Identification from Sensor 
  Pattern Noise" (2006)
- Chen et al.: "Determining Image Origin and Integrity Using Sensor Noise"
"""

from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger

try:
    from scipy.ndimage import uniform_filter
    from scipy.signal import wiener
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. PRNU extraction will be limited.")


@dataclass
class PRNUFeatures:
    """
    Container for PRNU analysis results.
    
    Attributes:
        noise_residual: Extracted noise pattern
        noise_strength: Overall noise strength metric
        noise_uniformity: How uniform the noise is across the image
        has_prnu_signature: Whether a valid PRNU signature was detected
        prnu_score: Score indicating likelihood of real camera origin (0-1)
        correlation: Correlation with reference fingerprint (if available)
        quality_score: Quality of the extraction
    """
    
    # Extracted noise
    noise_residual: Optional[np.ndarray] = None
    
    # Metrics
    noise_strength: float = 0.0
    noise_uniformity: float = 0.0
    has_prnu_signature: bool = False
    prnu_score: float = 0.5  # 0 = likely AI, 1 = likely real camera
    
    # Comparison results
    correlation: Optional[float] = None
    correlation_threshold: float = 0.02
    
    # Quality metrics
    quality_score: float = 0.0
    saturation_ratio: float = 0.0  # Ratio of saturated pixels
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding large arrays)."""
        return {
            "noise_strength": round(self.noise_strength, 6),
            "noise_uniformity": round(self.noise_uniformity, 4),
            "has_prnu_signature": self.has_prnu_signature,
            "prnu_score": round(self.prnu_score, 4),
            "correlation": round(self.correlation, 6) if self.correlation else None,
            "quality_score": round(self.quality_score, 4),
            "saturation_ratio": round(self.saturation_ratio, 4),
        }
    
    @property
    def is_likely_real(self) -> bool:
        """Check if image is likely from a real camera based on PRNU."""
        return self.has_prnu_signature and self.prnu_score > 0.5


class PRNUExtractor:
    """
    Extracts PRNU (sensor noise) patterns from images.
    
    PRNU is extracted by:
    1. Denoising the image to get a "clean" version
    2. Subtracting the clean version from original
    3. The residual contains the sensor noise pattern
    
    Usage:
        extractor = PRNUExtractor()
        
        # Extract PRNU from single image
        features = extractor.extract("photo.jpg")
        print(features.prnu_score)  # 0.72 = likely real
        
        # Build reference fingerprint from multiple images
        reference = extractor.build_reference(["img1.jpg", "img2.jpg", "img3.jpg"])
        
        # Compare new image to reference
        features = extractor.extract("test.jpg", reference=reference)
        print(features.correlation)  # High = same camera
    """
    
    def __init__(
        self,
        denoise_strength: float = 3.0,
        min_quality: float = 0.3,
    ):
        """
        Initialize the PRNU extractor.
        
        Args:
            denoise_strength: Strength of denoising filter
            min_quality: Minimum quality score for valid extraction
        """
        if not SCIPY_AVAILABLE:
            logger.warning(
                "scipy not available. Using simplified PRNU extraction."
            )
        
        self.denoise_strength = denoise_strength
        self.min_quality = min_quality
    
    def extract(
        self,
        source: Union[str, Path, Image.Image, np.ndarray],
        reference: Optional[np.ndarray] = None,
    ) -> PRNUFeatures:
        """
        Extract PRNU features from an image.
        
        Args:
            source: Image to analyze
            reference: Optional reference fingerprint for comparison
            
        Returns:
            PRNUFeatures with extraction results
        """
        # Load image
        image = self._load_image(source)
        
        # Convert to float
        img_float = image.astype(np.float64)
        
        # Check image quality
        quality_score, saturation_ratio = self._assess_quality(image)
        
        if quality_score < self.min_quality:
            logger.debug(f"Image quality too low: {quality_score:.2f}")
            return PRNUFeatures(
                quality_score=quality_score,
                saturation_ratio=saturation_ratio,
                prnu_score=0.5,  # Uncertain
            )
        
        # Extract noise residual
        noise_residual = self._extract_noise(img_float)
        
        # Compute metrics
        noise_strength = self._compute_noise_strength(noise_residual)
        noise_uniformity = self._compute_uniformity(noise_residual)
        
        # Determine if valid PRNU signature exists
        has_prnu = self._detect_prnu_signature(
            noise_residual, noise_strength, noise_uniformity
        )
        
        # Compute PRNU score
        prnu_score = self._compute_prnu_score(
            noise_strength, noise_uniformity, has_prnu
        )
        
        # Compare to reference if provided
        correlation = None
        if reference is not None:
            correlation = self._compute_correlation(noise_residual, reference)
            # Adjust PRNU score based on correlation
            if correlation is not None:
                if correlation > 0.02:  # Strong match
                    prnu_score = min(1.0, prnu_score + 0.2)
                elif correlation < 0.005:  # Weak match
                    prnu_score = max(0.0, prnu_score - 0.1)
        
        return PRNUFeatures(
            noise_residual=noise_residual,
            noise_strength=noise_strength,
            noise_uniformity=noise_uniformity,
            has_prnu_signature=has_prnu,
            prnu_score=prnu_score,
            correlation=correlation,
            quality_score=quality_score,
            saturation_ratio=saturation_ratio,
        )
    
    def build_reference(
        self,
        sources: list[Union[str, Path, Image.Image]],
        max_images: int = 50,
    ) -> np.ndarray:
        """
        Build a reference PRNU fingerprint from multiple images.
        
        For best results, use 20-50 images from the same camera,
        ideally of flat, well-lit surfaces (e.g., blue sky, white wall).
        
        Args:
            sources: List of images from the same camera
            max_images: Maximum number of images to use
            
        Returns:
            Reference fingerprint as numpy array
        """
        sources = sources[:max_images]
        
        if len(sources) < 3:
            raise ValueError("Need at least 3 images to build reference")
        
        logger.info(f"Building PRNU reference from {len(sources)} images")
        
        # Extract noise from each image
        noise_patterns = []
        reference_shape = None
        
        for source in sources:
            try:
                image = self._load_image(source)
                
                # Ensure consistent shape
                if reference_shape is None:
                    reference_shape = image.shape
                elif image.shape != reference_shape:
                    # Resize to match
                    from PIL import Image as PILImage
                    pil_img = PILImage.fromarray(image)
                    pil_img = pil_img.resize(
                        (reference_shape[1], reference_shape[0]),
                        PILImage.Resampling.LANCZOS
                    )
                    image = np.array(pil_img)
                
                img_float = image.astype(np.float64)
                noise = self._extract_noise(img_float)
                noise_patterns.append(noise)
                
            except Exception as e:
                logger.warning(f"Failed to process {source}: {e}")
                continue
        
        if len(noise_patterns) < 3:
            raise ValueError("Failed to extract noise from enough images")
        
        # Average the noise patterns (reduces content, amplifies PRNU)
        reference = np.mean(noise_patterns, axis=0)
        
        # Normalize
        reference = reference / (np.std(reference) + 1e-10)
        
        logger.info(f"Reference fingerprint built: shape={reference.shape}")
        
        return reference
    
    def _load_image(
        self,
        source: Union[str, Path, Image.Image, np.ndarray],
    ) -> np.ndarray:
        """Load image as numpy array."""
        if isinstance(source, np.ndarray):
            return source
        
        if isinstance(source, Image.Image):
            return np.array(source.convert("RGB"))
        
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        image = Image.open(path).convert("RGB")
        return np.array(image)
    
    def _extract_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Extract noise residual from image.
        
        Uses denoising filter to estimate the "clean" image,
        then subtracts to get noise.
        """
        if SCIPY_AVAILABLE:
            # Use Wiener filter for denoising (per channel)
            denoised = np.zeros_like(image)
            for c in range(image.shape[2] if image.ndim == 3 else 1):
                if image.ndim == 3:
                    channel = image[:, :, c]
                else:
                    channel = image
                
                # Wiener filter
                try:
                    denoised_channel = wiener(channel, mysize=int(self.denoise_strength))
                except Exception:
                    # Fallback to uniform filter
                    denoised_channel = uniform_filter(
                        channel, size=int(self.denoise_strength)
                    )
                
                if image.ndim == 3:
                    denoised[:, :, c] = denoised_channel
                else:
                    denoised = denoised_channel
        else:
            # Simple box filter fallback
            denoised = self._simple_denoise(image)
        
        # Noise residual = original - denoised
        noise = image - denoised
        
        # Zero-mean normalization
        noise = noise - np.mean(noise)
        
        return noise
    
    def _simple_denoise(self, image: np.ndarray) -> np.ndarray:
        """Simple box filter denoising (fallback when scipy unavailable)."""
        kernel_size = int(self.denoise_strength)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        denoised = np.zeros_like(image, dtype=np.float64)
        pad = kernel_size // 2
        
        for c in range(image.shape[2] if image.ndim == 3 else 1):
            if image.ndim == 3:
                channel = image[:, :, c]
            else:
                channel = image
            
            # Pad image
            padded = np.pad(channel, pad, mode='reflect')
            
            # Box filter
            for i in range(channel.shape[0]):
                for j in range(channel.shape[1]):
                    denoised[i, j, c] = np.mean(
                        padded[i:i+kernel_size, j:j+kernel_size]
                    )
        
        return denoised
    
    def _assess_quality(self, image: np.ndarray) -> tuple[float, float]:
        """
        Assess image quality for PRNU extraction.
        
        Returns:
            Tuple of (quality_score, saturation_ratio)
        """
        # Check for saturation (pixels at 0 or 255)
        saturated = np.sum((image <= 5) | (image >= 250))
        total_pixels = image.size
        saturation_ratio = saturated / total_pixels
        
        # Check for sufficient texture
        if image.ndim == 3:
            gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        else:
            gray = image
        
        texture = np.std(gray)
        
        # Quality score
        quality = 1.0
        
        # Penalize high saturation
        if saturation_ratio > 0.3:
            quality *= (1.0 - saturation_ratio)
        
        # Penalize low texture (flat images are actually good for PRNU, but
        # we need some texture to detect artifacts)
        if texture < 10:
            quality *= (texture / 10)
        
        return quality, saturation_ratio
    
    def _compute_noise_strength(self, noise: np.ndarray) -> float:
        """Compute overall noise strength."""
        return float(np.std(noise))
    
    def _compute_uniformity(self, noise: np.ndarray) -> float:
        """
        Compute noise uniformity across the image.
        
        Real PRNU is relatively uniform; AI artifacts may be localized.
        """
        # Divide into blocks and compute variance in each
        block_size = 64
        h, w = noise.shape[:2]
        
        if h < block_size * 2 or w < block_size * 2:
            return 1.0  # Image too small
        
        block_stds = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = noise[i:i+block_size, j:j+block_size]
                block_stds.append(np.std(block))
        
        if not block_stds:
            return 1.0
        
        # Uniformity = inverse of coefficient of variation
        mean_std = np.mean(block_stds)
        std_of_stds = np.std(block_stds)
        
        if mean_std < 1e-10:
            return 0.0  # No noise at all (suspicious)
        
        cv = std_of_stds / mean_std
        uniformity = 1.0 / (1.0 + cv * 2)
        
        return float(uniformity)
    
    def _detect_prnu_signature(
        self,
        noise: np.ndarray,
        strength: float,
        uniformity: float,
    ) -> bool:
        """
        Determine if a valid PRNU signature exists.
        
        Real cameras produce:
        - Consistent noise strength (not too weak, not too strong)
        - Uniform noise distribution
        - Specific spectral characteristics
        """
        # Check noise strength is in expected range
        if strength < 0.5 or strength > 15.0:
            return False
        
        # Check uniformity
        if uniformity < 0.4:
            return False
        
        # Additional spectral check
        spectral_score = self._compute_spectral_score(noise)
        if spectral_score < 0.3:
            return False
        
        return True
    
    def _compute_spectral_score(self, noise: np.ndarray) -> float:
        """
        Analyze frequency spectrum of noise.
        
        Real PRNU has specific spectral characteristics;
        AI-generated noise patterns differ.
        """
        # Use first channel if color
        if noise.ndim == 3:
            noise_2d = noise[:, :, 0]
        else:
            noise_2d = noise
        
        # Compute FFT
        fft = np.fft.fft2(noise_2d)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # Analyze high-frequency content
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency region (center)
        low_freq = magnitude[
            center_h - h//8:center_h + h//8,
            center_w - w//8:center_w + w//8
        ]
        
        # High frequency region (corners)
        high_freq = np.concatenate([
            magnitude[:h//4, :w//4].flatten(),
            magnitude[:h//4, -w//4:].flatten(),
            magnitude[-h//4:, :w//4].flatten(),
            magnitude[-h//4:, -w//4:].flatten(),
        ])
        
        # PRNU has relatively more high-frequency content
        low_energy = np.mean(low_freq)
        high_energy = np.mean(high_freq)
        
        if low_energy < 1e-10:
            return 0.0
        
        ratio = high_energy / low_energy
        
        # Score: PRNU typically has ratio in certain range
        if 0.1 < ratio < 2.0:
            return min(1.0, ratio / 0.5)
        else:
            return max(0.0, 1.0 - abs(ratio - 0.5) / 2.0)
    
    def _compute_prnu_score(
        self,
        strength: float,
        uniformity: float,
        has_prnu: bool,
    ) -> float:
        """
        Compute overall PRNU score (0 = likely AI, 1 = likely real).
        """
        if not has_prnu:
            return 0.3  # No PRNU detected, likely AI
        
        # Base score
        score = 0.5
        
        # Adjust based on noise characteristics
        # Optimal strength range for real photos: 1-8
        if 1.0 < strength < 8.0:
            score += 0.2
        elif strength < 0.5 or strength > 15.0:
            score -= 0.2
        
        # Uniformity bonus
        score += uniformity * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _compute_correlation(
        self,
        noise: np.ndarray,
        reference: np.ndarray,
    ) -> Optional[float]:
        """
        Compute normalized cross-correlation between noise and reference.
        """
        # Ensure same shape
        if noise.shape != reference.shape:
            # Resize noise to match reference
            min_h = min(noise.shape[0], reference.shape[0])
            min_w = min(noise.shape[1], reference.shape[1])
            noise = noise[:min_h, :min_w]
            reference = reference[:min_h, :min_w]
        
        # Flatten for correlation
        if noise.ndim == 3:
            noise_flat = noise.flatten()
            ref_flat = reference.flatten()
        else:
            noise_flat = noise.flatten()
            ref_flat = reference.flatten()
        
        # Normalized cross-correlation
        noise_norm = noise_flat - np.mean(noise_flat)
        ref_norm = ref_flat - np.mean(ref_flat)
        
        numerator = np.sum(noise_norm * ref_norm)
        denominator = np.sqrt(np.sum(noise_norm**2) * np.sum(ref_norm**2))
        
        if denominator < 1e-10:
            return None
        
        correlation = numerator / denominator
        
        return float(correlation)


# Convenience function
def extract_prnu(
    source: Union[str, Path, Image.Image, np.ndarray],
) -> PRNUFeatures:
    """
    Extract PRNU features from an image.
    
    Args:
        source: Image to analyze
        
    Returns:
        PRNUFeatures object
    """
    extractor = PRNUExtractor()
    return extractor.extract(source)
