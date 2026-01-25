"""
Frequency Domain Feature Extraction
=====================================

Extracts frequency-domain features from images using FFT and DCT.
AI-generated images often have different frequency characteristics than real photos:
- GAN/Diffusion models may leave artifacts in specific frequency bands
- Camera images have characteristic noise patterns
- Compression artifacts differ between synthetic and real images

Features extracted:
1. FFT magnitude spectrum statistics
2. DCT coefficient distributions
3. Patch-based frequency analysis
4. Azimuthal frequency profile (radial averaging)
"""

from typing import Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
from scipy import fftpack
from scipy.ndimage import uniform_filter
from PIL import Image
import torch
from loguru import logger


@dataclass
class FrequencyFeatures:
    """
    Container for extracted frequency features.
    
    Attributes:
        fft_magnitude: FFT magnitude spectrum (2D array)
        fft_phase: FFT phase spectrum (2D array)
        fft_stats: Statistical features from FFT
        dct_coeffs: DCT coefficients
        dct_stats: Statistical features from DCT
        azimuthal_profile: Radially averaged frequency profile
        patch_features: Per-patch frequency features
        feature_vector: Concatenated feature vector for classification
    """
    
    # Raw spectra
    fft_magnitude: Optional[np.ndarray] = None
    fft_phase: Optional[np.ndarray] = None
    dct_coeffs: Optional[np.ndarray] = None
    
    # Statistical features
    fft_stats: dict = field(default_factory=dict)
    dct_stats: dict = field(default_factory=dict)
    
    # Derived features
    azimuthal_profile: Optional[np.ndarray] = None
    patch_features: Optional[np.ndarray] = None
    
    # Final feature vector
    feature_vector: Optional[np.ndarray] = None
    
    @property
    def feature_dim(self) -> int:
        """Get dimension of feature vector."""
        if self.feature_vector is not None:
            return len(self.feature_vector)
        return 0


class FrequencyExtractor:
    """
    Extracts frequency-domain features from images.
    
    This extractor computes various frequency-domain representations
    that can help distinguish AI-generated from real images.
    
    Usage:
        extractor = FrequencyExtractor()
        features = extractor.extract(image)
        vector = features.feature_vector  # For classification
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        patch_size: int = 64,
        num_azimuthal_bins: int = 64,
        compute_patches: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize the frequency extractor.
        
        Args:
            image_size: Size to resize images to before processing
            patch_size: Size of patches for local frequency analysis
            num_azimuthal_bins: Number of bins for radial frequency profile
            compute_patches: Whether to compute patch-based features
            normalize: Whether to normalize features
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_azimuthal_bins = num_azimuthal_bins
        self.compute_patches = compute_patches
        self.normalize = normalize
        
        logger.debug(
            f"FrequencyExtractor initialized: size={image_size}, "
            f"patch_size={patch_size}, bins={num_azimuthal_bins}"
        )
    
    def extract(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        return_spectra: bool = False,
    ) -> FrequencyFeatures:
        """
        Extract frequency features from an image.
        
        Args:
            image: Input image (PIL, numpy, or tensor)
            return_spectra: Whether to include full spectra in output
            
        Returns:
            FrequencyFeatures containing extracted features
        """
        # Convert to grayscale numpy array
        gray = self._to_grayscale(image)
        
        # Resize to standard size
        gray = self._resize(gray)
        
        # Compute FFT features
        fft_mag, fft_phase, fft_stats = self._compute_fft_features(gray)
        
        # Compute DCT features
        dct_coeffs, dct_stats = self._compute_dct_features(gray)
        
        # Compute azimuthal profile
        azimuthal = self._compute_azimuthal_profile(fft_mag)
        
        # Compute patch features
        patch_features = None
        if self.compute_patches:
            patch_features = self._compute_patch_features(gray)
        
        # Build feature vector
        feature_vector = self._build_feature_vector(
            fft_stats, dct_stats, azimuthal, patch_features
        )
        
        # Create result
        features = FrequencyFeatures(
            fft_stats=fft_stats,
            dct_stats=dct_stats,
            azimuthal_profile=azimuthal,
            patch_features=patch_features,
            feature_vector=feature_vector,
        )
        
        # Optionally include full spectra
        if return_spectra:
            features.fft_magnitude = fft_mag
            features.fft_phase = fft_phase
            features.dct_coeffs = dct_coeffs
        
        return features
    
    def _to_grayscale(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Convert input to grayscale numpy array."""
        if isinstance(image, torch.Tensor):
            # Assume CHW or HW format
            arr = image.cpu().numpy()
            if arr.ndim == 3:
                if arr.shape[0] in [1, 3, 4]:  # CHW
                    arr = arr.transpose(1, 2, 0)
                if arr.shape[2] == 3:
                    arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
                elif arr.shape[2] == 1:
                    arr = arr[:, :, 0]
            return arr.astype(np.float32)
        
        elif isinstance(image, Image.Image):
            gray = image.convert("L")
            return np.array(gray, dtype=np.float32)
        
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                if image.shape[2] == 3:
                    return (0.299 * image[:, :, 0] + 
                            0.587 * image[:, :, 1] + 
                            0.114 * image[:, :, 2]).astype(np.float32)
                elif image.shape[2] == 1:
                    return image[:, :, 0].astype(np.float32)
            return image.astype(np.float32)
        
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _resize(self, gray: np.ndarray) -> np.ndarray:
        """Resize image to standard size."""
        if gray.shape == self.image_size:
            return gray
        
        # Use PIL for resizing
        img = Image.fromarray(gray.astype(np.uint8))
        img = img.resize(self.image_size, Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.float32)
    
    def _compute_fft_features(
        self,
        gray: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Compute FFT-based features.
        
        Returns:
            Tuple of (magnitude, phase, statistics_dict)
        """
        # Apply window to reduce edge effects
        window = np.outer(
            np.hanning(gray.shape[0]),
            np.hanning(gray.shape[1])
        )
        windowed = gray * window
        
        # Compute 2D FFT
        fft = np.fft.fft2(windowed)
        fft_shift = np.fft.fftshift(fft)
        
        # Magnitude and phase
        magnitude = np.abs(fft_shift)
        phase = np.angle(fft_shift)
        
        # Log magnitude (avoid log(0))
        log_magnitude = np.log1p(magnitude)
        
        # Compute statistics
        stats = {
            # Global statistics
            "mean": float(np.mean(log_magnitude)),
            "std": float(np.std(log_magnitude)),
            "max": float(np.max(log_magnitude)),
            "min": float(np.min(log_magnitude)),
            "median": float(np.median(log_magnitude)),
            
            # Percentiles
            "p25": float(np.percentile(log_magnitude, 25)),
            "p75": float(np.percentile(log_magnitude, 75)),
            "p90": float(np.percentile(log_magnitude, 90)),
            "p99": float(np.percentile(log_magnitude, 99)),
            
            # Frequency band statistics
            **self._compute_band_statistics(log_magnitude),
        }
        
        return magnitude, phase, stats
    
    def _compute_band_statistics(self, log_magnitude: np.ndarray) -> dict:
        """Compute statistics for different frequency bands."""
        h, w = log_magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Create distance map from center
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        
        # Define bands (low, mid-low, mid-high, high)
        bands = [
            ("low", 0, 0.15),
            ("mid_low", 0.15, 0.35),
            ("mid_high", 0.35, 0.65),
            ("high", 0.65, 1.0),
        ]
        
        band_stats = {}
        for name, r_min, r_max in bands:
            mask = (distance >= r_min * max_dist) & (distance < r_max * max_dist)
            if np.any(mask):
                band_values = log_magnitude[mask]
                band_stats[f"band_{name}_mean"] = float(np.mean(band_values))
                band_stats[f"band_{name}_std"] = float(np.std(band_values))
                band_stats[f"band_{name}_energy"] = float(np.sum(band_values ** 2))
        
        return band_stats
    
    def _compute_dct_features(
        self,
        gray: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute DCT-based features.
        
        DCT is commonly used in JPEG compression, so these features
        can reveal compression-related artifacts.
        
        Returns:
            Tuple of (dct_coefficients, statistics_dict)
        """
        # Compute 2D DCT
        dct = fftpack.dct(fftpack.dct(gray.T, norm='ortho').T, norm='ortho')
        
        # Log absolute values
        log_dct = np.log1p(np.abs(dct))
        
        # Statistics
        stats = {
            "dct_mean": float(np.mean(log_dct)),
            "dct_std": float(np.std(log_dct)),
            "dct_max": float(np.max(log_dct)),
            "dct_energy": float(np.sum(log_dct ** 2)),
        }
        
        # Zigzag scan statistics (captures JPEG-like coefficient ordering)
        zigzag = self._zigzag_scan(log_dct[:64, :64])
        stats["dct_zigzag_mean_early"] = float(np.mean(zigzag[:64]))
        stats["dct_zigzag_mean_mid"] = float(np.mean(zigzag[64:256]))
        stats["dct_zigzag_mean_late"] = float(np.mean(zigzag[256:]))
        
        # Ratio of high to low frequency energy
        h, w = log_dct.shape
        low_freq = log_dct[:h//4, :w//4]
        high_freq = log_dct[h//4:, w//4:]
        stats["dct_high_low_ratio"] = float(
            np.sum(high_freq ** 2) / (np.sum(low_freq ** 2) + 1e-10)
        )
        
        return dct, stats
    
    def _zigzag_scan(self, block: np.ndarray) -> np.ndarray:
        """Perform zigzag scan of a 2D block (like JPEG)."""
        h, w = block.shape
        result = []
        
        for s in range(h + w - 1):
            if s % 2 == 0:
                # Even diagonal: go up
                for i in range(min(s, h - 1), max(0, s - w + 1) - 1, -1):
                    j = s - i
                    if 0 <= j < w:
                        result.append(block[i, j])
            else:
                # Odd diagonal: go down
                for i in range(max(0, s - w + 1), min(s, h - 1) + 1):
                    j = s - i
                    if 0 <= j < w:
                        result.append(block[i, j])
        
        return np.array(result)
    
    def _compute_azimuthal_profile(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Compute azimuthally averaged frequency profile.
        
        This collapses the 2D spectrum into a 1D profile by averaging
        over angles at each radius (frequency).
        """
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Create distance map
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        
        # Bin by distance
        max_dist = min(center_x, center_y)
        bins = np.linspace(0, max_dist, self.num_azimuthal_bins + 1)
        
        profile = np.zeros(self.num_azimuthal_bins)
        log_mag = np.log1p(magnitude)
        
        for i in range(self.num_azimuthal_bins):
            mask = (distance >= bins[i]) & (distance < bins[i + 1])
            if np.any(mask):
                profile[i] = np.mean(log_mag[mask])
        
        # Normalize
        if self.normalize and np.std(profile) > 0:
            profile = (profile - np.mean(profile)) / np.std(profile)
        
        return profile
    
    def _compute_patch_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute frequency features for local patches.
        
        This can reveal spatial variations in frequency characteristics,
        which may differ between AI and real images.
        """
        h, w = gray.shape
        patch_h = h // self.patch_size
        patch_w = w // self.patch_size
        
        if patch_h == 0 or patch_w == 0:
            return np.array([])
        
        features_list = []
        
        for i in range(patch_h):
            for j in range(patch_w):
                # Extract patch
                y_start = i * self.patch_size
                x_start = j * self.patch_size
                patch = gray[y_start:y_start + self.patch_size,
                            x_start:x_start + self.patch_size]
                
                # Compute FFT of patch
                fft = np.fft.fft2(patch)
                fft_shift = np.fft.fftshift(fft)
                log_mag = np.log1p(np.abs(fft_shift))
                
                # Simple statistics per patch
                patch_feat = [
                    np.mean(log_mag),
                    np.std(log_mag),
                    np.max(log_mag),
                ]
                features_list.append(patch_feat)
        
        patch_features = np.array(features_list)
        
        # Aggregate patch statistics
        if len(patch_features) > 0:
            aggregated = np.concatenate([
                np.mean(patch_features, axis=0),
                np.std(patch_features, axis=0),
                np.max(patch_features, axis=0),
                np.min(patch_features, axis=0),
            ])
            return aggregated
        
        return np.array([])
    
    def _build_feature_vector(
        self,
        fft_stats: dict,
        dct_stats: dict,
        azimuthal: np.ndarray,
        patch_features: Optional[np.ndarray],
    ) -> np.ndarray:
        """Concatenate all features into a single vector."""
        parts = []
        
        # FFT statistics
        fft_values = [fft_stats[k] for k in sorted(fft_stats.keys())]
        parts.append(np.array(fft_values))
        
        # DCT statistics
        dct_values = [dct_stats[k] for k in sorted(dct_stats.keys())]
        parts.append(np.array(dct_values))
        
        # Azimuthal profile
        parts.append(azimuthal)
        
        # Patch features
        if patch_features is not None and len(patch_features) > 0:
            parts.append(patch_features)
        
        # Concatenate
        feature_vector = np.concatenate(parts).astype(np.float32)
        
        # Normalize if requested
        if self.normalize:
            mean = np.mean(feature_vector)
            std = np.std(feature_vector)
            if std > 0:
                feature_vector = (feature_vector - mean) / std
        
        return feature_vector
    
    def get_feature_dim(self) -> int:
        """
        Get the dimension of the output feature vector.
        
        Useful for initializing classifiers.
        """
        # Create dummy image and extract features to get dimension
        dummy = np.random.rand(*self.image_size).astype(np.float32)
        features = self.extract(dummy)
        return features.feature_dim


def extract_frequency_features(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    **kwargs,
) -> FrequencyFeatures:
    """
    Convenience function to extract frequency features.
    
    Args:
        image: Input image
        **kwargs: Arguments passed to FrequencyExtractor
        
    Returns:
        FrequencyFeatures object
    """
    extractor = FrequencyExtractor(**kwargs)
    return extractor.extract(image)
