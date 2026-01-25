"""
Metadata Analysis
==================

Analyzes image metadata (EXIF, XMP, etc.) for provenance signals.

Real camera photos typically have:
- Camera make/model
- Lens information
- GPS coordinates (sometimes)
- Consistent software tags

AI-generated images often:
- Lack EXIF entirely
- Have generic software tags ("Adobe Photoshop", "PIL", etc.)
- Missing camera-specific fields
- May have C2PA content credentials (newer models)

This module extracts metadata features that can indicate authenticity.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Any
import re
import json

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS, IFD
from loguru import logger


# Common camera manufacturers
CAMERA_MANUFACTURERS = frozenset({
    "canon", "nikon", "sony", "fujifilm", "fuji", "panasonic", "olympus",
    "pentax", "leica", "hasselblad", "samsung", "lg", "apple", "google",
    "huawei", "xiaomi", "oppo", "vivo", "oneplus", "motorola", "nokia",
})

# Software tags that suggest AI or heavy editing
AI_SOFTWARE_INDICATORS = frozenset({
    "midjourney", "stable diffusion", "dall-e", "dalle", "openai",
    "replicate", "runway", "firefly", "adobe firefly",
    "automatic1111", "comfyui", "invokeai",
})

# Generic editing software (not necessarily AI, but not camera)
EDITING_SOFTWARE = frozenset({
    "photoshop", "lightroom", "gimp", "paint", "snapseed",
    "picsart", "vsco", "afterlight", "pixlr",
})


@dataclass
class MetadataFeatures:
    """
    Container for extracted metadata features.
    
    Attributes:
        has_exif: Whether EXIF data exists
        has_camera_info: Whether camera make/model is present
        has_gps: Whether GPS data is present
        has_timestamp: Whether capture timestamp is present
        software_tag: Software field value (if any)
        camera_make: Camera manufacturer
        camera_model: Camera model
        is_suspicious: Overall suspicion flag
        suspicion_reasons: List of reasons for suspicion
        confidence_real: Confidence that image is from a camera
        raw_exif: Raw EXIF dictionary
        feature_vector: Numeric feature vector for classification
    """
    
    # Boolean flags
    has_exif: bool = False
    has_camera_info: bool = False
    has_gps: bool = False
    has_timestamp: bool = False
    has_lens_info: bool = False
    has_exposure_info: bool = False
    
    # Extracted values
    software_tag: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    datetime_original: Optional[str] = None
    
    # Analysis results
    is_suspicious: bool = False
    suspicion_reasons: list = field(default_factory=list)
    confidence_real: float = 0.5  # 0 = likely AI, 1 = likely real
    
    # Raw data
    raw_exif: dict = field(default_factory=dict)
    
    # Feature vector for classification
    feature_vector: Optional[list] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding raw_exif for brevity)."""
        return {
            "has_exif": self.has_exif,
            "has_camera_info": self.has_camera_info,
            "has_gps": self.has_gps,
            "has_timestamp": self.has_timestamp,
            "has_lens_info": self.has_lens_info,
            "has_exposure_info": self.has_exposure_info,
            "software_tag": self.software_tag,
            "camera_make": self.camera_make,
            "camera_model": self.camera_model,
            "datetime_original": self.datetime_original,
            "is_suspicious": self.is_suspicious,
            "suspicion_reasons": self.suspicion_reasons,
            "confidence_real": self.confidence_real,
        }


class MetadataParser:
    """
    Parses and analyzes image metadata for provenance signals.
    
    Usage:
        parser = MetadataParser()
        features = parser.parse("image.jpg")
        print(features.confidence_real)
    """
    
    def __init__(
        self,
        strict_mode: bool = False,
    ):
        """
        Initialize the metadata parser.
        
        Args:
            strict_mode: If True, any missing camera data is suspicious
        """
        self.strict_mode = strict_mode
    
    def parse(
        self,
        source: Union[str, Path, Image.Image],
    ) -> MetadataFeatures:
        """
        Parse metadata from an image.
        
        Args:
            source: Image path or PIL Image
            
        Returns:
            MetadataFeatures with extracted information
        """
        # Load image if needed
        if isinstance(source, (str, Path)):
            try:
                image = Image.open(source)
            except Exception as e:
                logger.warning(f"Could not open image for metadata: {e}")
                return MetadataFeatures()
        else:
            image = source
        
        # Extract EXIF
        raw_exif = self._extract_exif(image)
        
        # Initialize features
        features = MetadataFeatures(
            has_exif=bool(raw_exif),
            raw_exif=raw_exif,
        )
        
        if not raw_exif:
            features.suspicion_reasons.append("No EXIF data found")
            features.is_suspicious = True
            features.confidence_real = 0.3
            features.feature_vector = self._build_feature_vector(features)
            return features
        
        # Parse specific fields
        self._parse_camera_info(features, raw_exif)
        self._parse_datetime(features, raw_exif)
        self._parse_gps(features, raw_exif)
        self._parse_software(features, raw_exif)
        self._parse_exposure(features, raw_exif)
        self._parse_lens(features, raw_exif)
        
        # Analyze for suspicion
        self._analyze_suspicion(features)
        
        # Build feature vector
        features.feature_vector = self._build_feature_vector(features)
        
        return features
    
    def _extract_exif(self, image: Image.Image) -> dict:
        """Extract EXIF data from image."""
        exif_dict = {}
        
        try:
            exif = image.getexif()
            if not exif:
                return exif_dict
            
            # Basic EXIF tags
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                exif_dict[tag_name] = self._clean_value(value)
            
            # IFD data (nested EXIF)
            for ifd_id in IFD:
                try:
                    ifd = exif.get_ifd(ifd_id)
                    if ifd:
                        for tag_id, value in ifd.items():
                            tag_name = TAGS.get(tag_id, str(tag_id))
                            exif_dict[tag_name] = self._clean_value(value)
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"Error extracting EXIF: {e}")
        
        return exif_dict
    
    def _clean_value(self, value: Any) -> Any:
        """Clean EXIF value for storage."""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return str(value)[:100]
        elif isinstance(value, (int, float, str)):
            return value
        else:
            return str(value)[:200]
    
    def _parse_camera_info(self, features: MetadataFeatures, exif: dict) -> None:
        """Parse camera make and model."""
        make = exif.get("Make", "")
        model = exif.get("Model", "")
        
        if make:
            features.camera_make = str(make).strip()
        if model:
            features.camera_model = str(model).strip()
        
        features.has_camera_info = bool(make or model)
    
    def _parse_datetime(self, features: MetadataFeatures, exif: dict) -> None:
        """Parse datetime information."""
        datetime_original = exif.get("DateTimeOriginal")
        datetime_digitized = exif.get("DateTimeDigitized")
        datetime_basic = exif.get("DateTime")
        
        timestamp = datetime_original or datetime_digitized or datetime_basic
        
        if timestamp:
            features.datetime_original = str(timestamp)
            features.has_timestamp = True
    
    def _parse_gps(self, features: MetadataFeatures, exif: dict) -> None:
        """Parse GPS information."""
        gps_fields = ["GPSLatitude", "GPSLongitude", "GPSInfo", "GPSLatitudeRef"]
        features.has_gps = any(field in exif for field in gps_fields)
    
    def _parse_software(self, features: MetadataFeatures, exif: dict) -> None:
        """Parse software tag."""
        software = exif.get("Software", "")
        if software:
            features.software_tag = str(software).strip()
    
    def _parse_exposure(self, features: MetadataFeatures, exif: dict) -> None:
        """Parse exposure information."""
        exposure_fields = [
            "ExposureTime", "FNumber", "ISOSpeedRatings", "ISO",
            "ShutterSpeedValue", "ApertureValue",
        ]
        features.has_exposure_info = any(field in exif for field in exposure_fields)
    
    def _parse_lens(self, features: MetadataFeatures, exif: dict) -> None:
        """Parse lens information."""
        lens_fields = [
            "LensModel", "LensMake", "LensInfo", "FocalLength",
            "FocalLengthIn35mmFilm",
        ]
        features.has_lens_info = any(field in exif for field in lens_fields)
    
    def _analyze_suspicion(self, features: MetadataFeatures) -> None:
        """Analyze metadata for suspicious patterns."""
        confidence = 0.5  # Start neutral
        reasons = []
        
        # Positive signals (increase confidence in real)
        if features.has_camera_info:
            make_lower = (features.camera_make or "").lower()
            if any(m in make_lower for m in CAMERA_MANUFACTURERS):
                confidence += 0.15
            else:
                confidence += 0.05
        
        if features.has_gps:
            confidence += 0.1
        
        if features.has_timestamp:
            confidence += 0.05
        
        if features.has_exposure_info:
            confidence += 0.1
        
        if features.has_lens_info:
            confidence += 0.1
        
        # Negative signals (decrease confidence in real)
        if not features.has_exif:
            confidence -= 0.2
            reasons.append("No EXIF data")
        
        if features.software_tag:
            software_lower = features.software_tag.lower()
            
            # Check for AI software
            if any(ai in software_lower for ai in AI_SOFTWARE_INDICATORS):
                confidence -= 0.4
                reasons.append(f"AI software detected: {features.software_tag}")
            
            # Check for editing software (mild suspicion)
            elif any(ed in software_lower for ed in EDITING_SOFTWARE):
                confidence -= 0.1
                reasons.append(f"Editing software: {features.software_tag}")
        
        if not features.has_camera_info and not features.software_tag:
            confidence -= 0.15
            reasons.append("No camera or software information")
        
        # Clamp confidence
        features.confidence_real = max(0.0, min(1.0, confidence))
        
        # Determine if suspicious
        features.is_suspicious = (
            confidence < 0.4 or
            len(reasons) >= 2 or
            any("AI software" in r for r in reasons)
        )
        features.suspicion_reasons = reasons
    
    def _build_feature_vector(self, features: MetadataFeatures) -> list:
        """Build numeric feature vector for classification."""
        return [
            float(features.has_exif),
            float(features.has_camera_info),
            float(features.has_gps),
            float(features.has_timestamp),
            float(features.has_lens_info),
            float(features.has_exposure_info),
            float(features.software_tag is not None),
            float(features.is_suspicious),
            features.confidence_real,
        ]


def parse_metadata(
    source: Union[str, Path, Image.Image],
) -> MetadataFeatures:
    """
    Convenience function to parse image metadata.
    
    Args:
        source: Image path or PIL Image
        
    Returns:
        MetadataFeatures object
    """
    parser = MetadataParser()
    return parser.parse(source)
