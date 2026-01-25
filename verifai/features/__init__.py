"""
Features Module - Feature Extraction
=====================================

Feature extractors for different detection signals:
- Frequency domain features (FFT, DCT)
- Metadata parsing (EXIF, provenance)
- Temporal features (video) - Flicker, consistency analysis
- PRNU (camera sensor noise) - Phase 5

These features are combined in the ensemble for robust detection.
"""

from verifai.features.frequency import (
    FrequencyExtractor,
    FrequencyFeatures,
    extract_frequency_features,
)
from verifai.features.metadata import (
    MetadataParser,
    MetadataFeatures,
    parse_metadata,
)
from verifai.features.temporal import (
    TemporalAnalyzer,
    TemporalFeatures,
    analyze_temporal,
)

__all__ = [
    # Frequency
    "FrequencyExtractor",
    "FrequencyFeatures",
    "extract_frequency_features",
    # Metadata
    "MetadataParser",
    "MetadataFeatures",
    "parse_metadata",
    # Temporal (Video)
    "TemporalAnalyzer",
    "TemporalFeatures",
    "analyze_temporal",
]
