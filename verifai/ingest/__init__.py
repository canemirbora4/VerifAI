"""
Ingest Module - Media Loading and Preprocessing
================================================

Handles loading and preprocessing of images and videos:
- Format validation
- Image loading with PIL/OpenCV
- Video frame extraction with FFmpeg
- Preprocessing pipelines for model input
"""

from verifai.ingest.image_loader import ImageLoader, load_image
from verifai.ingest.utils import (
    validate_file_path,
    get_media_type,
    MediaType,
)
from verifai.ingest.video_loader import (
    VideoLoader,
    VideoData,
    VideoFrame,
    VideoMetadata,
    load_video,
)

__all__ = [
    # Image
    "ImageLoader",
    "load_image",
    # Video
    "VideoLoader",
    "VideoData",
    "VideoFrame",
    "VideoMetadata",
    "load_video",
    # Utils
    "validate_file_path",
    "get_media_type",
    "MediaType",
]
