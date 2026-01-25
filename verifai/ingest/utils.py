"""
Ingest Utilities
=================

Common utilities for media ingestion including file validation,
media type detection, and format handling.
"""

from enum import Enum
from pathlib import Path
from typing import Union, Optional
import os

from loguru import logger


class MediaType(Enum):
    """Supported media types."""
    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


# Supported file extensions
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"})


class FileValidationError(Exception):
    """Raised when file validation fails."""
    pass


class UnsupportedFormatError(Exception):
    """Raised when file format is not supported."""
    pass


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    allowed_types: Optional[set[MediaType]] = None,
) -> Path:
    """
    Validate a file path and return a resolved Path object.
    
    Args:
        file_path: Path to the file (string or Path object)
        must_exist: If True, raise error if file doesn't exist
        allowed_types: Set of allowed MediaTypes (None = allow all)
        
    Returns:
        Resolved Path object
        
    Raises:
        FileValidationError: If file doesn't exist or path is invalid
        UnsupportedFormatError: If file type is not allowed
    """
    # Convert to Path object
    path = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Resolve to absolute path
    path = path.resolve()
    
    # Check existence
    if must_exist and not path.exists():
        raise FileValidationError(f"File not found: {path}")
    
    if must_exist and not path.is_file():
        raise FileValidationError(f"Path is not a file: {path}")
    
    # Check file type if restrictions specified
    if allowed_types is not None:
        media_type = get_media_type(path)
        if media_type not in allowed_types:
            allowed_str = ", ".join(t.value for t in allowed_types)
            raise UnsupportedFormatError(
                f"File type '{media_type.value}' not allowed. Allowed types: {allowed_str}"
            )
    
    return path


def get_media_type(file_path: Union[str, Path]) -> MediaType:
    """
    Determine the media type based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MediaType enum value
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    suffix = path.suffix.lower()
    
    if suffix in IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    elif suffix in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    else:
        return MediaType.UNKNOWN


def get_file_info(file_path: Union[str, Path]) -> dict:
    """
    Get basic file information.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file info:
        - path: Absolute path
        - name: File name
        - extension: File extension
        - size_bytes: File size in bytes
        - size_human: Human-readable size
        - media_type: MediaType enum value
    """
    path = validate_file_path(file_path)
    stat = path.stat()
    
    return {
        "path": str(path),
        "name": path.name,
        "stem": path.stem,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_human": format_file_size(stat.st_size),
        "media_type": get_media_type(path),
        "modified_time": stat.st_mtime,
    }


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to the directory
        
    Returns:
        Resolved Path object
    """
    path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_media_files(
    directory: Union[str, Path],
    media_type: Optional[MediaType] = None,
    recursive: bool = False,
) -> list[Path]:
    """
    List all media files in a directory.
    
    Args:
        directory: Directory to search
        media_type: Filter by media type (None = all)
        recursive: If True, search subdirectories
        
    Returns:
        List of Path objects for matching files
    """
    path = Path(directory) if isinstance(directory, str) else directory
    path = path.resolve()
    
    if not path.is_dir():
        raise FileValidationError(f"Not a directory: {path}")
    
    # Determine which extensions to look for
    if media_type == MediaType.IMAGE:
        extensions = IMAGE_EXTENSIONS
    elif media_type == MediaType.VIDEO:
        extensions = VIDEO_EXTENSIONS
    else:
        extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    
    # Collect matching files
    files = []
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        files.extend(path.glob(f"{pattern}{ext}"))
        # Also check uppercase extensions
        files.extend(path.glob(f"{pattern}{ext.upper()}"))
    
    # Sort by name for consistent ordering
    files = sorted(set(files))
    
    logger.debug(f"Found {len(files)} media files in {path}")
    return files
