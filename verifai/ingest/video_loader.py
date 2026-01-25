"""
Video Loader
=============

Handles loading and frame extraction from video files.
Uses imageio-ffmpeg for cross-platform video decoding.

Features:
- Multiple sampling strategies (uniform, keyframe, scene-based)
- Efficient frame extraction
- Metadata extraction (duration, fps, resolution)
- Memory-efficient processing for long videos
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Generator, Literal
import tempfile
import subprocess
import json

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from loguru import logger

try:
    import imageio.v3 as iio
    from imageio_ffmpeg import get_ffmpeg_exe
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    logger.warning("imageio not available. Video loading will be limited.")


SamplingStrategy = Literal["uniform", "keyframe", "all", "fps"]


@dataclass
class VideoMetadata:
    """
    Metadata extracted from a video file.
    
    Attributes:
        path: Source file path
        duration: Duration in seconds
        fps: Frames per second
        total_frames: Total frame count
        width: Frame width
        height: Frame height
        codec: Video codec
        bitrate: Video bitrate (if available)
    """
    path: Optional[Path] = None
    duration: float = 0.0
    fps: float = 0.0
    total_frames: int = 0
    width: int = 0
    height: int = 0
    codec: str = ""
    bitrate: Optional[int] = None
    
    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0
    
    def to_dict(self) -> dict:
        return {
            "path": str(self.path) if self.path else None,
            "duration": self.duration,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "bitrate": self.bitrate,
        }


@dataclass
class VideoFrame:
    """
    Container for a single video frame.
    
    Attributes:
        image: Frame as PIL Image
        tensor: Preprocessed tensor (if requested)
        frame_number: Frame index in video
        timestamp: Timestamp in seconds
    """
    image: Image.Image
    tensor: Optional[torch.Tensor] = None
    frame_number: int = 0
    timestamp: float = 0.0
    
    @property
    def numpy(self) -> np.ndarray:
        return np.array(self.image)


@dataclass 
class VideoData:
    """
    Container for loaded video data.
    
    Attributes:
        frames: List of extracted frames
        metadata: Video metadata
        sampling_strategy: How frames were sampled
        frame_indices: Original frame indices
    """
    frames: list[VideoFrame] = field(default_factory=list)
    metadata: VideoMetadata = field(default_factory=VideoMetadata)
    sampling_strategy: str = "uniform"
    frame_indices: list[int] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> VideoFrame:
        return self.frames[idx]
    
    def __iter__(self):
        return iter(self.frames)
    
    @property
    def images(self) -> list[Image.Image]:
        """Get all frames as PIL Images."""
        return [f.image for f in self.frames]
    
    @property
    def tensors(self) -> Optional[torch.Tensor]:
        """Get all frames as batched tensor (N, C, H, W)."""
        if self.frames and self.frames[0].tensor is not None:
            return torch.stack([f.tensor for f in self.frames])
        return None
    
    @property
    def timestamps(self) -> list[float]:
        """Get timestamps for all frames."""
        return [f.timestamp for f in self.frames]


class VideoLoader:
    """
    Loads and extracts frames from video files.
    
    Usage:
        loader = VideoLoader()
        
        # Load with uniform sampling
        video = loader.load("video.mp4", num_frames=16)
        
        # Iterate over frames
        for frame in video:
            result = detector.detect(frame.tensor)
        
        # Get as batch
        batch = video.tensors  # (16, 3, 224, 224)
    """
    
    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        normalize_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, ...] = (0.229, 0.224, 0.225),
        max_frames: int = 1000,
    ):
        """
        Initialize the VideoLoader.
        
        Args:
            target_size: Target frame size for preprocessing
            normalize_mean: Normalization mean
            normalize_std: Normalization std
            max_frames: Maximum frames to extract (safety limit)
        """
        if not IMAGEIO_AVAILABLE:
            raise ImportError(
                "Video loading requires imageio. "
                "Install with: pip install imageio imageio-ffmpeg"
            )
        
        self.target_size = target_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.max_frames = max_frames
        
        # Build preprocessing transform
        self._transform = transforms.Compose([
            transforms.Resize(target_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ])
        
        logger.debug(f"VideoLoader initialized: target_size={target_size}")
    
    def load(
        self,
        source: Union[str, Path],
        num_frames: int = 16,
        strategy: SamplingStrategy = "uniform",
        preprocess: bool = True,
        target_fps: Optional[float] = None,
    ) -> VideoData:
        """
        Load a video and extract frames.
        
        Args:
            source: Path to video file
            num_frames: Number of frames to extract
            strategy: Sampling strategy
            preprocess: Apply preprocessing for model input
            target_fps: Resample to this FPS (for "fps" strategy)
            
        Returns:
            VideoData with extracted frames
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        
        # Get metadata
        metadata = self.get_metadata(path)
        
        # Determine frame indices to extract
        if strategy == "uniform":
            indices = self._uniform_sample(metadata.total_frames, num_frames)
        elif strategy == "keyframe":
            indices = self._keyframe_sample(path, num_frames)
        elif strategy == "fps" and target_fps:
            indices = self._fps_sample(metadata, target_fps, num_frames)
        elif strategy == "all":
            indices = list(range(min(metadata.total_frames, self.max_frames)))
        else:
            indices = self._uniform_sample(metadata.total_frames, num_frames)
        
        # Limit frames
        indices = indices[:self.max_frames]
        
        # Extract frames
        frames = self._extract_frames(path, indices, metadata.fps, preprocess)
        
        return VideoData(
            frames=frames,
            metadata=metadata,
            sampling_strategy=strategy,
            frame_indices=indices,
        )
    
    def get_metadata(self, source: Union[str, Path]) -> VideoMetadata:
        """
        Extract metadata from a video file.
        
        Args:
            source: Path to video file
            
        Returns:
            VideoMetadata object
        """
        path = Path(source)
        
        try:
            # Use ffprobe for accurate metadata
            metadata = self._ffprobe_metadata(path)
            if metadata:
                return metadata
        except Exception as e:
            logger.debug(f"ffprobe failed: {e}")
        
        # Fallback to imageio
        try:
            props = iio.improps(path, plugin="pyav")
            
            # Get duration and fps
            fps = props.get("fps", 30.0)
            if isinstance(fps, tuple):
                fps = fps[0] / fps[1] if fps[1] != 0 else 30.0
            
            n_frames = props.get("n_images", 0)
            shape = props.get("shape", (0, 0))
            
            duration = n_frames / fps if fps > 0 else 0
            
            return VideoMetadata(
                path=path,
                duration=duration,
                fps=float(fps),
                total_frames=n_frames,
                height=shape[0] if len(shape) >= 2 else 0,
                width=shape[1] if len(shape) >= 2 else 0,
            )
        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
            return VideoMetadata(path=path)
    
    def _ffprobe_metadata(self, path: Path) -> Optional[VideoMetadata]:
        """Use ffprobe to get video metadata."""
        try:
            ffmpeg_exe = get_ffmpeg_exe()
            ffprobe_exe = ffmpeg_exe.replace("ffmpeg", "ffprobe")
            
            cmd = [
                ffprobe_exe,
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            # Parse fps
            fps_str = video_stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(fps_str)
            
            # Get frame count
            n_frames = int(video_stream.get("nb_frames", 0))
            if n_frames == 0:
                # Estimate from duration
                duration = float(data.get("format", {}).get("duration", 0))
                n_frames = int(duration * fps)
            
            return VideoMetadata(
                path=path,
                duration=float(data.get("format", {}).get("duration", 0)),
                fps=fps,
                total_frames=n_frames,
                width=int(video_stream.get("width", 0)),
                height=int(video_stream.get("height", 0)),
                codec=video_stream.get("codec_name", ""),
                bitrate=int(video_stream.get("bit_rate", 0)) if video_stream.get("bit_rate") else None,
            )
            
        except Exception as e:
            logger.debug(f"ffprobe error: {e}")
            return None
    
    def _uniform_sample(self, total_frames: int, num_frames: int) -> list[int]:
        """Sample frames uniformly across the video."""
        if total_frames <= 0:
            return []
        
        if num_frames >= total_frames:
            return list(range(total_frames))
        
        # Uniform spacing
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return indices.tolist()
    
    def _keyframe_sample(self, path: Path, num_frames: int) -> list[int]:
        """
        Sample keyframes (I-frames) from the video.
        
        Keyframes often contain scene changes and are more informative.
        """
        try:
            ffmpeg_exe = get_ffmpeg_exe()
            ffprobe_exe = ffmpeg_exe.replace("ffmpeg", "ffprobe")
            
            cmd = [
                ffprobe_exe,
                "-v", "quiet",
                "-select_streams", "v:0",
                "-show_frames",
                "-show_entries", "frame=pict_type,pts_time",
                "-of", "json",
                str(path),
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.warning("Keyframe extraction failed, falling back to uniform")
                metadata = self.get_metadata(path)
                return self._uniform_sample(metadata.total_frames, num_frames)
            
            data = json.loads(result.stdout)
            
            # Find I-frames
            keyframes = []
            for i, frame in enumerate(data.get("frames", [])):
                if frame.get("pict_type") == "I":
                    keyframes.append(i)
            
            if not keyframes:
                metadata = self.get_metadata(path)
                return self._uniform_sample(metadata.total_frames, num_frames)
            
            # Sample from keyframes
            if len(keyframes) <= num_frames:
                return keyframes
            
            indices = np.linspace(0, len(keyframes) - 1, num_frames, dtype=int)
            return [keyframes[i] for i in indices]
            
        except Exception as e:
            logger.warning(f"Keyframe extraction error: {e}")
            metadata = self.get_metadata(path)
            return self._uniform_sample(metadata.total_frames, num_frames)
    
    def _fps_sample(
        self,
        metadata: VideoMetadata,
        target_fps: float,
        max_frames: int,
    ) -> list[int]:
        """Sample frames at a target FPS."""
        if metadata.fps <= 0 or target_fps <= 0:
            return self._uniform_sample(metadata.total_frames, max_frames)
        
        # Calculate frame interval
        interval = metadata.fps / target_fps
        
        indices = []
        current = 0.0
        
        while current < metadata.total_frames and len(indices) < max_frames:
            indices.append(int(current))
            current += interval
        
        return indices
    
    def _extract_frames(
        self,
        path: Path,
        indices: list[int],
        fps: float,
        preprocess: bool,
    ) -> list[VideoFrame]:
        """Extract specific frames from video."""
        frames = []
        
        try:
            # Read all frames (imageio handles seeking internally)
            reader = iio.imiter(path, plugin="pyav")
            
            indices_set = set(indices)
            frame_idx = 0
            
            for frame_array in reader:
                if frame_idx in indices_set:
                    # Convert to PIL
                    image = Image.fromarray(frame_array)
                    
                    # Calculate timestamp
                    timestamp = frame_idx / fps if fps > 0 else 0.0
                    
                    # Preprocess if requested
                    tensor = None
                    if preprocess:
                        tensor = self._transform(image)
                    
                    frames.append(VideoFrame(
                        image=image,
                        tensor=tensor,
                        frame_number=frame_idx,
                        timestamp=timestamp,
                    ))
                    
                    # Early exit if we have all frames
                    if len(frames) >= len(indices):
                        break
                
                frame_idx += 1
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
        
        # Sort by frame number (in case of out-of-order extraction)
        frames.sort(key=lambda f: f.frame_number)
        
        return frames
    
    def stream_frames(
        self,
        source: Union[str, Path],
        preprocess: bool = True,
        skip_frames: int = 0,
    ) -> Generator[VideoFrame, None, None]:
        """
        Stream frames from video one at a time (memory efficient).
        
        Args:
            source: Path to video file
            preprocess: Apply preprocessing
            skip_frames: Skip every N frames (0 = no skip)
            
        Yields:
            VideoFrame objects
        """
        path = Path(source)
        metadata = self.get_metadata(path)
        fps = metadata.fps or 30.0
        
        reader = iio.imiter(path, plugin="pyav")
        
        frame_idx = 0
        for frame_array in reader:
            if skip_frames == 0 or frame_idx % (skip_frames + 1) == 0:
                image = Image.fromarray(frame_array)
                timestamp = frame_idx / fps
                
                tensor = None
                if preprocess:
                    tensor = self._transform(image)
                
                yield VideoFrame(
                    image=image,
                    tensor=tensor,
                    frame_number=frame_idx,
                    timestamp=timestamp,
                )
            
            frame_idx += 1
            
            if frame_idx >= self.max_frames:
                break


def load_video(
    source: Union[str, Path],
    num_frames: int = 16,
    strategy: SamplingStrategy = "uniform",
) -> VideoData:
    """
    Convenience function to load a video.
    
    Args:
        source: Path to video file
        num_frames: Number of frames to extract
        strategy: Sampling strategy
        
    Returns:
        VideoData object
    """
    loader = VideoLoader()
    return loader.load(source, num_frames=num_frames, strategy=strategy)
