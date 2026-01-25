#!/usr/bin/env python3
"""
Data Preparation Script for VerifAI
=====================================

This script helps download and prepare datasets for training and evaluation.

Supported datasets:
1. CIFAKE - Small, quick to download (Kaggle)
2. GenImage - Large, comprehensive (requires manual download)
3. Custom - Organize your own images

Usage:
    python scripts/prepare_data.py --dataset cifake --output data/cifake
    python scripts/prepare_data.py --dataset custom --real-dir my_real/ --ai-dir my_ai/
    python scripts/prepare_data.py --info
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from typing import Optional
import json

from loguru import logger

# Try to import optional dependencies
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


# ============================================================================
# Dataset Information
# ============================================================================

DATASETS_INFO = {
    "cifake": {
        "name": "CIFAKE",
        "description": "60K real + 60K AI-generated images (32x32, based on CIFAR-10)",
        "size": "~500 MB",
        "source": "Kaggle: birdy654/cifake-real-and-ai-generated-synthetic-images",
        "generators": ["Stable Diffusion"],
        "real_source": "CIFAR-10",
        "download": "kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images",
        "hf_id": None,  # Not on HuggingFace
    },
    "genimage": {
        "name": "GenImage",
        "description": "1.3M+ images from 8 generators + real images",
        "size": "~100 GB",
        "source": "GitHub: GenImage-Dataset/GenImage",
        "generators": ["Midjourney", "SD 1.4", "SD 1.5", "ADM", "GLIDE", "Wukong", "VQDM", "BigGAN"],
        "real_source": "ImageNet",
        "download": "See: https://github.com/GenImage-Dataset/GenImage",
        "hf_id": None,
    },
    "diffusiondb": {
        "name": "DiffusionDB",
        "description": "14M Stable Diffusion images with prompts",
        "size": "~1.7 TB (full) or ~1.5 GB (2M subset)",
        "source": "HuggingFace: poloclub/diffusiondb",
        "generators": ["Stable Diffusion"],
        "real_source": None,
        "download": "Automatic via HuggingFace datasets",
        "hf_id": "poloclub/diffusiondb",
    },
    "aiartbench": {
        "name": "AIArtBench",
        "description": "180K AI art images across different styles",
        "size": "~50 GB",
        "source": "HuggingFace: competitions/aiartbench",
        "generators": ["Multiple"],
        "real_source": None,
        "download": "Automatic via HuggingFace datasets",
        "hf_id": "competitions/aiartbench",
    },
}


# ============================================================================
# Download Functions
# ============================================================================

def download_diffusiondb_subset(
    output_dir: Path,
    subset: str = "2m_random_1k",
    max_images: int = 10000,
) -> None:
    """
    Download a subset of DiffusionDB from HuggingFace.
    
    Args:
        output_dir: Where to save images
        subset: Which subset to download
        max_images: Maximum images to download
    """
    if not HF_AVAILABLE:
        logger.error("HuggingFace datasets not installed. Run: pip install datasets")
        return
    
    logger.info(f"Downloading DiffusionDB subset: {subset}")
    
    # Create output directory
    ai_dir = output_dir / "ai_generated" / "stable_diffusion"
    ai_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(
        "poloclub/diffusiondb",
        subset,
        split="train",
        trust_remote_code=True,
    )
    
    # Save images
    count = 0
    for i, item in enumerate(tqdm(dataset, desc="Saving images")):
        if count >= max_images:
            break
        
        image = item["image"]
        prompt = item.get("prompt", "")
        
        # Save image
        image_path = ai_dir / f"diffusiondb_{i:06d}.png"
        image.save(image_path)
        
        count += 1
    
    logger.success(f"Downloaded {count} images to {ai_dir}")


def download_hf_dataset(
    dataset_id: str,
    output_dir: Path,
    max_images: int = 10000,
    image_column: str = "image",
) -> None:
    """
    Download a dataset from HuggingFace Hub.
    """
    if not HF_AVAILABLE:
        logger.error("HuggingFace datasets not installed. Run: pip install datasets")
        return
    
    logger.info(f"Downloading from HuggingFace: {dataset_id}")
    
    try:
        dataset = load_dataset(dataset_id, split="train", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for i, item in enumerate(tqdm(dataset, desc="Saving images")):
        if count >= max_images:
            break
        
        if image_column in item:
            image = item[image_column]
            image_path = output_dir / f"image_{i:06d}.png"
            image.save(image_path)
            count += 1
    
    logger.success(f"Downloaded {count} images to {output_dir}")


# ============================================================================
# Data Organization
# ============================================================================

def organize_custom_data(
    real_dir: Path,
    ai_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Organize custom data into train/val/test splits.
    
    Args:
        real_dir: Directory containing real images
        ai_dir: Directory containing AI-generated images
        output_dir: Where to create the organized dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        
    Returns:
        Statistics about the organized dataset
    """
    random.seed(seed)
    
    # Collect all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    def collect_images(directory: Path) -> list[Path]:
        images = []
        for ext in image_extensions:
            images.extend(directory.glob(f"**/*{ext}"))
            images.extend(directory.glob(f"**/*{ext.upper()}"))
        return sorted(set(images))
    
    real_images = collect_images(real_dir)
    ai_images = collect_images(ai_dir)
    
    logger.info(f"Found {len(real_images)} real images")
    logger.info(f"Found {len(ai_images)} AI-generated images")
    
    if not real_images or not ai_images:
        logger.error("Need both real and AI-generated images!")
        return {}
    
    # Shuffle
    random.shuffle(real_images)
    random.shuffle(ai_images)
    
    # Split function
    def split_list(items: list, train_r: float, val_r: float) -> tuple:
        n = len(items)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))
        return items[:train_end], items[train_end:val_end], items[val_end:]
    
    real_train, real_val, real_test = split_list(real_images, train_ratio, val_ratio)
    ai_train, ai_val, ai_test = split_list(ai_images, train_ratio, val_ratio)
    
    # Create directory structure
    splits = {
        "train": (real_train, ai_train),
        "val": (real_val, ai_val),
        "test": (real_test, ai_test),
    }
    
    stats = {"train": {}, "val": {}, "test": {}}
    
    for split_name, (real_split, ai_split) in splits.items():
        # Create directories
        real_out = output_dir / split_name / "real"
        ai_out = output_dir / split_name / "ai_generated"
        real_out.mkdir(parents=True, exist_ok=True)
        ai_out.mkdir(parents=True, exist_ok=True)
        
        # Copy/link images
        for i, src in enumerate(tqdm(real_split, desc=f"{split_name}/real")):
            dst = real_out / f"real_{i:06d}{src.suffix}"
            shutil.copy2(src, dst)
        
        for i, src in enumerate(tqdm(ai_split, desc=f"{split_name}/ai_generated")):
            dst = ai_out / f"ai_{i:06d}{src.suffix}"
            shutil.copy2(src, dst)
        
        stats[split_name] = {
            "real": len(real_split),
            "ai_generated": len(ai_split),
            "total": len(real_split) + len(ai_split),
        }
    
    # Save stats
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.success(f"Dataset organized at {output_dir}")
    logger.info(f"Stats saved to {stats_path}")
    
    return stats


def create_sample_dataset(output_dir: Path, num_samples: int = 100) -> None:
    """
    Create a tiny sample dataset for testing (random noise images).
    
    This is just for testing the pipeline - not for actual training!
    """
    import numpy as np
    from PIL import Image
    
    logger.info(f"Creating sample dataset with {num_samples} images per class")
    
    for split in ["train", "val", "test"]:
        for label in ["real", "ai_generated"]:
            dir_path = output_dir / split / label
            dir_path.mkdir(parents=True, exist_ok=True)
            
            n = num_samples if split == "train" else num_samples // 5
            
            for i in range(n):
                # Create random image (just for testing!)
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(dir_path / f"{label}_{i:04d}.jpg")
    
    logger.success(f"Sample dataset created at {output_dir}")
    logger.warning("This is random noise - only use for testing the pipeline!")


# ============================================================================
# Main CLI
# ============================================================================

def print_dataset_info():
    """Print information about available datasets."""
    print("\n" + "=" * 70)
    print("Available Datasets for AI-Generated Image Detection")
    print("=" * 70)
    
    for key, info in DATASETS_INFO.items():
        print(f"\n📦 {info['name']} ({key})")
        print(f"   {info['description']}")
        print(f"   Size: {info['size']}")
        print(f"   Generators: {', '.join(info['generators'])}")
        print(f"   Real source: {info['real_source'] or 'N/A'}")
        print(f"   Download: {info['download']}")
    
    print("\n" + "=" * 70)
    print("Recommended Setup:")
    print("=" * 70)
    print("""
1. For quick experiments:
   - Download CIFAKE from Kaggle (~500 MB)
   - Small images (32x32) but good for initial testing

2. For serious training:
   - Download GenImage dataset (~100 GB)
   - Multiple generators for cross-generator evaluation
   - Higher resolution images

3. For production:
   - Combine multiple datasets
   - Generate fresh AI images from latest models
   - Include diverse real image sources
""")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for VerifAI training and evaluation"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a dataset")
    download_parser.add_argument(
        "--dataset",
        choices=["diffusiondb", "sample"],
        required=True,
        help="Dataset to download",
    )
    download_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output directory",
    )
    download_parser.add_argument(
        "--max-images",
        type=int,
        default=10000,
        help="Maximum images to download",
    )
    
    # Organize command
    organize_parser = subparsers.add_parser("organize", help="Organize custom data")
    organize_parser.add_argument(
        "--real-dir",
        type=Path,
        required=True,
        help="Directory with real images",
    )
    organize_parser.add_argument(
        "--ai-dir",
        type=Path,
        required=True,
        help="Directory with AI-generated images",
    )
    organize_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/organized"),
        help="Output directory",
    )
    organize_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    organize_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    if args.command == "info" or args.command is None:
        print_dataset_info()
    
    elif args.command == "download":
        if args.dataset == "diffusiondb":
            download_diffusiondb_subset(
                args.output,
                max_images=args.max_images,
            )
        elif args.dataset == "sample":
            create_sample_dataset(args.output / "sample", num_samples=100)
    
    elif args.command == "organize":
        organize_custom_data(
            args.real_dir,
            args.ai_dir,
            args.output,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
