#!/usr/bin/env python3
"""
Download and cache model weights for offline use.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model google/vit-large-patch16-224
"""

import argparse
from pathlib import Path

from transformers import AutoModelForImageClassification, AutoImageProcessor
from loguru import logger


DEFAULT_MODELS = [
    "google/vit-base-patch16-224",
]

OPTIONAL_MODELS = [
    "google/vit-large-patch16-224",
    "facebook/convnext-base-224",
    "facebook/convnext-tiny-224",
    "microsoft/swin-base-patch4-window7-224",
]


def download_model(model_name: str, cache_dir: Path | None = None) -> None:
    """Download a model and its processor."""
    logger.info(f"Downloading: {model_name}")
    
    try:
        # Download model
        _ = AutoModelForImageClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        
        # Download processor
        _ = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        
        logger.success(f"✓ Downloaded: {model_name}")
        
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download VerifAI model weights"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to download (default: download all default models)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models (including optional)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Custom cache directory for models",
    )
    
    args = parser.parse_args()
    
    if args.model:
        models = [args.model]
    elif args.all:
        models = DEFAULT_MODELS + OPTIONAL_MODELS
    else:
        models = DEFAULT_MODELS
    
    logger.info(f"Downloading {len(models)} model(s)...")
    
    for model in models:
        download_model(model, args.cache_dir)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
