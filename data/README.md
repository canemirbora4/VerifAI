# Data for VerifAI

This directory contains datasets for training and evaluating VerifAI.

## Quick Start

### Option 1: Create a Sample Dataset (For Testing Only)

```bash
python scripts/prepare_data.py download --dataset sample --output data/
```

This creates random noise images - only useful for testing the pipeline works!

---

## Recommended Datasets

### 🏆 GenImage (Best for Training)

**The most comprehensive dataset for AI detection research.**

- **Size:** 1.3M+ images
- **Real images:** ImageNet subset
- **AI generators:** Midjourney, Stable Diffusion 1.4/1.5, ADM, GLIDE, Wukong, VQDM, BigGAN

**Download:**
1. Visit: https://github.com/GenImage-Dataset/GenImage
2. Follow their download instructions (requires Google Drive)
3. Organize into this structure:

```
data/genimage/
├── train/
│   ├── real/
│   │   └── *.jpg
│   └── ai_generated/
│       └── *.jpg
├── val/
│   ├── real/
│   └── ai_generated/
└── test/
    ├── real/
    └── ai_generated/
```

---

###  CIFAKE (Quick Experiments)

**Small dataset, perfect for initial testing.**

- **Size:** 120K images (60K real, 60K fake)
- **Resolution:** 32×32 pixels
- **Generator:** Stable Diffusion

**Download from Kaggle:**

```bash
# Install Kaggle CLI
pip install kaggle

# Set up API key (https://www.kaggle.com/docs/api)
# Download and extract
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
unzip cifake-real-and-ai-generated-synthetic-images.zip -d data/cifake/
```

---

###  DiffusionDB (AI Images Only)

**14M Stable Diffusion images with prompts.**

- **Size:** 1.7 TB (full) or smaller subsets available
- **Includes:** Prompts, parameters, seeds

**Download subset via HuggingFace:**

```bash
python scripts/prepare_data.py download --dataset diffusiondb --output data/ --max-images 10000
```

---

## Organizing Your Own Data

If you have your own images:

```bash
python scripts/prepare_data.py organize \
    --real-dir /path/to/real/images \
    --ai-dir /path/to/ai/images \
    --output data/custom \
    --train-ratio 0.8
```

This will:
1. Split into train/val/test (80/10/10)
2. Copy images to organized structure
3. Generate `dataset_stats.json`

---

## Expected Directory Structure

VerifAI expects data in this format:

```
data/
├── <dataset_name>/
│   ├── train/
│   │   ├── real/
│   │   │   ├── image_001.jpg
│   │   │   └── ...
│   │   └── ai_generated/
│   │       ├── image_001.jpg
│   │       └── ...
│   ├── val/
│   │   ├── real/
│   │   └── ai_generated/
│   └── test/
│       ├── real/
│       └── ai_generated/
└── dataset_stats.json  (optional)
```

---

## Generating Fresh AI Images

For testing against the latest generators:

```python
# Stable Diffusion XL
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompts = [
    "a photo of a cat sitting on a windowsill",
    "a landscape photograph of mountains at sunset",
    # ... more prompts
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f"data/sdxl/ai_generated/sdxl_{i:04d}.png")
```

---

## Data Sources Summary

| Dataset | Size | Generators | Use Case |
|---------|------|------------|----------|
| **GenImage** | 1.3M | 8 generators | Full training |
| **CIFAKE** | 120K | SD only | Quick experiments |
| **DiffusionDB** | 14M | SD only | AI images source |
| **COCO** | 330K | N/A | Real images |
| **ImageNet** | 14M | N/A | Real images |

---

## Cross-Generator Evaluation

For robust evaluation, train on some generators and test on others:

```
Training: Stable Diffusion 1.5, GLIDE, BigGAN
Testing:  Midjourney, DALL-E 3, Stable Diffusion XL
```

This tests whether your model generalizes to unseen generators!
