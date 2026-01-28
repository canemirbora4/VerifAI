# VerifAI

**AI-Generated Media Detector** — Detect AI-generated images and videos with calibrated confidence scores, localized evidence, and robustness evaluation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **Fusion Detection (97% accuracy)** — CLIP ViT-L/14 + Frequency ensemble for best results
- **Calibrated Confidence Scores** — Not just yes/no, but reliable probability estimates
- **Multi-Signal Detection** — Ensemble of neural, frequency, and metadata signals
- **Localized Evidence** — Heatmaps showing which regions triggered detection
- **Robustness Evaluation** — Test performance under real-world transformations (JPEG, resize, etc.)
- **Image & Video Support** — Process both still images and video files
- **Fast Inference** — Optimized for both CPU and GPU/MPS with FP16 support

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/canemirbora4/VerifAI
cd VerifAI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

### Basic Usage

**Command Line:**

```bash
# Detect a single image
verifai detect image.jpg

# Detect with JSON output
verifai detect image.jpg --json

# Detect all images in a directory
verifai detect photos/ --recursive

# Save results to file
verifai detect image.jpg --output results.json

# Show model info
verifai info
```

**Python API:**

```python
from verifai import VerifAI

# Initialize detector
detector = VerifAI()

# Detect single image
result = detector.detect("image.jpg")

print(f"Label: {result.label}")           # 'real' or 'ai_generated'
print(f"Confidence: {result.confidence:.1%}")  # e.g., '87.3%'
print(f"Is AI: {result.is_ai_generated}")      # True or False

# Get detailed JSON output
print(result.to_json())

# Detect video (frame-by-frame analysis with temporal consistency)
result = detector.detect("video.mp4")

print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Temporal consistency: {result.temporal_consistency:.1%}")
print(f"Suspicious frames: {result.suspicious_frames}")
print(f"Frames analyzed: {result.num_frames_analyzed}")
```

**With Heatmaps & Evidence:**

```python
from verifai import VerifAI

# Heatmaps are enabled by default
detector = VerifAI()

# Get detection with evidence
result = detector.detect("image.jpg", return_evidence=True)

# Access heatmap (numpy array showing suspicious regions)
if result.heatmap is not None:
    print(f"Heatmap shape: {result.heatmap.shape}")
    # Red = suspicious, Blue = normal
    
# Access detailed evidence
print(result.evidence)
# {'neural': {'clip_score': 0.99, 'frequency_score': 0.38, ...}}

# Disable heatmaps for faster inference
fast_detector = VerifAI(generate_heatmaps=False)
```

---

## Documentation

### Detection Pipeline

VerifAI uses an ensemble approach combining multiple detection signals:

**Default (Enabled):**
1. **Fusion Detector** — CLIP ViT-L/14 + Frequency ensemble (97.0% accuracy)
2. **Metadata Analyzer** — EXIF parsing for AI tool markers

**Optional (Disabled by default):**
3. **PRNU Detector** — Camera sensor noise fingerprint (requires reference images)
4. **Provenance Checker** — C2PA credential verification (most images lack C2PA)
5. **Temporal Analyzer** — Video consistency and flicker detection *(videos only)*

**Benchmark Results (800 images from 2 datasets):**

| Dataset | Real Acc | Fake Acc | Overall |
|---------|----------|----------|---------|
| Defactify (SD3, SDXL, DALL-E 3, MidJourney) | 97.0% | 100.0% | 98.5% |
| COCO_AI (SD, FLUX, Ideogram) | 92.5% | 98.5% | 95.5% |
| **Average** | **94.8%** | **99.2%** | **97.0%** |

*Inference Speed: ~120ms/image on Apple M-series*

### Video Detection

Video detection analyzes multiple frames and temporal consistency:

```python
from verifai import VerifAI

detector = VerifAI()
result = detector.detect("suspicious_video.mp4")

# Video-specific outputs
print(f"Video duration: {result.video_duration:.1f}s")
print(f"Frames analyzed: {result.num_frames_analyzed}")
print(f"Temporal consistency: {result.temporal_consistency:.1%}")

# Per-frame scores
for fs in result.frame_scores:
    if fs.is_suspicious:
        print(f"  Frame {fs.frame_number} @ {fs.timestamp:.2f}s: {fs.score:.1%}")

# Suspicious frame indices
if result.suspicious_frames:
    print(f"Suspicious frames: {result.suspicious_frames}")
```

**Temporal Analysis Features:**
- **Flicker Detection** — Detects unnatural brightness variations between frames
- **Consistency Score** — Measures overall temporal coherence
- **Motion Smoothness** — Analyzes physically plausible motion
- **Noise Pattern Analysis** — Checks for consistent sensor noise across frames

### PRNU Analysis (Camera Fingerprinting)

PRNU (Photo-Response Non-Uniformity) is a unique fingerprint caused by manufacturing imperfections in camera sensors. Real photos carry this fingerprint; AI images don't.

```python
from verifai.features import PRNUExtractor, extract_prnu

# Quick analysis
features = extract_prnu("photo.jpg")

print(f"Has PRNU signature: {features.has_prnu_signature}")
print(f"PRNU score: {features.prnu_score:.1%}")  # Higher = more likely real
print(f"Is likely real: {features.is_likely_real}")

# Build reference fingerprint from multiple images (same camera)
extractor = PRNUExtractor()
reference = extractor.build_reference([
    "camera_photo1.jpg",
    "camera_photo2.jpg",
    "camera_photo3.jpg",
])

# Compare new image to reference
features = extractor.extract("test_image.jpg", reference=reference)
print(f"Correlation with camera: {features.correlation:.4f}")
```

### Provenance & C2PA (Content Credentials)

C2PA is an open standard for digital content provenance. VerifAI checks for Content Credentials and analyzes provenance data.

```python
from verifai.features import ProvenanceAnalyzer, analyze_provenance

# Quick analysis
features = analyze_provenance("image.jpg")

print(f"Has C2PA manifest: {features.has_c2pa}")
print(f"Valid signature: {features.has_valid_signature}")
print(f"Is verified: {features.is_verified}")
print(f"Creation tool: {features.creation_tool}")

# Check trust and risk indicators
print(f"Trust indicators: {features.trust_indicators}")
print(f"Risk indicators: {features.risk_indicators}")
print(f"Provenance score: {features.provenance_score:.1%}")
```

**What the analyzer checks:**
- **C2PA manifests** — Cryptographically signed content credentials
- **XMP metadata** — Creation tool, edit history
- **AI generation markers** — Known AI tool signatures
- **Camera metadata** — Indicators of real camera origin

### Supported Formats

**Images:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)
- TIFF (.tiff, .tif)

**Videos:**
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)

### Configuration

VerifAI can be configured via YAML files or programmatically:

```python
from verifai import VerifAI

# Custom configuration
detector = VerifAI(
    device="cuda",                    # Force GPU (auto-detects by default)
    threshold=0.6,                    # Higher threshold (default: 0.5)
    generate_heatmaps=False,          # Disable for faster inference
    use_metadata=False,               # Disable metadata analysis
)

# CLIP-only mode (no frequency fusion)
detector = VerifAI(use_fusion=False, use_clip=True)
```

See `config/default.yaml` for all available options.

### Available Detectors

#### Fusion Detector (Default - Recommended)

The Fusion Detector combines CLIP semantic features with frequency-domain analysis for the best accuracy.

| Detector | Accuracy | Speed | Description |
|----------|----------|-------|-------------|
| **FusionDetector** | **97.0%** | ⚡⚡ | CLIP + Frequency ensemble (Default) |
| CLIPDetector | 95.0% | ⚡⚡⚡ | CLIP ViT-L/14 with trained head |
| FrequencyDetector | 67.0% | ⚡⚡⚡⚡ | FFT/DCT-based classifier |

**Usage:**
```python
from verifai import VerifAI

# Default: Fusion Detector (best accuracy, includes heatmaps)
detector = VerifAI()
result = detector.detect("image.jpg")

# Fast mode (no heatmaps)
detector = VerifAI(generate_heatmaps=False)

# Direct detector access
from verifai.models import FusionDetector
detector = FusionDetector()
detector.load()
result = detector.detect(image)
```

#### Training Details

The Fusion Detector was trained on:
- **Defactify Dataset** — Modern AI generators (SD3, SDXL, DALL-E 3, MidJourney v6)
- **VCT² COCO_AI** — COCO images + AI-generated variants (SD3, SD3.5, SDXL, DALL-E 3, Midjourney)

Weights: CLIP=0.80, Frequency=0.20

---

## Evaluation

### Running Evaluation

VerifAI includes tools for evaluating detection performance:

```bash
# Evaluate on a labeled dataset
verifai evaluate dataset/ --output results/

# Expected directory structure:
# dataset/
# ├── real/
# │   ├── image1.jpg
# │   └── ...
# └── ai_generated/
#     ├── image1.jpg
#     └── ...
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **Precision** | Positive predictive value |
| **Recall** | True positive rate |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve |
| **PR-AUC** | Area under Precision-Recall curve |
| **ECE** | Expected Calibration Error |

### Programmatic Evaluation

```python
from verifai.eval import compute_metrics
import numpy as np

# Your predictions
y_true = np.array([0, 0, 1, 1, 0, 1])  # 0=real, 1=ai
y_prob = np.array([0.2, 0.3, 0.8, 0.9, 0.4, 0.7])  # probabilities

# Compute metrics
metrics = compute_metrics(y_true, y_prob)

print(metrics.summary())
# Accuracy:  0.9700
# ROC-AUC:   0.9900
# ECE:       0.0350
```

---

## Project Structure

```
VerifAI/
├── verifai/                 # Main package
│   ├── __init__.py          # Package exports
│   ├── pipeline.py          # Main detection pipeline (VerifAI class)
│   ├── ingest/              # Media loading
│   │   ├── image_loader.py  # Image preprocessing
│   │   ├── video_loader.py  # Video frame extraction
│   │   └── utils.py         # File utilities
│   ├── models/              # Detection models
│   │   ├── base.py          # Abstract interfaces
│   │   ├── fusion_detector.py  # CLIP + Frequency ensemble (DEFAULT)
│   │   ├── clip_detector.py    # CLIP ViT-L/14 detector
│   │   └── neural_detector.py  # Legacy ViT-based detector
│   ├── features/            # Feature extractors
│   │   ├── frequency.py     # FFT/DCT analysis
│   │   ├── prnu.py          # Camera fingerprint
│   │   ├── provenance.py    # C2PA/metadata
│   │   └── temporal.py      # Video temporal analysis
│   ├── fusion/              # Ensemble & explainability
│   │   ├── ensemble.py      # Multi-signal fusion
│   │   ├── calibration.py   # Probability calibration
│   │   └── explainer.py     # Heatmap generation
│   └── eval/                # Evaluation tools
│       ├── metrics.py       # Performance metrics
│       └── benchmark.py     # Dataset benchmarking
├── models/                  # Trained model weights
│   ├── modern_ai_detector.pt    # CLIP classification head
│   └── frequency_classifier.joblib  # Frequency classifier
├── config/                  # Configuration files
│   └── models.yaml          # Model registry
├── cli/                     # Command-line interface
└── tests/                   # Test suite
```

---

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run tests excluding slow ones (no model downloads)
pytest tests/ -m "not slow"

# Run with coverage
pytest tests/ --cov=verifai

# Format code
black verifai/ cli/ tests/

# Lint
ruff check verifai/ cli/ tests/

# Type check
mypy verifai/
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_ingest.py

# Specific test
pytest tests/test_ingest.py::TestImageLoader::test_load_from_pil

# With verbose output
pytest -v

# Skip slow tests (model downloads)
pytest -m "not slow"
```

---

##  Roadmap

### Phase 1 Foundation
- [x] Project scaffolding
- [x] Image ingestion pipeline
- [x] Neural detector (ViT)
- [x] Basic CLI
- [x] Evaluation metrics

### Phase 2 Multi-Signal Detection
- [x] Frequency domain features (FFT/DCT)
- [x] Ensemble fusion
- [x] Probability calibration
- [x] Heatmap generation

### Phase 3 Robustness Evaluation
- [x] Corruption harness (JPEG, resize, blur)
- [x] Robustness curves
- [x] Benchmark reports
- [x] Cross-generator evaluation

### Phase 4 Video Pipeline
- [x] Video frame extraction
- [x] Per-frame detection
- [x] Temporal aggregation
- [x] Video corruption tests

### Phase 5 Advanced Detection
- [x] PRNU analysis
- [x] C2PA integration
- [x] Provenance analyzer
- [x] Final ensemble

### Phase 6 Deployment
- [ ] FastAPI server
- [ ] Gradio UI
- [ ] Docker container
- [ ] CI/CD pipeline

---

## ⚠️ Disclaimer

**Detection is not perfect.** AI-generated media detection is an ongoing arms race. VerifAI provides probabilistic assessments, not definitive verdicts. Always:

- Consider the confidence score, not just the label
- Look at the evidence (heatmaps, metadata)
- Use multiple verification methods when stakes are high
- Stay updated as both generators and detectors evolve

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

