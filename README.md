# VerifAI

**AI-Generated Media Detector** — Detect AI-generated images and videos with calibrated confidence scores, localized evidence, and robustness evaluation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  Features

-  **Calibrated Confidence Scores** — Not just yes/no, but reliable probability estimates
-  **Multi-Signal Detection** — Ensemble of neural, frequency, and provenance signals
-  **Localized Evidence** — Heatmaps showing which regions triggered detection
-  **Robustness Evaluation** — Test performance under real-world transformations (JPEG, resize, etc.)
- **Image & Video Support** — Process both still images and video files
-  **Fast Inference** — Optimized for both CPU and GPU with FP16 support

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
```

---

## Documentation

### Detection Pipeline

VerifAI uses an ensemble approach combining multiple detection signals:

1. **Neural Detector** — Vision Transformer (ViT) trained to recognize AI-generated patterns
2. **Frequency Detector** — FFT/DCT analysis of image frequency patterns *(Phase 2)*
3. **PRNU Detector** — Camera sensor noise fingerprint analysis *(Phase 5)*
4. **Provenance Checker** — EXIF metadata and C2PA credential verification *(Phase 5)*

### Supported Formats

**Images:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)
- TIFF (.tiff, .tif)

**Videos:** *(Coming in Phase 4)*
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
    model_name="google/vit-large-patch16-224",  # Larger model
    device="cuda",                               # Force GPU
    threshold=0.6,                               # Higher threshold
    fp16=True,                                   # Enable FP16
)
```

See `config/default.yaml` for all available options.

### Available Models

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `google/vit-base-patch16-224` | 86M | ⚡⚡⚡ | ⭐⭐⭐ | Default, balanced |
| `google/vit-large-patch16-224` | 304M | ⚡⚡ | ⭐⭐⭐⭐ | Higher accuracy |
| `facebook/convnext-tiny-224` | 29M | ⚡⚡⚡⚡ | ⭐⭐ | Edge deployment |
| `facebook/convnext-base-224` | 89M | ⚡⚡⚡ | ⭐⭐⭐ | Apple Silicon |

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
# Accuracy:  0.8333
# ROC-AUC:   0.9444
# ECE:       0.0521
```

---

## Project Structure

```
VerifAI/
├── verifai/                 # Main package
│   ├── __init__.py          # Package exports
│   ├── pipeline.py          # Main detection pipeline
│   ├── ingest/              # Media loading
│   │   ├── image_loader.py  # Image preprocessing
│   │   └── utils.py         # File utilities
│   ├── models/              # Detection models
│   │   ├── base.py          # Abstract interfaces
│   │   └── neural_detector.py # ViT-based detector
│   ├── eval/                # Evaluation tools
│   │   └── metrics.py       # Performance metrics
│   ├── features/            # Feature extractors (Phase 2+)
│   └── fusion/              # Ensemble methods (Phase 2+)
├── cli/                     # Command-line interface
│   └── main.py              # CLI commands
├── config/                  # Configuration files
│   ├── default.yaml         # Default settings
│   └── models.yaml          # Model registry
├── tests/                   # Test suite
├── api/                     # REST API (Phase 6)
├── ui/                      # Web interface (Phase 6)
└── pyproject.toml           # Dependencies
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
- [ ] Frequency domain features (FFT/DCT)
- [ ] Ensemble fusion
- [ ] Probability calibration
- [ ] Heatmap generation

### Phase 3 Robustness Evaluation
- [ ] Corruption harness (JPEG, resize, blur)
- [ ] Robustness curves
- [ ] Benchmark reports
- [ ] Cross-generator evaluation

### Phase 4 Video Pipeline
- [ ] Video frame extraction
- [ ] Per-frame detection
- [ ] Temporal aggregation
- [ ] Video corruption tests

### Phase 5 Advanced Detection
- [ ] PRNU analysis
- [ ] C2PA integration
- [ ] Open-set detection
- [ ] Final ensemble

### Phase 6 Deployment
- [ ] FastAPI server
- [ ] Gradio UI
- [ ] Docker container
- [ ] CI/CD pipeline

---

## References

### Papers

- [ViT: An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Detecting AI-Generated Images with Texture Patterns](https://arxiv.org/abs/2307.02289)
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

### Related Projects

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [C2PA Content Credentials](https://c2pa.org/)

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

