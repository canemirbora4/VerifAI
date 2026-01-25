"""
Benchmark Runner
=================

Comprehensive benchmarking for AI-generated image detection.

Features:
1. Performance metrics on clean images
2. Robustness curves under corruptions
3. Cross-generator evaluation
4. Auto-generated reports (Markdown + JSON)

Usage:
    benchmark = Benchmark(detector)
    results = benchmark.run(dataset_path)
    benchmark.generate_report(results, output_dir)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Callable
import json
import time

import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger

from verifai.eval.metrics import compute_metrics, MetricsResult
from verifai.eval.corruptions import (
    ImageCorruptor,
    CorruptionType,
    CorruptionConfig,
)


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark runs.
    
    Attributes:
        name: Benchmark name
        corruption_types: Types of corruptions to test
        severity_levels: Severity levels to test for each corruption
        num_samples: Max samples per class (None = all)
        include_clean: Include clean (no corruption) evaluation
        save_corrupted: Save corrupted images for inspection
    """
    name: str = "verifai_benchmark"
    corruption_types: list = field(default_factory=lambda: [
        CorruptionType.JPEG_COMPRESSION,
        CorruptionType.RESIZE,
        CorruptionType.GAUSSIAN_BLUR,
        CorruptionType.GAUSSIAN_NOISE,
    ])
    severity_levels: list = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    num_samples: Optional[int] = None
    include_clean: bool = True
    save_corrupted: bool = False


@dataclass 
class RobustnessResult:
    """
    Results for a single corruption type across severities.
    
    Attributes:
        corruption_type: Type of corruption
        severities: List of severity levels tested
        metrics: MetricsResult for each severity
        scores_by_severity: Raw scores at each severity
    """
    corruption_type: str
    severities: list
    metrics: list  # List of MetricsResult
    auc_by_severity: list  # ROC-AUC at each severity
    
    def to_dict(self) -> dict:
        return {
            "corruption_type": self.corruption_type,
            "severities": self.severities,
            "auc_by_severity": [round(a, 4) for a in self.auc_by_severity],
            "auc_degradation": round(self.auc_degradation, 4),
            "metrics_by_severity": [m.to_dict() for m in self.metrics],
        }
    
    @property
    def auc_degradation(self) -> float:
        """AUC drop from clean to max severity."""
        if len(self.auc_by_severity) >= 2:
            return self.auc_by_severity[0] - self.auc_by_severity[-1]
        return 0.0


@dataclass
class BenchmarkResult:
    """
    Complete benchmark results.
    
    Attributes:
        config: Benchmark configuration used
        clean_metrics: Metrics on clean images
        robustness_results: Results per corruption type
        timing: Timing information
        metadata: Additional metadata
    """
    config: BenchmarkConfig
    clean_metrics: Optional[MetricsResult] = None
    robustness_results: dict = field(default_factory=dict)  # corruption_type -> RobustnessResult
    timing: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        result = {
            "benchmark_name": self.config.name,
            "timestamp": datetime.now().isoformat(),
            "metadata": self.metadata,
            "timing": self.timing,
        }
        
        if self.clean_metrics:
            result["clean_metrics"] = self.clean_metrics.to_dict()
        
        result["robustness"] = {
            k: v.to_dict() for k, v in self.robustness_results.items()
        }
        
        # Summary statistics
        result["summary"] = self._compute_summary()
        
        return result
    
    def _compute_summary(self) -> dict:
        """Compute summary statistics."""
        summary = {}
        
        if self.clean_metrics:
            summary["clean_roc_auc"] = self.clean_metrics.roc_auc
            summary["clean_accuracy"] = self.clean_metrics.accuracy
        
        # Average degradation across corruptions
        degradations = [r.auc_degradation for r in self.robustness_results.values()]
        if degradations:
            summary["mean_auc_degradation"] = float(np.mean(degradations))
            summary["max_auc_degradation"] = float(np.max(degradations))
            summary["most_harmful_corruption"] = max(
                self.robustness_results.keys(),
                key=lambda k: self.robustness_results[k].auc_degradation,
            )
        
        return summary
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class Benchmark:
    """
    Runs comprehensive benchmarks for detection robustness.
    
    Usage:
        from verifai import VerifAI
        
        detector = VerifAI()
        benchmark = Benchmark(detector)
        
        results = benchmark.run("data/test/")
        benchmark.generate_report(results, "reports/")
    """
    
    def __init__(
        self,
        detector,  # VerifAI instance
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            detector: VerifAI detector instance
            config: Benchmark configuration
        """
        self.detector = detector
        self.config = config or BenchmarkConfig()
        self.corruptor = ImageCorruptor(seed=42)
        
        logger.info(f"Benchmark initialized: {self.config.name}")
    
    def run(
        self,
        dataset_path: Union[str, Path],
        progress: bool = True,
    ) -> BenchmarkResult:
        """
        Run full benchmark suite.
        
        Args:
            dataset_path: Path to dataset with real/ and ai_generated/ subdirs
            progress: Show progress bars
            
        Returns:
            BenchmarkResult with all metrics
        """
        dataset_path = Path(dataset_path)
        start_time = time.time()
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        images, labels, paths = self._load_dataset(dataset_path)
        
        if len(images) == 0:
            raise ValueError(f"No images found in {dataset_path}")
        
        logger.info(f"Loaded {len(images)} images ({sum(labels)} AI, {len(labels) - sum(labels)} real)")
        
        result = BenchmarkResult(
            config=self.config,
            metadata={
                "dataset_path": str(dataset_path),
                "num_images": len(images),
                "num_real": len(labels) - sum(labels),
                "num_ai": sum(labels),
            }
        )
        
        # Clean evaluation
        if self.config.include_clean:
            logger.info("Evaluating on clean images...")
            clean_metrics = self._evaluate_clean(images, labels, progress)
            result.clean_metrics = clean_metrics
            logger.info(f"Clean ROC-AUC: {clean_metrics.roc_auc:.4f}")
        
        # Robustness evaluation
        for corruption_type in self.config.corruption_types:
            logger.info(f"Evaluating robustness to {corruption_type.value}...")
            robustness = self._evaluate_robustness(
                images, labels, corruption_type, progress
            )
            result.robustness_results[corruption_type.value] = robustness
            logger.info(
                f"  AUC degradation: {robustness.auc_degradation:.4f} "
                f"({robustness.auc_by_severity[0]:.3f} → {robustness.auc_by_severity[-1]:.3f})"
            )
        
        # Timing
        result.timing["total_seconds"] = time.time() - start_time
        result.timing["images_per_second"] = len(images) / result.timing["total_seconds"]
        
        logger.info(f"Benchmark complete in {result.timing['total_seconds']:.1f}s")
        
        return result
    
    def _load_dataset(
        self,
        dataset_path: Path,
    ) -> tuple[list[Image.Image], list[int], list[Path]]:
        """Load images from dataset directory."""
        images = []
        labels = []
        paths = []
        
        # Load real images
        real_dir = dataset_path / "real"
        if real_dir.exists():
            for img_path in self._find_images(real_dir):
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    labels.append(0)  # Real
                    paths.append(img_path)
                    
                    if self.config.num_samples and len(images) >= self.config.num_samples:
                        break
                except Exception as e:
                    logger.warning(f"Could not load {img_path}: {e}")
        
        # Load AI images
        ai_dir = dataset_path / "ai_generated"
        if ai_dir.exists():
            count_before = len(images)
            for img_path in self._find_images(ai_dir):
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    labels.append(1)  # AI
                    paths.append(img_path)
                    
                    if self.config.num_samples and (len(images) - count_before) >= self.config.num_samples:
                        break
                except Exception as e:
                    logger.warning(f"Could not load {img_path}: {e}")
        
        return images, labels, paths
    
    def _find_images(self, directory: Path) -> list[Path]:
        """Find all image files in directory."""
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = []
        
        for ext in extensions:
            images.extend(directory.glob(f"**/*{ext}"))
            images.extend(directory.glob(f"**/*{ext.upper()}"))
        
        return sorted(set(images))
    
    def _evaluate_clean(
        self,
        images: list[Image.Image],
        labels: list[int],
        progress: bool,
    ) -> MetricsResult:
        """Evaluate on clean (uncorrupted) images."""
        scores = []
        
        iterator = tqdm(images, desc="Clean eval", disable=not progress)
        for img in iterator:
            try:
                result = self.detector.detect(img)
                scores.append(result.confidence)
            except Exception as e:
                logger.warning(f"Detection failed: {e}")
                scores.append(0.5)  # Neutral on failure
        
        return compute_metrics(labels, scores)
    
    def _evaluate_robustness(
        self,
        images: list[Image.Image],
        labels: list[int],
        corruption_type: CorruptionType,
        progress: bool,
    ) -> RobustnessResult:
        """Evaluate robustness to a specific corruption."""
        severities = self.config.severity_levels
        metrics_list = []
        auc_list = []
        
        for severity in severities:
            scores = []
            
            desc = f"{corruption_type.value} s={severity:.1f}"
            iterator = tqdm(images, desc=desc, disable=not progress)
            
            for img in iterator:
                try:
                    # Apply corruption
                    config = CorruptionConfig(
                        corruption_type=corruption_type,
                        severity=severity,
                    )
                    corrupted = self.corruptor.apply_corruption(img, config).image
                    
                    # Detect
                    result = self.detector.detect(corrupted)
                    scores.append(result.confidence)
                except Exception as e:
                    logger.warning(f"Failed: {e}")
                    scores.append(0.5)
            
            # Compute metrics
            metrics = compute_metrics(labels, scores)
            metrics_list.append(metrics)
            auc_list.append(metrics.roc_auc)
        
        return RobustnessResult(
            corruption_type=corruption_type.value,
            severities=severities,
            metrics=metrics_list,
            auc_by_severity=auc_list,
        )
    
    def generate_report(
        self,
        result: BenchmarkResult,
        output_dir: Union[str, Path],
    ) -> Path:
        """
        Generate benchmark report.
        
        Args:
            result: Benchmark results
            output_dir: Directory to save report
            
        Returns:
            Path to generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_dir / f"{result.config.name}_results.json"
        json_path.write_text(result.to_json())
        logger.info(f"Saved JSON results to {json_path}")
        
        # Generate Markdown report
        md_path = output_dir / f"{result.config.name}_report.md"
        md_content = self._generate_markdown_report(result)
        md_path.write_text(md_content)
        logger.info(f"Saved Markdown report to {md_path}")
        
        return md_path
    
    def _generate_markdown_report(self, result: BenchmarkResult) -> str:
        """Generate Markdown report content."""
        lines = []
        
        # Header
        lines.append(f"# VerifAI Benchmark Report: {result.config.name}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Dataset info
        lines.append("## Dataset")
        lines.append("")
        lines.append(f"- **Path:** `{result.metadata.get('dataset_path', 'N/A')}`")
        lines.append(f"- **Total Images:** {result.metadata.get('num_images', 0)}")
        lines.append(f"- **Real Images:** {result.metadata.get('num_real', 0)}")
        lines.append(f"- **AI Images:** {result.metadata.get('num_ai', 0)}")
        lines.append("")
        
        # Clean performance
        if result.clean_metrics:
            lines.append("## Clean Image Performance")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| **ROC-AUC** | {result.clean_metrics.roc_auc:.4f} |")
            lines.append(f"| **PR-AUC** | {result.clean_metrics.pr_auc:.4f} |")
            lines.append(f"| **Accuracy** | {result.clean_metrics.accuracy:.4f} |")
            lines.append(f"| **Precision** | {result.clean_metrics.precision:.4f} |")
            lines.append(f"| **Recall** | {result.clean_metrics.recall:.4f} |")
            lines.append(f"| **F1 Score** | {result.clean_metrics.f1:.4f} |")
            lines.append(f"| **ECE** | {result.clean_metrics.ece:.4f} |")
            lines.append("")
        
        # Robustness results
        lines.append("## Robustness Evaluation")
        lines.append("")
        lines.append("Performance (ROC-AUC) under different corruption severities:")
        lines.append("")
        
        # Create robustness table
        if result.robustness_results:
            severities = result.config.severity_levels
            
            # Header row
            header = "| Corruption |"
            for s in severities:
                header += f" s={s:.1f} |"
            header += " Δ AUC |"
            lines.append(header)
            
            # Separator
            sep = "|" + "---|" * (len(severities) + 2)
            lines.append(sep)
            
            # Data rows
            for ctype, robustness in result.robustness_results.items():
                row = f"| **{ctype}** |"
                for auc in robustness.auc_by_severity:
                    row += f" {auc:.3f} |"
                row += f" -{robustness.auc_degradation:.3f} |"
                lines.append(row)
            
            lines.append("")
        
        # Summary
        summary = result._compute_summary()
        lines.append("## Summary")
        lines.append("")
        
        if "clean_roc_auc" in summary:
            lines.append(f"- **Clean ROC-AUC:** {summary['clean_roc_auc']:.4f}")
        if "mean_auc_degradation" in summary:
            lines.append(f"- **Mean AUC Degradation:** {summary['mean_auc_degradation']:.4f}")
        if "most_harmful_corruption" in summary:
            lines.append(f"- **Most Harmful Corruption:** {summary['most_harmful_corruption']}")
        
        lines.append("")
        
        # Interpretation
        lines.append("## Interpretation")
        lines.append("")
        
        if result.clean_metrics:
            if result.clean_metrics.roc_auc > 0.9:
                lines.append("- ✅ **Excellent** clean performance (AUC > 0.9)")
            elif result.clean_metrics.roc_auc > 0.8:
                lines.append("- ⚠️ **Good** clean performance (AUC > 0.8)")
            else:
                lines.append("- ❌ **Needs improvement** on clean images (AUC < 0.8)")
        
        if summary.get("mean_auc_degradation", 0) < 0.1:
            lines.append("- ✅ **Robust** to corruptions (mean degradation < 0.1)")
        elif summary.get("mean_auc_degradation", 0) < 0.2:
            lines.append("- ⚠️ **Moderately robust** (mean degradation < 0.2)")
        else:
            lines.append("- ❌ **Sensitive** to corruptions (mean degradation > 0.2)")
        
        lines.append("")
        
        # Timing
        if result.timing:
            lines.append("## Performance")
            lines.append("")
            lines.append(f"- **Total Time:** {result.timing.get('total_seconds', 0):.1f}s")
            lines.append(f"- **Throughput:** {result.timing.get('images_per_second', 0):.1f} images/sec")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("*Report generated by VerifAI*")
        
        return "\n".join(lines)


def run_quick_benchmark(
    detector,
    dataset_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> BenchmarkResult:
    """
    Run a quick benchmark with default settings.
    
    Args:
        detector: VerifAI detector
        dataset_path: Path to test dataset
        output_dir: Optional output directory for report
        
    Returns:
        BenchmarkResult
    """
    config = BenchmarkConfig(
        name="quick_benchmark",
        corruption_types=[
            CorruptionType.JPEG_COMPRESSION,
            CorruptionType.RESIZE,
        ],
        severity_levels=[0.0, 0.5, 1.0],
        num_samples=100,  # Limit for speed
    )
    
    benchmark = Benchmark(detector, config)
    result = benchmark.run(dataset_path)
    
    if output_dir:
        benchmark.generate_report(result, output_dir)
    
    return result
