"""
VerifAI CLI Main Module
========================

Command-line interface for VerifAI AI-generated media detection.
"""

import sys
import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from loguru import logger

# Configure loguru
logger.remove()  # Remove default handler


console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = "DEBUG" if verbose else "WARNING"
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="VerifAI")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    VerifAI - AI-Generated Media Detector
    
    Detect AI-generated images and videos with calibrated confidence scores,
    localized evidence, and robustness evaluation.
    
    \b
    Examples:
        verifai detect image.jpg
        verifai detect photo.png --output result.json
        verifai detect images/ --recursive
        verifai info
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output file for results (JSON format)",
)
@click.option(
    "-t", "--threshold",
    type=float,
    default=0.5,
    help="Classification threshold (default: 0.5)",
)
@click.option(
    "-m", "--model",
    type=str,
    default="google/vit-base-patch16-224",
    help="Model to use for detection",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
    help="Device for inference",
)
@click.option(
    "-r", "--recursive",
    is_flag=True,
    help="Process directories recursively",
)
@click.option(
    "--evidence",
    is_flag=True,
    help="Include evidence (attention maps) in output",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output results as JSON",
)
@click.pass_context
def detect(
    ctx: click.Context,
    input_path: str,
    output: Optional[str],
    threshold: float,
    model: str,
    device: str,
    recursive: bool,
    evidence: bool,
    json_output: bool,
) -> None:
    """
    Detect AI-generated media in images or videos.
    
    INPUT_PATH can be a single file or a directory.
    
    \b
    Examples:
        verifai detect image.jpg
        verifai detect photos/ --recursive
        verifai detect video.mp4 --output result.json
    """
    from verifai import VerifAI
    from verifai.ingest.utils import list_media_files, get_media_type, MediaType
    
    path = Path(input_path)
    
    # Collect files to process
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list_media_files(path, recursive=recursive)
        if not files:
            console.print(f"[yellow]No media files found in {path}[/yellow]")
            return
        console.print(f"Found [bold]{len(files)}[/bold] media files")
    else:
        console.print(f"[red]Invalid path: {input_path}[/red]")
        sys.exit(1)
    
    # Filter to only images for now (video coming in Phase 4)
    image_files = [f for f in files if get_media_type(f) == MediaType.IMAGE]
    video_files = [f for f in files if get_media_type(f) == MediaType.VIDEO]
    
    if video_files:
        console.print(
            f"[yellow]Note: Found {len(video_files)} video files. "
            f"Video detection coming in Phase 4![/yellow]"
        )
    
    if not image_files:
        console.print("[red]No supported image files found[/red]")
        sys.exit(1)
    
    # Initialize detector
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading VerifAI model...", total=None)
        
        try:
            detector = VerifAI(
                model_name=model,
                device=None if device == "auto" else device,
                threshold=threshold,
            )
            detector.load()
        except Exception as e:
            console.print(f"[red]Failed to load model: {e}[/red]")
            if ctx.obj.get("verbose"):
                raise
            sys.exit(1)
    
    # Process files
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(image_files))
        
        for file_path in image_files:
            progress.update(task, description=f"Processing {file_path.name}...")
            
            try:
                result = detector.detect(file_path, return_evidence=evidence)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
            
            progress.advance(task)
    
    if not results:
        console.print("[red]No files were successfully processed[/red]")
        sys.exit(1)
    
    # Output results
    if json_output or output:
        # JSON output
        output_data = {
            "version": "0.1.0",
            "model": model,
            "threshold": threshold,
            "results": [r.to_dict() for r in results],
        }
        
        if output:
            output_path = Path(output)
            output_path.write_text(json.dumps(output_data, indent=2))
            console.print(f"[green]Results saved to {output_path}[/green]")
        else:
            print(json.dumps(output_data, indent=2))
    else:
        # Rich formatted output
        if len(results) == 1:
            # Single file: detailed output
            result = results[0]
            display_single_result(result)
        else:
            # Multiple files: table output
            display_results_table(results)


def display_single_result(result) -> None:
    """Display a single detection result with rich formatting."""
    # Determine colors and emoji based on result
    if result.label == "ai_generated":
        emoji = "🤖"
        label_color = "red"
        verdict = "AI-GENERATED"
    elif result.label == "real":
        emoji = "📷"
        label_color = "green"
        verdict = "REAL"
    else:
        emoji = "❓"
        label_color = "yellow"
        verdict = result.label.upper()
    
    # Confidence bar
    conf_int = int(result.confidence * 20)
    conf_bar = "█" * conf_int + "░" * (20 - conf_int)
    
    # Build panel content
    content = []
    content.append(f"[bold {label_color}]{emoji} {verdict}[/bold {label_color}]")
    content.append("")
    content.append(f"Confidence: [{label_color}]{conf_bar}[/{label_color}] {result.confidence:.1%}")
    content.append("")
    
    if result.input_path:
        content.append(f"[dim]File:[/dim] {Path(result.input_path).name}")
    if result.input_size:
        content.append(f"[dim]Size:[/dim] {result.input_size[0]} × {result.input_size[1]} px")
    if result.processing_time_ms > 0:
        content.append(f"[dim]Time:[/dim] {result.processing_time_ms:.0f} ms")
    
    # Detector scores
    if result.detector_scores:
        content.append("")
        content.append("[dim]Detector Scores:[/dim]")
        for name, score in result.detector_scores.items():
            content.append(f"  {name}: {score:.4f}")
    
    panel = Panel(
        "\n".join(content),
        title="[bold]VerifAI Detection Result[/bold]",
        border_style="blue",
        padding=(1, 2),
    )
    
    console.print(panel)


def display_results_table(results: list) -> None:
    """Display multiple results in a table format."""
    table = Table(
        title="VerifAI Detection Results",
        show_header=True,
        header_style="bold blue",
    )
    
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Verdict", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Time", justify="right", style="dim")
    
    for result in results:
        # File name
        if result.input_path:
            filename = Path(result.input_path).name
            if len(filename) > 30:
                filename = filename[:27] + "..."
        else:
            filename = "unknown"
        
        # Verdict with color
        if result.label == "ai_generated":
            verdict = "[red]🤖 AI[/red]"
        elif result.label == "real":
            verdict = "[green]📷 Real[/green]"
        else:
            verdict = f"[yellow]❓ {result.label}[/yellow]"
        
        # Confidence
        confidence = f"{result.confidence:.1%}"
        
        # Time
        time_str = f"{result.processing_time_ms:.0f}ms"
        
        table.add_row(filename, verdict, confidence, time_str)
    
    console.print(table)
    
    # Summary
    ai_count = sum(1 for r in results if r.label == "ai_generated")
    real_count = sum(1 for r in results if r.label == "real")
    
    console.print(
        f"\n[dim]Summary:[/dim] "
        f"[green]{real_count} real[/green], "
        f"[red]{ai_count} AI-generated[/red], "
        f"{len(results)} total"
    )


@cli.command()
@click.option(
    "-m", "--model",
    type=str,
    default="google/vit-base-patch16-224",
    help="Model to show info for",
)
def info(model: str) -> None:
    """
    Show information about VerifAI and the detection model.
    """
    from verifai import __version__
    
    content = []
    content.append(f"[bold]Version:[/bold] {__version__}")
    content.append(f"[bold]Model:[/bold] {model}")
    content.append("")
    
    # Check device availability
    import torch
    
    if torch.cuda.is_available():
        device = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = "MPS (Apple Silicon)"
    else:
        device = "CPU"
    
    content.append(f"[bold]Available Device:[/bold] {device}")
    content.append("")
    
    # Model info
    content.append("[bold]Supported Models:[/bold]")
    models = [
        ("google/vit-base-patch16-224", "ViT Base (default)"),
        ("google/vit-large-patch16-224", "ViT Large"),
        ("facebook/convnext-base-224", "ConvNeXt Base"),
        ("facebook/convnext-tiny-224", "ConvNeXt Tiny"),
        ("microsoft/swin-base-patch4-window7-224", "Swin Base"),
    ]
    
    for model_id, description in models:
        content.append(f"  • {model_id}")
        content.append(f"    [dim]{description}[/dim]")
    
    panel = Panel(
        "\n".join(content),
        title="[bold blue]VerifAI Information[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )
    
    console.print(panel)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default=None,
    help="Output directory for evaluation report",
)
def evaluate(input_path: str, output: Optional[str]) -> None:
    """
    Evaluate detection performance on a labeled dataset.
    
    INPUT_PATH should be a directory with subdirectories 'real' and 'ai_generated'
    containing labeled samples.
    
    \b
    Expected structure:
        dataset/
        ├── real/
        │   ├── image1.jpg
        │   └── ...
        └── ai_generated/
            ├── image1.jpg
            └── ...
    """
    from verifai import VerifAI
    from verifai.eval import compute_metrics
    from verifai.ingest.utils import list_media_files, MediaType
    import numpy as np
    
    path = Path(input_path)
    
    # Check directory structure
    real_dir = path / "real"
    ai_dir = path / "ai_generated"
    
    if not real_dir.exists() or not ai_dir.exists():
        console.print(
            "[red]Expected directory structure not found![/red]\n"
            "Please organize your data as:\n"
            "  dataset/\n"
            "  ├── real/\n"
            "  └── ai_generated/"
        )
        sys.exit(1)
    
    # Collect files
    real_files = list_media_files(real_dir, media_type=MediaType.IMAGE, recursive=True)
    ai_files = list_media_files(ai_dir, media_type=MediaType.IMAGE, recursive=True)
    
    console.print(f"Found [green]{len(real_files)}[/green] real images")
    console.print(f"Found [red]{len(ai_files)}[/red] AI-generated images")
    
    if not real_files or not ai_files:
        console.print("[red]Need at least one sample in each category[/red]")
        sys.exit(1)
    
    # Initialize detector
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading model...", total=None)
        detector = VerifAI()
        detector.load()
    
    # Run detection and collect predictions
    y_true = []
    y_prob = []
    
    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in ai_files]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(all_files))
        
        for file_path, label in all_files:
            try:
                result = detector.detect(file_path)
                y_true.append(label)
                y_prob.append(result.confidence)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
            
            progress.advance(task)
    
    # Compute metrics
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    metrics = compute_metrics(y_true, y_prob)
    
    # Display results
    console.print("\n")
    console.print(Panel(
        metrics.summary(),
        title="[bold]Evaluation Results[/bold]",
        border_style="green",
    ))
    
    # Save report if output specified
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_path = output_path / "evaluation_report.json"
        report_path.write_text(json.dumps(metrics.to_dict(), indent=2))
        
        console.print(f"\n[green]Report saved to {report_path}[/green]")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default="benchmark_results",
    help="Output directory for benchmark report",
)
@click.option(
    "--corruptions",
    type=str,
    default="jpeg,resize,blur,noise",
    help="Comma-separated list of corruptions to test",
)
@click.option(
    "--severities",
    type=str,
    default="0.0,0.2,0.4,0.6,0.8,1.0",
    help="Comma-separated severity levels",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum samples per class (for faster testing)",
)
@click.option(
    "--quick",
    is_flag=True,
    help="Run quick benchmark (fewer corruptions and severities)",
)
@click.pass_context
def benchmark(
    ctx: click.Context,
    input_path: str,
    output: str,
    corruptions: str,
    severities: str,
    max_samples: Optional[int],
    quick: bool,
) -> None:
    """
    Run robustness benchmark on a dataset.
    
    Tests detector performance under various image corruptions
    (JPEG compression, resize, blur, noise) and generates a report.
    
    INPUT_PATH should be a directory with 'real' and 'ai_generated' subdirs.
    
    \b
    Examples:
        verifai benchmark data/test/ --output reports/
        verifai benchmark data/test/ --quick
        verifai benchmark data/test/ --corruptions jpeg,resize --max-samples 50
    """
    from verifai import VerifAI
    from verifai.eval import (
        Benchmark,
        BenchmarkConfig,
        CorruptionType,
    )
    
    path = Path(input_path)
    output_path = Path(output)
    
    # Check directory structure
    real_dir = path / "real"
    ai_dir = path / "ai_generated"
    
    if not real_dir.exists() or not ai_dir.exists():
        console.print(
            "[red]Expected directory structure not found![/red]\n"
            "Please organize your data as:\n"
            "  dataset/\n"
            "  ├── real/\n"
            "  └── ai_generated/"
        )
        sys.exit(1)
    
    # Parse corruptions
    corruption_map = {
        "jpeg": CorruptionType.JPEG_COMPRESSION,
        "resize": CorruptionType.RESIZE,
        "blur": CorruptionType.GAUSSIAN_BLUR,
        "noise": CorruptionType.GAUSSIAN_NOISE,
        "crop": CorruptionType.CROP,
        "brightness": CorruptionType.BRIGHTNESS,
        "contrast": CorruptionType.CONTRAST,
        "screenshot": CorruptionType.SCREENSHOT,
    }
    
    if quick:
        corruption_types = [CorruptionType.JPEG_COMPRESSION, CorruptionType.RESIZE]
        severity_levels = [0.0, 0.5, 1.0]
        max_samples = max_samples or 50
    else:
        corruption_types = []
        for c in corruptions.split(","):
            c = c.strip().lower()
            if c in corruption_map:
                corruption_types.append(corruption_map[c])
            else:
                console.print(f"[yellow]Unknown corruption: {c}[/yellow]")
        
        severity_levels = [float(s.strip()) for s in severities.split(",")]
    
    if not corruption_types:
        corruption_types = [CorruptionType.JPEG_COMPRESSION, CorruptionType.RESIZE]
    
    # Create config
    config = BenchmarkConfig(
        name="verifai_benchmark",
        corruption_types=corruption_types,
        severity_levels=severity_levels,
        num_samples=max_samples,
        include_clean=True,
    )
    
    console.print(Panel(
        f"[bold]Corruptions:[/bold] {', '.join(c.value for c in corruption_types)}\n"
        f"[bold]Severities:[/bold] {severity_levels}\n"
        f"[bold]Max samples:[/bold] {max_samples or 'all'}",
        title="[bold blue]Benchmark Configuration[/bold blue]",
        border_style="blue",
    ))
    
    # Initialize detector
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading VerifAI model...", total=None)
        detector = VerifAI()
        detector.load()
    
    # Run benchmark
    console.print("\n[bold]Running benchmark...[/bold]\n")
    
    bench = Benchmark(detector, config)
    
    try:
        results = bench.run(path, progress=True)
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        if ctx.obj.get("verbose"):
            raise
        sys.exit(1)
    
    # Display summary
    console.print("\n")
    
    summary_lines = []
    if results.clean_metrics:
        summary_lines.append(f"[bold]Clean Performance:[/bold]")
        summary_lines.append(f"  ROC-AUC: {results.clean_metrics.roc_auc:.4f}")
        summary_lines.append(f"  Accuracy: {results.clean_metrics.accuracy:.4f}")
        summary_lines.append("")
    
    summary_lines.append("[bold]Robustness (AUC at max severity):[/bold]")
    for ctype, robustness in results.robustness_results.items():
        clean_auc = robustness.auc_by_severity[0]
        worst_auc = robustness.auc_by_severity[-1]
        degradation = robustness.auc_degradation
        
        if degradation < 0.1:
            status = "[green]✓[/green]"
        elif degradation < 0.2:
            status = "[yellow]⚠[/yellow]"
        else:
            status = "[red]✗[/red]"
        
        summary_lines.append(
            f"  {ctype}: {clean_auc:.3f} → {worst_auc:.3f} "
            f"(Δ={degradation:.3f}) {status}"
        )
    
    console.print(Panel(
        "\n".join(summary_lines),
        title="[bold]Benchmark Results[/bold]",
        border_style="green",
    ))
    
    # Generate report
    report_path = bench.generate_report(results, output_path)
    
    console.print(f"\n[green]Report saved to {report_path}[/green]")
    console.print(f"[dim]JSON results: {output_path / 'verifai_benchmark_results.json'}[/dim]")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
