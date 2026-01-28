"""
VerifAI Gradio Web Interface
============================

Beautiful web UI for AI-generated image detection.

Usage:
    python ui/app.py
    
    # Or with custom port
    python ui/app.py --port 7860 --share
"""

import argparse
import numpy as np
from PIL import Image
import gradio as gr
from loguru import logger

from verifai import VerifAI


# Global detector
detector = None


def load_detector():
    """Load detector on first use."""
    global detector
    if detector is None:
        logger.info("Loading VerifAI detector...")
        detector = VerifAI(generate_heatmaps=True)
        detector.load()
        logger.info("Detector loaded!")
    return detector


def analyze_image(image: np.ndarray | Image.Image | None) -> tuple:
    """
    Analyze an image for AI generation.
    
    Returns:
        tuple: (label_html, confidence, heatmap, details)
    """
    if image is None:
        return (
            "<div style='text-align:center; color:#888;'>Upload an image to analyze</div>",
            None,
            None,
            ""
        )
    
    try:
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Load detector
        det = load_detector()
        
        # Run detection
        result = det.detect(image, return_evidence=True)
        
        # Format label with color
        if result.is_ai_generated:
            label_html = f"""
            <div style='text-align:center; padding:20px;'>
                <span style='font-size:48px;'>🤖</span>
                <h2 style='color:#e74c3c; margin:10px 0;'>AI-Generated</h2>
                <p style='font-size:24px; color:#e74c3c;'>{result.confidence:.1%} confident</p>
            </div>
            """
        else:
            label_html = f"""
            <div style='text-align:center; padding:20px;'>
                <span style='font-size:48px;'>📷</span>
                <h2 style='color:#27ae60; margin:10px 0;'>Real Photo</h2>
                <p style='font-size:24px; color:#27ae60;'>{result.confidence:.1%} confident</p>
            </div>
            """
        
        # Confidence gauge value (0-100)
        confidence_pct = result.confidence * 100
        
        # Get heatmap if available
        heatmap = None
        if result.heatmap is not None:
            # Create colored heatmap overlay
            heatmap_colored = create_heatmap_overlay(image, result.heatmap)
            heatmap = heatmap_colored
        
        # Format details
        details = format_evidence(result)
        
        return label_html, confidence_pct, heatmap, details
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return (
            f"<div style='color:#e74c3c;'>Error: {str(e)}</div>",
            None,
            None,
            ""
        )


def create_heatmap_overlay(
    original: Image.Image, 
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> Image.Image:
    """Create a colored heatmap overlay on the original image."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Resize heatmap to match original
    original_np = np.array(original.convert("RGB"))
    h, w = original_np.shape[:2]
    
    # Resize heatmap
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))
    ) / 255.0
    
    # Apply colormap (red = suspicious, blue = normal)
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # Remove alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend with original
    blended = (
        original_np * (1 - alpha) + heatmap_colored * alpha
    ).astype(np.uint8)
    
    return Image.fromarray(blended)


def format_evidence(result) -> str:
    """Format detection evidence as markdown."""
    lines = ["### Detection Evidence\n"]
    
    if result.evidence:
        neural = result.evidence.get("neural", {})
        
        if "clip_score" in neural:
            lines.append(f"- **CLIP Score:** {neural['clip_score']:.3f}")
        if "frequency_score" in neural:
            lines.append(f"- **Frequency Score:** {neural['frequency_score']:.3f}")
        if "fusion_score" in neural:
            lines.append(f"- **Fusion Score:** {neural['fusion_score']:.3f}")
    
    # Generate explanation from result
    if result.is_ai_generated:
        lines.append(f"\n**Explanation:** AI-generated image detected with {result.confidence:.1%} confidence")
    else:
        lines.append(f"\n**Explanation:** Real image detected with {result.confidence:.1%} confidence")
    
    return "\n".join(lines)


def create_ui():
    """Create the Gradio interface."""
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(
        title="VerifAI - AI Image Detector",
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🔍 VerifAI - AI-Generated Image Detector
        
        Upload an image to detect if it was generated by AI (Stable Diffusion, DALL-E, MidJourney, etc.)
        
        **Accuracy:** 97% on modern AI generators | **Model:** CLIP ViT-L/14 + Frequency Fusion
        """)
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400,
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg",
                )
                
                gr.Markdown("""
                **Supported formats:** JPEG, PNG, WebP, BMP, TIFF
                
                **Tips:**
                - Higher resolution images give better results
                - Works best on uncompressed images
                """)
            
            # Right column - Results
            with gr.Column(scale=1):
                result_label = gr.HTML(
                    value="<div style='text-align:center; color:#888; padding:50px;'>Upload an image to analyze</div>",
                    label="Result",
                )
                
                confidence_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Confidence (%)",
                    interactive=False,
                )
                
                heatmap_output = gr.Image(
                    label="Suspicious Regions (Red = AI artifacts)",
                    type="pil",
                    height=300,
                )
                
                evidence_output = gr.Markdown(
                    label="Evidence",
                )
        
        # Examples
        gr.Markdown("### Example Images")
        gr.Markdown("*Try with your own images or use examples from the web*")
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[result_label, confidence_slider, heatmap_output, evidence_output],
        )
        
        # Also analyze on image upload
        input_image.change(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[result_label, confidence_slider, heatmap_output, evidence_output],
        )
        
        # Footer
        gr.Markdown("""
        ---
        **VerifAI** | [GitHub](https://github.com/canemirbora4/VerifAI) | 
        [Model Weights](https://huggingface.co/canemirbora/verifai-models)
        
        *Note: Detection is probabilistic. Always verify with multiple sources for critical decisions.*
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="VerifAI Web Interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    args = parser.parse_args()
    
    # Pre-load detector
    logger.info("Pre-loading detector...")
    load_detector()
    
    # Create and launch UI
    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


# Also allow creating demo without launching for imports
def get_demo():
    """Get demo instance without launching."""
    return create_ui()


if __name__ == "__main__":
    main()
