"""
VerifAI FastAPI Server
======================

REST API for AI-generated media detection.

Usage:
    uvicorn api.server:app --reload --port 8000
    
    # Or with the CLI
    verifai serve --port 8000

Endpoints:
    POST /detect          - Detect AI-generated media
    GET  /health          - Health check
    GET  /info            - Model information
    GET  /                - API documentation redirect
"""

import io
import time
import base64
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from loguru import logger

from verifai import VerifAI


# Global detector instance
detector: Optional[VerifAI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize detector on startup."""
    global detector
    logger.info("Starting VerifAI API server...")
    
    # Initialize detector (models will be downloaded if needed)
    detector = VerifAI(generate_heatmaps=True)
    detector.load()
    
    logger.info("VerifAI detector loaded and ready")
    yield
    
    # Cleanup
    logger.info("Shutting down VerifAI API server")


# Create FastAPI app
app = FastAPI(
    title="VerifAI API",
    description="AI-Generated Media Detection API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Response Models
# ============================================================================

class DetectionResponse(BaseModel):
    """Detection result response."""
    success: bool = Field(description="Whether detection succeeded")
    label: str = Field(description="Detection label: 'real' or 'ai_generated'")
    confidence: float = Field(description="Confidence score (0-1)")
    is_ai_generated: bool = Field(description="True if AI-generated")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    
    # Optional detailed fields
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    evidence: Optional[dict] = Field(None, description="Detailed detection evidence")
    heatmap_base64: Optional[str] = Field(None, description="Base64-encoded heatmap PNG")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "label": "ai_generated",
                "confidence": 0.95,
                "is_ai_generated": True,
                "processing_time_ms": 120.5,
                "explanation": "High confidence AI-generated image detected",
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Server status")
    model_loaded: bool = Field(description="Whether model is loaded")
    version: str = Field(description="API version")


class InfoResponse(BaseModel):
    """Model information response."""
    model_name: str = Field(description="Active model name")
    accuracy: str = Field(description="Model accuracy")
    supported_formats: list[str] = Field(description="Supported image formats")
    features: list[str] = Field(description="Enabled features")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns server status and model availability.
    """
    return HealthResponse(
        status="healthy" if detector and detector._is_loaded else "starting",
        model_loaded=detector._is_loaded if detector else False,
        version="1.0.0",
    )


@app.get("/info", response_model=InfoResponse, tags=["System"])
async def model_info():
    """
    Get model information.
    
    Returns details about the active detection model.
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    return InfoResponse(
        model_name="FusionDetector (CLIP + Frequency)",
        accuracy="97.0%",
        supported_formats=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        features=[
            "Fusion Detection (CLIP + Frequency)",
            "Heatmap Generation",
            "Confidence Calibration",
            "Metadata Analysis",
        ],
    )


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    return_heatmap: bool = Query(False, description="Include heatmap in response"),
    return_evidence: bool = Query(False, description="Include detailed evidence"),
):
    """
    Detect if an image is AI-generated.
    
    Upload an image file and get a detection result with confidence score.
    
    - **file**: Image file (JPEG, PNG, WebP, etc.)
    - **return_heatmap**: If true, includes base64-encoded heatmap
    - **return_evidence**: If true, includes detailed detection evidence
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Expected image/*"
        )
    
    try:
        # Read image
        start_time = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run detection
        result = detector.detect(image, return_evidence=return_evidence)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Generate explanation from result
        if result.is_ai_generated:
            explanation = f"AI-generated image detected with {result.confidence:.1%} confidence"
        else:
            explanation = f"Real image detected with {result.confidence:.1%} confidence"
        
        # Build response
        response = DetectionResponse(
            success=True,
            label=result.label,
            confidence=result.confidence,
            is_ai_generated=result.is_ai_generated,
            processing_time_ms=round(processing_time, 2),
            explanation=explanation,
        )
        
        # Add evidence if requested (convert numpy types to native Python)
        if return_evidence and result.evidence:
            import json
            # Convert to JSON and back to ensure all numpy types are converted
            response.evidence = json.loads(
                json.dumps(result.evidence, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
            )
        
        # Add heatmap if requested and available
        if return_heatmap and result.heatmap is not None:
            import numpy as np
            from PIL import Image as PILImage
            
            # Convert heatmap to PNG
            heatmap_img = PILImage.fromarray(
                (result.heatmap * 255).astype(np.uint8)
            )
            buffer = io.BytesIO()
            heatmap_img.save(buffer, format="PNG")
            response.heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return response
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect/url", response_model=DetectionResponse, tags=["Detection"])
async def detect_from_url(
    url: str = Query(..., description="URL of image to analyze"),
    return_heatmap: bool = Query(False, description="Include heatmap in response"),
    return_evidence: bool = Query(False, description="Include detailed evidence"),
):
    """
    Detect if an image from URL is AI-generated.
    
    - **url**: Direct URL to an image file
    - **return_heatmap**: If true, includes base64-encoded heatmap
    - **return_evidence**: If true, includes detailed detection evidence
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        import httpx
        
        start_time = time.time()
        
        # Download image
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Run detection
        result = detector.detect(image, return_evidence=return_evidence)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Generate explanation from result
        if result.is_ai_generated:
            explanation = f"AI-generated image detected with {result.confidence:.1%} confidence"
        else:
            explanation = f"Real image detected with {result.confidence:.1%} confidence"
        
        # Build response
        detection_response = DetectionResponse(
            success=True,
            label=result.label,
            confidence=result.confidence,
            is_ai_generated=result.is_ai_generated,
            processing_time_ms=round(processing_time, 2),
            explanation=explanation,
        )
        
        if return_evidence and result.evidence:
            import json
            detection_response.evidence = json.loads(
                json.dumps(result.evidence, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
            )
        
        if return_heatmap and result.heatmap is not None:
            import numpy as np
            from PIL import Image as PILImage
            
            heatmap_img = PILImage.fromarray(
                (result.heatmap * 255).astype(np.uint8)
            )
            buffer = io.BytesIO()
            heatmap_img.save(buffer, format="PNG")
            detection_response.heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return detection_response
        
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


# ============================================================================
# Run server (for direct execution)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
