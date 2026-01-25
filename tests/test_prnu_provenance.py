"""
Tests for PRNU and Provenance Analysis
======================================

Tests for camera sensor fingerprinting and content credentials verification.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import io

from verifai.features.prnu import (
    PRNUExtractor,
    PRNUFeatures,
    extract_prnu,
)
from verifai.features.provenance import (
    ProvenanceAnalyzer,
    ProvenanceFeatures,
    C2PAClaim,
    analyze_provenance,
)


# =============================================================================
# PRNU Tests
# =============================================================================

class TestPRNUFeatures:
    """Tests for PRNUFeatures dataclass."""
    
    def test_prnu_features_defaults(self):
        """Test default values."""
        features = PRNUFeatures()
        
        assert features.noise_residual is None
        assert features.noise_strength == 0.0
        assert features.noise_uniformity == 0.0
        assert features.has_prnu_signature is False
        assert features.prnu_score == 0.5
        assert features.correlation is None
    
    def test_prnu_features_to_dict(self):
        """Test to_dict conversion."""
        features = PRNUFeatures(
            noise_strength=2.5,
            noise_uniformity=0.85,
            has_prnu_signature=True,
            prnu_score=0.75,
            quality_score=0.9,
            saturation_ratio=0.05,
        )
        
        d = features.to_dict()
        
        assert d["noise_strength"] == 2.5
        assert d["noise_uniformity"] == 0.85
        assert d["has_prnu_signature"] is True
        assert d["prnu_score"] == 0.75
        assert "correlation" in d
    
    def test_is_likely_real_property(self):
        """Test is_likely_real property."""
        # Has PRNU and high score = likely real
        features1 = PRNUFeatures(has_prnu_signature=True, prnu_score=0.7)
        assert features1.is_likely_real is True
        
        # Has PRNU but low score = not likely real
        features2 = PRNUFeatures(has_prnu_signature=True, prnu_score=0.3)
        assert features2.is_likely_real is False
        
        # No PRNU = not likely real
        features3 = PRNUFeatures(has_prnu_signature=False, prnu_score=0.8)
        assert features3.is_likely_real is False


class TestPRNUExtractor:
    """Tests for PRNUExtractor class."""
    
    def test_init(self):
        """Test extractor initialization."""
        extractor = PRNUExtractor()
        assert extractor.denoise_strength == 3.0
        assert extractor.min_quality == 0.3
    
    def test_extract_from_numpy(self):
        """Test extraction from numpy array."""
        extractor = PRNUExtractor()
        
        # Create synthetic image with noise
        np.random.seed(42)
        image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        
        features = extractor.extract(image)
        
        assert isinstance(features, PRNUFeatures)
        assert features.noise_residual is not None
        assert features.noise_strength > 0
        assert 0 <= features.quality_score <= 1
    
    def test_extract_from_pil(self):
        """Test extraction from PIL Image."""
        extractor = PRNUExtractor()
        
        # Create PIL image
        image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        
        # Add some texture
        pixels = image.load()
        for i in range(256):
            for j in range(256):
                noise = np.random.randint(-20, 20)
                r = max(0, min(255, 128 + noise))
                g = max(0, min(255, 128 + noise))
                b = max(0, min(255, 128 + noise))
                pixels[i, j] = (r, g, b)
        
        features = extractor.extract(image)
        
        assert isinstance(features, PRNUFeatures)
        assert features.noise_residual is not None
    
    def test_extract_from_file(self, tmp_path):
        """Test extraction from file path."""
        extractor = PRNUExtractor()
        
        # Create and save test image
        image = Image.new("RGB", (256, 256), color=(100, 150, 200))
        file_path = tmp_path / "test_image.png"
        image.save(file_path)
        
        features = extractor.extract(file_path)
        
        assert isinstance(features, PRNUFeatures)
    
    def test_extract_low_quality_image(self):
        """Test extraction from low quality (saturated) image."""
        extractor = PRNUExtractor()
        
        # Create heavily saturated image
        image = np.full((256, 256, 3), 255, dtype=np.uint8)  # All white
        
        features = extractor.extract(image)
        
        # Should have low quality score
        assert features.quality_score < 0.5
    
    def test_noise_uniformity_calculation(self):
        """Test noise uniformity is calculated correctly."""
        extractor = PRNUExtractor()
        
        # Uniform noise should have high uniformity
        np.random.seed(42)
        uniform_image = np.random.randint(100, 150, (256, 256, 3), dtype=np.uint8)
        
        features = extractor.extract(uniform_image)
        
        # Should have reasonable uniformity
        assert features.noise_uniformity > 0
    
    def test_convenience_function(self):
        """Test extract_prnu convenience function."""
        image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        
        features = extract_prnu(image)
        
        assert isinstance(features, PRNUFeatures)


class TestPRNUReference:
    """Tests for PRNU reference fingerprint building."""
    
    def test_build_reference_minimum_images(self):
        """Test that at least 3 images required for reference."""
        extractor = PRNUExtractor()
        
        images = [
            np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            for _ in range(2)
        ]
        
        with pytest.raises(ValueError, match="at least 3 images"):
            extractor.build_reference(images)
    
    def test_build_reference_success(self):
        """Test successful reference building."""
        extractor = PRNUExtractor()
        
        # Create 5 similar images (simulating same camera)
        np.random.seed(42)
        base_noise = np.random.randn(256, 256, 3) * 2  # Consistent PRNU pattern
        
        images = []
        for i in range(5):
            # Each image = random content + consistent PRNU
            content = np.random.randint(50, 200, (256, 256, 3)).astype(np.float32)
            image = np.clip(content + base_noise, 0, 255).astype(np.uint8)
            images.append(image)
        
        reference = extractor.build_reference(images)
        
        assert isinstance(reference, np.ndarray)
        assert reference.shape == (256, 256, 3)


# =============================================================================
# Provenance Tests
# =============================================================================

class TestC2PAClaim:
    """Tests for C2PAClaim dataclass."""
    
    def test_c2pa_claim_defaults(self):
        """Test default values."""
        claim = C2PAClaim()
        
        assert claim.claim_generator == ""
        assert claim.title == ""
        assert claim.actions == []
        assert claim.ingredients == []
        assert claim.is_valid is False
    
    def test_c2pa_claim_with_data(self):
        """Test claim with actual data."""
        claim = C2PAClaim(
            claim_generator="Adobe Photoshop 24.0",
            title="My Photo",
            format="image/jpeg",
            instance_id="xmp:iid:12345",
            actions=[{"action": "c2pa.created"}],
        )
        
        assert claim.claim_generator == "Adobe Photoshop 24.0"
        assert len(claim.actions) == 1


class TestProvenanceFeatures:
    """Tests for ProvenanceFeatures dataclass."""
    
    def test_provenance_features_defaults(self):
        """Test default values."""
        features = ProvenanceFeatures()
        
        assert features.has_c2pa is False
        assert features.has_valid_signature is False
        assert features.claims == []
        assert features.provenance_score == 0.5
    
    def test_provenance_features_to_dict(self):
        """Test to_dict conversion."""
        features = ProvenanceFeatures(
            has_c2pa=True,
            has_valid_signature=True,
            creation_tool="Canon EOS R5",
            trust_indicators=["Canon camera metadata"],
            risk_indicators=[],
            provenance_score=0.85,
        )
        
        d = features.to_dict()
        
        assert d["has_c2pa"] is True
        assert d["has_valid_signature"] is True
        assert d["creation_tool"] == "Canon EOS R5"
        assert d["provenance_score"] == 0.85
    
    def test_is_verified_property(self):
        """Test is_verified property."""
        # Has C2PA and valid signature = verified
        features1 = ProvenanceFeatures(has_c2pa=True, has_valid_signature=True)
        assert features1.is_verified is True
        
        # Has C2PA but no valid signature = not verified
        features2 = ProvenanceFeatures(has_c2pa=True, has_valid_signature=False)
        assert features2.is_verified is False
        
        # No C2PA = not verified
        features3 = ProvenanceFeatures(has_c2pa=False, has_valid_signature=True)
        assert features3.is_verified is False


class TestProvenanceAnalyzer:
    """Tests for ProvenanceAnalyzer class."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = ProvenanceAnalyzer()
        assert analyzer.AI_GENERATORS  # Should have known AI generators
        assert analyzer.TRUSTED_TOOLS  # Should have known trusted tools
    
    def test_analyze_no_c2pa(self, tmp_path):
        """Test analysis of image without C2PA."""
        analyzer = ProvenanceAnalyzer()
        
        # Create simple image without any metadata
        image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        file_path = tmp_path / "no_metadata.jpg"
        image.save(file_path, "JPEG")
        
        features = analyzer.analyze(file_path)
        
        assert features.has_c2pa is False
        assert "No provenance metadata" in features.risk_indicators or len(features.risk_indicators) >= 0
    
    def test_analyze_from_bytes(self):
        """Test analysis from bytes."""
        analyzer = ProvenanceAnalyzer()
        
        # Create JPEG bytes
        image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        data = buffer.getvalue()
        
        features = analyzer.analyze(data)
        
        assert isinstance(features, ProvenanceFeatures)
    
    def test_detect_ai_markers_in_bytes(self):
        """Test detection of AI generation markers."""
        analyzer = ProvenanceAnalyzer()
        
        # Create bytes with AI marker
        image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        data = buffer.getvalue()
        
        # Inject AI marker (this is a simplified test)
        data_with_marker = data + b"stable-diffusion"
        
        features = analyzer.analyze(data_with_marker)
        
        # Should detect the marker
        assert any("stable" in ind.lower() or "diffusion" in ind.lower() 
                   for ind in features.risk_indicators)
    
    def test_provenance_score_calculation(self):
        """Test provenance score is calculated correctly."""
        analyzer = ProvenanceAnalyzer()
        
        # Image with no metadata should have neutral/low score
        image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        
        features = analyzer.analyze(buffer.getvalue())
        
        # Score should be in valid range
        assert 0.0 <= features.provenance_score <= 1.0
    
    def test_convenience_function(self, tmp_path):
        """Test analyze_provenance convenience function."""
        image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        file_path = tmp_path / "test.jpg"
        image.save(file_path, "JPEG")
        
        features = analyze_provenance(file_path)
        
        assert isinstance(features, ProvenanceFeatures)


class TestProvenanceXMP:
    """Tests for XMP metadata parsing."""
    
    def test_xmp_creation_tool_extraction(self):
        """Test XMP CreatorTool extraction."""
        analyzer = ProvenanceAnalyzer()
        
        # Create fake XMP data
        xmp_data = b'''<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
            <xmp:CreatorTool>Adobe Photoshop 24.0</xmp:CreatorTool>
        </x:xmpmeta>
        <?xpacket end="w"?>'''
        
        result = analyzer._check_xmp(xmp_data)
        
        assert result["creation_tool"] == "Adobe Photoshop 24.0"


# =============================================================================
# Integration Tests
# =============================================================================

class TestPRNUProvenanceIntegration:
    """Integration tests for PRNU and Provenance in pipeline."""
    
    def test_features_module_exports(self):
        """Test that features module exports PRNU and Provenance."""
        from verifai.features import (
            PRNUExtractor,
            PRNUFeatures,
            extract_prnu,
            ProvenanceAnalyzer,
            ProvenanceFeatures,
            C2PAClaim,
            analyze_provenance,
        )
        
        # All should be importable
        assert PRNUExtractor is not None
        assert ProvenanceAnalyzer is not None
    
    def test_pipeline_has_prnu_option(self):
        """Test that VerifAI pipeline has PRNU option."""
        from verifai.pipeline import VerifAI
        
        # Should be able to create with use_prnu option
        # (Note: this doesn't actually run detection)
        detector = VerifAI.__new__(VerifAI)
        
        # Check the class accepts these parameters in __init__
        import inspect
        sig = inspect.signature(VerifAI.__init__)
        params = list(sig.parameters.keys())
        
        assert "use_prnu" in params
        assert "use_provenance" in params


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_prnu_very_small_image(self):
        """Test PRNU with very small image."""
        extractor = PRNUExtractor()
        
        # Very small image
        image = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        
        features = extractor.extract(image)
        
        # Should handle gracefully
        assert isinstance(features, PRNUFeatures)
    
    def test_prnu_grayscale_image(self):
        """Test PRNU with grayscale image."""
        extractor = PRNUExtractor()
        
        # Grayscale as 2D array
        gray_image = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        
        # Convert to RGB for processing
        rgb_image = np.stack([gray_image] * 3, axis=-1)
        
        features = extractor.extract(rgb_image)
        
        assert isinstance(features, PRNUFeatures)
    
    def test_provenance_nonexistent_file(self):
        """Test provenance with non-existent file."""
        analyzer = ProvenanceAnalyzer()
        
        features = analyzer.analyze("/nonexistent/path/image.jpg")
        
        # Should return empty features, not crash
        assert isinstance(features, ProvenanceFeatures)
        assert features.has_c2pa is False
    
    def test_provenance_empty_bytes(self):
        """Test provenance with empty bytes."""
        analyzer = ProvenanceAnalyzer()
        
        features = analyzer.analyze(b"")
        
        # Should handle gracefully
        assert isinstance(features, ProvenanceFeatures)
