"""
Tests for the fusion module.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from verifai.fusion import (
    Ensemble,
    EnsembleConfig,
    EnsembleOutput,
    FusionMethod,
    Calibrator,
    CalibrationResult,
    calibrate_scores,
    Explainer,
    ExplanationResult,
    create_metadata_detector_output,
)
from verifai.models.base import DetectorOutput, Label


class TestEnsemble:
    """Tests for ensemble fusion."""
    
    @pytest.fixture
    def ensemble(self):
        """Create an ensemble with default config."""
        return Ensemble()
    
    @pytest.fixture
    def detector_outputs(self):
        """Create sample detector outputs."""
        return {
            "neural": DetectorOutput(
                raw_score=0.8,
                confidence=0.8,
                label=Label.AI_GENERATED,
            ),
            "frequency": DetectorOutput(
                raw_score=0.6,
                confidence=0.6,
                label=Label.AI_GENERATED,
            ),
            "metadata": DetectorOutput(
                raw_score=0.3,
                confidence=0.3,
                label=Label.REAL,
            ),
        }
    
    def test_ensemble_initialization(self, ensemble):
        """Test ensemble initialization."""
        assert ensemble.config is not None
        assert ensemble.config.method == FusionMethod.WEIGHTED
    
    def test_fuse_weighted(self, ensemble, detector_outputs):
        """Test weighted fusion."""
        result = ensemble.fuse(detector_outputs)
        
        assert isinstance(result, EnsembleOutput)
        assert 0 <= result.final_score <= 1
        assert result.final_label in [Label.REAL, Label.AI_GENERATED]
    
    def test_fuse_average(self, detector_outputs):
        """Test average fusion."""
        config = EnsembleConfig(method=FusionMethod.AVERAGE)
        ensemble = Ensemble(config)
        
        result = ensemble.fuse(detector_outputs)
        
        # Average should be (0.8 + 0.6 + 0.3) / 3 ≈ 0.567
        expected = (0.8 + 0.6 + 0.3) / 3
        assert abs(result.final_score - expected) < 0.01
    
    def test_fuse_max(self, detector_outputs):
        """Test max fusion."""
        config = EnsembleConfig(method=FusionMethod.MAX)
        ensemble = Ensemble(config)
        
        result = ensemble.fuse(detector_outputs)
        
        assert result.final_score == 0.8  # Max is neural's 0.8
    
    def test_fuse_empty(self, ensemble):
        """Test fusion with no outputs."""
        result = ensemble.fuse({})
        
        assert result.final_score == 0.5  # Neutral
        assert result.final_label == Label.UNCERTAIN
    
    def test_detector_scores_preserved(self, ensemble, detector_outputs):
        """Test that individual scores are preserved."""
        result = ensemble.fuse(detector_outputs)
        
        assert "neural" in result.detector_scores
        assert "frequency" in result.detector_scores
        assert result.detector_scores["neural"] == 0.8
    
    def test_update_weights(self, ensemble):
        """Test updating weights."""
        ensemble.update_weights({"neural": 0.9, "frequency": 0.1})
        
        # Weights should be normalized
        total = sum(ensemble.config.weights.values())
        assert abs(total - 1.0) < 0.01
    
    def test_add_detector(self, ensemble):
        """Test adding a detector."""
        ensemble.add_detector("new_detector", weight=0.2)
        
        assert "new_detector" in ensemble.config.detectors
        assert "new_detector" in ensemble.config.weights
    
    def test_remove_detector(self, ensemble):
        """Test removing a detector."""
        ensemble.remove_detector("metadata")
        
        assert "metadata" not in ensemble.config.detectors


class TestCalibrator:
    """Tests for probability calibration."""
    
    @pytest.fixture
    def scores(self):
        """Sample uncalibrated scores."""
        return np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    @pytest.fixture
    def labels(self):
        """Sample labels."""
        return np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    def test_isotonic_calibrator(self, scores, labels):
        """Test isotonic calibration."""
        calibrator = Calibrator(method="isotonic")
        calibrator.fit(scores, labels)
        
        assert calibrator.is_fitted
        
        # Calibrate a score
        calibrated = calibrator.calibrate(0.5)
        assert 0 <= calibrated <= 1
    
    def test_platt_calibrator(self, scores, labels):
        """Test Platt scaling."""
        calibrator = Calibrator(method="platt")
        calibrator.fit(scores, labels)
        
        assert calibrator.is_fitted
        
        calibrated = calibrator.calibrate(0.5)
        assert 0 <= calibrated <= 1
    
    def test_temperature_calibrator(self, scores, labels):
        """Test temperature scaling."""
        calibrator = Calibrator(method="temperature")
        calibrator.fit(scores, labels)
        
        assert calibrator.is_fitted
        assert hasattr(calibrator, "_temperature")
    
    def test_no_calibration(self):
        """Test 'none' calibration method."""
        calibrator = Calibrator(method="none")
        
        result = calibrator.calibrate(0.7)
        assert result == 0.7  # No change
    
    def test_calibrate_array(self, scores, labels):
        """Test calibrating array of scores."""
        calibrator = Calibrator(method="isotonic")
        calibrator.fit(scores, labels)
        
        test_scores = np.array([0.3, 0.5, 0.7])
        calibrated = calibrator.calibrate(test_scores)
        
        assert len(calibrated) == 3
        assert all(0 <= s <= 1 for s in calibrated)
    
    def test_calibrate_with_result(self, scores, labels):
        """Test calibrate_with_result method."""
        calibrator = Calibrator(method="isotonic")
        calibrator.fit(scores, labels)
        
        result = calibrator.calibrate_with_result(0.6)
        
        assert isinstance(result, CalibrationResult)
        assert result.original_score == 0.6
        assert result.method == "isotonic"
    
    def test_save_and_load(self, scores, labels, tmp_path):
        """Test saving and loading calibrator."""
        calibrator = Calibrator(method="isotonic")
        calibrator.fit(scores, labels)
        
        # Save
        path = tmp_path / "calibrator.pkl"
        calibrator.save(path)
        
        # Load
        loaded = Calibrator.load(path)
        
        assert loaded.is_fitted
        assert loaded.method == "isotonic"
        
        # Check same result
        orig = calibrator.calibrate(0.5)
        loaded_result = loaded.calibrate(0.5)
        assert abs(orig - loaded_result) < 0.01
    
    def test_convenience_function(self, scores, labels):
        """Test calibrate_scores convenience function."""
        calibrator, calibrated = calibrate_scores(scores, labels, method="isotonic")
        
        assert calibrator.is_fitted
        assert len(calibrated) == len(scores)


class TestExplainer:
    """Tests for explanation generation."""
    
    @pytest.fixture
    def explainer(self):
        """Create an explainer."""
        return Explainer(alpha=0.5)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        from PIL import Image
        return Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
    
    def test_explainer_initialization(self, explainer):
        """Test explainer initialization."""
        assert explainer.alpha == 0.5
    
    def test_explain_without_model(self, explainer, sample_image):
        """Test explanation without model (fallback)."""
        result = explainer.explain(sample_image)
        
        assert isinstance(result, ExplanationResult)
        assert result.heatmap is not None
        assert result.overlay is not None
    
    def test_heatmap_shape(self, explainer, sample_image):
        """Test heatmap shape matches image."""
        result = explainer.explain(sample_image)
        
        assert result.heatmap.shape == (224, 224)
    
    def test_heatmap_values(self, explainer, sample_image):
        """Test heatmap values are in [0, 1]."""
        result = explainer.explain(sample_image)
        
        assert result.heatmap.min() >= 0
        assert result.heatmap.max() <= 1


class TestMetadataDetectorOutput:
    """Tests for metadata to detector output conversion."""
    
    def test_create_from_suspicious(self):
        """Test creating output from suspicious metadata."""
        from verifai.features import MetadataFeatures
        
        features = MetadataFeatures(
            has_exif=False,
            confidence_real=0.2,  # Low = suspicious
            is_suspicious=True,
        )
        
        output = create_metadata_detector_output(features)
        
        assert isinstance(output, DetectorOutput)
        assert output.confidence == 0.8  # 1 - 0.2
        assert output.label == Label.AI_GENERATED
    
    def test_create_from_confident(self):
        """Test creating output from confident real metadata."""
        from verifai.features import MetadataFeatures
        
        features = MetadataFeatures(
            has_exif=True,
            has_camera_info=True,
            confidence_real=0.9,
            is_suspicious=False,
        )
        
        output = create_metadata_detector_output(features)
        
        assert output.confidence == 0.1  # 1 - 0.9
        assert output.label == Label.REAL
