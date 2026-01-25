"""
Tests for the models module.
"""

import pytest
import torch
import numpy as np

from verifai.models.base import BaseDetector, DetectorOutput, Label


class TestLabel:
    """Tests for the Label enum."""
    
    def test_label_values(self):
        """Test label enum values."""
        assert Label.REAL.value == "real"
        assert Label.AI_GENERATED.value == "ai_generated"
        assert Label.UNCERTAIN.value == "uncertain"


class TestDetectorOutput:
    """Tests for DetectorOutput dataclass."""
    
    def test_basic_creation(self):
        """Test basic DetectorOutput creation."""
        output = DetectorOutput(
            raw_score=0.8,
            confidence=0.8,
            label=Label.AI_GENERATED,
        )
        
        assert output.raw_score == 0.8
        assert output.confidence == 0.8
        assert output.label == Label.AI_GENERATED
    
    def test_confidence_clipping(self):
        """Test that confidence is clipped to [0, 1]."""
        output = DetectorOutput(
            raw_score=1.5,
            confidence=1.5,
            label=Label.AI_GENERATED,
        )
        assert output.confidence == 1.0
        
        output = DetectorOutput(
            raw_score=-0.5,
            confidence=-0.5,
            label=Label.REAL,
        )
        assert output.confidence == 0.0
    
    def test_default_probabilities(self):
        """Test that default probabilities are computed."""
        output = DetectorOutput(
            raw_score=0.7,
            confidence=0.7,
            label=Label.AI_GENERATED,
        )
        
        assert "real" in output.probabilities
        assert "ai_generated" in output.probabilities
        assert output.probabilities["ai_generated"] == 0.7
        assert output.probabilities["real"] == 0.3
    
    def test_is_ai_generated(self):
        """Test is_ai_generated property."""
        ai_output = DetectorOutput(
            raw_score=0.8,
            confidence=0.8,
            label=Label.AI_GENERATED,
        )
        assert ai_output.is_ai_generated is True
        
        real_output = DetectorOutput(
            raw_score=0.2,
            confidence=0.2,
            label=Label.REAL,
        )
        assert real_output.is_ai_generated is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = DetectorOutput(
            raw_score=0.75,
            confidence=0.75,
            label=Label.AI_GENERATED,
            metadata={"model": "test"},
        )
        
        d = output.to_dict()
        
        assert d["raw_score"] == 0.75
        assert d["confidence"] == 0.75
        assert d["label"] == "ai_generated"
        assert d["metadata"]["model"] == "test"
    
    def test_features_array(self):
        """Test storing numpy features."""
        features = np.random.randn(768).astype(np.float32)
        
        output = DetectorOutput(
            raw_score=0.5,
            confidence=0.5,
            label=Label.UNCERTAIN,
            features=features,
        )
        
        assert output.features is not None
        assert output.features.shape == (768,)


class TestBaseDetectorInterface:
    """Tests for BaseDetector abstract interface."""
    
    def test_device_resolution_auto(self):
        """Test automatic device resolution."""
        # Create a concrete implementation for testing
        class DummyDetector(BaseDetector):
            def load(self):
                self._is_loaded = True
            
            def detect(self, input_tensor, return_features=False, return_evidence=False):
                return DetectorOutput(
                    raw_score=0.5,
                    confidence=0.5,
                    label=Label.UNCERTAIN,
                )
        
        detector = DummyDetector(device=None)
        
        # Should resolve to one of the valid devices
        assert detector.device.type in ["cuda", "mps", "cpu"]
    
    def test_device_resolution_explicit(self):
        """Test explicit device specification."""
        class DummyDetector(BaseDetector):
            def load(self):
                self._is_loaded = True
            
            def detect(self, input_tensor, return_features=False, return_evidence=False):
                return DetectorOutput(
                    raw_score=0.5,
                    confidence=0.5,
                    label=Label.UNCERTAIN,
                )
        
        detector = DummyDetector(device="cpu")
        assert detector.device.type == "cpu"
    
    def test_score_to_label(self):
        """Test score to label conversion."""
        class DummyDetector(BaseDetector):
            def load(self):
                pass
            
            def detect(self, input_tensor, return_features=False, return_evidence=False):
                pass
        
        detector = DummyDetector(threshold=0.5)
        
        assert detector._score_to_label(0.7) == Label.AI_GENERATED
        assert detector._score_to_label(0.5) == Label.AI_GENERATED
        assert detector._score_to_label(0.3) == Label.REAL
    
    def test_repr(self):
        """Test string representation."""
        class DummyDetector(BaseDetector):
            def load(self):
                pass
            
            def detect(self, input_tensor, return_features=False, return_evidence=False):
                pass
        
        detector = DummyDetector(name="test_detector", device="cpu")
        repr_str = repr(detector)
        
        assert "DummyDetector" in repr_str
        assert "test_detector" in repr_str
        assert "cpu" in repr_str


class TestNeuralDetector:
    """Tests for NeuralDetector (integration tests - may require model download)."""
    
    @pytest.fixture
    def sample_tensor(self):
        """Create a sample input tensor."""
        return torch.randn(3, 224, 224)
    
    @pytest.fixture
    def batch_tensor(self):
        """Create a batch of sample tensors."""
        return torch.randn(4, 3, 224, 224)
    
    @pytest.mark.slow
    def test_neural_detector_load(self):
        """Test loading the neural detector model."""
        from verifai.models import NeuralDetector
        
        detector = NeuralDetector(
            model_name="google/vit-base-patch16-224",
            device="cpu",
            fp16=False,  # CPU doesn't support FP16
        )
        
        assert not detector.is_loaded
        detector.load()
        assert detector.is_loaded
    
    @pytest.mark.slow
    def test_neural_detector_inference(self, sample_tensor):
        """Test running inference with neural detector."""
        from verifai.models import NeuralDetector
        
        detector = NeuralDetector(
            model_name="google/vit-base-patch16-224",
            device="cpu",
            fp16=False,
        )
        detector.load()
        
        result = detector.detect(sample_tensor)
        
        assert isinstance(result, DetectorOutput)
        assert 0 <= result.confidence <= 1
        assert result.label in [Label.REAL, Label.AI_GENERATED]
    
    @pytest.mark.slow
    def test_neural_detector_batch(self, batch_tensor):
        """Test batch inference with neural detector."""
        from verifai.models import NeuralDetector
        
        detector = NeuralDetector(
            model_name="google/vit-base-patch16-224",
            device="cpu",
            fp16=False,
        )
        detector.load()
        
        results = detector.detect_batch(batch_tensor)
        
        assert len(results) == 4
        for result in results:
            assert isinstance(result, DetectorOutput)
    
    @pytest.mark.slow
    def test_neural_detector_with_features(self, sample_tensor):
        """Test extracting features from neural detector."""
        from verifai.models import NeuralDetector
        
        detector = NeuralDetector(
            model_name="google/vit-base-patch16-224",
            device="cpu",
            fp16=False,
        )
        detector.load()
        
        result = detector.detect(sample_tensor, return_features=True)
        
        assert result.features is not None
        assert isinstance(result.features, np.ndarray)
    
    @pytest.mark.slow
    def test_neural_detector_model_info(self):
        """Test getting model information."""
        from verifai.models import NeuralDetector
        
        detector = NeuralDetector(
            model_name="google/vit-base-patch16-224",
            device="cpu",
        )
        
        # Before loading
        info = detector.get_model_info()
        assert info["loaded"] is False
        
        # After loading
        detector.load()
        info = detector.get_model_info()
        assert info["loaded"] is True
        assert info["total_parameters"] > 0
