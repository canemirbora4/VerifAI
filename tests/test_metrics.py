"""
Tests for the evaluation metrics module.
"""

import pytest
import numpy as np

from verifai.eval import (
    compute_metrics,
    compute_binary_metrics,
    compute_calibration_error,
    MetricsResult,
)
from verifai.eval.metrics import find_optimal_threshold


class TestMetricsResult:
    """Tests for MetricsResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic MetricsResult creation."""
        result = MetricsResult(
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1=0.865,
            roc_auc=0.95,
            pr_auc=0.92,
        )
        
        assert result.accuracy == 0.9
        assert result.roc_auc == 0.95
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MetricsResult(
            accuracy=0.85,
            precision=0.8,
            recall=0.9,
            f1=0.848,
            num_samples=100,
        )
        
        d = result.to_dict()
        
        assert d["accuracy"] == 0.85
        assert d["num_samples"] == 100
    
    def test_summary(self):
        """Test human-readable summary."""
        result = MetricsResult(
            accuracy=0.85,
            precision=0.8,
            recall=0.9,
            f1=0.848,
            roc_auc=0.92,
            pr_auc=0.88,
            ece=0.05,
            num_samples=100,
            num_positive=40,
            num_negative=60,
        )
        
        summary = result.summary()
        
        assert "Accuracy" in summary
        assert "0.85" in summary
        assert "ROC-AUC" in summary


class TestBinaryMetrics:
    """Tests for binary classification metrics."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.95])
        
        result = compute_binary_metrics(y_true, y_pred, y_prob)
        
        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
    
    def test_worst_predictions(self):
        """Test metrics with worst possible predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])  # All wrong
        
        result = compute_binary_metrics(y_true, y_pred)
        
        assert result.accuracy == 0.0
        assert result.recall == 0.0
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        result = compute_binary_metrics(y_true, y_pred)
        
        # Confusion matrix: [[TN, FP], [FN, TP]]
        assert result.confusion_matrix[0, 0] == 1  # TN
        assert result.confusion_matrix[0, 1] == 1  # FP
        assert result.confusion_matrix[1, 0] == 1  # FN
        assert result.confusion_matrix[1, 1] == 1  # TP
    
    def test_roc_auc_computation(self):
        """Test ROC-AUC calculation."""
        # Well-separated classes
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.35, 0.7, 0.8, 0.85, 0.9])
        
        result = compute_binary_metrics(y_true, (y_prob > 0.5).astype(int), y_prob)
        
        assert result.roc_auc > 0.9  # Should be high for well-separated classes
    
    def test_edge_case_all_positive(self):
        """Test with all positive samples."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        
        result = compute_binary_metrics(y_true, y_pred)
        
        assert result.num_positive == 4
        assert result.num_negative == 0
        assert result.recall == 1.0
    
    def test_edge_case_all_negative(self):
        """Test with all negative samples."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        
        result = compute_binary_metrics(y_true, y_pred)
        
        assert result.num_positive == 0
        assert result.num_negative == 4
        assert result.accuracy == 1.0


class TestCalibrationError:
    """Tests for calibration error computation."""
    
    def test_perfect_calibration(self):
        """Test ECE with perfectly calibrated probabilities."""
        # If we predict 0.8, we should be correct 80% of the time
        # This is hard to test exactly, but we can test extreme cases
        
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        ece, mce, _ = compute_calibration_error(y_true, y_prob, n_bins=10)
        
        # Perfect calibration (0 vs 1 probs match labels)
        assert ece < 0.2  # Should be low
    
    def test_poor_calibration(self):
        """Test ECE with poorly calibrated probabilities."""
        # All confident predictions that are wrong
        y_true = np.array([0, 0, 0, 0, 0])
        y_prob = np.array([0.95, 0.9, 0.85, 0.9, 0.95])  # High confidence but all wrong
        
        ece, mce, _ = compute_calibration_error(y_true, y_prob, n_bins=10)
        
        # Should have high calibration error
        assert ece > 0.5
    
    def test_calibration_curve_data(self):
        """Test that calibration curve data is returned."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        
        ece, mce, curve_data = compute_calibration_error(y_true, y_prob, n_bins=5)
        
        assert "bin_edges" in curve_data
        assert "bin_accuracies" in curve_data
        assert "bin_confidences" in curve_data
        assert "bin_counts" in curve_data
        assert len(curve_data["bin_edges"]) == 6  # n_bins + 1 edges


class TestComputeMetrics:
    """Tests for the compute_metrics convenience function."""
    
    def test_basic_usage(self):
        """Test basic usage with probabilities."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_prob = [0.2, 0.3, 0.7, 0.9, 0.4, 0.8]
        
        result = compute_metrics(y_true, y_prob, threshold=0.5)
        
        assert isinstance(result, MetricsResult)
        assert result.num_samples == 6
        assert result.threshold == 0.5
    
    def test_threshold_variation(self):
        """Test that different thresholds give different results."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.3, 0.4, 0.45, 0.55, 0.6, 0.7])
        
        result_low = compute_metrics(y_true, y_prob, threshold=0.3)
        result_high = compute_metrics(y_true, y_prob, threshold=0.7)
        
        # Lower threshold = more positive predictions
        # Higher threshold = fewer positive predictions
        assert result_low.recall >= result_high.recall


class TestFindOptimalThreshold:
    """Tests for optimal threshold finding."""
    
    def test_find_optimal_f1(self):
        """Test finding optimal threshold for F1."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.35, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        threshold, f1 = find_optimal_threshold(y_true, y_prob, metric="f1")
        
        assert 0.4 <= threshold <= 0.6  # Should be somewhere in the middle
        assert f1 > 0.8  # Should be reasonably high
    
    def test_find_optimal_accuracy(self):
        """Test finding optimal threshold for accuracy."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        threshold, acc = find_optimal_threshold(y_true, y_prob, metric="accuracy")
        
        assert acc >= 0.75  # Should achieve decent accuracy
    
    def test_find_optimal_youden(self):
        """Test finding optimal threshold using Youden's J."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        
        threshold, j = find_optimal_threshold(y_true, y_prob, metric="youden")
        
        assert 0.0 <= j <= 1.0  # Youden's J is in [0, 1] for good classifiers
