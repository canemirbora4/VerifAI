"""
Evaluation Metrics
===================

Comprehensive metrics for evaluating AI-generated media detection.
Includes standard classification metrics and calibration measures.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from loguru import logger


@dataclass
class MetricsResult:
    """
    Container for evaluation metrics results.
    
    Attributes:
        accuracy: Classification accuracy
        precision: Precision for positive class (AI-generated)
        recall: Recall for positive class (AI-generated)
        f1: F1 score for positive class
        roc_auc: Area under ROC curve
        pr_auc: Area under Precision-Recall curve
        ece: Expected Calibration Error
        confusion_matrix: Confusion matrix [[TN, FP], [FN, TP]]
        threshold: Classification threshold used
        num_samples: Total number of samples
        num_positive: Number of positive samples (AI-generated)
        num_negative: Number of negative samples (real)
    """
    
    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # Ranking metrics
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    
    # Calibration metrics
    ece: float = 0.0
    mce: float = 0.0  # Maximum Calibration Error
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    
    # Curve data for plotting
    roc_curve: Optional[dict] = None
    pr_curve: Optional[dict] = None
    calibration_curve: Optional[dict] = None
    
    # Metadata
    threshold: float = 0.5
    num_samples: int = 0
    num_positive: int = 0
    num_negative: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "ece": self.ece,
            "mce": self.mce,
            "threshold": self.threshold,
            "num_samples": self.num_samples,
            "num_positive": self.num_positive,
            "num_negative": self.num_negative,
        }
        
        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix.tolist()
        
        return result
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 50,
            "Evaluation Results",
            "=" * 50,
            f"Samples: {self.num_samples} (positive: {self.num_positive}, negative: {self.num_negative})",
            f"Threshold: {self.threshold:.3f}",
            "-" * 50,
            "Classification Metrics:",
            f"  Accuracy:  {self.accuracy:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall:    {self.recall:.4f}",
            f"  F1 Score:  {self.f1:.4f}",
            "-" * 50,
            "Ranking Metrics:",
            f"  ROC-AUC:   {self.roc_auc:.4f}",
            f"  PR-AUC:    {self.pr_auc:.4f}",
            "-" * 50,
            "Calibration Metrics:",
            f"  ECE:       {self.ece:.4f}",
            f"  MCE:       {self.mce:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> MetricsResult:
    """
    Compute binary classification metrics.
    
    Args:
        y_true: Ground truth labels (0 = real, 1 = AI-generated)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities for positive class (optional)
        threshold: Classification threshold
        
    Returns:
        MetricsResult with computed metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # If y_prob provided, compute threshold-based predictions
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= threshold).astype(int)
    
    # Basic counts
    num_samples = len(y_true)
    num_positive = int(np.sum(y_true == 1))
    num_negative = int(np.sum(y_true == 0))
    
    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Handle edge cases for precision/recall
    if num_positive == 0:
        logger.warning("No positive samples in ground truth")
        prec = 0.0
        rec = 0.0
        f1 = 0.0
    else:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Ranking metrics (require probabilities)
    roc_auc = 0.0
    pr_auc = 0.0
    roc_curve_data = None
    pr_curve_data = None
    
    if y_prob is not None and num_positive > 0 and num_negative > 0:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            
            # Compute curves for plotting
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
            roc_curve_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
            }
            
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                y_true, y_prob
            )
            pr_curve_data = {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist(),
                "thresholds": pr_thresholds.tolist(),
            }
            
        except Exception as e:
            logger.warning(f"Could not compute ranking metrics: {e}")
    
    # Calibration metrics (require probabilities)
    ece = 0.0
    mce = 0.0
    calibration_curve_data = None
    
    if y_prob is not None:
        ece, mce, calibration_curve_data = compute_calibration_error(
            y_true, y_prob, n_bins=10
        )
    
    return MetricsResult(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        ece=float(ece),
        mce=float(mce),
        confusion_matrix=cm,
        roc_curve=roc_curve_data,
        pr_curve=pr_curve_data,
        calibration_curve=calibration_curve_data,
        threshold=threshold,
        num_samples=num_samples,
        num_positive=num_positive,
        num_negative=num_negative,
    )


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, dict]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    ECE measures how well the predicted probabilities match actual outcomes.
    A well-calibrated model has ECE close to 0.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
        
    Returns:
        Tuple of (ECE, MCE, calibration_curve_data)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    
    # Compute statistics per bin
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_count = np.sum(mask)
        else:
            bin_acc = 0.0
            bin_conf = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_count = 0
        
        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
        bin_counts.append(bin_count)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    # Compute ECE (weighted average of |accuracy - confidence|)
    total_samples = np.sum(bin_counts)
    if total_samples > 0:
        ece = np.sum(
            bin_counts * np.abs(bin_accuracies - bin_confidences)
        ) / total_samples
    else:
        ece = 0.0
    
    # Compute MCE (maximum |accuracy - confidence|)
    gaps = np.abs(bin_accuracies - bin_confidences)
    # Only consider bins with samples
    non_empty = bin_counts > 0
    if np.any(non_empty):
        mce = np.max(gaps[non_empty])
    else:
        mce = 0.0
    
    # Calibration curve data for plotting
    calibration_data = {
        "bin_edges": bin_edges.tolist(),
        "bin_accuracies": bin_accuracies.tolist(),
        "bin_confidences": bin_confidences.tolist(),
        "bin_counts": bin_counts.tolist(),
    }
    
    return float(ece), float(mce), calibration_data


def compute_metrics(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> MetricsResult:
    """
    Convenience function to compute all metrics from probabilities.
    
    Args:
        y_true: Ground truth labels (0 = real, 1 = AI-generated)
        y_prob: Predicted probabilities for AI-generated class
        threshold: Classification threshold
        
    Returns:
        MetricsResult with all computed metrics
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    return compute_binary_metrics(y_true, y_pred, y_prob, threshold)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """
    Find the optimal threshold that maximizes a given metric.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ("f1", "accuracy", "youden")
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    # Test thresholds
    thresholds = np.linspace(0.1, 0.9, 81)
    
    best_threshold = 0.5
    best_value = 0.0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        if metric == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            value = accuracy_score(y_true, y_pred)
        elif metric == "youden":
            # Youden's J statistic = TPR + TNR - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            value = tpr + tnr - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if value > best_value:
            best_value = value
            best_threshold = thresh
    
    return float(best_threshold), float(best_value)
