"""
Probability Calibration
========================

Calibrates model outputs to produce reliable probability estimates.

Why calibration matters:
- Raw model outputs are often overconfident or underconfident
- "85% confident" should mean ~85% accuracy at that confidence level
- Proper calibration enables rational decision-making

Calibration methods:
1. Isotonic Regression - Non-parametric, flexible
2. Platt Scaling - Parametric (sigmoid), simpler
3. Temperature Scaling - Single parameter, preserves ranking

This module provides calibration that can be fitted on validation data
and applied during inference.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Union
import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from loguru import logger


CalibrationMethod = Literal["isotonic", "platt", "temperature", "none"]


@dataclass
class CalibrationResult:
    """
    Result of applying calibration.
    
    Attributes:
        original_score: Score before calibration
        calibrated_score: Score after calibration
        method: Calibration method used
    """
    original_score: float
    calibrated_score: float
    method: str


class Calibrator:
    """
    Calibrates probability scores from detectors.
    
    Usage:
        # Fitting
        calibrator = Calibrator(method="isotonic")
        calibrator.fit(val_scores, val_labels)
        calibrator.save("calibration.pkl")
        
        # Inference
        calibrator = Calibrator.load("calibration.pkl")
        calibrated = calibrator.calibrate(0.73)
    """
    
    def __init__(
        self,
        method: CalibrationMethod = "isotonic",
    ):
        """
        Initialize the calibrator.
        
        Args:
            method: Calibration method to use
        """
        self.method = method
        self._is_fitted = False
        
        # Initialize calibration model based on method
        if method == "isotonic":
            self._model = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
        elif method == "platt":
            self._model = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
            )
        elif method == "temperature":
            self._temperature = 1.0
            self._model = None
        elif method == "none":
            self._model = None
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        logger.debug(f"Calibrator initialized with method: {method}")
    
    @property
    def is_fitted(self) -> bool:
        """Check if calibrator has been fitted."""
        return self._is_fitted
    
    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> "Calibrator":
        """
        Fit the calibration model.
        
        Args:
            scores: Raw model scores (N,)
            labels: Ground truth labels (N,) - 0 for real, 1 for AI
            
        Returns:
            self for chaining
        """
        scores = np.asarray(scores).ravel()
        labels = np.asarray(labels).ravel()
        
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have same length")
        
        if len(scores) < 10:
            logger.warning("Very few samples for calibration fitting")
        
        logger.info(f"Fitting calibrator on {len(scores)} samples")
        
        if self.method == "isotonic":
            self._model.fit(scores, labels)
            
        elif self.method == "platt":
            # Platt scaling: fit logistic regression on scores
            self._model.fit(scores.reshape(-1, 1), labels)
            
        elif self.method == "temperature":
            # Temperature scaling: find optimal temperature
            self._temperature = self._find_optimal_temperature(scores, labels)
            logger.info(f"Optimal temperature: {self._temperature:.4f}")
            
        elif self.method == "none":
            pass  # No fitting needed
        
        self._is_fitted = True
        return self
    
    def _find_optimal_temperature(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        num_temps: int = 100,
    ) -> float:
        """Find optimal temperature for temperature scaling."""
        # Search over temperatures
        temperatures = np.linspace(0.1, 5.0, num_temps)
        
        best_temp = 1.0
        best_ece = float("inf")
        
        for temp in temperatures:
            # Apply temperature scaling
            calibrated = self._apply_temperature(scores, temp)
            
            # Compute ECE
            ece = self._compute_ece(calibrated, labels)
            
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        
        return best_temp
    
    def _apply_temperature(self, scores: np.ndarray, temp: float) -> np.ndarray:
        """Apply temperature scaling to scores."""
        # Convert to logits, scale, convert back
        # Clamp to avoid log(0)
        scores = np.clip(scores, 1e-7, 1 - 1e-7)
        logits = np.log(scores / (1 - scores))
        scaled_logits = logits / temp
        return 1 / (1 + np.exp(-scaled_logits))
    
    def _compute_ece(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(scores, bin_edges[1:-1])
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(labels[mask])
                bin_confidence = np.mean(scores[mask])
                bin_size = np.sum(mask) / len(scores)
                ece += bin_size * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def calibrate(
        self,
        score: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calibrate a score or array of scores.
        
        Args:
            score: Raw score(s) to calibrate
            
        Returns:
            Calibrated score(s)
        """
        if not self._is_fitted and self.method != "none":
            logger.warning("Calibrator not fitted, returning original scores")
            return score
        
        is_scalar = np.isscalar(score)
        scores = np.atleast_1d(score)
        
        if self.method == "isotonic":
            calibrated = self._model.predict(scores)
            
        elif self.method == "platt":
            # Platt scaling returns probability from logistic regression
            calibrated = self._model.predict_proba(scores.reshape(-1, 1))[:, 1]
            
        elif self.method == "temperature":
            calibrated = self._apply_temperature(scores, self._temperature)
            
        elif self.method == "none":
            calibrated = scores
        
        else:
            calibrated = scores
        
        # Clip to [0, 1]
        calibrated = np.clip(calibrated, 0.0, 1.0)
        
        if is_scalar:
            return float(calibrated[0])
        return calibrated
    
    def calibrate_with_result(self, score: float) -> CalibrationResult:
        """
        Calibrate a score and return detailed result.
        
        Args:
            score: Raw score to calibrate
            
        Returns:
            CalibrationResult with original and calibrated scores
        """
        calibrated = self.calibrate(score)
        return CalibrationResult(
            original_score=score,
            calibrated_score=calibrated,
            method=self.method,
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save calibrator to file.
        
        Args:
            path: Path to save file
        """
        path = Path(path)
        
        state = {
            "method": self.method,
            "is_fitted": self._is_fitted,
        }
        
        if self.method == "isotonic":
            state["model"] = self._model
        elif self.method == "platt":
            state["model"] = self._model
        elif self.method == "temperature":
            state["temperature"] = self._temperature
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved calibrator to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Calibrator":
        """
        Load calibrator from file.
        
        Args:
            path: Path to saved file
            
        Returns:
            Loaded Calibrator
        """
        path = Path(path)
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        calibrator = cls(method=state["method"])
        calibrator._is_fitted = state["is_fitted"]
        
        if state["method"] == "isotonic":
            calibrator._model = state["model"]
        elif state["method"] == "platt":
            calibrator._model = state["model"]
        elif state["method"] == "temperature":
            calibrator._temperature = state["temperature"]
        
        logger.info(f"Loaded calibrator from {path}")
        return calibrator
    
    def get_info(self) -> dict:
        """Get calibrator information."""
        info = {
            "method": self.method,
            "is_fitted": self._is_fitted,
        }
        
        if self.method == "temperature" and self._is_fitted:
            info["temperature"] = self._temperature
        
        return info


def calibrate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    method: CalibrationMethod = "isotonic",
) -> tuple[Calibrator, np.ndarray]:
    """
    Convenience function to fit calibrator and transform scores.
    
    Args:
        scores: Raw scores
        labels: Ground truth labels
        method: Calibration method
        
    Returns:
        Tuple of (fitted calibrator, calibrated scores)
    """
    calibrator = Calibrator(method=method)
    calibrator.fit(scores, labels)
    calibrated = calibrator.calibrate(scores)
    return calibrator, calibrated
