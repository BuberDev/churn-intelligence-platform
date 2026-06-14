"""Abstract base class for all churn prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelMetadata:
    """Tracks model identity and training provenance."""

    name: str
    version: str
    trained_at: datetime = field(default_factory=datetime.utcnow)
    params: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    mlflow_run_id: str | None = None


class BaseChurnModel(ABC):
    """Contract that every churn model must fulfil.

    Concrete subclasses must implement:
    - ``fit`` — train the model
    - ``predict_proba`` — return probability of churn (class 1)
    - ``metadata`` — expose model identity

    Prediction and threshold-based classification are provided here so all
    models behave consistently at the API layer.
    """

    DEFAULT_THRESHOLD = 0.50

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> BaseChurnModel:
        """Train the model.

        Args:
            X_train: Training feature matrix.
            y_train: Binary training target.
            X_val: Optional validation features (used for early stopping).
            y_val: Optional validation target.

        Returns:
            Self (fitted model).
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return churn probability for each sample.

        Args:
            X: Feature matrix with the same columns as training data.

        Returns:
            1-D array of churn probabilities in [0, 1].
        """

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Model name, version, and training provenance."""

    def predict(
        self, X: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
    ) -> np.ndarray:
        """Return binary churn predictions using the given threshold.

        Args:
            X: Feature matrix.
            threshold: Decision boundary — tune via ``Evaluator.optimal_threshold``.

        Returns:
            Binary array where 1 = predicted churner.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_with_confidence(
        self, X: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
    ) -> pd.DataFrame:
        """Return predictions with calibrated confidence tier labels.

        Returns:
            DataFrame with columns: ``churn_probability``, ``predicted_churned``,
            ``confidence`` (high / medium / low).
        """
        proba = self.predict_proba(X)
        predictions = (proba >= threshold).astype(int)

        confidence = np.where(
            proba >= 0.80, "high",
            np.where(proba >= 0.60, "medium", "low"),
        )

        return pd.DataFrame(
            {
                "churn_probability": proba.round(4),
                "predicted_churned": predictions,
                "confidence": confidence,
            },
            index=X.index,
        )
