"""Logistic Regression baseline — fast, interpretable, and production-safe."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseChurnModel, ModelMetadata


class LogisticRegressionModel(BaseChurnModel):
    """Regularised Logistic Regression with class-weight balancing.

    Serves as a fast, interpretable baseline. Coefficient magnitudes
    provide direct insight into feature contributions, which is useful
    for stakeholder communication.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1_000,
        solver: str = "lbfgs",
        class_weight: str = "balanced",
        random_state: int = 42,
    ) -> None:
        self._params = {
            "C": C,
            "max_iter": max_iter,
            "solver": solver,
            "class_weight": class_weight,
            "random_state": random_state,
        }
        self._model = LogisticRegression(**self._params)
        self._metadata = ModelMetadata(
            name="logistic_regression",
            version="1.0.0",
            params=self._params,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> LogisticRegressionModel:
        self._model.fit(X_train, y_train)
        self._metadata.feature_names = X_train.columns.tolist()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def feature_coefficients(self) -> pd.Series:
        """Return feature coefficients sorted by absolute magnitude."""
        return (
            pd.Series(
                self._model.coef_[0],
                index=self._metadata.feature_names,
            )
            .abs()
            .sort_values(ascending=False)
        )
