"""Random Forest classifier with Optuna hyperparameter tuning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseChurnModel, ModelMetadata


class RandomForestModel(BaseChurnModel):
    """Ensemble tree model with optional Optuna-based hyperparameter search.

    Random Forest is robust to outliers and provides built-in feature
    importance via mean decrease in impurity (MDI).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 12,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self._params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        self._model = RandomForestClassifier(**self._params)
        self._metadata = ModelMetadata(
            name="random_forest",
            version="1.0.0",
            params=self._params,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> RandomForestModel:
        logger.info(f"Training RandomForest with {self._params['n_estimators']} trees")
        self._model.fit(X_train, y_train)
        self._metadata.feature_names = X_train.columns.tolist()
        logger.info("RandomForest training complete")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def feature_importance(self) -> pd.Series:
        """Return MDI-based feature importances sorted descending."""
        return (
            pd.Series(
                self._model.feature_importances_,
                index=self._metadata.feature_names,
            )
            .sort_values(ascending=False)
        )

    @classmethod
    def tune(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> RandomForestModel:
        """Find optimal hyperparameters with Optuna and return a fitted model.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features for AUC scoring.
            y_val: Validation target.
            n_trials: Number of Optuna trials.
            random_state: Seed for reproducibility.

        Returns:
            Fitted ``RandomForestModel`` with best hyperparameters.
        """
        import optuna
        from sklearn.metrics import roc_auc_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 4, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "class_weight": "balanced",
                "n_jobs": -1,
                "random_state": random_state,
            }
            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)
            proba = rf.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, proba)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        logger.info(f"Best RandomForest AUC: {study.best_value:.4f} | params: {best}")

        model = cls(**best, random_state=random_state)
        model.fit(X_train, y_train)
        return model
