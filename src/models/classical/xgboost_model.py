"""XGBoost model — typically the best-performing classical baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

from src.models.base import BaseChurnModel, ModelMetadata


class XGBoostModel(BaseChurnModel):
    """Gradient-boosted trees with early stopping and scale_pos_weight.

    ``scale_pos_weight`` is set to handle class imbalance without
    undersampling, preserving all training signal.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 3.0,
        eval_metric: str = "auc",
        early_stopping_rounds: int = 30,
        random_state: int = 42,
    ) -> None:
        self._params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": eval_metric,
            "early_stopping_rounds": early_stopping_rounds,
            "random_state": random_state,
            "use_label_encoder": False,
            "verbosity": 0,
        }
        self._model = xgb.XGBClassifier(**self._params)
        self._metadata = ModelMetadata(
            name="xgboost",
            version="1.0.0",
            params=self._params,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> XGBoostModel:
        has_val = X_val is not None and y_val is not None
        eval_set = [(X_val, y_val)] if has_val else None
        logger.info("Training XGBoost")

        # early_stopping_rounds requires eval_set — disable when no validation data
        if not has_val and self._model.early_stopping_rounds is not None:
            self._model.set_params(early_stopping_rounds=None)

        self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self._metadata.feature_names = X_train.columns.tolist()
        best_iter = getattr(self._model, "best_iteration", None)
        logger.info(f"XGBoost training complete — best iteration: {best_iter}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Return XGBoost feature importances.

        Args:
            importance_type: One of ``"gain"``, ``"weight"``, ``"cover"``.
        """
        scores = self._model.get_booster().get_score(importance_type=importance_type)
        return (
            pd.Series(scores)
            .reindex(self._metadata.feature_names, fill_value=0)
            .sort_values(ascending=False)
        )

    @classmethod
    def tune(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 100,
        random_state: int = 42,
    ) -> XGBoostModel:
        """Tune hyperparameters with Optuna and return the best fitted model."""
        import optuna
        from sklearn.metrics import roc_auc_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
                "random_state": random_state,
                "verbosity": 0,
                "use_label_encoder": False,
                "eval_metric": "auc",
                "early_stopping_rounds": 20,
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        logger.info(f"Best XGBoost AUC: {study.best_value:.4f} | params: {best}")

        model = cls(**best, random_state=random_state)
        model.fit(X_train, y_train, X_val, y_val)
        return model
