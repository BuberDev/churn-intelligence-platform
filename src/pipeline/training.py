"""End-to-end training pipeline with MLflow experiment tracking."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

from src.data.preprocessor import TARGET_COLUMN, DataPreprocessor
from src.data.validator import DataValidator
from src.evaluation.metrics import BusinessMetrics, EvaluationResult, ModelEvaluator
from src.features.engineering import FeatureEngineer
from src.models.base import BaseChurnModel
from src.models.classical import LogisticRegressionModel, RandomForestModel, XGBoostModel
from src.models.deep_learning import NeuralNetworkModel, TabularTransformerModel


class TrainingPipeline:
    """Orchestrates the full training lifecycle.

    Steps:
    1. Load and validate data
    2. Feature engineering
    3. Preprocessing (split → fit_transform)
    4. Train enabled models
    5. Evaluate on held-out test set
    6. Log everything to MLflow
    7. Persist best model artifact

    Each model run is a nested MLflow run under the parent experiment run.
    """

    def __init__(self, config_path: str | Path) -> None:
        with open(config_path) as f:
            self._cfg: dict[str, Any] = yaml.safe_load(f)

        mlflow.set_tracking_uri(self._cfg["experiment"]["tracking_uri"])
        mlflow.set_experiment(self._cfg["experiment"]["name"])

        self._evaluator = ModelEvaluator(
            business_metrics=BusinessMetrics(
                cost_false_negative=self._cfg["evaluation"]["business_metrics"][
                    "cost_false_negative"
                ],
                cost_false_positive=self._cfg["evaluation"]["business_metrics"][
                    "cost_false_positive"
                ],
            )
        )
        self._validator = DataValidator()
        self._preprocessor = DataPreprocessor()
        self._engineer = FeatureEngineer()

    def run(self, df: pd.DataFrame) -> list[EvaluationResult]:
        """Execute the full training pipeline.

        Args:
            df: Raw customer DataFrame (output of DataLoader or SyntheticDataGenerator).

        Returns:
            List of evaluation results, one per trained model.
        """
        report = self._validator.validate(df)
        if not report.passed:
            raise ValueError("Data validation failed — check the ValidationReport logs")

        df = self._engineer.fit_transform(df)

        X = df.drop(columns=[TARGET_COLUMN, "customer_id"], errors="ignore")
        y = df[TARGET_COLUMN]

        cfg_data = self._cfg["data"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg_data["test_size"],
            random_state=cfg_data["random_state"],
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=cfg_data["val_size"],
            random_state=cfg_data["random_state"],
            stratify=y_train,
        )

        X_train_p, _ = self._preprocessor.fit_transform(X_train)
        X_val_p = self._preprocessor.transform(X_val)
        X_test_p = self._preprocessor.transform(X_test)

        results: list[EvaluationResult] = []

        with mlflow.start_run(run_name="churn-training"):
            mlflow.log_params({"n_train": len(X_train), "n_test": len(X_test)})

            for model in self._build_models():
                result = self._train_and_evaluate(
                    model, X_train_p, y_train, X_val_p, y_val, X_test_p, y_test
                )
                results.append(result)

            best = max(results, key=lambda r: r.roc_auc)
            logger.info(
                f"Best model: {best.model_name} — AUC {best.roc_auc:.4f}"
            )
            mlflow.set_tag("best_model", best.model_name)
            mlflow.set_tag("best_roc_auc", f"{best.roc_auc:.4f}")

        return results

    # ------------------------------------------------------------------

    def _build_models(self) -> list[BaseChurnModel]:
        models: list[BaseChurnModel] = []
        cfg = self._cfg["models"]

        if cfg.get("logistic_regression", {}).get("enabled"):
            models.append(LogisticRegressionModel(**cfg["logistic_regression"]["params"]))

        if cfg.get("random_forest", {}).get("enabled"):
            models.append(RandomForestModel(**cfg["random_forest"]["params"]))

        if cfg.get("xgboost", {}).get("enabled"):
            models.append(XGBoostModel(**cfg["xgboost"]["params"]))

        if cfg.get("neural_network", {}).get("enabled"):
            models.append(NeuralNetworkModel(**cfg["neural_network"]["params"]))

        if cfg.get("tabular_transformer", {}).get("enabled"):
            models.append(TabularTransformerModel(**cfg["tabular_transformer"]["params"]))

        return models

    def _train_and_evaluate(
        self,
        model: BaseChurnModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> EvaluationResult:
        name = model.metadata.name
        logger.info(f"Training {name}")

        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params(model.metadata.params)

            t0 = time.perf_counter()
            model.fit(X_train, y_train, X_val, y_val)
            train_time = time.perf_counter() - t0

            proba = model.predict_proba(X_test)
            result = self._evaluator.evaluate(proba, y_test, name)

            mlflow.log_metrics(
                {
                    "roc_auc": result.roc_auc,
                    "avg_precision": result.avg_precision,
                    "f1": result.f1,
                    "precision": result.precision,
                    "recall": result.recall,
                    "train_time_sec": train_time,
                    "business_savings_pln": result.business_savings,
                }
            )

            artifact_path = Path("artifacts") / name
            artifact_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, artifact_path / "model.joblib")
            joblib.dump(self._preprocessor, artifact_path / "preprocessor.joblib")
            mlflow.log_artifacts(str(artifact_path), artifact_path=name)

            logger.info(
                f"{name} — AUC: {result.roc_auc:.4f} | F1: {result.f1:.4f} "
                f"| savings: {result.business_savings:,.0f} PLN "
                f"| train: {train_time:.1f}s"
            )

        return result
