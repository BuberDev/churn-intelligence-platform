"""Model evaluation metrics including business-cost-aware threshold optimisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class BusinessMetrics:
    """Translates ML confusion matrix into financial impact.

    Args:
        cost_false_negative: Revenue lost per missed churner (e.g. average
            customer value). Typically 500–2 000 PLN.
        cost_false_positive: Cost of retention intervention per customer
            (e.g. discount, agent call). Typically 30–100 PLN.
    """

    cost_false_negative: float = 500.0
    cost_false_positive: float = 50.0

    def total_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        _tn, fp, fn, _tp = confusion_matrix(y_true, y_pred).ravel()
        return fn * self.cost_false_negative + fp * self.cost_false_positive

    def savings_vs_baseline(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Compare model cost against a naive 'flag everyone' baseline."""
        model_cost = self.total_cost(y_true, y_pred)
        baseline_cost = len(y_true) * self.cost_false_positive
        return baseline_cost - model_cost


@dataclass
class EvaluationResult:
    """Full evaluation report for one model."""

    model_name: str
    roc_auc: float
    avg_precision: float
    f1: float
    precision: float
    recall: float
    threshold: float
    business_savings: float
    classification_report: str


class ModelEvaluator:
    """Computes comprehensive classification and business metrics.

    Usage::

        evaluator = ModelEvaluator(business_metrics=BusinessMetrics())
        result = evaluator.evaluate(model, X_test, y_test)
        evaluator.log_to_mlflow(result, run_id)
    """

    def __init__(
        self,
        business_metrics: BusinessMetrics | None = None,
    ) -> None:
        self._business = business_metrics or BusinessMetrics()

    def evaluate(
        self,
        proba: np.ndarray,
        y_true: pd.Series,
        model_name: str,
        optimize_threshold: bool = True,
    ) -> EvaluationResult:
        """Evaluate predictions from ``model.predict_proba``.

        Args:
            proba: Predicted churn probabilities (1-D array).
            y_true: Ground-truth labels.
            model_name: Model identifier for reporting.
            optimize_threshold: If True, selects the threshold that maximises F1.

        Returns:
            :class:`EvaluationResult` with all metrics.
        """
        y_arr = y_true.to_numpy()
        roc_auc = roc_auc_score(y_arr, proba)
        avg_precision = average_precision_score(y_arr, proba)

        threshold = (
            self._optimal_threshold(y_arr, proba) if optimize_threshold else 0.5
        )
        y_pred = (proba >= threshold).astype(int)

        f1 = f1_score(y_arr, y_pred)
        from sklearn.metrics import precision_score, recall_score

        precision = precision_score(y_arr, y_pred, zero_division=0)
        recall = recall_score(y_arr, y_pred, zero_division=0)
        savings = self._business.savings_vs_baseline(y_arr, y_pred)
        report = classification_report(y_arr, y_pred, target_names=["retained", "churned"])

        return EvaluationResult(
            model_name=model_name,
            roc_auc=roc_auc,
            avg_precision=avg_precision,
            f1=f1,
            precision=precision,
            recall=recall,
            threshold=threshold,
            business_savings=savings,
            classification_report=report,
        )

    def compare(self, results: list[EvaluationResult]) -> pd.DataFrame:
        """Return a sorted comparison DataFrame from a list of evaluation results."""
        rows = [
            {
                "model": r.model_name,
                "roc_auc": r.roc_auc,
                "avg_precision": r.avg_precision,
                "f1": r.f1,
                "precision": r.precision,
                "recall": r.recall,
                "threshold": r.threshold,
                "business_savings_pln": r.business_savings,
            }
            for r in results
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("roc_auc", ascending=False)
            .reset_index(drop=True)
        )

    def log_to_mlflow(self, result: EvaluationResult, run_id: str | None = None) -> None:
        """Log metrics to an active or specified MLflow run."""
        import mlflow

        metrics = {
            "roc_auc": result.roc_auc,
            "avg_precision": result.avg_precision,
            "f1": result.f1,
            "precision": result.precision,
            "recall": result.recall,
            "optimal_threshold": result.threshold,
            "business_savings_pln": result.business_savings,
        }
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metrics(metrics)

    # ------------------------------------------------------------------

    @staticmethod
    def _optimal_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
        """Return the decision threshold that maximises the F1 score."""
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        f1_scores = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
        return float(thresholds[np.argmax(f1_scores[:-1])])
