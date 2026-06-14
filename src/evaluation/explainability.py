"""Model explainability using SHAP — critical for business stakeholder trust."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger


class ShapExplainer:
    """Computes and visualises SHAP values for any tree or black-box model.

    SHAP (SHapley Additive exPlanations) provides both global feature
    importance and individual prediction explanations — essential for
    presenting model decisions to non-technical stakeholders and satisfying
    regulatory explainability requirements.
    """

    def __init__(self, model: Any, X_background: pd.DataFrame) -> None:
        """
        Args:
            model: Fitted model with a ``predict_proba`` method.
            X_background: Representative sample used by KernelExplainer as
                the background distribution (100–500 rows is sufficient).
        """
        self._model = model
        self._X_background = X_background
        self._explainer: shap.Explainer | None = None
        self._shap_values: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> ShapExplainer:
        """Compute SHAP values for the given dataset.

        Tries TreeExplainer first (fast, exact) and falls back to
        KernelExplainer (model-agnostic) for non-tree models.

        Args:
            X: Feature matrix to explain.

        Returns:
            Self (fitted explainer).
        """
        logger.info("Computing SHAP values")
        try:
            self._explainer = shap.TreeExplainer(self._model)
            shap_output = self._explainer.shap_values(X)
            # TreeExplainer returns list [class0, class1] for binary classifiers
            self._shap_values = (
                shap_output[1] if isinstance(shap_output, list) else shap_output
            )
        except Exception:
            logger.info("Falling back to KernelExplainer")
            background = shap.kmeans(self._X_background, k=50)
            self._explainer = shap.KernelExplainer(
                lambda x: self._model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1],
                background,
            )
            self._shap_values = self._explainer.shap_values(X, nsamples=100)

        logger.info("SHAP values computed")
        return self

    def global_importance(self) -> pd.Series:
        """Return mean absolute SHAP values per feature (global importance)."""
        assert self._shap_values is not None, "Call fit() first"
        return pd.Series(
            np.abs(self._shap_values).mean(axis=0),
            index=self._X_background.columns,
        ).sort_values(ascending=False)

    def explain_instance(
        self, X: pd.DataFrame, idx: int = 0
    ) -> pd.Series:
        """Return SHAP values for a single prediction.

        Args:
            X: Feature matrix (same columns as training).
            idx: Row index to explain.

        Returns:
            Series of SHAP values with feature names as index.
        """
        assert self._shap_values is not None
        return pd.Series(
            self._shap_values[idx], index=X.columns
        ).sort_values(key=abs, ascending=False)

    def plot_summary(self, X: pd.DataFrame, output_path: str | Path | None = None) -> None:
        """Generate a SHAP beeswarm summary plot."""
        assert self._shap_values is not None
        plt.figure(figsize=(10, 7))
        shap.summary_plot(self._shap_values, X, show=False)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary saved to {output_path}")
        else:
            plt.show()
        plt.close()

    def plot_waterfall(
        self,
        X: pd.DataFrame,
        idx: int,
        expected_value: float | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        """Waterfall chart explaining one prediction.

        Args:
            X: Feature matrix.
            idx: Sample index to explain.
            expected_value: Model base rate (mean prediction).
            output_path: Optional path to save the figure.
        """
        assert self._shap_values is not None and self._explainer is not None
        ev = expected_value if expected_value is not None else float(
            getattr(self._explainer, "expected_value", 0.5)
        )
        shap_exp = shap.Explanation(
            values=self._shap_values[idx],
            base_values=ev,
            data=X.iloc[idx].values,
            feature_names=X.columns.tolist(),
        )
        plt.figure()
        shap.plots.waterfall(shap_exp, show=False)
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()
