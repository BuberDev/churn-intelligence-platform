"""Plotly-based visualisation utilities for EDA and model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_curve


class ChurnPlotter:
    """Generates interactive Plotly charts for churn analysis reports.

    All methods return ``go.Figure`` objects so they can be displayed in
    Jupyter notebooks, exported to HTML, or embedded in a Streamlit dashboard.
    """

    COLORS = {
        "churned": "#EF4444",
        "retained": "#22C55E",
        "primary": "#6366F1",
        "secondary": "#F59E0B",
        "neutral": "#94A3B8",
    }

    def churn_distribution(self, y: pd.Series) -> go.Figure:
        """Pie + bar chart showing churn vs. retention distribution."""
        counts = y.value_counts()
        labels = ["Retained", "Churned"]
        values = [counts.get(0, 0), counts.get(1, 0)]

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=[self.COLORS["retained"], self.COLORS["churned"]],
                hole=0.4,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=[self.COLORS["retained"], self.COLORS["churned"]],
                text=[f"{v:,}" for v in values],
                textposition="outside",
            ),
            row=1, col=2,
        )
        fig.update_layout(
            title="Churn Distribution",
            showlegend=False,
            height=400,
        )
        return fig

    def roc_curves(
        self,
        results: list[tuple[str, np.ndarray, np.ndarray]],
    ) -> go.Figure:
        """Overlay ROC curves for multiple models.

        Args:
            results: List of ``(model_name, y_true, y_proba)`` tuples.
        """
        fig = go.Figure()

        for name, y_true, proba in results:
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc = float(np.trapezoid(tpr, fpr))
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode="lines",
                    name=f"{name} (AUC={auc:.3f})",
                    line={"width": 2},
                )
            )

        fig.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line={"dash": "dash", "color": self.COLORS["neutral"], "width": 1},
        )
        fig.update_layout(
            title="ROC Curves — Model Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
        )
        return fig

    def precision_recall_curves(
        self,
        results: list[tuple[str, np.ndarray, np.ndarray]],
    ) -> go.Figure:
        """Precision-Recall curves (more informative than ROC for imbalanced data)."""
        fig = go.Figure()

        for name, y_true, proba in results:
            precision, recall, _ = precision_recall_curve(y_true, proba)
            ap = float(np.mean(precision))
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision,
                    mode="lines",
                    name=f"{name} (AP={ap:.3f})",
                    line={"width": 2},
                )
            )

        fig.update_layout(
            title="Precision-Recall Curves",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=500,
        )
        return fig

    def feature_importance(
        self,
        importances: pd.Series,
        top_n: int = 20,
        title: str = "Feature Importance",
    ) -> go.Figure:
        """Horizontal bar chart of top feature importances."""
        top = importances.nlargest(top_n).sort_values()
        fig = go.Figure(
            go.Bar(
                x=top.values,
                y=top.index.tolist(),
                orientation="h",
                marker_color=self.COLORS["primary"],
            )
        )
        fig.update_layout(title=title, xaxis_title="Importance", height=max(400, top_n * 22))
        return fig

    def churn_by_segment(
        self, df: pd.DataFrame, segment_col: str, target_col: str = "churned"
    ) -> go.Figure:
        """Grouped bar chart — churn rate per segment value."""
        rates = (
            df.groupby(segment_col)[target_col]
            .agg(["mean", "count"])
            .rename(columns={"mean": "churn_rate", "count": "n"})
            .sort_values("churn_rate", ascending=False)
        )

        fig = go.Figure(
            go.Bar(
                x=rates.index.tolist(),
                y=rates["churn_rate"].round(3),
                text=[f"{v:.1%}" for v in rates["churn_rate"]],
                textposition="outside",
                marker_color=self.COLORS["churned"],
                customdata=rates["n"].tolist(),
                hovertemplate=(
                    "<b>%{x}</b><br>Churn rate: %{y:.1%}<br>n=%{customdata}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=f"Churn Rate by {segment_col}",
            yaxis_title="Churn Rate",
            yaxis_tickformat=".0%",
            height=400,
        )
        return fig

    def score_distribution(
        self, proba: np.ndarray, y_true: np.ndarray
    ) -> go.Figure:
        """Overlaid histogram of predicted probabilities by true label."""
        fig = go.Figure()
        for label, name, color in [
            (0, "Retained", self.COLORS["retained"]),
            (1, "Churned", self.COLORS["churned"]),
        ]:
            scores = proba[y_true == label]
            fig.add_trace(
                go.Histogram(
                    x=scores,
                    name=name,
                    nbinsx=40,
                    opacity=0.65,
                    marker_color=color,
                )
            )
        fig.update_layout(
            barmode="overlay",
            title="Predicted Score Distribution by True Label",
            xaxis_title="Churn Probability",
            yaxis_title="Count",
            height=400,
        )
        return fig
