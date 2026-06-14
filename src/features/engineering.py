"""Domain-driven feature engineering for churn prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives business-relevant features from raw customer data.

    Inherits from sklearn's ``BaseEstimator`` and ``TransformerMixin`` so this
    transformer can be inserted into a ``Pipeline`` and serialised with joblib.

    New features created:
    - ``clv_estimate`` — simplified Customer Lifetime Value proxy
    - ``monthly_charge_tier`` — bucketed pricing tier
    - ``is_at_risk`` — heuristic risk flag (short tenure + high charges)
    - ``engagement_score`` — composite product/support engagement metric
    - ``tenure_segment`` — categorical lifecycle stage
    - ``charge_to_tenure_ratio`` — rate of spending over lifetime
    """

    TIER_BINS = [0, 35, 65, 90, float("inf")]
    TIER_LABELS = ["low", "medium", "high", "premium"]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df["clv_estimate"] = (
            df["monthly_charges"] * df["tenure_months"] * 1.2
        ).round(2)

        df["monthly_charge_tier"] = pd.cut(
            df["monthly_charges"],
            bins=self.TIER_BINS,
            labels=self.TIER_LABELS,
            right=False,
        )

        df["is_at_risk"] = (
            (df["tenure_months"] < 6)
            & (df["monthly_charges"] > 65)
            & (df["contract_type"] == "month-to-month")
        ).astype(int)

        df["engagement_score"] = (
            df["num_products"] * 0.4
            + df["has_tech_support"].astype(int) * 0.2
            + df["has_online_backup"].astype(int) * 0.2
            - df["num_support_calls"] * 0.05
        ).clip(0, 1)

        df["tenure_segment"] = pd.cut(
            df["tenure_months"],
            bins=[0, 12, 24, 48, float("inf")],
            labels=["new", "growing", "mature", "loyal"],
            right=True,
        )

        df["charge_to_tenure_ratio"] = (
            df["monthly_charges"] / df["tenure_months"].clip(lower=1)
        ).round(4)

        df["products_x_tenure"] = df["num_products"] * np.log1p(df["tenure_months"])

        return df
