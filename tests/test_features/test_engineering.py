"""Tests for FeatureEngineer."""

import pandas as pd

from src.features.engineering import FeatureEngineer

NEW_FEATURES = [
    "clv_estimate",
    "is_at_risk",
    "engagement_score",
    "tenure_segment",
    "charge_to_tenure_ratio",
    "products_x_tenure",
]


def test_engineer_adds_expected_columns(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(raw_df)
    for col in NEW_FEATURES:
        assert col in result.columns, f"Missing column: {col}"


def test_clv_estimate_is_positive(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(raw_df)
    assert (result["clv_estimate"] >= 0).all()


def test_is_at_risk_is_binary(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(raw_df)
    assert set(result["is_at_risk"].unique()).issubset({0, 1})


def test_engagement_score_bounded(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(raw_df)
    assert (result["engagement_score"] >= 0).all()
    assert (result["engagement_score"] <= 1).all()


def test_original_columns_preserved(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(raw_df)
    for col in raw_df.columns:
        assert col in result.columns


def test_no_nulls_introduced(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(raw_df)
    numeric_new = [c for c in NEW_FEATURES if c not in ("monthly_charge_tier", "tenure_segment")]
    assert result[numeric_new].isnull().sum().sum() == 0


def test_at_risk_flag_higher_churn_rate(raw_df: pd.DataFrame) -> None:
    result = FeatureEngineer().fit_transform(raw_df)
    if result["is_at_risk"].sum() >= 10:
        at_risk_churn = result[result["is_at_risk"] == 1]["churned"].mean()
        safe_churn = result[result["is_at_risk"] == 0]["churned"].mean()
        assert at_risk_churn > safe_churn, "is_at_risk flag should predict higher churn"
