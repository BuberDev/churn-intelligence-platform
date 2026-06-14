"""Data preprocessing pipeline with sklearn-compatible transformers."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "num_support_calls",
    "avg_charge_per_product",
    "support_call_rate",
    "total_charges_log",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
    "segment",
]

BOOLEAN_FEATURES = [
    "has_tech_support",
    "has_online_backup",
    "is_senior_citizen",
    "has_partner",
    "has_dependents",
]

DROP_COLUMNS = ["customer_id"]
TARGET_COLUMN = "churned"


class DataPreprocessor:
    """Sklearn-compatible preprocessing pipeline.

    Handles missing values, scaling, and one-hot encoding in a single
    ``ColumnTransformer`` so the same pipeline can be persisted and applied
    identically at training and inference time.
    """

    def __init__(self, scaling: str = "robust") -> None:
        """
        Args:
            scaling: Scaler to apply to numeric columns — ``"robust"`` (default)
                is less sensitive to outliers than ``"standard"``.
        """
        scaler = RobustScaler() if scaling == "robust" else StandardScaler()

        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="first",
                    ),
                ),
            ]
        )

        self.transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, NUMERIC_FEATURES),
                ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
                ("boolean", "passthrough", BOOLEAN_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def fit_transform(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """Fit on training data and return transformed DataFrame with feature names.

        Args:
            df: Raw input DataFrame.
            y: Ignored — present for sklearn compatibility.

        Returns:
            Transformed DataFrame and list of output feature names.
        """
        df = _cast_booleans(df)
        X = self.transformer.fit_transform(df)
        feature_names = self.transformer.get_feature_names_out().tolist()
        return pd.DataFrame(X, columns=feature_names, index=df.index), feature_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted transformer.

        Args:
            df: Raw input DataFrame (must have same columns as training data).

        Returns:
            Transformed DataFrame.
        """
        df = _cast_booleans(df)
        X = self.transformer.transform(df)
        feature_names = self.transformer.get_feature_names_out().tolist()
        return pd.DataFrame(X, columns=feature_names, index=df.index)


def _cast_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns to int so sklearn transformers handle them correctly."""
    df = df.copy()
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df
