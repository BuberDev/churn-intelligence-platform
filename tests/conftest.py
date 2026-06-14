"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.generator import SyntheticDataGenerator
from src.data.preprocessor import TARGET_COLUMN, DataPreprocessor
from src.features.engineering import FeatureEngineer


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Small synthetic dataset for fast unit tests."""
    return SyntheticDataGenerator(random_state=0).generate(n_samples=500)


@pytest.fixture(scope="session")
def engineered_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer().fit_transform(raw_df)


@pytest.fixture(scope="session")
def X_y(engineered_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = engineered_df[TARGET_COLUMN]
    X = engineered_df.drop(columns=[TARGET_COLUMN, "customer_id"], errors="ignore")
    return X, y


@pytest.fixture(scope="session")
def processed_X_y(
    X_y: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.Series]:
    X, y = X_y
    preprocessor = DataPreprocessor()
    X_processed, _ = preprocessor.fit_transform(X)
    return X_processed, y
