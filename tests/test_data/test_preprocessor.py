"""Tests for DataPreprocessor."""

import numpy as np
import pandas as pd

from src.data.preprocessor import DataPreprocessor


def test_fit_transform_returns_dataframe(X_y: tuple) -> None:
    X, _ = X_y
    preprocessor = DataPreprocessor()
    X_out, feature_names = preprocessor.fit_transform(X)
    assert isinstance(X_out, pd.DataFrame)
    assert len(feature_names) == X_out.shape[1]


def test_no_nan_after_preprocessing(X_y: tuple) -> None:
    X, _ = X_y
    preprocessor = DataPreprocessor()
    X_out, _ = preprocessor.fit_transform(X)
    assert not X_out.isnull().any().any()


def test_transform_produces_same_shape_as_fit_transform(X_y: tuple) -> None:
    X, _ = X_y
    half = len(X) // 2
    preprocessor = DataPreprocessor()
    X_train, _ = preprocessor.fit_transform(X.iloc[:half])
    X_test = preprocessor.transform(X.iloc[half:])
    assert X_train.shape[1] == X_test.shape[1]


def test_robust_scaler_does_not_explode_on_outliers(X_y: tuple) -> None:
    X, _ = X_y
    X_extreme = X.copy()
    X_extreme.loc[X_extreme.index[0], "monthly_charges"] = 1_000_000
    preprocessor = DataPreprocessor(scaling="robust")
    X_out, _ = preprocessor.fit_transform(X_extreme)
    assert np.isfinite(X_out.values).all()
