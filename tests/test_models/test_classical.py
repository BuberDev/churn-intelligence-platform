"""Tests for classical ML models."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.models.base import BaseChurnModel
from src.models.classical import LogisticRegressionModel, RandomForestModel, XGBoostModel


@pytest.fixture(scope="module")
def train_test(
    processed_X_y: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = processed_X_y
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@pytest.mark.parametrize(
    "model_cls",
    [LogisticRegressionModel, RandomForestModel, XGBoostModel],
    ids=["logistic_regression", "random_forest", "xgboost"],
)
def test_fit_predict_proba_shape(
    model_cls: type[BaseChurnModel],
    train_test: tuple,
) -> None:
    X_train, X_test, y_train, _ = train_test
    model = model_cls()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test),)


@pytest.mark.parametrize(
    "model_cls",
    [LogisticRegressionModel, RandomForestModel, XGBoostModel],
    ids=["logistic_regression", "random_forest", "xgboost"],
)
def test_predict_proba_values_in_range(
    model_cls: type[BaseChurnModel],
    train_test: tuple,
) -> None:
    X_train, X_test, y_train, _ = train_test
    model = model_cls()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert (proba >= 0).all() and (proba <= 1).all()


@pytest.mark.parametrize(
    "model_cls",
    [LogisticRegressionModel, RandomForestModel, XGBoostModel],
    ids=["logistic_regression", "random_forest", "xgboost"],
)
def test_predict_returns_binary(
    model_cls: type[BaseChurnModel],
    train_test: tuple,
) -> None:
    X_train, X_test, y_train, _ = train_test
    model = model_cls()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1})


def test_xgboost_uses_validation_for_early_stopping(
    train_test: tuple,
) -> None:
    X_train, X_test, y_train, y_test = train_test
    model = XGBoostModel(n_estimators=200, early_stopping_rounds=5)
    model.fit(X_train, y_train, X_test, y_test)
    # early stopping means actual trees < n_estimators
    assert model._model.best_iteration < 200


def test_logistic_regression_feature_coefficients(train_test: tuple) -> None:
    X_train, _, y_train, _ = train_test
    model = LogisticRegressionModel()
    model.fit(X_train, y_train)
    coefs = model.feature_coefficients()
    assert len(coefs) == X_train.shape[1]
    assert (coefs >= 0).all()


def test_metadata_name_set_correctly() -> None:
    assert LogisticRegressionModel().metadata.name == "logistic_regression"
    assert RandomForestModel().metadata.name == "random_forest"
    assert XGBoostModel().metadata.name == "xgboost"
