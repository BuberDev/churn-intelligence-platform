"""Tests for deep learning models."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.models.base import BaseChurnModel
from src.models.deep_learning import NeuralNetworkModel, TabularTransformerModel


@pytest.fixture(scope="module")
def small_train_test(
    processed_X_y: tuple[pd.DataFrame, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = processed_X_y
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (NeuralNetworkModel, {"hidden_dims": [32, 16], "max_epochs": 3, "patience": 2}),
        (TabularTransformerModel, {"d_model": 16, "n_heads": 2, "n_layers": 1, "max_epochs": 3, "patience": 2}),
    ],
    ids=["neural_network", "tabular_transformer"],
)
def test_fit_and_predict_proba(
    model_cls: type[BaseChurnModel],
    kwargs: dict,
    small_train_test: tuple,
) -> None:
    X_train, X_test, y_train, _ = small_train_test
    model = model_cls(**kwargs)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_neural_network_early_stopping(small_train_test: tuple) -> None:
    X_train, X_val, y_train, y_val = small_train_test
    model = NeuralNetworkModel(
        hidden_dims=[32], max_epochs=50, patience=2, batch_size=32
    )
    model.fit(X_train, y_train, X_val, y_val)
    assert model._net is not None


def test_metadata_names() -> None:
    assert NeuralNetworkModel().metadata.name == "neural_network"
    assert TabularTransformerModel().metadata.name == "tabular_transformer"
