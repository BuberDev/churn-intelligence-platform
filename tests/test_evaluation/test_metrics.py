"""Tests for ModelEvaluator and BusinessMetrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import BusinessMetrics, ModelEvaluator


@pytest.fixture
def perfect_proba() -> tuple[np.ndarray, pd.Series]:
    y = pd.Series([0, 0, 0, 1, 1, 1])
    proba = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
    return proba, y


@pytest.fixture
def random_proba(processed_X_y: tuple) -> tuple[np.ndarray, pd.Series]:
    _, y = processed_X_y
    rng = np.random.default_rng(0)
    return rng.uniform(size=len(y)), y


def test_evaluate_returns_result(random_proba: tuple) -> None:
    proba, y = random_proba
    result = ModelEvaluator().evaluate(proba, y, "test_model")
    assert result.model_name == "test_model"
    assert 0.0 <= result.roc_auc <= 1.0


def test_perfect_predictions_give_auc_one(perfect_proba: tuple) -> None:
    proba, y = perfect_proba
    result = ModelEvaluator().evaluate(proba, y, "perfect")
    assert result.roc_auc == pytest.approx(1.0)


def test_random_predictions_give_auc_near_half(random_proba: tuple) -> None:
    proba, y = random_proba
    result = ModelEvaluator().evaluate(proba, y, "random", optimize_threshold=False)
    assert 0.3 < result.roc_auc < 0.7


def test_optimal_threshold_improves_f1(perfect_proba: tuple) -> None:
    proba, y = perfect_proba
    r_default = ModelEvaluator().evaluate(proba, y, "m", optimize_threshold=False)
    r_opt = ModelEvaluator().evaluate(proba, y, "m", optimize_threshold=True)
    assert r_opt.f1 >= r_default.f1


def test_compare_sorts_by_roc_auc(random_proba: tuple) -> None:
    proba, y = random_proba
    ev = ModelEvaluator()
    r1 = ev.evaluate(proba, y, "a")
    r2 = ev.evaluate(np.ones_like(proba) * 0.5, y, "b")
    comparison = ev.compare([r2, r1])
    assert comparison.iloc[0]["roc_auc"] >= comparison.iloc[1]["roc_auc"]


class TestBusinessMetrics:
    def test_perfect_model_has_lower_cost(self) -> None:
        bm = BusinessMetrics(cost_false_negative=500, cost_false_positive=50)
        y = np.array([1, 1, 0, 0])
        perfect = np.array([1, 1, 0, 0])
        random_pred = np.array([0, 1, 1, 0])
        assert bm.total_cost(y, perfect) < bm.total_cost(y, random_pred)

    def test_savings_vs_baseline_positive_for_good_model(self) -> None:
        bm = BusinessMetrics()
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        # Model correctly flags churners, avoids false positives
        good_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        assert bm.savings_vs_baseline(y, good_pred) > 0
