"""Integration tests for FastAPI endpoints using a mocked InferencePipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline._model.metadata.name = "xgboost"
    pipeline._model.metadata.version = "1.0.0"
    pipeline._cache = None
    pipeline.predict_single.return_value = {
        "churn_probability": 0.82,
        "predicted_churned": True,
        "confidence": "high",
        "model_name": "xgboost",
        "cache_hit": False,
    }
    pipeline.predict_batch.return_value = pd.DataFrame(
        [
            {"churn_probability": 0.82, "predicted_churned": 1, "confidence": "high"},
            {"churn_probability": 0.21, "predicted_churned": 0, "confidence": "low"},
        ]
    )
    return pipeline


@pytest.fixture
def client(mock_pipeline: MagicMock) -> TestClient:
    app.state.pipeline = mock_pipeline
    return TestClient(app)


SAMPLE_CUSTOMER = {
    "tenure_months": 3,
    "monthly_charges": 79.5,
    "total_charges": 238.5,
    "contract_type": "month-to-month",
    "payment_method": "electronic_check",
    "internet_service": "Fiber optic",
    "segment": "Consumer",
    "num_products": 2,
    "num_support_calls": 5,
    "has_tech_support": False,
    "has_online_backup": False,
    "is_senior_citizen": False,
    "has_partner": False,
    "has_dependents": False,
}


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_name"] == "xgboost"


def test_predict_single_returns_200(client: TestClient) -> None:
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200


def test_predict_single_response_schema(client: TestClient) -> None:
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    data = response.json()
    assert "churn_probability" in data
    assert "predicted_churned" in data
    assert "confidence" in data
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_single_invalid_contract_type_returns_422(
    client: TestClient,
) -> None:
    bad = {**SAMPLE_CUSTOMER, "contract_type": "weekly"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_batch_returns_200(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER] * 2})
    assert response.status_code == 200


def test_predict_batch_response_counts(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER] * 2})
    data = response.json()
    assert data["total"] == 2
    assert "churn_rate" in data
    assert 0.0 <= data["churn_rate"] <= 1.0


def test_predict_batch_empty_list_returns_422(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422
