"""Pydantic v2 request/response schemas with validation and OpenAPI examples."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class ContractType(StrEnum):
    MONTH_TO_MONTH = "month-to-month"
    ONE_YEAR = "one_year"
    TWO_YEAR = "two_year"


class PaymentMethod(StrEnum):
    ELECTRONIC_CHECK = "electronic_check"
    MAILED_CHECK = "mailed_check"
    BANK_TRANSFER = "bank_transfer"
    CREDIT_CARD = "credit_card"


class InternetService(StrEnum):
    DSL = "DSL"
    FIBER = "Fiber optic"
    NONE = "No"


class CustomerFeatures(BaseModel):
    """Input features for a single customer churn prediction."""

    model_config = {"json_schema_extra": {"examples": [
        {
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
    ]}}

    tenure_months: Annotated[int, Field(ge=0, le=120, description="Months as a customer")]
    monthly_charges: Annotated[float, Field(ge=0, le=500)]
    total_charges: Annotated[float, Field(ge=0)]
    contract_type: ContractType
    payment_method: PaymentMethod
    internet_service: InternetService
    segment: str = Field(default="Consumer", description="Customer segment")
    num_products: Annotated[int, Field(ge=1, le=20)] = 1
    num_support_calls: Annotated[int, Field(ge=0, le=100)] = 0
    has_tech_support: bool = False
    has_online_backup: bool = False
    is_senior_citizen: bool = False
    has_partner: bool = False
    has_dependents: bool = False

    @field_validator("total_charges")
    @classmethod
    def total_must_be_plausible(cls, v: float, info: object) -> float:
        return max(v, 0.0)


class PredictionResponse(BaseModel):
    """Single-customer prediction result."""

    churn_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_churned: bool
    confidence: str = Field(..., description="'high', 'medium', or 'low'")
    model_name: str
    cache_hit: bool = False


class BatchRequest(BaseModel):
    """Request body for batch prediction endpoint."""

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=10_000)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    total: int
    predicted_churners: int
    churn_rate: float
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_name: str
    model_version: str
    cache_available: bool
