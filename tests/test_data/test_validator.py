"""Tests for DataValidator."""

import pandas as pd
import pytest

from src.data.validator import DataValidator


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator()


def test_valid_dataframe_passes(raw_df: pd.DataFrame, validator: DataValidator) -> None:
    report = validator.validate(raw_df)
    assert report.passed


def test_missing_column_fails(validator: DataValidator) -> None:
    df = pd.DataFrame({"tenure_months": [12]})
    report = validator.validate(df)
    assert not report.passed
    assert any("monthly_charges" in e for e in report.schema_errors)


def test_range_violation_detected(raw_df: pd.DataFrame, validator: DataValidator) -> None:
    df = raw_df.copy()
    df.loc[0, "tenure_months"] = -5
    report = validator.validate(df)
    assert any("tenure_months" in v for v in report.range_violations)


def test_missing_values_above_threshold_fail(
    raw_df: pd.DataFrame, validator: DataValidator
) -> None:
    df = raw_df.copy()
    df.loc[:, "monthly_charges"] = None
    report = validator.validate(df)
    assert not report.passed
    assert "monthly_charges" in report.missing_values


def test_unknown_categorical_produces_warning(
    raw_df: pd.DataFrame, validator: DataValidator
) -> None:
    df = raw_df.copy()
    df.loc[0, "contract_type"] = "quarterly"
    report = validator.validate(df)
    assert any("contract_type" in w for w in report.warnings)
