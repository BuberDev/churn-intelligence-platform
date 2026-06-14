"""Tests for SyntheticDataGenerator."""

import pandas as pd
import pytest

from src.data.generator import SyntheticDataGenerator


@pytest.fixture
def generator() -> SyntheticDataGenerator:
    return SyntheticDataGenerator(random_state=42)


def test_generate_returns_expected_row_count(generator: SyntheticDataGenerator) -> None:
    df = generator.generate(n_samples=200)
    assert len(df) == 200


def test_generate_contains_target_column(generator: SyntheticDataGenerator) -> None:
    df = generator.generate(n_samples=100)
    assert "churned" in df.columns


def test_target_is_binary(generator: SyntheticDataGenerator) -> None:
    df = generator.generate(n_samples=500)
    assert set(df["churned"].unique()).issubset({0, 1})


def test_churn_rate_in_realistic_range(generator: SyntheticDataGenerator) -> None:
    df = generator.generate(n_samples=5_000)
    rate = df["churned"].mean()
    assert 0.10 < rate < 0.50, f"Churn rate {rate:.1%} outside expected range"


def test_no_null_values_in_key_columns(generator: SyntheticDataGenerator) -> None:
    df = generator.generate(n_samples=300)
    key_cols = ["tenure_months", "monthly_charges", "contract_type", "churned"]
    assert df[key_cols].isnull().sum().sum() == 0


def test_reproducibility_with_same_seed() -> None:
    df1 = SyntheticDataGenerator(random_state=7).generate(n_samples=100)
    df2 = SyntheticDataGenerator(random_state=7).generate(n_samples=100)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data() -> None:
    df1 = SyntheticDataGenerator(random_state=1).generate(n_samples=100)
    df2 = SyntheticDataGenerator(random_state=2).generate(n_samples=100)
    assert not df1["churned"].equals(df2["churned"])
