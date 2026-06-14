"""Synthetic customer dataset generator for development and testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from numpy.random import Generator


class SyntheticDataGenerator:
    """Generates realistic synthetic customer churn data.

    The generator produces correlated features that mimic real-world churn
    patterns: short tenure, high charges, and month-to-month contracts are
    strong churn predictors.
    """

    CONTRACT_TYPES = ["month-to-month", "one_year", "two_year"]
    PAYMENT_METHODS = ["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
    INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
    SEGMENTS = ["SMB", "Enterprise", "Consumer"]

    def __init__(self, random_state: int = 42) -> None:
        self._rng: Generator = np.random.default_rng(random_state)

    def generate(self, n_samples: int = 10_000) -> pd.DataFrame:
        """Generate a synthetic customer dataset.

        Args:
            n_samples: Number of customer records to generate.

        Returns:
            DataFrame with customer features and a binary ``churned`` target.
        """
        logger.info(f"Generating {n_samples:,} synthetic customer records")

        df = self._build_base_features(n_samples)
        df = self._add_derived_features(df)
        df["churned"] = self._assign_churn_labels(df)

        churn_rate = df["churned"].mean()
        logger.info(f"Dataset generated — churn rate: {churn_rate:.1%}")
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_base_features(self, n: int) -> pd.DataFrame:
        tenure = self._rng.integers(1, 73, n)  # 1–72 months

        contract = self._rng.choice(
            self.CONTRACT_TYPES, n, p=[0.55, 0.25, 0.20]
        )

        # Monthly charges correlated with contract type
        base_charge = np.where(
            contract == "month-to-month",
            self._rng.normal(70, 20, n),
            np.where(
                contract == "one_year",
                self._rng.normal(55, 15, n),
                self._rng.normal(45, 12, n),
            ),
        )
        monthly_charges = np.clip(base_charge, 18.0, 120.0)
        total_charges = monthly_charges * tenure + self._rng.normal(0, 50, n)

        return pd.DataFrame(
            {
                "customer_id": [f"C{i:07d}" for i in range(n)],
                "tenure_months": tenure.astype(int),
                "contract_type": contract,
                "monthly_charges": monthly_charges.round(2),
                "total_charges": np.clip(total_charges, 0, None).round(2),
                "payment_method": self._rng.choice(self.PAYMENT_METHODS, n),
                "internet_service": self._rng.choice(
                    self.INTERNET_SERVICES, n, p=[0.35, 0.45, 0.20]
                ),
                "segment": self._rng.choice(self.SEGMENTS, n, p=[0.50, 0.20, 0.30]),
                "num_products": self._rng.integers(1, 7, n),
                "num_support_calls": self._rng.integers(0, 11, n),
                "has_tech_support": self._rng.choice([True, False], n, p=[0.45, 0.55]),
                "has_online_backup": self._rng.choice([True, False], n),
                "is_senior_citizen": self._rng.choice([True, False], n, p=[0.16, 0.84]),
                "has_partner": self._rng.choice([True, False], n, p=[0.48, 0.52]),
                "has_dependents": self._rng.choice([True, False], n, p=[0.30, 0.70]),
            }
        )

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["avg_charge_per_product"] = (
            df["monthly_charges"] / df["num_products"]
        ).round(2)
        df["support_call_rate"] = (
            df["num_support_calls"] / df["tenure_months"].clip(lower=1)
        ).round(4)
        df["total_charges_log"] = np.log1p(df["total_charges"])
        return df

    def _assign_churn_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Compute churn probability from feature-driven logit."""
        logit = (
            -2.5
            + 1.8 * (df["contract_type"] == "month-to-month").astype(float)
            - 0.04 * df["tenure_months"]
            + 0.015 * df["monthly_charges"]
            + 0.6 * (df["payment_method"] == "electronic_check").astype(float)
            - 0.4 * df["has_tech_support"].astype(float)
            + 0.3 * df["num_support_calls"]
            - 0.2 * df["num_products"]
            + 0.4 * df["is_senior_citizen"].astype(float)
            + self._rng.normal(0, 0.3, len(df))
        )
        prob = 1 / (1 + np.exp(-logit))
        return (self._rng.uniform(size=len(df)) < prob).astype(int)
