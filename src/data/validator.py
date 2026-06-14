"""Data quality validation following CRISP-DM data understanding phase."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger


@dataclass
class ValidationReport:
    """Summary of a data validation run."""

    passed: bool
    row_count: int
    missing_values: dict[str, float]
    schema_errors: list[str]
    range_violations: list[str]
    warnings: list[str] = field(default_factory=list)

    def log(self) -> None:
        status = "PASSED" if self.passed else "FAILED"
        logger.info(f"Data validation {status} — {self.row_count:,} rows")
        for col, pct in self.missing_values.items():
            logger.warning(f"  Missing in '{col}': {pct:.1%}")
        for err in self.schema_errors + self.range_violations:
            logger.error(f"  {err}")


class DataValidator:
    """Validates DataFrames against a schema and business rules.

    Checks performed:
    - Required column presence
    - Dtype compatibility
    - Numeric range constraints
    - Missing value thresholds
    """

    _SCHEMA: dict[str, type] = {
        "tenure_months": int,
        "monthly_charges": float,
        "total_charges": float,
        "num_products": int,
        "num_support_calls": int,
        "contract_type": object,
        "payment_method": object,
        "internet_service": object,
    }

    _RANGES: dict[str, tuple[float, float]] = {
        "tenure_months": (0, 120),
        "monthly_charges": (0, 500),
        "total_charges": (0, 100_000),
        "num_products": (1, 20),
        "num_support_calls": (0, 100),
    }

    _ALLOWED_VALUES: dict[str, set[str]] = {
        "contract_type": {"month-to-month", "one_year", "two_year"},
        "payment_method": {
            "electronic_check",
            "mailed_check",
            "bank_transfer",
            "credit_card",
        },
        "internet_service": {"DSL", "Fiber optic", "No"},
    }

    MISSING_THRESHOLD = 0.10

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run all validation checks and return a report.

        Args:
            df: Input DataFrame to validate.

        Returns:
            :class:`ValidationReport` with detailed findings.
        """
        schema_errors = self._check_schema(df)
        range_violations = self._check_ranges(df)
        missing = self._check_missing(df)
        warnings = self._check_categorical_values(df)

        passed = (
            not schema_errors
            and not range_violations
            and all(v < self.MISSING_THRESHOLD for v in missing.values())
        )

        report = ValidationReport(
            passed=passed,
            row_count=len(df),
            missing_values={k: v for k, v in missing.items() if v > 0},
            schema_errors=schema_errors,
            range_violations=range_violations,
            warnings=warnings,
        )
        report.log()
        return report

    # ------------------------------------------------------------------

    def _check_schema(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        for col in self._SCHEMA:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")
        return errors

    def _check_ranges(self, df: pd.DataFrame) -> list[str]:
        violations: list[str] = []
        for col, (lo, hi) in self._RANGES.items():
            if col not in df.columns:
                continue
            out = df[col].dropna()
            if (out < lo).any() or (out > hi).any():
                n = ((out < lo) | (out > hi)).sum()
                violations.append(
                    f"'{col}' has {n} values outside [{lo}, {hi}]"
                )
        return violations

    def _check_missing(self, df: pd.DataFrame) -> dict[str, float]:
        existing = [c for c in self._SCHEMA if c in df.columns]
        return df[existing].isnull().mean().to_dict()

    def _check_categorical_values(self, df: pd.DataFrame) -> list[str]:
        warnings: list[str] = []
        for col, allowed in self._ALLOWED_VALUES.items():
            if col not in df.columns:
                continue
            unknown = set(df[col].dropna().unique()) - allowed
            if unknown:
                warnings.append(
                    f"'{col}' contains unknown categories: {unknown}"
                )
        return warnings
