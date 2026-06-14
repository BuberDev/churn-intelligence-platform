"""Feature selection using permutation importance and correlation filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class FeatureSelector:
    """Selects the most informative features using two-stage filtering.

    Stage 1 — Correlation filter: removes features with pairwise correlation
    above ``corr_threshold`` (keeps the one with higher target correlation).

    Stage 2 — Permutation importance: drops features whose importance falls
    below ``importance_threshold`` relative to a quick random forest fit.
    """

    def __init__(
        self,
        corr_threshold: float = 0.90,
        importance_threshold: float = 0.005,
        n_estimators: int = 50,
        random_state: int = 42,
    ) -> None:
        self.corr_threshold = corr_threshold
        self.importance_threshold = importance_threshold
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.selected_features_: list[str] = []
        self._dropped_corr_: list[str] = []
        self._dropped_importance_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeatureSelector:
        """Fit the selector on training data.

        Args:
            X: Feature matrix.
            y: Binary target.

        Returns:
            Self (fitted selector).
        """
        features = X.columns.tolist()

        # Stage 1: remove highly correlated features
        features = self._drop_correlated(X[features], y, features)

        # Stage 2: permutation importance
        features = self._drop_low_importance(X[features], y, features)

        self.selected_features_ = features
        logger.info(
            f"Selected {len(features)} features "
            f"(dropped {len(self._dropped_corr_)} correlated, "
            f"{len(self._dropped_importance_)} low-importance)"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------

    def _drop_correlated(
        self, X: pd.DataFrame, y: pd.Series, features: list[str]
    ) -> list[str]:
        numeric = X[features].select_dtypes(include="number")
        corr_matrix = numeric.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        target_corr = numeric.corrwith(y).abs()
        to_drop: set[str] = set()

        for col in upper.columns:
            highly_corr = upper[col][upper[col] > self.corr_threshold].index.tolist()
            for other in highly_corr:
                drop = col if target_corr.get(col, 0) < target_corr.get(other, 0) else other
                to_drop.add(drop)

        self._dropped_corr_ = list(to_drop)
        return [f for f in features if f not in to_drop]

    def _drop_low_importance(
        self, X: pd.DataFrame, y: pd.Series, features: list[str]
    ) -> list[str]:
        numeric_features = X[features].select_dtypes(include="number").columns.tolist()
        if not numeric_features:
            return features

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X[numeric_features], y)

        result = permutation_importance(
            rf,
            X[numeric_features],
            y,
            n_repeats=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        importances = pd.Series(
            result.importances_mean, index=numeric_features
        )
        low_importance = importances[importances < self.importance_threshold].index.tolist()
        self._dropped_importance_ = low_importance

        non_numeric = [f for f in features if f not in numeric_features]
        return [f for f in features if f not in low_importance] + [
            f for f in non_numeric if f not in low_importance
        ]
