"""Inference pipeline with Redis caching and batch processing support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import redis
from loguru import logger

from src.data.preprocessor import DataPreprocessor
from src.models.base import BaseChurnModel


class InferencePipeline:
    """Loads a trained model artifact and serves predictions.

    Features:
    - Redis-backed prediction cache (TTL-controlled) to avoid redundant
      computation for recently-seen customer IDs
    - Batch inference with configurable chunk size for large datasets
    - Confidence tiers attached to each prediction
    """

    CACHE_TTL_SECONDS = 3_600

    def __init__(
        self,
        artifact_dir: str | Path,
        redis_url: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        artifact_dir = Path(artifact_dir)
        self._model: BaseChurnModel = joblib.load(artifact_dir / "model.joblib")
        self._preprocessor: DataPreprocessor = joblib.load(
            artifact_dir / "preprocessor.joblib"
        )
        self._threshold = threshold

        self._cache: redis.Redis | None = None  # type: ignore[type-arg]
        if redis_url:
            try:
                self._cache = redis.from_url(redis_url, decode_responses=True)
                self._cache.ping()
                logger.info("Redis prediction cache connected")
            except redis.ConnectionError:
                logger.warning("Redis unavailable — running without cache")
                self._cache = None

    def predict_single(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict churn for a single customer.

        Args:
            features: Raw feature dictionary (as received from the API).

        Returns:
            Dict with ``churn_probability``, ``predicted_churned``,
            ``confidence``, and ``model_name``.
        """
        cache_key = self._cache_key(features)

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                logger.debug("Cache hit")
                result: dict[str, Any] = json.loads(str(cached))
                result["cache_hit"] = True
                return result

        df = pd.DataFrame([features])
        X = self._preprocessor.transform(df)
        pred_df = self._model.predict_with_confidence(X, self._threshold)

        result = {
            "churn_probability": float(pred_df["churn_probability"].iloc[0]),
            "predicted_churned": bool(pred_df["predicted_churned"].iloc[0]),
            "confidence": pred_df["confidence"].iloc[0],
            "model_name": self._model.metadata.name,
            "cache_hit": False,
        }

        if self._cache:
            self._cache.setex(cache_key, self.CACHE_TTL_SECONDS, json.dumps(result))

        return result

    def predict_batch(
        self,
        df: pd.DataFrame,
        chunk_size: int = 1_000,
    ) -> pd.DataFrame:
        """Run batch inference with chunked processing.

        Args:
            df: Raw feature DataFrame (without target column).
            chunk_size: Number of rows processed per chunk (controls memory usage).

        Returns:
            DataFrame with prediction columns appended.
        """
        logger.info(f"Batch inference: {len(df):,} rows in chunks of {chunk_size}")
        chunks: list[pd.DataFrame] = []

        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            X = self._preprocessor.transform(chunk)
            preds = self._model.predict_with_confidence(X, self._threshold)
            chunks.append(preds)

        result = pd.concat(chunks, ignore_index=True)
        logger.info(
            f"Batch complete — predicted churners: "
            f"{result['predicted_churned'].sum():,} / {len(result):,}"
        )
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(features: dict[str, Any]) -> str:
        payload = json.dumps(features, sort_keys=True)
        return "churn:pred:" + hashlib.sha256(payload.encode()).hexdigest()[:16]
