"""Data loading utilities supporting CSV, Parquet, PostgreSQL, and MongoDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text


class DataLoader:
    """Unified data loading interface for multiple storage backends.

    Supports:
    - CSV and Parquet files
    - PostgreSQL via SQLAlchemy
    - MongoDB via pymongo
    """

    def from_csv(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = Path(path)
        logger.info(f"Loading CSV: {path}")
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Loaded {len(df):,} rows from {path.name}")
        return df

    def from_parquet(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        path = Path(path)
        logger.info(f"Loading Parquet: {path}")
        df = pd.read_parquet(path, **kwargs)
        logger.info(f"Loaded {len(df):,} rows from {path.name}")
        return df

    def from_sql(
        self,
        query: str,
        database_url: str,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            query: SQL query string.
            database_url: SQLAlchemy connection URL.
            params: Optional query parameters (prevents SQL injection).

        Returns:
            Query result as a DataFrame.
        """
        logger.info("Loading data from PostgreSQL")
        engine = create_engine(database_url)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params or {})
        logger.info(f"Loaded {len(df):,} rows from database")
        return df

    def from_mongo(
        self,
        mongo_url: str,
        database: str,
        collection: str,
        filter_: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Load documents from a MongoDB collection into a DataFrame.

        Args:
            mongo_url: MongoDB connection URL.
            database: Database name.
            collection: Collection name.
            filter_: Optional MongoDB filter document.

        Returns:
            Collection documents as a DataFrame.
        """
        from pymongo import MongoClient  # deferred import — optional dependency

        logger.info(f"Loading from MongoDB: {database}.{collection}")
        client: MongoClient[dict[str, Any]] = MongoClient(mongo_url)
        col = client[database][collection]
        docs = list(col.find(filter_ or {}, {"_id": 0}))
        df = pd.DataFrame(docs)
        logger.info(f"Loaded {len(df):,} documents")
        return df

    def save_parquet(
        self,
        df: pd.DataFrame,
        path: str | Path,
        compression: str = "snappy",
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression=compression, index=False)
        logger.info(f"Saved {len(df):,} rows to {path}")
