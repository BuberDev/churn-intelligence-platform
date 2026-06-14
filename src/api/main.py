"""FastAPI application — entrypoint for the churn prediction REST API."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.endpoints.health import router as health_router
from src.api.endpoints.predict import router as predict_router
from src.pipeline.inference import InferencePipeline


def _create_pipeline() -> InferencePipeline:
    artifact_dir = os.getenv("MODEL_ARTIFACT_DIR", "artifacts/xgboost")
    redis_url = os.getenv("REDIS_URL")
    threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    return InferencePipeline(
        artifact_dir=artifact_dir,
        redis_url=redis_url,
        threshold=threshold,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading model artifact…")
    app.state.pipeline = _create_pipeline()
    logger.info(
        f"Model loaded: {app.state.pipeline._model.metadata.name} "
        f"v{app.state.pipeline._model.metadata.version}"
    )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Churn Intelligence Platform API",
    description=(
        "REST API for real-time and batch customer churn prediction. "
        "Powered by an ensemble of ML and Deep Learning models trained on "
        "customer behavioural data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router, tags=["Health"])
app.include_router(predict_router, prefix="/predict", tags=["Predictions"])
