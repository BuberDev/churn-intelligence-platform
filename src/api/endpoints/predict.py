"""Prediction endpoints — single customer and batch."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from loguru import logger

from src.api.schemas import (
    BatchPredictionResponse,
    BatchRequest,
    CustomerFeatures,
    PredictionResponse,
)

router = APIRouter()


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict churn for a single customer",
    status_code=status.HTTP_200_OK,
)
async def predict_single(
    request: Request,
    body: CustomerFeatures,
) -> PredictionResponse:
    """Return churn probability and binary decision for one customer.

    The response includes a **confidence** tier (high / medium / low) to help
    customer success teams prioritise their outreach.
    """
    pipeline = request.app.state.pipeline
    try:
        result = pipeline.predict_single(body.model_dump())
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictionResponse(**result)


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict churn for a batch of customers",
    status_code=status.HTTP_200_OK,
)
async def predict_batch(
    request: Request,
    body: BatchRequest,
) -> BatchPredictionResponse:
    """Run batch inference and return aggregated churn statistics.

    Processes up to 10 000 customers per request. Suitable for scheduled
    nightly scoring of the full customer base.
    """
    pipeline = request.app.state.pipeline
    df = pd.DataFrame([c.model_dump() for c in body.customers])

    try:
        preds_df = pipeline.predict_batch(df)
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        ) from exc

    model_name = pipeline._model.metadata.name
    predictions = [
        PredictionResponse(
            churn_probability=row["churn_probability"],
            predicted_churned=bool(row["predicted_churned"]),
            confidence=row["confidence"],
            model_name=model_name,
        )
        for _, row in preds_df.iterrows()
    ]

    n_churners = int(preds_df["predicted_churned"].sum())

    return BatchPredictionResponse(
        total=len(predictions),
        predicted_churners=n_churners,
        churn_rate=round(n_churners / len(predictions), 4),
        predictions=predictions,
    )
