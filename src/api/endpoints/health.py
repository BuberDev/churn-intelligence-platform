"""Health check endpoint."""

from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="API health check")
async def health(request: Request) -> HealthResponse:
    pipeline = request.app.state.pipeline
    model = pipeline._model
    return HealthResponse(
        status="ok",
        model_name=model.metadata.name,
        model_version=model.metadata.version,
        cache_available=pipeline._cache is not None,
    )
