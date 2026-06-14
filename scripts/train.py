"""CLI: run the full training pipeline."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from src.data.loader import DataLoader
from src.evaluation.metrics import ModelEvaluator
from src.pipeline.training import TrainingPipeline

app = typer.Typer(help="Train churn prediction models")


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/model_config.yaml"), "--config", "-c", help="Path to model config"
    ),
    data: Path = typer.Option(
        Path("data/raw/customers.parquet"), "--data", "-d", help="Input data file"
    ),
) -> None:
    """Execute the training pipeline and log all experiments to MLflow."""
    if not data.exists():
        typer.echo(
            f"Data file '{data}' not found. Run: python scripts/generate_data.py",
            err=True,
        )
        raise typer.Exit(1)

    loader = DataLoader()
    df = loader.from_parquet(data)

    pipeline = TrainingPipeline(config_path=config)
    results = pipeline.run(df)

    evaluator = ModelEvaluator()
    comparison = evaluator.compare(results)

    typer.echo("\n=== Model Comparison ===")
    typer.echo(comparison.to_string(index=False))

    best = comparison.iloc[0]
    logger.success(
        f"\nBest model: {best['model']} | "
        f"AUC: {best['roc_auc']:.4f} | "
        f"F1: {best['f1']:.4f} | "
        f"Savings: {best['business_savings_pln']:,.0f} PLN"
    )


if __name__ == "__main__":
    app()
