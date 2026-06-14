"""CLI: evaluate the best model from an MLflow experiment."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from src.data.loader import DataLoader
from src.evaluation.explainability import ShapExplainer
from src.pipeline.inference import InferencePipeline

app = typer.Typer(help="Evaluate a trained churn model")


@app.command()
def main(
    artifact_dir: Path = typer.Option(
        Path("artifacts/xgboost"), "--artifact-dir", "-a"
    ),
    data: Path = typer.Option(
        Path("data/raw/customers.parquet"), "--data", "-d"
    ),
    shap: bool = typer.Option(False, "--shap", help="Generate SHAP explainability report"),
    output_dir: Path = typer.Option(Path("reports"), "--output-dir", "-o"),
) -> None:
    """Load a trained artifact and produce evaluation plots and SHAP analysis."""
    loader = DataLoader()
    df = loader.from_parquet(data)

    from src.data.preprocessor import TARGET_COLUMN, DataPreprocessor
    from src.features.engineering import FeatureEngineer

    df = FeatureEngineer().fit_transform(df)
    X = df.drop(columns=[TARGET_COLUMN, "customer_id"], errors="ignore")

    preprocessor = DataPreprocessor()
    X_proc, _ = preprocessor.fit_transform(X)

    pipeline = InferencePipeline(artifact_dir=artifact_dir)
    model = pipeline._model

    logger.info(f"Model: {model.metadata.name} v{model.metadata.version}")

    if shap:
        output_dir.mkdir(parents=True, exist_ok=True)
        background = X_proc.sample(n=200, random_state=42)
        explainer = ShapExplainer(model._model, background)
        explainer.fit(X_proc.sample(n=500, random_state=0))
        explainer.plot_summary(
            X_proc.sample(n=500, random_state=0),
            output_path=output_dir / "shap_summary.png",
        )
        importance = explainer.global_importance()
        logger.info(f"\nTop 10 SHAP features:\n{importance.head(10).to_string()}")

    logger.success("Evaluation complete")


if __name__ == "__main__":
    app()
