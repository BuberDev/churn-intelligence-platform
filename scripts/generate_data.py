"""CLI: generate synthetic customer data and save to Parquet."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from src.data.generator import SyntheticDataGenerator
from src.data.loader import DataLoader

app = typer.Typer(help="Generate synthetic churn dataset")


@app.command()
def main(
    samples: int = typer.Option(50_000, "--samples", "-n", help="Number of customer records"),
    output: Path = typer.Option(
        Path("data/raw/customers.parquet"), "--output", "-o", help="Output file path"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
) -> None:
    """Generate a synthetic customer dataset and persist it to Parquet."""
    logger.info(f"Generating {samples:,} records (seed={seed})")
    df = SyntheticDataGenerator(random_state=seed).generate(n_samples=samples)
    DataLoader().save_parquet(df, output)
    logger.success(f"Saved to {output} ({len(df):,} rows, {df.memory_usage().sum() / 1e6:.1f} MB)")


if __name__ == "__main__":
    app()
