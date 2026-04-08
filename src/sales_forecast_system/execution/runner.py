from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..data.connectors import DataIntegrator, ExternalAPIConnector, InternalAPIConnector, InternalCSVConnector
from ..engine.forecast_engine import ForecastEngine
from ..features.engineering import FeatureEngineer
from ..modeling.registry import ModelRegistry


def build_engine(model_dir: Path) -> ForecastEngine:
    cfg = load_config()
    feature_engineer = FeatureEngineer(
        date_col=cfg.feature.date_col,
        target_col=cfg.feature.target_col,
        item_col=cfg.feature.item_col,
        lags=cfg.feature.lags,
        rolling_windows=cfg.feature.rolling_windows,
    )
    model_registry = ModelRegistry(model_dir=model_dir)
    return ForecastEngine(feature_engineer=feature_engineer, model_registry=model_registry, target_col=cfg.feature.target_col)


def run_pipeline(
    train_data: Path,
    output_dir: Path,
    model_name: str = "random_forest",
    external_api_url: str | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    internal_df = InternalCSVConnector(train_data).load()

    external_df = None
    if external_api_url:
        external_df = ExternalAPIConnector(external_api_url).load()

    merged = DataIntegrator(date_col="date").merge(internal_df, external_df)

    engine = build_engine(output_dir / "models")
    train_result = engine.train(merged, model_name=model_name)
    prediction_df = engine.predict(merged, model_name=model_name)

    prediction_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"

    prediction_df.to_csv(prediction_path, index=False)
    pd.Series(train_result.metrics).to_json(metrics_path, indent=2)

    return {
        "model": train_result.artifact.model_name,
        "model_path": str(train_result.artifact.model_path),
        "prediction_path": str(prediction_path),
        "metrics": train_result.metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sales Forecast Pipeline Runner")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-name", type=str, default="random_forest")
    parser.add_argument("--external-api-url", type=str, default=None)

    args = parser.parse_args()
    result = run_pipeline(
        train_data=args.train_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        external_api_url=args.external_api_url,
    )
    print(result)


if __name__ == "__main__":
    main()
