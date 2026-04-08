from pathlib import Path

import pandas as pd

from sales_forecast_system.engine.forecast_engine import ForecastEngine
from sales_forecast_system.features.engineering import FeatureEngineer
from sales_forecast_system.modeling.registry import ModelRegistry


def test_engine_train_and_predict(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=70, freq="D"),
            "item_id": ["A"] * 70,
            "sales": [100 + i * 0.3 for i in range(70)],
            "price": [20 + (i % 5) for i in range(70)],
            "promo": [1 if i % 10 == 0 else 0 for i in range(70)],
        }
    )

    engine = ForecastEngine(
        feature_engineer=FeatureEngineer(),
        model_registry=ModelRegistry(tmp_path / "models"),
    )

    result = engine.train(df, model_name="ridge")
    assert "mae" in result.metrics

    pred = engine.predict(df, model_name="ridge")
    assert {"date", "item_id", "sales", "prediction"}.issubset(pred.columns)
    assert len(pred) > 0
