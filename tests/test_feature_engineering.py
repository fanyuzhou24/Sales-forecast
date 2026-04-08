import pandas as pd

from sales_forecast_system.features.engineering import FeatureEngineer


def test_feature_engineering_generates_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=40, freq="D"),
            "item_id": ["A"] * 40,
            "sales": [float(i + 1) for i in range(40)],
            "price": [10.0] * 40,
            "promo": [0, 1] * 20,
        }
    )

    fe = FeatureEngineer()
    out = fe.transform(df, dropna=True)

    assert "lag_1" in out.columns
    assert "rolling_mean_7" in out.columns
    assert "is_weekend" in out.columns
    assert len(out) > 0
