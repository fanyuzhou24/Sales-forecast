import pandas as pd

from sales_forecast_system.collaboration.workflow import CollaborationManager, ManualOverride
from sales_forecast_system.forecasting.service import MultiDimensionForecaster, accuracy_metrics


def test_multi_dimension_and_grain_forecast_summary() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=420, freq="D"),
            "sales": [100 + (i % 30) for i in range(420)],
            "product_id": ["P1"] * 420,
            "customer_id": ["C1"] * 420,
            "region": ["CN-SH"] * 420,
            "channel": ["online"] * 420,
        }
    )
    svc = MultiDimensionForecaster()
    out = svc.forecast(df, dimension="product", grain="month", dimension_value="P1")

    assert out.prediction >= 0
    assert out.upper >= out.lower
    assert isinstance(out.trend, str)


def test_override_and_metrics() -> None:
    mgr = CollaborationManager()
    mgr.add_override(
        ManualOverride(
            dimension="product",
            dimension_value="P1",
            grain="month",
            override_value=123.4,
            reason="大促调整",
            operator="alice",
        )
    )
    latest = mgr.latest_override("product", "P1", "month")
    assert latest is not None
    assert latest.override_value == 123.4

    metric = accuracy_metrics(
        pd.Series([100, 110, 120]),
        pd.Series([98, 112, 121]),
        target=pd.Series([105, 115, 125]),
    )
    assert "accuracy" in metric
    assert "target_gap" in metric
