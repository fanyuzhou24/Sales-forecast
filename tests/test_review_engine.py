import pandas as pd

from sales_forecast_system.review.service import ReviewConfig, ReviewEngine


def test_review_run_and_action_tracking() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=90, freq="D"),
            "actual": [100 + i % 20 for i in range(90)],
            "forecast": [98 + i % 20 for i in range(90)],
            "product_id": ["P1"] * 90,
            "region": ["华东"] * 90,
        }
    )

    engine = ReviewEngine()
    engine.register_metric("custom_mean_error", lambda y, yhat: float((yhat - y).mean()))
    out = engine.run_review(df, ReviewConfig(cycle="monthly", manual_trigger=True), trigger="manual")

    assert "mae" in out.metrics
    assert "custom_mean_error" in out.metrics
    assert out.bias_type in {"systematic", "random", "unknown"}

    action = engine.track_action(owner="ops", action="补充活动数据", due_date="2026-05-01")
    assert action["status"] == "open"
