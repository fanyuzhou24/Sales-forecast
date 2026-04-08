from pathlib import Path

from sales_forecast_system.modeling.recommender import ModelRuntimeParams, ScenarioModelRecommender
from sales_forecast_system.storage.model_store import ModelMeta, ModelMetadataStore


def test_scenario_recommendation_and_runtime_validation() -> None:
    recommender = ScenarioModelRecommender()
    models = recommender.recommend("爆品")
    assert "xgboost" in models

    recommender.validate_runtime(ModelRuntimeParams(history_months=12, forecast_months=6, confidence_level=0.95))


def test_model_version_rollback(tmp_path: Path) -> None:
    store = ModelMetadataStore(tmp_path / "versions.json")
    store.save(ModelMeta(model_name="random_forest", model_path="/m/v1.pkl", created_at="2026-04-08", metrics={"mae": 1.2}))
    store.save(ModelMeta(model_name="random_forest", model_path="/m/v2.pkl", created_at="2026-04-09", metrics={"mae": 1.0}))

    rolled = store.rollback("random_forest", 0)
    assert rolled["model_path"] == "/m/v1.pkl"
    assert rolled["rollback_to_version"] == 0
