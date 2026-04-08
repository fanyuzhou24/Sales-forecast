from __future__ import annotations

from dataclasses import dataclass


SCENARIO_MODEL_MAP = {
    "常规品": ["random_forest", "lightgbm", "fusion"],
    "爆品": ["xgboost", "lightgbm", "fusion"],
    "季节品": ["prophet", "arima", "fusion"],
    "新品上市": ["lightgbm", "random_forest", "fusion"],
}


@dataclass
class ModelRuntimeParams:
    history_months: int = 12
    forecast_months: int = 3
    confidence_level: float = 0.95


class ScenarioModelRecommender:
    """按业务场景推荐模型组合。"""

    def recommend(self, scenario: str) -> list[str]:
        return SCENARIO_MODEL_MAP.get(scenario, ["random_forest", "fusion"])

    def validate_runtime(self, params: ModelRuntimeParams) -> None:
        if params.history_months not in (6, 12, 24):
            raise ValueError("history_months 仅支持 6/12/24")
        if not (1 <= params.forecast_months <= 12):
            raise ValueError("forecast_months 需在 1~12")
        if not (0.5 <= params.confidence_level < 1.0):
            raise ValueError("confidence_level 需在 [0.5, 1.0)")
