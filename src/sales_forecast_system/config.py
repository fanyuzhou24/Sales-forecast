from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class DataSourceConfig(BaseModel):
    internal_csv_path: Path | None = None
    internal_api_url: str | None = None
    external_api_url: str | None = None
    external_api_key: str | None = None


class FeatureConfig(BaseModel):
    lags: tuple[int, ...] = (1, 7, 14)
    rolling_windows: tuple[int, ...] = (7, 14, 28)
    target_col: str = "sales"
    date_col: str = "date"
    item_col: str = "item_id"


class ModelConfig(BaseModel):
    default_model_name: str = "random_forest"
    random_state: int = 42


class AppConfig(BaseModel):
    project_name: str = "Sales Forecast System"
    model_dir: Path = Field(default=Path("artifacts/models"))
    data: DataSourceConfig = Field(default_factory=DataSourceConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_config() -> AppConfig:
    return AppConfig()
