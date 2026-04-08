from __future__ import annotations

from pydantic import BaseModel


class TrainRequest(BaseModel):
    train_data_path: str
    output_dir: str = "artifacts"
    model_name: str = "random_forest"
    external_api_url: str | None = None
    scenario: str = "常规品"
    history_months: int = 12
    forecast_months: int = 3
    confidence_level: float = 0.95


class PredictRequest(BaseModel):
    data_path: str
    model_name: str = "random_forest"
    model_dir: str = "artifacts/models"
    confidence_level: float = 0.95


class MultiDimForecastRequest(BaseModel):
    data_path: str
    dimension: str = "product"
    dimension_value: str | None = None
    grain: str = "month"


class ManualOverrideRequest(BaseModel):
    dimension: str
    dimension_value: str
    grain: str
    override_value: float
    reason: str
    operator: str


class ExportReportRequest(BaseModel):
    report_data_path: str
    export_dir: str = "artifacts/reports"
    roles: list[str] = ["sales_manager"]


class RecommendModelRequest(BaseModel):
    scenario: str = "常规品"


class RollbackModelRequest(BaseModel):
    model_name: str
    version_index: int


class RunReviewRequest(BaseModel):
    review_data_path: str
    cycle: str = "monthly"
    trigger: str = "manual"


class TrackActionRequest(BaseModel):
    owner: str
    action: str
    due_date: str
    status: str = "open"


class ExportReviewRequest(BaseModel):
    review_data_path: str
    export_dir: str = "artifacts/review"
    cycle: str = "monthly"
    roles: list[str] = ["sales_manager"]
