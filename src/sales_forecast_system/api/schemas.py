from __future__ import annotations

from pydantic import BaseModel


class TrainRequest(BaseModel):
    train_data_path: str
    output_dir: str = "artifacts"
    model_name: str = "random_forest"
    external_api_url: str | None = None


class PredictRequest(BaseModel):
    data_path: str
    model_name: str = "random_forest"
    model_dir: str = "artifacts/models"
