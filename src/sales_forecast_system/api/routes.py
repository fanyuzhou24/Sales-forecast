from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..engine.forecast_engine import ForecastEngine
from ..execution.runner import build_engine, run_pipeline
from ..features.engineering import FeatureEngineer
from ..modeling.registry import ModelRegistry
from .schemas import PredictRequest, TrainRequest

router = APIRouter(prefix="/api/v1", tags=["forecast"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/train")
def train_model(req: TrainRequest) -> dict:
    data_path = Path(req.train_data_path)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"训练数据不存在: {data_path}")

    result = run_pipeline(
        train_data=data_path,
        output_dir=Path(req.output_dir),
        model_name=req.model_name,
        external_api_url=req.external_api_url,
    )
    return result


@router.post("/predict")
def predict(req: PredictRequest) -> list[dict]:
    data_path = Path(req.data_path)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"待预测数据不存在: {data_path}")

    df = pd.read_csv(data_path)
    engine = build_engine(Path(req.model_dir).parent)
    try:
        pred_df = engine.predict(df, model_name=req.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return pred_df.to_dict(orient="records")
