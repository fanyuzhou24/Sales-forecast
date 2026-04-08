from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..collaboration.workflow import CollaborationManager, ManualOverride
from ..execution.runner import build_engine, run_pipeline
from ..forecasting.service import MultiDimensionForecaster, accuracy_metrics
from ..modeling.recommender import ModelRuntimeParams, ScenarioModelRecommender
from ..notifications.notifier import ConsoleNotifier
from ..reporting.exporter import ReportExporter
from ..review.service import ReviewConfig, ReviewEngine
from .schemas import (
    ExportReportRequest,
    ExportReviewRequest,
    ManualOverrideRequest,
    MultiDimForecastRequest,
    PredictRequest,
    RecommendModelRequest,
    RollbackModelRequest,
    RunReviewRequest,
    TrackActionRequest,
    TrainRequest,
)

router = APIRouter(prefix="/api/v1", tags=["forecast"])
collab_manager = CollaborationManager()
notifier = ConsoleNotifier()
report_exporter = ReportExporter()
recommender = ScenarioModelRecommender()
review_engine = ReviewEngine()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/train")
def train_model(req: TrainRequest) -> dict:
    # 训练接口：由客户端传入训练数据路径与模型参数。
    data_path = Path(req.train_data_path)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"训练数据不存在: {data_path}")

    runtime = ModelRuntimeParams(
        history_months=req.history_months,
        forecast_months=req.forecast_months,
        confidence_level=req.confidence_level,
    )
    recommender.validate_runtime(runtime)
    recommended = recommender.recommend(req.scenario)

    result = run_pipeline(
        train_data=data_path,
        output_dir=Path(req.output_dir),
        model_name=req.model_name if req.model_name else recommended[0],
        external_api_url=req.external_api_url,
        history_months=req.history_months,
        confidence_level=req.confidence_level,
    )
    result["scenario"] = req.scenario
    result["recommended_models"] = recommended
    result["runtime"] = runtime.__dict__
    return result


@router.post("/predict")
def predict(req: PredictRequest) -> list[dict]:
    # 推理接口：读取批量数据并返回逐行预测结果。
    data_path = Path(req.data_path)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"待预测数据不存在: {data_path}")

    df = pd.read_csv(data_path)
    engine = build_engine(Path(req.model_dir).parent)
    try:
        pred_df = engine.predict(df, model_name=req.model_name, confidence_level=req.confidence_level)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return pred_df.to_dict(orient="records")


@router.post("/model/recommend")
def recommend_model(req: RecommendModelRequest) -> dict:
    return {"scenario": req.scenario, "models": recommender.recommend(req.scenario)}


@router.post("/model/rollback")
def rollback_model(req: RollbackModelRequest) -> dict:
    registry = build_engine(Path("artifacts/models")).model_registry
    result = registry.rollback(req.model_name, req.version_index)
    notifier.send(
        title="模型版本已回滚",
        content=f"model={req.model_name}, version_index={req.version_index}",
        recipients=["ml_admin"],
    )
    return result


@router.post("/forecast/multi-dim")
def multi_dim_forecast(req: MultiDimForecastRequest) -> dict:
    data_path = Path(req.data_path)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"数据不存在: {data_path}")

    df = pd.read_csv(data_path)
    forecaster = MultiDimensionForecaster(date_col="date", target_col="sales")
    summary = forecaster.forecast(
        df,
        dimension=req.dimension,
        grain=req.grain,
        dimension_value=req.dimension_value,
    )
    return summary.__dict__


@router.post("/forecast/override")
def apply_manual_override(req: ManualOverrideRequest) -> dict:
    override = collab_manager.add_override(
        ManualOverride(
            dimension=req.dimension,
            dimension_value=req.dimension_value,
            grain=req.grain,
            override_value=req.override_value,
            reason=req.reason,
            operator=req.operator,
        )
    )
    notifier.send(
        title="预测人工干预已提交",
        content=f"{req.operator} 对 {req.dimension}:{req.dimension_value} 提交了人工调整，值={req.override_value}",
        recipients=["forecast_admin"],
    )
    return {
        "status": "ok",
        "override": {
            "dimension": override.dimension,
            "dimension_value": override.dimension_value,
            "grain": override.grain,
            "override_value": override.override_value,
            "reason": override.reason,
            "operator": override.operator,
        },
    }


@router.post("/report/export")
def export_report(req: ExportReportRequest) -> dict:
    source = Path(req.report_data_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"报告数据不存在: {source}")
    df = pd.read_csv(source)

    export_dir = Path(req.export_dir)
    excel_path = report_exporter.export_excel(df, export_dir / "forecast_report.xlsx")
    pdf_path = report_exporter.export_pdf(df, export_dir / "forecast_report.pdf")

    share_excel = report_exporter.share(excel_path, req.roles)
    share_pdf = report_exporter.share(pdf_path, req.roles)

    notifier.send(
        title="预测评估报告已导出",
        content=f"Excel: {excel_path}\\nPDF: {pdf_path}",
        recipients=req.roles,
    )

    # 若报告中含 actual/forecast/target 列，额外返回准确率、MAE、目标偏差等评估指标。
    metrics = {}
    if {"actual", "forecast"}.issubset(df.columns):
        target = df["target"] if "target" in df.columns else None
        metrics = accuracy_metrics(df["actual"], df["forecast"], target=target)

    return {
        "excel": share_excel,
        "pdf": share_pdf,
        "metrics": metrics,
    }


@router.post("/review/run")
def run_review(req: RunReviewRequest) -> dict:
    path = Path(req.review_data_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"复盘数据不存在: {path}")

    df = pd.read_csv(path)
    result = review_engine.run_review(
        df=df,
        config=ReviewConfig(cycle=req.cycle, manual_trigger=True),
        trigger=req.trigger,
    )
    return result.__dict__


@router.post("/review/action")
def track_review_action(req: TrackActionRequest) -> dict:
    item = review_engine.track_action(
        owner=req.owner,
        action=req.action,
        due_date=req.due_date,
        status=req.status,
    )
    notifier.send(
        title="复盘优化措施已登记",
        content=f"{req.owner} 提交措施: {req.action}",
        recipients=["forecast_admin"],
    )
    return item


@router.post("/review/export")
def export_review(req: ExportReviewRequest) -> dict:
    path = Path(req.review_data_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"复盘数据不存在: {path}")
    df = pd.read_csv(path)

    review = review_engine.run_review(
        df=df,
        config=ReviewConfig(cycle=req.cycle, manual_trigger=True),
        trigger="manual",
    )
    exported = report_exporter.export_review_report(
        detail_df=df,
        bias_breakdown=review.bias_breakdown,
        recommendations=review.recommendations,
        actions=review_engine.action_tracker,
        output_dir=Path(req.export_dir),
    )
    notifier.send(
        title="复盘报告已导出",
        content=str(exported),
        recipients=req.roles,
    )
    return {
        "review": review.__dict__,
        "files": exported,
        "shared_roles": req.roles,
        "actions": review_engine.action_tracker,
    }
