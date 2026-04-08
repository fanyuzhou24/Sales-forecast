from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd


REVIEW_FREQ = {
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
}


@dataclass
class ReviewConfig:
    cycle: str = "monthly"
    manual_trigger: bool = True


@dataclass
class ReviewResult:
    cycle: str
    triggered_at: str
    metrics: dict[str, float]
    bias_breakdown: dict[str, dict[str, float]]
    bias_type: str
    recommendations: list[str]


class ReviewEngine:
    """预测复盘引擎：支持周期配置、手动触发、偏差分析与优化建议。"""

    def __init__(self) -> None:
        self.custom_metrics: dict[str, Callable[[pd.Series, pd.Series], float]] = {}
        self.action_tracker: list[dict] = []

    def register_metric(self, name: str, fn: Callable[[pd.Series, pd.Series], float]) -> None:
        self.custom_metrics[name] = fn

    def _base_metrics(self, actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
        actual = actual.astype(float)
        forecast = forecast.astype(float)
        mae = float((actual - forecast).abs().mean())
        mse = float(((actual - forecast) ** 2).mean())
        mape = float((((actual - forecast).abs() / actual.replace(0, np.nan)).fillna(0)).mean() * 100)
        accuracy = float(max(0.0, 100 - mape))
        bias_rate = float(((forecast - actual).mean() / (actual.mean() + 1e-6)) * 100)
        return {
            "accuracy": accuracy,
            "mae": mae,
            "mse": mse,
            "bias_rate": bias_rate,
        }

    def _bias_breakdown(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for dim in ("product_id", "region"):
            if dim in df.columns:
                grp = df.groupby(dim).apply(lambda x: float((x["forecast"] - x["actual"]).mean()))
                out[dim] = {str(k): float(v) for k, v in grp.to_dict().items()}

        if "date" in df.columns:
            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp["date"])
            tmp["ym"] = tmp["date"].dt.to_period("M").astype(str)
            grp = tmp.groupby("ym").apply(lambda x: float((x["forecast"] - x["actual"]).mean()))
            out["time"] = {str(k): float(v) for k, v in grp.to_dict().items()}
        return out

    def _classify_bias(self, bias_breakdown: dict[str, dict[str, float]]) -> str:
        values = []
        for _, m in bias_breakdown.items():
            values.extend(list(m.values()))
        if not values:
            return "unknown"
        std = float(np.std(values))
        mean = float(np.mean(values))
        if abs(mean) > 0.01 and std < abs(mean):
            return "systematic"
        return "random"

    def _suggestions(self, metrics: dict[str, float], bias_type: str) -> list[str]:
        rec = []
        if metrics.get("mse", 0) > 1000:
            rec.append("建议调整模型参数（如树深、学习率、正则化）")
        if abs(metrics.get("bias_rate", 0)) > 10:
            rec.append("建议补充影响因子维度（价格、活动、政策）")
        if bias_type == "systematic":
            rec.append("存在系统性偏差，建议分场景建模并引入分层特征")
        else:
            rec.append("偏差以偶然性为主，建议优化业务执行与数据采集流程")
        return rec

    def run_review(self, df: pd.DataFrame, config: ReviewConfig, trigger: str = "manual") -> ReviewResult:
        if config.cycle not in REVIEW_FREQ:
            raise ValueError("cycle 仅支持 weekly/monthly/quarterly")
        if trigger == "manual" and not config.manual_trigger:
            raise PermissionError("当前配置不允许手动触发")
        if {"actual", "forecast"}.issubset(df.columns) is False:
            raise ValueError("复盘数据必须包含 actual 与 forecast 列")

        metrics = self._base_metrics(df["actual"], df["forecast"])
        for name, fn in self.custom_metrics.items():
            metrics[name] = float(fn(df["actual"], df["forecast"]))

        bias_breakdown = self._bias_breakdown(df)
        bias_type = self._classify_bias(bias_breakdown)
        rec = self._suggestions(metrics, bias_type)

        return ReviewResult(
            cycle=config.cycle,
            triggered_at=datetime.utcnow().isoformat(),
            metrics=metrics,
            bias_breakdown=bias_breakdown,
            bias_type=bias_type,
            recommendations=rec,
        )

    def track_action(self, owner: str, action: str, due_date: str, status: str = "open") -> dict:
        item = {
            "owner": owner,
            "action": action,
            "due_date": due_date,
            "status": status,
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.action_tracker.append(item)
        return item
