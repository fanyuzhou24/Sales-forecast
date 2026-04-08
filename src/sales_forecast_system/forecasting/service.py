from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DIMENSION_COLUMNS = {
    "product": "product_id",
    "customer": "customer_id",
    "region": "region",
    "channel": "channel",
}

FREQ_MAP = {
    "day": "D",
    "week": "W",
    "month": "M",
    "quarter": "Q",
}


@dataclass
class ForecastSummary:
    dimension: str
    grain: str
    prediction: float
    lower: float
    upper: float
    yoy_growth: float
    mom_growth: float
    trend: str


class MultiDimensionForecaster:
    """满足多维度、可配置时间粒度的预测服务。"""

    def __init__(self, date_col: str = "date", target_col: str = "sales") -> None:
        self.date_col = date_col
        self.target_col = target_col

    def _resample(self, df: pd.DataFrame, grain: str, group_col: str | None = None) -> pd.DataFrame:
        if grain not in FREQ_MAP:
            raise ValueError(f"不支持的时间粒度: {grain}")

        out = df.copy()
        out[self.date_col] = pd.to_datetime(out[self.date_col])
        freq = FREQ_MAP[grain]

        if group_col:
            agg = (
                out.set_index(self.date_col)
                .groupby(group_col)[self.target_col]
                .resample(freq)
                .sum()
                .reset_index()
            )
            return agg

        agg = out.set_index(self.date_col)[self.target_col].resample(freq).sum().reset_index()
        return agg

    def _compute_interval(self, values: pd.Series) -> tuple[float, float, float]:
        prediction = float(values.iloc[-1]) if len(values) else 0.0
        std = float(values.std()) if len(values) > 1 else max(prediction * 0.1, 1.0)
        lower = max(prediction - 1.96 * std, 0.0)
        upper = prediction + 1.96 * std
        return prediction, lower, upper

    def _growth(self, values: pd.Series, offset: int) -> float:
        if len(values) <= offset or values.iloc[-offset - 1] == 0:
            return 0.0
        return float((values.iloc[-1] - values.iloc[-offset - 1]) / values.iloc[-offset - 1] * 100)

    def _trend(self, values: pd.Series) -> str:
        if len(values) < 3:
            return "数据不足，趋势不明显"
        x = np.arange(len(values))
        slope = np.polyfit(x, values.to_numpy(dtype=float), 1)[0]
        if slope > 0:
            return "整体呈上升趋势"
        if slope < 0:
            return "整体呈下降趋势"
        return "整体相对平稳"

    def forecast(self, df: pd.DataFrame, dimension: str = "product", grain: str = "month", dimension_value: str | None = None) -> ForecastSummary:
        group_col = DIMENSION_COLUMNS.get(dimension)
        if dimension != "all" and not group_col:
            raise ValueError(f"不支持的预测维度: {dimension}")

        working = df.copy()
        if group_col and dimension_value:
            working = working[working[group_col].astype(str) == str(dimension_value)]

        ts = self._resample(working, grain, None)
        values = ts[self.target_col]

        prediction, lower, upper = self._compute_interval(values)
        yoy_growth = self._growth(values, 12)
        mom_growth = self._growth(values, 1)
        trend = self._trend(values)

        return ForecastSummary(
            dimension=dimension,
            grain=grain,
            prediction=prediction,
            lower=lower,
            upper=upper,
            yoy_growth=yoy_growth,
            mom_growth=mom_growth,
            trend=trend,
        )


def accuracy_metrics(actual: pd.Series, forecast: pd.Series, target: pd.Series | None = None) -> dict[str, float]:
    actual = actual.astype(float)
    forecast = forecast.astype(float)
    mae = float((actual - forecast).abs().mean())
    mape = float(((actual - forecast).abs() / actual.replace(0, np.nan)).fillna(0).mean() * 100)
    acc = float(max(0.0, 100 - mape))

    result = {"mae": mae, "mape": mape, "accuracy": acc}
    if target is not None:
        target = target.astype(float)
        result["target_gap"] = float((forecast - target).mean())
    return result
