from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series, epsilon: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def wape(y_true: pd.Series, y_pred: pd.Series, epsilon: float = 1e-6) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + epsilon) * 100)


def evaluate_all(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }
