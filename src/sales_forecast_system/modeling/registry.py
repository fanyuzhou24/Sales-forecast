from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge


@dataclass
class ModelArtifact:
    model_name: str
    model_path: Path
    feature_columns: list[str]


class ModelRegistry:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def build_model(self, model_name: str, random_state: int = 42) -> Any:
        model_name = model_name.lower()
        if model_name == "random_forest":
            return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
        if model_name == "gbr":
            return GradientBoostingRegressor(random_state=random_state)
        if model_name == "ridge":
            return Ridge(alpha=1.0)
        raise ValueError(f"不支持的模型: {model_name}")

    def fit(self, model_name: str, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> ModelArtifact:
        model = self.build_model(model_name, random_state=random_state)
        model.fit(X, y)
        model_path = self.model_dir / f"{model_name}.pkl"
        payload = {
            "model": model,
            "feature_columns": list(X.columns),
            "model_name": model_name,
        }
        joblib.dump(payload, model_path)
        return ModelArtifact(model_name=model_name, model_path=model_path, feature_columns=list(X.columns))

    def load(self, model_name: str) -> dict[str, Any]:
        model_path = self.model_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        return joblib.load(model_path)
