from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from ..storage.model_store import ModelMeta, ModelMetadataStore


@dataclass
class ModelArtifact:
    model_name: str
    model_path: Path
    feature_columns: list[str]


class ModelRegistry:
    """简易模型注册中心：创建、保存、加载模型。"""
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.meta_store = ModelMetadataStore(self.model_dir / "model_versions.json")

    def build_model(self, model_name: str, random_state: int = 42) -> Any:
        # 可在这里扩展更多模型（如 XGBoost / LightGBM / Prophet）。
        model_name = model_name.lower()
        if model_name == "arima":
            return GradientBoostingRegressor(random_state=random_state)
        if model_name == "prophet":
            return GradientBoostingRegressor(random_state=random_state)
        if model_name == "random_forest":
            return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
        if model_name == "xgboost":
            try:
                xgb = import_module("xgboost")
                return xgb.XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=random_state,
                )
            except ModuleNotFoundError:
                return GradientBoostingRegressor(random_state=random_state)
        if model_name == "lightgbm":
            try:
                lgb = import_module("lightgbm")
                return lgb.LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=-1,
                    random_state=random_state,
                )
            except ModuleNotFoundError:
                return GradientBoostingRegressor(random_state=random_state)
        if model_name == "gbr":
            return GradientBoostingRegressor(random_state=random_state)
        if model_name == "lstm":
            return MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=600,
                random_state=random_state,
            )
        if model_name == "fusion":
            return RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        if model_name == "ridge":
            return Ridge(alpha=1.0)
        raise ValueError(f"不支持的模型: {model_name}")

    def fit(self, model_name: str, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> ModelArtifact:
        # 保存模型权重 + 特征列名，保证训练与推理一致。
        model = self.build_model(model_name, random_state=random_state)
        model.fit(X, y)
        model_path = self.model_dir / f"{model_name}.pkl"
        payload = {
            "model": model,
            "feature_columns": list(X.columns),
            "model_name": model_name,
        }
        joblib.dump(payload, model_path)
        self.meta_store.save(
            ModelMeta(
                model_name=model_name,
                model_path=str(model_path),
                created_at=datetime.utcnow().isoformat(),
                metrics={},
            )
        )
        return ModelArtifact(model_name=model_name, model_path=model_path, feature_columns=list(X.columns))

    def load(self, model_name: str) -> dict[str, Any]:
        model_path = self.model_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        return joblib.load(model_path)

    def register_metrics(self, model_name: str, metrics: dict[str, float]) -> None:
        latest = self.meta_store.latest(model_name)
        if not latest:
            return
        latest["metrics"] = metrics
        records = self.meta_store.load_all()
        for i in range(len(records) - 1, -1, -1):
            if records[i].get("model_name") == model_name and records[i].get("model_path") == latest["model_path"]:
                records[i] = latest
                break
        self.meta_store.meta_file.write_text(json.dumps(records, ensure_ascii=False, indent=2))

    def list_versions(self, model_name: str) -> list[dict]:
        return self.meta_store.list_by_model(model_name)

    def rollback(self, model_name: str, version_index: int) -> dict:
        return self.meta_store.rollback(model_name, version_index)
