from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..evaluation.metrics import evaluate_all
from ..features.engineering import FeatureEngineer
from ..modeling.registry import ModelArtifact, ModelRegistry


@dataclass
class TrainResult:
    artifact: ModelArtifact
    metrics: dict[str, float]


class ForecastEngine:
    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        model_registry: ModelRegistry,
        target_col: str = "sales",
    ) -> None:
        self.feature_engineer = feature_engineer
        self.model_registry = model_registry
        self.target_col = target_col

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        excluded = {"date", "item_id", self.target_col}
        return [c for c in df.columns if c not in excluded]

    def train(self, train_df: pd.DataFrame, model_name: str = "random_forest", random_state: int = 42) -> TrainResult:
        transformed = self.feature_engineer.transform(train_df, dropna=True)
        feature_cols = self._select_features(transformed)
        X = transformed[feature_cols]
        y = transformed[self.target_col]

        split_idx = int(len(transformed) * 0.8)
        X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

        artifact = self.model_registry.fit(model_name, X_train, y_train, random_state=random_state)
        payload = self.model_registry.load(model_name)
        pred = payload["model"].predict(X_valid)
        metrics = evaluate_all(y_valid.reset_index(drop=True), pd.Series(pred))
        return TrainResult(artifact=artifact, metrics=metrics)

    def predict(self, df: pd.DataFrame, model_name: str = "random_forest") -> pd.DataFrame:
        transformed = self.feature_engineer.transform(df, dropna=True)
        payload = self.model_registry.load(model_name)
        model = payload["model"]
        feature_columns = payload["feature_columns"]

        for col in feature_columns:
            if col not in transformed.columns:
                transformed[col] = 0.0
        X = transformed[feature_columns]
        transformed["prediction"] = model.predict(X)
        return transformed[["date", "item_id", self.target_col, "prediction"]].copy()
