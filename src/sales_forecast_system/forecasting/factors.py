from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FactorWeight:
    name: str
    weight: float


class ExternalInternalFactorAdjuster:
    """配置内外部因素权重，并对基础预测做加权修正。"""

    def __init__(self, factors: list[FactorWeight] | None = None) -> None:
        self.factors = factors or []

    def set_factors(self, factors: list[FactorWeight]) -> None:
        self.factors = factors

    def impact_score(self, row: pd.Series) -> float:
        score = 0.0
        for f in self.factors:
            score += float(row.get(f.name, 0.0)) * f.weight
        return score

    def adjust(self, df: pd.DataFrame, base_col: str = "prediction") -> pd.DataFrame:
        out = df.copy()
        if base_col not in out.columns:
            raise ValueError(f"缺少基础预测列: {base_col}")
        out["factor_impact"] = out.apply(self.impact_score, axis=1)
        out["adjusted_prediction"] = (out[base_col].astype(float) * (1 + out["factor_impact"]))
        return out
