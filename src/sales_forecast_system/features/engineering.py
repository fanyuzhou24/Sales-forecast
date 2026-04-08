from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """统一特征工程入口。

    包含：
    1) 时间特征
    2) 滞后特征
    3) 滚动统计特征
    4) 价格与促销交叉特征
    """
    def __init__(
        self,
        date_col: str = "date",
        target_col: str = "sales",
        item_col: str = "item_id",
        lags: tuple[int, ...] = (1, 7, 14),
        rolling_windows: tuple[int, ...] = (7, 14, 28),
    ) -> None:
        self.date_col = date_col
        self.target_col = target_col
        self.item_col = item_col
        self.lags = lags
        self.rolling_windows = rolling_windows

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 生成日历类特征，帮助模型学习周内/月份季节性。
        out = df.copy()
        out[self.date_col] = pd.to_datetime(out[self.date_col])
        out["year"] = out[self.date_col].dt.year
        out["month"] = out[self.date_col].dt.month
        out["day"] = out[self.date_col].dt.day
        out["day_of_week"] = out[self.date_col].dt.dayofweek
        out["week_of_year"] = out[self.date_col].dt.isocalendar().week.astype(int)
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
        return out

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 使用历史销量作为监督学习特征，避免使用未来信息。
        out = df.copy()
        out = out.sort_values([self.item_col, self.date_col])
        for lag in self.lags:
            out[f"lag_{lag}"] = out.groupby(self.item_col)[self.target_col].shift(lag)
        return out

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 通过滚动均值/波动率刻画趋势与不确定性。
        out = df.copy()
        out = out.sort_values([self.item_col, self.date_col])
        for window in self.rolling_windows:
            g = out.groupby(self.item_col)[self.target_col]
            out[f"rolling_mean_{window}"] = g.shift(1).rolling(window).mean()
            out[f"rolling_std_{window}"] = g.shift(1).rolling(window).std()
        return out

    def add_price_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 价格变化和促销交叉能反映价格弹性与营销影响。
        out = df.copy()
        if "price" in out.columns:
            out["price"] = out["price"].astype(float)
            out["price_change_1d"] = out.groupby(self.item_col)["price"].diff(1)
        if "promo" in out.columns:
            out["promo"] = out["promo"].fillna(0).astype(int)
            if "price" in out.columns:
                out["promo_price_interaction"] = out["promo"] * out["price"]
        return out

    def transform(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        # 串联所有步骤，形成可直接喂给模型的训练/推理特征表。
        out = self.add_time_features(df)
        out = self.add_lag_features(out)
        out = self.add_rolling_features(out)
        out = self.add_price_promo_features(out)

        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

        if dropna:
            out = out.dropna().reset_index(drop=True)
        return out
