from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


@dataclass
class InternalCSVConnector:
    file_path: Path

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"内部数据文件不存在: {self.file_path}")
        df = pd.read_csv(self.file_path)
        return df


@dataclass
class InternalAPIConnector:
    base_url: str
    timeout: int = 30

    def load(self, endpoint: str = "/sales") -> pd.DataFrame:
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        return pd.DataFrame(payload)


@dataclass
class ExternalAPIConnector:
    base_url: str
    api_key: str | None = None
    timeout: int = 30

    def load(self, params: dict[str, Any] | None = None) -> pd.DataFrame:
        params = params.copy() if params else {}
        if self.api_key:
            params.setdefault("api_key", self.api_key)
        response = requests.get(self.base_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        return pd.DataFrame(payload)


class DataIntegrator:
    """合并内部销售与外部因子数据。"""

    def __init__(self, date_col: str = "date") -> None:
        self.date_col = date_col

    def merge(self, internal_df: pd.DataFrame, external_df: pd.DataFrame | None = None) -> pd.DataFrame:
        df = internal_df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        if external_df is None or external_df.empty:
            return df.sort_values(self.date_col).reset_index(drop=True)

        ext = external_df.copy()
        ext[self.date_col] = pd.to_datetime(ext[self.date_col])
        merged = df.merge(ext, on=self.date_col, how="left")
        return merged.sort_values(self.date_col).reset_index(drop=True)
