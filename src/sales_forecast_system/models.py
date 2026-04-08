from datetime import date
from pydantic import BaseModel, Field


class SalesPoint(BaseModel):
    sku_id: str = Field(..., description="SKU 编号")
    date: date = Field(..., description="销售日期")
    quantity: float = Field(..., ge=0)
    revenue: float = Field(..., ge=0)


class ForecastRequest(BaseModel):
    sku_id: str
    horizon_days: int = Field(default=30, ge=1, le=365)


class ForecastPoint(BaseModel):
    date: date
    yhat: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    sku_id: str
    model_name: str
    points: list[ForecastPoint]
