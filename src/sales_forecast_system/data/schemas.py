from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class SalesRecord(BaseModel):
    date: datetime
    item_id: str
    sales: float = Field(ge=0)
    price: float | None = Field(default=None, ge=0)
    promo: int | None = Field(default=0)


class ExternalSignalRecord(BaseModel):
    date: datetime
    location: str
    temperature: float | None = None
    rainfall: float | None = None
    event_level: float | None = None
