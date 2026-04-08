from datetime import date, timedelta
from fastapi import APIRouter, HTTPException

from ..database import InMemoryDatabase
from ..models import ForecastPoint, ForecastRequest, ForecastResponse, SalesPoint
from ..services.pipeline import ForecastPipeline

router = APIRouter(prefix="/api/v1", tags=["forecast"])
_db = InMemoryDatabase()
_pipeline = ForecastPipeline(_db)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/sales")
def add_sale(point: SalesPoint) -> dict:
    _pipeline.ingest(point.sku_id, [(point.date, point.quantity, point.revenue)])
    return {"message": "inserted", "sku_id": point.sku_id}


@router.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    if req.horizon_days < 1:
        raise HTTPException(status_code=400, detail="horizon_days must be positive")
    result = _pipeline.run(req.sku_id, req.horizon_days)
    points = [
        ForecastPoint(date=d, yhat=p.center, lower=p.lower, upper=p.upper)
        for d, p in zip(result.dates, result.predictions)
    ]
    return ForecastResponse(sku_id=req.sku_id, model_name=result.model_name, points=points)


@router.get("/insights/1/{sku_id}")
def insights_1(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 1,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/2/{sku_id}")
def insights_2(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 2,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/3/{sku_id}")
def insights_3(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 3,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/4/{sku_id}")
def insights_4(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 4,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/5/{sku_id}")
def insights_5(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 5,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/6/{sku_id}")
def insights_6(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 6,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/7/{sku_id}")
def insights_7(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 7,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/8/{sku_id}")
def insights_8(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 8,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/9/{sku_id}")
def insights_9(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 9,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/10/{sku_id}")
def insights_10(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 10,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/11/{sku_id}")
def insights_11(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 11,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/12/{sku_id}")
def insights_12(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 12,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/13/{sku_id}")
def insights_13(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 13,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/14/{sku_id}")
def insights_14(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 14,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/15/{sku_id}")
def insights_15(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 15,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/16/{sku_id}")
def insights_16(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 16,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/17/{sku_id}")
def insights_17(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 17,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/18/{sku_id}")
def insights_18(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 18,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/19/{sku_id}")
def insights_19(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 19,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/20/{sku_id}")
def insights_20(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 20,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/21/{sku_id}")
def insights_21(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 21,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/22/{sku_id}")
def insights_22(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 22,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/23/{sku_id}")
def insights_23(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 23,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/24/{sku_id}")
def insights_24(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 24,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/25/{sku_id}")
def insights_25(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 25,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/26/{sku_id}")
def insights_26(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 26,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/27/{sku_id}")
def insights_27(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 27,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/28/{sku_id}")
def insights_28(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 28,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/29/{sku_id}")
def insights_29(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 29,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/30/{sku_id}")
def insights_30(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 30,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/31/{sku_id}")
def insights_31(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 31,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/32/{sku_id}")
def insights_32(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 32,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/33/{sku_id}")
def insights_33(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 33,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/34/{sku_id}")
def insights_34(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 34,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/35/{sku_id}")
def insights_35(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 35,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/36/{sku_id}")
def insights_36(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 36,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/37/{sku_id}")
def insights_37(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 37,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/38/{sku_id}")
def insights_38(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 38,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/39/{sku_id}")
def insights_39(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 39,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/40/{sku_id}")
def insights_40(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 40,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/41/{sku_id}")
def insights_41(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 41,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/42/{sku_id}")
def insights_42(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 42,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/43/{sku_id}")
def insights_43(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 43,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/44/{sku_id}")
def insights_44(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 44,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/45/{sku_id}")
def insights_45(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 45,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/46/{sku_id}")
def insights_46(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 46,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/47/{sku_id}")
def insights_47(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 47,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/48/{sku_id}")
def insights_48(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 48,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/49/{sku_id}")
def insights_49(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 49,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/50/{sku_id}")
def insights_50(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 50,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/51/{sku_id}")
def insights_51(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 51,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/52/{sku_id}")
def insights_52(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 52,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/53/{sku_id}")
def insights_53(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 53,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/54/{sku_id}")
def insights_54(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 54,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/55/{sku_id}")
def insights_55(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 55,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/56/{sku_id}")
def insights_56(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 56,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/57/{sku_id}")
def insights_57(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 57,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/58/{sku_id}")
def insights_58(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 58,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/59/{sku_id}")
def insights_59(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 59,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/60/{sku_id}")
def insights_60(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 60,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/61/{sku_id}")
def insights_61(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 61,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/62/{sku_id}")
def insights_62(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 62,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/63/{sku_id}")
def insights_63(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 63,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/64/{sku_id}")
def insights_64(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 64,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/65/{sku_id}")
def insights_65(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 65,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/66/{sku_id}")
def insights_66(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 66,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/67/{sku_id}")
def insights_67(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 67,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/68/{sku_id}")
def insights_68(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 68,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/69/{sku_id}")
def insights_69(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 69,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/70/{sku_id}")
def insights_70(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 70,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/71/{sku_id}")
def insights_71(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 71,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/72/{sku_id}")
def insights_72(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 72,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/73/{sku_id}")
def insights_73(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 73,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/74/{sku_id}")
def insights_74(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 74,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/75/{sku_id}")
def insights_75(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 75,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/76/{sku_id}")
def insights_76(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 76,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/77/{sku_id}")
def insights_77(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 77,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/78/{sku_id}")
def insights_78(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 78,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/79/{sku_id}")
def insights_79(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 79,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/80/{sku_id}")
def insights_80(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 80,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/81/{sku_id}")
def insights_81(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 81,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/82/{sku_id}")
def insights_82(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 82,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/83/{sku_id}")
def insights_83(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 83,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/84/{sku_id}")
def insights_84(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 84,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/85/{sku_id}")
def insights_85(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 85,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/86/{sku_id}")
def insights_86(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 86,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/87/{sku_id}")
def insights_87(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 87,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/88/{sku_id}")
def insights_88(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 88,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/89/{sku_id}")
def insights_89(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 89,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/90/{sku_id}")
def insights_90(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 90,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/91/{sku_id}")
def insights_91(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 91,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/92/{sku_id}")
def insights_92(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 92,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/93/{sku_id}")
def insights_93(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 93,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/94/{sku_id}")
def insights_94(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 94,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/95/{sku_id}")
def insights_95(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 95,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/96/{sku_id}")
def insights_96(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 96,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/97/{sku_id}")
def insights_97(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 97,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/98/{sku_id}")
def insights_98(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 98,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/99/{sku_id}")
def insights_99(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 99,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/100/{sku_id}")
def insights_100(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 100,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/101/{sku_id}")
def insights_101(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 101,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/102/{sku_id}")
def insights_102(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 102,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/103/{sku_id}")
def insights_103(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 103,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/104/{sku_id}")
def insights_104(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 104,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/105/{sku_id}")
def insights_105(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 105,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/106/{sku_id}")
def insights_106(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 106,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/107/{sku_id}")
def insights_107(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 107,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/108/{sku_id}")
def insights_108(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 108,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/109/{sku_id}")
def insights_109(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 109,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/110/{sku_id}")
def insights_110(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 110,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/111/{sku_id}")
def insights_111(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 111,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/112/{sku_id}")
def insights_112(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 112,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/113/{sku_id}")
def insights_113(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 113,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/114/{sku_id}")
def insights_114(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 114,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/115/{sku_id}")
def insights_115(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 115,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/116/{sku_id}")
def insights_116(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 116,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/117/{sku_id}")
def insights_117(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 117,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/118/{sku_id}")
def insights_118(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 118,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }


@router.get("/insights/119/{sku_id}")
def insights_119(sku_id: str) -> dict:
    records = _db.get_sales(sku_id)
    if not records:
        raise HTTPException(status_code=404, detail="sku not found")
    quantities = [r.quantity for r in records]
    revenues = [r.revenue for r in records]
    avg_q = sum(quantities) / len(quantities)
    avg_r = sum(revenues) / len(revenues)
    max_q = max(quantities)
    min_q = min(quantities)
    trend = quantities[-1] - quantities[0]
    score = (avg_q * 0.7 + avg_r * 0.3) / (1 + abs(trend))
    bucket = "A" if score > 100 else "B" if score > 50 else "C"
    return {
        "insight_id": 119,
        "sku_id": sku_id,
        "avg_quantity": avg_q,
        "avg_revenue": avg_r,
        "max_quantity": max_q,
        "min_quantity": min_q,
        "trend": trend,
        "score": score,
        "bucket": bucket,
        "record_count": len(records),
    }
