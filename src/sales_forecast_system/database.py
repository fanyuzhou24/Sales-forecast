from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List


@dataclass
class SalesRecord:
    sku_id: str
    record_date: date
    quantity: float
    revenue: float


@dataclass
class InMemoryDatabase:
    sales: Dict[str, List[SalesRecord]] = field(default_factory=dict)

    def insert_sale(self, record: SalesRecord) -> None:
        self.sales.setdefault(record.sku_id, []).append(record)

    def get_sales(self, sku_id: str) -> List[SalesRecord]:
        return list(self.sales.get(sku_id, []))

    def list_skus(self) -> List[str]:
        return sorted(self.sales.keys())
