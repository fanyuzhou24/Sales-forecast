from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ModelMeta:
    model_name: str
    model_path: str
    created_at: str
    metrics: dict[str, float]


class ModelMetadataStore:
    def __init__(self, meta_file: Path) -> None:
        self.meta_file = meta_file
        self.meta_file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, meta: ModelMeta) -> None:
        records = self.load_all()
        records.append({
            "model_name": meta.model_name,
            "model_path": meta.model_path,
            "created_at": meta.created_at,
            "metrics": meta.metrics,
        })
        self.meta_file.write_text(json.dumps(records, ensure_ascii=False, indent=2))

    def load_all(self) -> list[dict]:
        if not self.meta_file.exists():
            return []
        return json.loads(self.meta_file.read_text())
