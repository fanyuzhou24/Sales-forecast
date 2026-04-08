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

    def list_by_model(self, model_name: str) -> list[dict]:
        return [x for x in self.load_all() if x.get("model_name") == model_name]

    def latest(self, model_name: str) -> dict | None:
        items = self.list_by_model(model_name)
        return items[-1] if items else None

    def rollback(self, model_name: str, version_index: int) -> dict:
        items = self.list_by_model(model_name)
        if not items:
            raise ValueError(f"未找到模型版本: {model_name}")
        if version_index < 0 or version_index >= len(items):
            raise IndexError("version_index 超出范围")
        selected = items[version_index]
        marker = {
            "model_name": model_name,
            "model_path": selected["model_path"],
            "created_at": selected["created_at"],
            "metrics": selected.get("metrics", {}),
            "rollback_to_version": version_index,
        }
        all_records = self.load_all()
        all_records.append(marker)
        self.meta_file.write_text(json.dumps(all_records, ensure_ascii=False, indent=2))
        return marker
