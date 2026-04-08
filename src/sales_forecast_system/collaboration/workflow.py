from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


Status = Literal["draft", "submitted", "approved", "rejected", "published"]


@dataclass
class ManualOverride:
    dimension: str
    dimension_value: str
    grain: str
    override_value: float
    reason: str
    operator: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ForecastTask:
    task_id: str
    owner: str
    reviewers: list[str]
    status: Status = "draft"
    comments: list[str] = field(default_factory=list)


class CollaborationManager:
    """协同管理与人工干预。"""

    def __init__(self) -> None:
        self.tasks: dict[str, ForecastTask] = {}
        self.overrides: list[ManualOverride] = []

    def create_task(self, task: ForecastTask) -> ForecastTask:
        self.tasks[task.task_id] = task
        return task

    def submit(self, task_id: str) -> ForecastTask:
        task = self.tasks[task_id]
        task.status = "submitted"
        return task

    def review(self, task_id: str, reviewer: str, approve: bool, comment: str) -> ForecastTask:
        task = self.tasks[task_id]
        if reviewer not in task.reviewers:
            raise PermissionError("当前用户不在审核人列表")
        task.comments.append(f"{reviewer}: {comment}")
        task.status = "approved" if approve else "rejected"
        return task

    def publish(self, task_id: str) -> ForecastTask:
        task = self.tasks[task_id]
        if task.status != "approved":
            raise ValueError("仅已审批任务可发布")
        task.status = "published"
        return task

    def add_override(self, item: ManualOverride) -> ManualOverride:
        self.overrides.append(item)
        return item

    def latest_override(self, dimension: str, dimension_value: str, grain: str) -> ManualOverride | None:
        matched = [
            x for x in self.overrides
            if x.dimension == dimension and x.dimension_value == dimension_value and x.grain == grain
        ]
        return matched[-1] if matched else None
