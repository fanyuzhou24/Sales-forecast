from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


class Notifier(Protocol):
    def send(self, title: str, content: str, recipients: list[str]) -> None:
        ...


@dataclass
class ConsoleNotifier:
    def send(self, title: str, content: str, recipients: list[str]) -> None:
        print(f"[通知] {title} -> {', '.join(recipients)}")
        print(content)


@dataclass
class WebhookNotifier:
    webhook_url: str

    def send(self, title: str, content: str, recipients: list[str]) -> None:
        requests.post(
            self.webhook_url,
            json={"title": title, "content": content, "recipients": recipients},
            timeout=15,
        ).raise_for_status()
