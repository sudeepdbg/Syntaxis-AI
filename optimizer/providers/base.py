"""Base provider interfaces for Syntaxis-AI."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ProviderAdapter(ABC):
    """Base class for provider adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def map_model_id(self, model: str) -> str:
        ...

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        ...


class Provider(ABC):
    """Client-side provider interface."""

    @abstractmethod
    def send_request(
        self, model: str, messages: list[dict], **kwargs
    ) -> dict[str, Any]:
        ...


class TokenCounter:
    """Simple token counter wrapper."""

    def count_text(self, text: str, model: str = "gpt-4o") -> int:
        return max(1, len(text) // 4)
