"""OpenAI Provider and Adapter for Syntaxis-AI."""
from __future__ import annotations

from typing import Any

from .base import Provider, ProviderAdapter


class OpenAIProvider(Provider):
    def send_request(
        self, model: str, messages: list[dict], **kwargs
    ) -> dict[str, Any]:
        return {"model": model, "messages": messages}


class OpenAIAdapter(ProviderAdapter):
    @property
    def name(self) -> str:
        return "openai"

    def map_model_id(self, model: str) -> str:
        return model

    def supports_model(self, model: str) -> bool:
        return True
