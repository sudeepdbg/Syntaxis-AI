"""Bedrock Adapter for Syntaxis-AI (wraps LiteLLM)."""
from __future__ import annotations

from .base import ProviderAdapter


class BedrockAdapter(ProviderAdapter):
    def __init__(self, region: str = "us-east-1"):
        self.region = region

    @property
    def name(self) -> str:
        return "bedrock"

    def map_model_id(self, model: str) -> str:
        return f"bedrock/{model}"

    def supports_model(self, model: str) -> bool:
        return True
