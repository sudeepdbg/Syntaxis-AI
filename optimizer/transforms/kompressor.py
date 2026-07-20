"""Lossy ML compression via Kompress model for Syntaxis-AI."""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

DEFAULT_MODEL = "chopratejas/kompress-v2-base"


class _KompressModel:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self._loaded = False

    @classmethod
    def get(cls, model_id: str = DEFAULT_MODEL):
        if not _ML_AVAILABLE:
            return None
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_id)
        return cls._instance

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            self._loaded = True
            return True
        except Exception as e:
            logger.warning(f"Failed to load Kompress: {e}")
            return False

    def compress(self, text: str, target_ratio: float = 0.3) -> str:
        if not self._ensure_loaded():
            return text
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=2048
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            max_new = max(32, int(len(inputs["input_ids"][0]) * target_ratio))
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception:
            return text


class LossyKompressor:
    """Transform wrapper for Kompress."""

    def __init__(
        self, model_id: str = DEFAULT_MODEL, target_ratio: float = 0.3
    ):
        self._model = _KompressModel.get(model_id)
        self.target_ratio = target_ratio

    def apply(
        self, messages: list[dict[str, Any]], **kwargs
    ) -> list[dict[str, Any]]:
        if self._model is None:
            return messages
        out = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 500:
                msg = {
                    **msg,
                    "content": self._model.compress(content, self.target_ratio),
                }
            out.append(msg)
        return out


def kompress_text(text: str, target_ratio: float = 0.3) -> str:
    model = _KompressModel.get()
    return model.compress(text, target_ratio) if model else text
