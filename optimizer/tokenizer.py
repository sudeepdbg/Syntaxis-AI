"""Token counting — tiktoken with char-heuristic fallback.

Caches tokenizer instances per model. Thread-safe.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False


# Model -> tiktoken encoding name
_MODEL_TO_ENCODING = {
    # OpenAI
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    # Anthropic (use cl100k_base as a close approximation)
    "claude-": "cl100k_base",
    "claude-3": "cl100k_base",
    "claude-sonnet": "cl100k_base",
    "claude-opus": "cl100k_base",
    "claude-haiku": "cl100k_base",
}

# Rough chars-per-token heuristics per vendor
_CHARS_PER_TOKEN = {
    "openai": 4.0,
    "anthropic": 3.5,
    "bedrock": 3.5,
    "vertex": 3.5,
    "gemini": 4.0,
    "default": 4.0,
}


def _detect_vendor(model: str) -> str:
    m = model.lower()
    if m.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"
    if "bedrock" in m:
        return "bedrock"
    if "vertex" in m or "gemini" in m:
        return "gemini"
    return "default"


class TokenCounter:
    """Thread-safe token counter with tokenizer caching."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _get_encoding(self, model: str) -> Any:
        if not _TIKTOKEN_AVAILABLE:
            return None
        with self._lock:
            if model in self._cache:
                return self._cache[model]
            enc_name = None
            # Try exact match first
            if model in _MODEL_TO_ENCODING:
                enc_name = _MODEL_TO_ENCODING[model]
            else:
                # Prefix match
                for prefix, name in _MODEL_TO_ENCODING.items():
                    if model.lower().startswith(prefix):
                        enc_name = name
                        break
            if enc_name is None:
                enc_name = "cl100k_base"
            try:
                enc = tiktoken.get_encoding(enc_name)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            self._cache[model] = enc
            return enc

    def count_text(self, text: str, model: str = "gpt-4o") -> int:
        if not text:
            return 0
        enc = self._get_encoding(model)
        if enc is not None:
            try:
                return len(enc.encode(text))
            except Exception:
                pass
        # Fallback: char heuristic
        vendor = _detect_vendor(model)
        cpt = _CHARS_PER_TOKEN.get(vendor, _CHARS_PER_TOKEN["default"])
        return max(1, int(len(text) / cpt))

    def count_messages(self, messages: list[dict[str, Any]], model: str = "gpt-4o") -> int:
        """Count tokens across a list of messages.

        Adds per-message overhead (~4 tokens per message for role/formatting).
        """
        total = 0
        for msg in messages:
            total += 4  # role + formatting overhead
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_text(content, model)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total += self.count_text(block.get("text", ""), model)
                        elif block.get("type") == "tool_use":
                            total += self.count_text(str(block.get("input", "")), model)
                            total += self.count_text(block.get("name", ""), model)
                        elif block.get("type") == "tool_result":
                            c = block.get("content", "")
                            if isinstance(c, str):
                                total += self.count_text(c, model)
                            elif isinstance(c, list):
                                for sub in c:
                                    if isinstance(sub, dict) and sub.get("type") == "text":
                                        total += self.count_text(sub.get("text", ""), model)
        total += 3  # priming tokens
        return total

    @classmethod
    def for_model(cls, model: str) -> "TokenCounter":
        """Convenience: return a fresh counter (counters are cheap to create)."""
        return cls()


# Module-level singleton
_default_counter = TokenCounter()


def count_tokens_text(text: str, model: str = "gpt-4o") -> int:
    return _default_counter.count_text(text, model)


def count_tokens_messages(messages: list[dict[str, Any]], model: str = "gpt-4o") -> int:
    return _default_counter.count_messages(messages, model)
