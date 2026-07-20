"""Cross-turn deduplication transform for Syntaxis-AI."""
from __future__ import annotations

import hashlib
from typing import Any


class CrossTurnDeduper:
    """Collapse identical content across messages."""

    def __init__(self, min_content_chars: int = 200):
        self.min_content_chars = min_content_chars
        self._seen: dict[str, int] = {}

    def apply(
        self, messages: list[dict[str, Any]], **kwargs
    ) -> list[dict[str, Any]]:
        out = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > self.min_content_chars:
                h = hashlib.sha256(content.encode()).hexdigest()[:16]
                if h in self._seen:
                    msg = {
                        **msg,
                        "content": f"[deduplicated: see msg {self._seen[h]}]",
                    }
                else:
                    self._seen[h] = len(out)
            out.append(msg)
        return out
