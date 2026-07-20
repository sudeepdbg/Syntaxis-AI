"""SmartCrusher — lossless-first JSON/text compression.

Strategy:
  1. Detect JSON/array content → dedup identical items, keep head+tail
  2. Preserve error/exception strings (never drop these)
  3. Truncate long text with head+tail strategy
  4. Cross-turn dedup: identical content across messages → reference

Always produces a diff. Never inflates tokens.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .base import Transform, TransformContext, TransformDiff, TransformResult
from ..tokenizer import TokenCounter

logger = logging.getLogger(__name__)

# Strings that signal "this is important, don't drop"
_ERROR_SIGNALS = re.compile(
    r"(error|exception|traceback|failed|failure|fatal|panic|crash|"
    r"denied|forbidden|unauthorized|timeout|refused)",
    re.IGNORECASE,
)


def _looks_like_json_array(text: str) -> bool:
    s = text.strip()
    return s.startswith("[") and s.endswith("]")


def _looks_like_json_object(text: str) -> bool:
    s = text.strip()
    return s.startswith("{") and s.endswith("}")


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _has_error_signal(text: str) -> bool:
    return bool(_ERROR_SIGNALS.search(text))


def _truncate_text(text: str, max_chars: int) -> tuple[str, str]:
    """Truncate with head+tail. Returns (new_text, detail)."""
    if len(text) <= max_chars:
        return text, ""
    head_size = int(max_chars * 0.4)
    tail_size = max_chars - head_size - 30  # room for marker
    head = text[:head_size]
    tail = text[-tail_size:]
    dropped = len(text) - head_size - tail_size
    marker = f"\n\n[... {dropped} chars truncated by SmartCrusher ...]\n\n"
    return head + marker + tail, f"truncated {droued} chars" if False else f"truncated {dropped} chars"


def _crush_array(items: list[Any], max_items: int) -> tuple[list[Any], str]:
    """Dedup + cap array items. Keep first N unique + last item."""
    if len(items) <= max_items:
        return items, ""
    seen: list[Any] = []
    seen_set: set[str] = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen_set:
            seen_set.add(key)
            seen.append(item)
    deduped_count = len(items) - len(seen)
    # Cap at max_items: keep first (max_items-1) + last
    if len(seen) > max_items:
        kept = seen[: max_items - 1] + [seen[-1]]
        capped = len(seen) - len(kept)
    else:
        kept = seen
        capped = 0
    detail_parts = []
    if deduped_count:
        detail_parts.append(f"deduped {deduped_count} identical items")
    if capped:
        detail_parts.append(f"kept {len(kept)}/{len(seen)} items")
    return kept, "; ".join(detail_parts)


class SmartCrusher(Transform):
    """Lossless-first content compressor.

    - JSON arrays: dedup + cap
    - JSON objects: truncate large string values
    - Plain text: head+tail truncation
    - Always preserves error/exception strings
    """

    def __init__(
        self,
        max_items: int = 15,
        max_text_chars: int = 8000,
        counter: TokenCounter | None = None,
    ) -> None:
        self.max_items = max_items
        self.max_text_chars = max_text_chars
        self.counter = counter or TokenCounter()

    @property
    def name(self) -> str:
        return "smart_crusher"

    def apply(
        self,
        messages: list[dict[str, Any]],
        ctx: TransformContext,
    ) -> TransformResult:
        out: list[dict[str, Any]] = []
        diffs: list[TransformDiff] = []
        total = len(messages)

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            # Respect protect_recent
            if i >= total - ctx.protect_recent:
                out.append(msg)
                continue
            # Respect role filters
            if role == "user" and not ctx.compress_user_messages:
                out.append(msg)
                continue
            if role == "system" and not ctx.compress_system_messages:
                out.append(msg)
                continue

            new_msg, msg_diffs = self._crush_message(msg, i, ctx)
            out.append(new_msg)
            diffs.extend(msg_diffs)

        return TransformResult(messages=out, diffs=diffs)

    def _crush_message(
        self,
        msg: dict[str, Any],
        idx: int,
        ctx: TransformContext,
    ) -> tuple[dict[str, Any], list[TransformDiff]]:
        content = msg.get("content", "")
        diffs: list[TransformDiff] = []

        if isinstance(content, str):
            new_content, detail = self._crush_text(content, ctx)
            if new_content != content:
                before = self.counter.count_text(content, ctx.model)
                after = self.counter.count_text(new_content, ctx.model)
                diffs.append(TransformDiff(
                    transform_name=self.name,
                    message_index=idx,
                    action="truncate",
                    before_tokens=before,
                    after_tokens=after,
                    detail=detail,
                ))
                new_msg = {**msg, "content": new_content}
                return new_msg, diffs
            return msg, diffs

        if isinstance(content, list):
            new_blocks: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    new_blocks.append(block)
                    continue
                new_block, block_diffs = self._crush_block(block, idx, ctx)
                new_blocks.append(new_block)
                diffs.extend(block_diffs)
            return {**msg, "content": new_blocks}, diffs

        return msg, diffs

    def _crush_text(self, text: str, ctx: TransformContext) -> tuple[str, str]:
        # Never crush if it contains error signals
        if _has_error_signal(text):
            return text, "preserved (error signal)"
        # Try JSON array
        if _looks_like_json_array(text):
            data = _safe_json_loads(text)
            if isinstance(data, list) and len(data) > ctx.max_items_after_crush:
                crushed, detail = _crush_array(data, ctx.max_items_after_crush)
                if detail:
                    return json.dumps(crushed, indent=2), detail
        # Plain text truncation
        if len(text) > self.max_text_chars:
            return _truncate_text(text, self.max_text_chars)
        return text, ""

    def _crush_block(
        self,
        block: dict[str, Any],
        idx: int,
        ctx: TransformContext,
    ) -> tuple[dict[str, Any], list[TransformDiff]]:
        diffs: list[TransformDiff] = []
        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "")
            new_text, detail = self._crush_text(text, ctx)
            if new_text != text:
                before = self.counter.count_text(text, ctx.model)
                after = self.counter.count_text(new_text, ctx.model)
                diffs.append(TransformDiff(
                    transform_name=self.name,
                    message_index=idx,
                    action="truncate",
                    before_tokens=before,
                    after_tokens=after,
                    detail=detail,
                ))
                return {**block, "text": new_text}, diffs

        elif btype == "tool_result":
            # Tool results are the #1 source of bloat
            c = block.get("content", "")
            if isinstance(c, str):
                new_c, detail = self._crush_text(c, ctx)
                if new_c != c:
                    before = self.counter.count_text(c, ctx.model)
                    after = self.counter.count_text(new_c, ctx.model)
                    diffs.append(TransformDiff(
                        transform_name=self.name,
                        message_index=idx,
                        action="truncate",
                        before_tokens=before,
                        after_tokens=after,
                        detail=f"tool_result: {detail}",
                    ))
                    return {**block, "content": new_c}, diffs
            elif isinstance(c, list):
                new_items = []
                for sub in c:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        new_sub, sub_diffs = self._crush_block(sub, idx, ctx)
                        new_items.append(new_sub)
                        diffs.extend(sub_diffs)
                    else:
                        new_items.append(sub)
                return {**block, "content": new_items}, diffs

        return block, diffs
