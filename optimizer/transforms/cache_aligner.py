"""CacheAligner — stabilize prefix for provider KV cache hits.

Providers cache by **prefix match**:
  - Anthropic: 1024-token blocks, breakpoints via `cache_control`
  - OpenAI: automatic prefix caching, 1024-token granularity
  - Bedrock: `cachePoint` on content blocks

Strategy:
  1. Keep system prompt + first few user turns byte-identical
  2. Emit cache_control breakpoints on the last 2 system blocks
  3. Emit cache_control on the last tool_result (moving breakpoint)
  4. Never reorder messages

This is the 50-90% cost lever for agentic workloads.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import Transform, TransformContext, TransformDiff, TransformResult

logger = logging.getLogger(__name__)


class CacheAligner(Transform):
    """Emit provider-specific cache breakpoints.

    Does NOT compress — only annotates messages so the provider caches the prefix.
    """

    def __init__(self, provider: str = "anthropic") -> None:
        self.provider = provider.lower()

    @property
    def name(self) -> str:
        return "cache_aligner"

    def apply(
        self,
        messages: list[dict[str, Any]],
        ctx: TransformContext,
    ) -> TransformResult:
        if self.provider in ("anthropic", "bedrock", "vertex"):
            return self._align_anthropic_style(messages, ctx)
        if self.provider in ("openai",):
            return self._align_openai(messages, ctx)
        # Unknown provider — pass through
        return TransformResult(messages=messages)

    def _align_anthropic_style(
        self,
        messages: list[dict[str, Any]],
        ctx: TransformContext,
    ) -> TransformResult:
        """Anthropic/Bedrock/Vertex: add cache_control breakpoints."""
        out: list[dict[str, Any]] = []
        diffs: list[TransformDiff] = []

        # Find system messages (they may be in messages[0] with role=system,
        # or in a top-level `system` field — we handle the messages-list form).
        system_indices = [i for i, m in enumerate(messages) if m.get("role") == "system"]
        # Find last tool_result message (moving breakpoint)
        last_tool_result_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            content = msg.get("content", "")
            if isinstance(content, list):
                if any(b.get("type") == "tool_result" for b in content if isinstance(b, dict)):
                    last_tool_result_idx = i
                    break

        for i, msg in enumerate(messages):
            new_msg = dict(msg)
            content = msg.get("content", "")

            # System messages: add cache_control on the last 2
            if i in system_indices:
                pos_in_system = system_indices.index(i)
                is_near_end = pos_in_system >= len(system_indices) - 2
                if is_near_end and isinstance(content, list):
                    new_blocks = []
                    for j, block in enumerate(content):
                        if isinstance(block, dict) and j == len(content) - 1:
                            new_block = dict(block)
                            if "cache_control" not in new_block:
                                new_block["cache_control"] = {"type": "ephemeral"}
                                diffs.append(TransformDiff(
                                    transform_name=self.name,
                                    message_index=i,
                                    action="cache_breakpoint",
                                    detail="system:ephemeral",
                                ))
                            new_blocks.append(new_block)
                        else:
                            new_blocks.append(block)
                    new_msg["content"] = new_blocks
                elif is_near_end and isinstance(content, str):
                    # Wrap string content in a block with cache_control
                    new_msg["content"] = [
                        {"type": "text", "text": content,
                         "cache_control": {"type": "ephemeral"}}
                    ]
                    diffs.append(TransformDiff(
                        transform_name=self.name,
                        message_index=i,
                        action="cache_breakpoint",
                        detail="system:ephemeral (wrapped)",
                    ))

            # Last tool_result: moving breakpoint
            elif i == last_tool_result_idx and isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        new_block = dict(block)
                        if "cache_control" not in new_block:
                            new_block["cache_control"] = {"type": "ephemeral"}
                            diffs.append(TransformDiff(
                                transform_name=self.name,
                                message_index=i,
                                action="cache_breakpoint",
                                detail="tool_result:ephemeral",
                            ))
                        new_blocks.append(new_block)
                    else:
                        new_blocks.append(block)
                new_msg["content"] = new_blocks

            out.append(new_msg)

        return TransformResult(
            messages=out,
            diffs=diffs,
            metadata={"cache_breakpoints": len(diffs)},
        )

    def _align_openai(
        self,
        messages: list[dict[str, Any]],
        ctx: TransformContext,
    ) -> TransformResult:
        """OpenAI: automatic prefix caching — just stabilize order.

        OpenAI caches automatically; we don't need to emit markers.
        But we DO need to ensure the prefix is byte-identical across requests,
        which means no reordering. This transform is a no-op marker.
        """
        return TransformResult(
            messages=messages,
            diffs=[TransformDiff(
                transform_name=self.name,
                message_index=-1,
                action="cache_breakpoint",
                detail="openai:automatic (no reordering)",
            )],
            metadata={"cache_breakpoints": 0},
        )
