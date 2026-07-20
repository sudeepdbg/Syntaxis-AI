"""Transform ABC and result types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransformDiff:
    """Audit record of what a transform changed.

    Lossless-first: every transform must produce a diff so callers can
    reconstruct or inspect what was dropped.
    """
    transform_name: str
    message_index: int
    action: str  # "truncate", "dedup", "drop", "rewrite", "cache_breakpoint"
    before_tokens: int = 0
    after_tokens: int = 0
    detail: str = ""

    @property
    def saved(self) -> int:
        return max(0, self.before_tokens - self.after_tokens)


@dataclass
class TransformResult:
    """Output of a single transform stage."""
    messages: list[dict[str, Any]]
    diffs: list[TransformDiff] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformContext:
    """Context passed to each transform."""
    model: str
    model_limit: int = 200_000
    user_query: str = ""
    compress_user_messages: bool = False
    compress_system_messages: bool = True
    protect_recent: int = 0
    protect_analysis_context: bool = True
    min_tokens_to_compress: int = 250
    max_items_after_crush: int = 15
    frozen_message_count: int = 0
    provider: str = "anthropic"
    extra: dict[str, Any] = field(default_factory=dict)


class Transform(ABC):
    """Base class for all transforms.

    Transforms are stateless, composable, and produce a diff for every change.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def apply(
        self,
        messages: list[dict[str, Any]],
        ctx: TransformContext,
    ) -> TransformResult:
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
