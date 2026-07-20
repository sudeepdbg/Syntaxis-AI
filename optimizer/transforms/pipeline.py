"""TransformPipeline — chain transforms, collect diffs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .base import Transform, TransformContext, TransformDiff, TransformResult
from ..tokenizer import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Final output of the pipeline."""
    messages: list[dict[str, Any]]
    diffs: list[TransformDiff] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    transforms_applied: list[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return self.tokens_saved / self.tokens_before


class TransformPipeline:
    """Ordered chain of transforms.

    Each transform receives the output of the previous one.
    All diffs are collected for auditability.
    """

    def __init__(
        self,
        stages: list[Transform] | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        self.stages = list(stages) if stages else self._default_stages()
        self.counter = counter or TokenCounter()

    @staticmethod
    def _default_stages() -> list[Transform]:
        from .crusher import SmartCrusher
        from .cache_aligner import CacheAligner
        return [
            SmartCrusher(),
            CacheAligner(provider="anthropic"),
        ]

    def apply(
        self,
        messages: list[dict[str, Any]],
        ctx: TransformContext,
    ) -> PipelineResult:
        tokens_before = self.counter.count_messages(messages, ctx.model)
        current = messages
        all_diffs: list[TransformDiff] = []
        applied: list[str] = []

        for stage in self.stages:
            try:
                result = stage.apply(current, ctx)
                current = result.messages
                if result.diffs:
                    all_diffs.extend(result.diffs)
                    applied.append(stage.name)
            except Exception as e:
                logger.warning("transform %s failed: %s", stage.name, e)
                continue

        tokens_after = self.counter.count_messages(current, ctx.model)

        return PipelineResult(
            messages=current,
            diffs=all_diffs,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=applied,
        )
