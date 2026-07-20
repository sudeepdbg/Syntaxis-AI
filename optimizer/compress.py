"""One-function compression API.

The simplest way to use the optimizer:
    from optimizer import compress
    result = compress(messages, model="claude-sonnet-4-6")
    result.messages          # Compressed messages
    result.tokens_saved      # Tokens saved
    result.compression_ratio # e.g. 0.65 = 65% saved
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

from .profiles import apply_agent_savings_profile, get_agent_savings_profile
from .tokenizer import TokenCounter
from .transforms import (
    SmartCrusher,
    CacheAligner,
    TransformContext,
    TransformPipeline,
)
from .transforms.pipeline import PipelineResult
from .cache.providers import detect_provider
from .observability import get_otel_metrics

logger = logging.getLogger(__name__)


_pipeline: TransformPipeline | None = None
_pipeline_lock = threading.Lock()


@dataclass
class CompressConfig:
    """User-facing compression options.

    Pass to `compress()` or use a named profile via `savings_profile`.
    """
    # What to compress
    compress_user_messages: bool = False
    compress_system_messages: bool = True
    protect_recent: int = 4
    protect_analysis_context: bool = True
    frozen_message_count: int = 0
    # How aggressive
    target_ratio: float | None = None
    min_tokens_to_compress: int = 250
    max_items_after_crush: int = 15
    # Profile
    savings_profile: str | None = None


@dataclass
class CompressResult:
    """Result of compressing messages."""
    messages: list[dict[str, Any]]
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 0.0
    transforms_applied: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cache_hit_rate: float = 0.0


def _extract_user_query(messages: list[dict[str, Any]]) -> str:
    """Pull the most recent user message for relevance scoring."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content[:500]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")[:500]
    return ""


def _get_pipeline(provider: str) -> TransformPipeline:
    """Get or create the singleton pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        _pipeline = TransformPipeline(
            stages=[
                SmartCrusher(),
                CacheAligner(provider=provider),
            ]
        )
        return _pipeline


def compress(
    messages: list[dict[str, Any]],
    model: str = "claude-sonnet-4-6",
    model_limit: int = 200_000,
    optimize: bool = True,
    config: CompressConfig | None = None,
    **kwargs: Any,
) -> CompressResult:
    """Compress messages using the full pipeline.

    Args:
        messages: List of messages (Anthropic or OpenAI format).
        model: Model name (for token counting + provider detection).
        model_limit: Model's context window.
        optimize: False = passthrough (for A/B testing).
        config: CompressConfig. Overrides defaults.
        **kwargs: Shorthand for CompressConfig fields.

    Returns:
        CompressResult with compressed messages + metrics.
    """
    t0 = time.time()

    if not messages or not optimize:
        return CompressResult(messages=messages, latency_ms=(time.time() - t0) * 1000)

    # Build config
    cfg = replace(config) if config is not None else CompressConfig()
    config_fields = {f.name for f in cfg.__dataclass_fields__.values()}
    for key, value in kwargs.items():
        if key in config_fields:
            setattr(cfg, key, value)
    if cfg.savings_profile:
        apply_agent_savings_profile(cfg, cfg.savings_profile)

    provider = detect_provider(model)
    pipeline = _get_pipeline(provider)
    counter = TokenCounter()

    try:
        # Extract user query for relevance context
        context_query = _extract_user_query(messages)

        ctx = TransformContext(
            model=model,
            model_limit=model_limit,
            user_query=context_query,
            compress_user_messages=cfg.compress_user_messages,
            compress_system_messages=cfg.compress_system_messages,
            protect_recent=cfg.protect_recent,
            protect_analysis_context=cfg.protect_analysis_context,
            min_tokens_to_compress=cfg.min_tokens_to_compress,
            max_items_after_crush=cfg.max_items_after_crush,
            frozen_message_count=cfg.frozen_message_count,
            provider=provider,
        )

        result: PipelineResult = pipeline.apply(messages, ctx)
        tokens_before = result.tokens_before
        tokens_after = result.tokens_after
        compressed_messages = result.messages

        # --- Inflation guard ---
        # If compression made things bigger, revert. Mirrors Headroom's
        # proxy handlers — the library path needs this too.
        if tokens_after > tokens_before:
            logger.warning(
                "Optimization inflated tokens (%d -> %d); reverting",
                tokens_before, tokens_after,
            )
            get_otel_metrics().record_compression_failure(model, "inflation")
            return CompressResult(
                messages=messages,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                tokens_saved=0,
                compression_ratio=0.0,
                transforms_applied=["inflation_guard:reverted"],
                latency_ms=(time.time() - t0) * 1000,
            )

        tokens_saved = tokens_before - tokens_after
        ratio = tokens_saved / tokens_before if tokens_before > 0 else 0.0

        # Record metrics
        get_otel_metrics().record_compression(
            model=model,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

        # Record to dashboard (best-effort)
        try:
            from .dashboard import get_state
            get_state().record_compression(
                model=model,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                transforms=result.transforms_applied,
                latency_ms=(time.time() - t0) * 1000,
                profile=cfg.savings_profile or "default",
            )
        except Exception:
            pass  # dashboard is optional

        return CompressResult(
            messages=compressed_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            compression_ratio=ratio,
            transforms_applied=result.transforms_applied,
            latency_ms=(time.time() - t0) * 1000,
        )

    except Exception as e:
        logger.warning("Compression failed, returning originals: %s", e)
        get_otel_metrics().record_compression_failure(model, type(e).__name__)
        return CompressResult(
            messages=messages,
            tokens_before=0,
            tokens_after=0,
            tokens_saved=0,
            compression_ratio=0.0,
            latency_ms=(time.time() - t0) * 1000,
        )
