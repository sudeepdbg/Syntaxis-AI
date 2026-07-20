"""Token Optimizer — cut LLM costs 50-90% without losing accuracy.

Quick start:
    from optimizer import compress, start_dashboard

    result = compress(messages, model="claude-sonnet-4-6")
    print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.0%})")

    start_dashboard()  # http://127.0.0.1:8765/dashboard
"""
from __future__ import annotations

from ._version import version  # noqa: F401
from .compress import (
    CompressConfig,
    CompressResult,
    compress,
)
from .profiles import (
    AgentSavingsProfile,
    get_agent_savings_profile,
    apply_agent_savings_profile,
    DEFAULT_PROFILE,
)
from .tokenizer import TokenCounter, count_tokens_text, count_tokens_messages
from .transforms import (
    Transform,
    TransformDiff,
    TransformResult,
    TransformPipeline,
    SmartCrusher,
    CacheAligner,
)
from .observability import (
    get_otel_metrics,
    configure_otel_metrics,
    HeadroomOtelMetrics,
)
from .dashboard import start_dashboard, stop_dashboard, get_state, CompressionEvent

__all__ = [
    # Core API
    "compress",
    "CompressConfig",
    "CompressResult",
    # Profiles
    "AgentSavingsProfile",
    "get_agent_savings_profile",
    "apply_agent_savings_profile",
    "DEFAULT_PROFILE",
    # Tokenizer
    "TokenCounter",
    "count_tokens_text",
    "count_tokens_messages",
    # Transforms
    "Transform",
    "TransformDiff",
    "TransformResult",
    "TransformPipeline",
    "SmartCrusher",
    "CacheAligner",
    # Observability
    "get_otel_metrics",
    "configure_otel_metrics",
    "HeadroomOtelMetrics",
    # Dashboard
    "start_dashboard",
    "stop_dashboard",
    "get_state",
    "CompressionEvent",
    # Version
    "version",
]
