"""Provider detection and cache constants."""
from __future__ import annotations

# Minimum token granularity for provider prefix caches
CACHE_BREAKPOINT_TOKENS = {
    "anthropic": 1024,
    "openai": 1024,
    "bedrock": 1024,
    "vertex": 1024,
    "gemini": 1024,
}


def detect_provider(model: str) -> str:
    """Guess provider from model name."""
    m = model.lower()
    if m.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"
    if "bedrock" in m:
        return "bedrock"
    if "vertex" in m:
        return "vertex"
    if "gemini" in m:
        return "gemini"
    return "anthropic"  # default to Anthropic format
