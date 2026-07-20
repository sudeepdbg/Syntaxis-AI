"""Named savings profiles — the API surface, not individual knobs.

Users pick a profile ("coding", "balanced", "agent-90", "general"); the profile
bundles ~20 correlated knobs into a coherent posture. This is the hardest part
of the API to change later, so get it right now.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Protocol

logger = logging.getLogger(__name__)

AGENT_90_PROFILE = "agent-90"
FALLBACK_PROFILE = "balanced"
DEFAULT_PROFILE = "coding"


class CompressConfigLike(Protocol):
    compress_user_messages: bool
    compress_system_messages: bool
    protect_recent: int
    protect_analysis_context: bool
    target_ratio: float | None
    min_tokens_to_compress: int


@dataclass(frozen=True)
class AgentSavingsProfile:
    """Reusable policy for agent compression.

    Each profile encodes a coherent posture — don't mix knobs across profiles
    unless you know what you're doing.
    """
    name: str
    target_savings: float           # Nominal target (display only for emergent profiles)
    target_ratio: float | None      # Keep ratio for lossy compressor. None = adaptive.
    compress_user_messages: bool
    compress_system_messages: bool
    protect_recent: int             # Last N messages untouched
    protect_analysis_context: bool  # Detect "analyze/review" intent
    min_tokens_to_compress: int     # Below this, skip
    max_items_after_crush: int      # Cap on array items after SmartCrusher
    smart_crusher_with_compaction: bool
    force_kompress: bool            # Force lossy ML compression
    proxy_mode: str                 # "token" or "cache"
    accuracy_guard: str             # "strict" | "relaxed"
    # Advanced toggles
    cross_turn_dedup: bool = False
    lossless_then_lossy: bool = False
    protect_reads: bool = False     # Never lossy-compress file reads
    code_aware: bool = True

    @property
    def savings_percent(self) -> int:
        return round(self.target_savings * 100)


_PROFILES: dict[str, AgentSavingsProfile] = {
    AGENT_90_PROFILE: AgentSavingsProfile(
        name=AGENT_90_PROFILE,
        target_savings=0.90,
        target_ratio=0.10,
        compress_user_messages=True,
        compress_system_messages=True,
        protect_recent=2,
        protect_analysis_context=True,
        min_tokens_to_compress=120,
        max_items_after_crush=8,
        smart_crusher_with_compaction=False,
        force_kompress=True,
        proxy_mode="token",
        accuracy_guard="strict",
    ),
    FALLBACK_PROFILE: AgentSavingsProfile(
        name=FALLBACK_PROFILE,
        target_savings=0.70,
        target_ratio=0.30,
        compress_user_messages=False,
        compress_system_messages=False,
        protect_recent=4,
        protect_analysis_context=True,
        min_tokens_to_compress=250,
        max_items_after_crush=15,
        smart_crusher_with_compaction=True,
        force_kompress=False,
        proxy_mode="token",
        accuracy_guard="strict",
    ),
    DEFAULT_PROFILE: AgentSavingsProfile(
        name=DEFAULT_PROFILE,
        target_savings=0.50,  # emergent
        target_ratio=None,
        compress_user_messages=True,
        compress_system_messages=False,  # system = hottest cache
        protect_recent=0,                # delta-only in cache mode
        protect_analysis_context=True,
        min_tokens_to_compress=10,
        max_items_after_crush=15,
        smart_crusher_with_compaction=True,
        force_kompress=False,
        proxy_mode="cache",
        accuracy_guard="strict",
        cross_turn_dedup=True,
        lossless_then_lossy=True,
        protect_reads=True,
        code_aware=True,
    ),
    "general": AgentSavingsProfile(
        name="general",
        target_savings=0.60,
        target_ratio=None,
        compress_user_messages=False,
        compress_system_messages=False,
        protect_recent=0,
        protect_analysis_context=True,
        min_tokens_to_compress=25,
        max_items_after_crush=15,
        smart_crusher_with_compaction=True,
        force_kompress=False,
        proxy_mode="token",
        accuracy_guard="strict",
    ),
}


def get_agent_savings_profile(name: str | None = None) -> AgentSavingsProfile:
    """Return named profile. Unknown names fall back to `balanced` with a warning."""
    key = (name or DEFAULT_PROFILE).strip().lower()
    profile = _PROFILES.get(key)
    if profile is not None:
        return profile
    valid = ", ".join(sorted(_PROFILES))
    logger.warning(
        "unknown savings profile %r; falling back to %r (known: %s)",
        name, FALLBACK_PROFILE, valid,
    )
    return _PROFILES[FALLBACK_PROFILE]


def apply_agent_savings_profile(
    config: CompressConfigLike,
    profile: AgentSavingsProfile | str | None = None,
) -> CompressConfigLike:
    """Apply a profile to an existing CompressConfig-like object."""
    resolved = (
        get_agent_savings_profile(profile)
        if isinstance(profile, str) or profile is None
        else profile
    )
    config.compress_user_messages = resolved.compress_user_messages
    config.compress_system_messages = resolved.compress_system_messages
    config.protect_recent = resolved.protect_recent
    config.protect_analysis_context = resolved.protect_analysis_context
    if resolved.target_ratio is not None:
        config.target_ratio = resolved.target_ratio
    config.min_tokens_to_compress = resolved.min_tokens_to_compress
    return config


def list_profiles() -> list[str]:
    return sorted(_PROFILES.keys())
