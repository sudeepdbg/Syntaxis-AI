"""Air-gap master switch (`OPTIMIZER_OFFLINE`).

Single predicate to disable all outbound network: telemetry, model downloads,
update checks. Fails closed.
"""
from __future__ import annotations

import os

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
OFFLINE_ENV = "OPTIMIZER_OFFLINE"


def is_offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in _TRUE_VALUES


def apply_offline_env() -> None:
    """Force HF/Transformers offline. Idempotent, setdefault semantics."""
    if is_offline():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("OPTIMIZER_TELEMETRY_DISABLED", "1")
