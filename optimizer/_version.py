"""Package version metadata."""
from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as _pkg_version

UNKNOWN_VERSION = "unknown"


def get_version() -> str:
    env = os.environ.get("OPTIMIZER_VERSION")
    if env:
        return env.strip() or UNKNOWN_VERSION
    try:
        return _pkg_version("token-optimizer")
    except PackageNotFoundError:
        return "0.1.0-dev"


version = get_version()
