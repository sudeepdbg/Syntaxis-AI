"""OTel metrics — optional, no-op if not installed.

Records compression events, cache stats, errors. Integrates with the
dashboard via the CompressionEvent stream.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    _OTEL_AVAILABLE = True
except ImportError:
    otel_metrics = None  # type: ignore
    _OTEL_AVAILABLE = False


@dataclass
class _Counters:
    """In-memory counters (always available, even without OTel)."""
    compressions_total: int = 0
    tokens_before_total: int = 0
    tokens_after_total: int = 0
    tokens_saved_total: int = 0
    cache_hits_total: int = 0
    cache_writes_total: int = 0
    failures_total: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self,
        model: str,
        tokens_before: int,
        tokens_after: int,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> None:
        with self._lock:
            self.compressions_total += 1
            self.tokens_before_total += tokens_before
            self.tokens_after_total += tokens_after
            self.tokens_saved_total += max(0, tokens_before - tokens_after)
            self.cache_hits_total += cache_read
            self.cache_writes_total += cache_write
            if model not in self.by_model:
                self.by_model[model] = {
                    "requests": 0, "tokens_saved": 0, "tokens_sent": 0,
                }
            self.by_model[model]["requests"] += 1
            self.by_model[model]["tokens_saved"] += max(0, tokens_before - tokens_after)
            self.by_model[model]["tokens_sent"] += tokens_after

    def record_failure(self, model: str, error_type: str) -> None:
        with self._lock:
            self.failures_total += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "compressions_total": self.compressions_total,
                "tokens_before_total": self.tokens_before_total,
                "tokens_after_total": self.tokens_after_total,
                "tokens_saved_total": self.tokens_saved_total,
                "cache_hits_total": self.cache_hits_total,
                "cache_writes_total": self.cache_writes_total,
                "failures_total": self.failures_total,
                "by_model": dict(self.by_model),
            }


class HeadroomOtelMetrics:
    """OTel metrics facade. No-op if OTel not installed or telemetry disabled."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and _OTEL_AVAILABLE and not self._telemetry_disabled()
        self._counters = _Counters()
        self._meter = None
        self._compression_counter = None
        self._tokens_saved_counter = None
        self._failure_counter = None
        if self.enabled:
            self._init_otel()

    @staticmethod
    def _telemetry_disabled() -> bool:
        return os.environ.get("OPTIMIZER_TELEMETRY_DISABLED", "").lower() in ("1", "true", "yes")

    def _init_otel(self) -> None:
        try:
            self._meter = otel_metrics.get_meter("token-optimizer", "0.1.0")
            self._compression_counter = self._meter.create_counter(
                name="optimizer.compressions.total",
                description="Total compression operations",
            )
            self._tokens_saved_counter = self._meter.create_counter(
                name="optimizer.tokens.saved",
                description="Total tokens saved by compression",
            )
            self._failure_counter = self._meter.create_counter(
                name="optimizer.compressions.failures",
                description="Total compression failures",
            )
        except Exception as e:
            logger.warning("OTel init failed, falling back to in-memory: %s", e)
            self.enabled = False

    def record_compression(
        self,
        model: str,
        tokens_before: int,
        tokens_after: int,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> None:
        self._counters.record(
            model=model,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            cache_read=cache_read,
            cache_write=cache_write,
        )
        if self.enabled and self._compression_counter:
            try:
                self._compression_counter.add(1, {"model": model})
                saved = max(0, tokens_before - tokens_after)
                if saved and self._tokens_saved_counter:
                    self._tokens_saved_counter.add(saved, {"model": model})
            except Exception as e:
                logger.debug("OTel record failed: %s", e)

    def record_compression_failure(self, model: str, error_type: str) -> None:
        self._counters.record_failure(model, error_type)
        if self.enabled and self._failure_counter:
            try:
                self._failure_counter.add(1, {"model": model, "error": error_type})
            except Exception:
                pass

    def snapshot(self) -> dict[str, Any]:
        return self._counters.snapshot()


# --- singleton ---------------------------------------------------------

_metrics: HeadroomOtelMetrics | None = None
_metrics_lock = threading.Lock()


def get_otel_metrics() -> HeadroomOtelMetrics:
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = HeadroomOtelMetrics()
    return _metrics


def configure_otel_metrics(enabled: bool = True) -> HeadroomOtelMetrics:
    global _metrics
    with _metrics_lock:
        _metrics = HeadroomOtelMetrics(enabled=enabled)
    return _metrics


def reset_otel_metrics() -> None:
    global _metrics
    with _metrics_lock:
        _metrics = None
