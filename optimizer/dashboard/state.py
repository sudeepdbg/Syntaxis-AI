"""Thread-safe dashboard state with ring buffer + aggregates."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class CompressionEvent:
    timestamp: float
    model: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    transforms: list[str]
    cache_hit_rate: float = 0.0
    latency_ms: float = 0.0
    profile: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Aggregates:
    requests: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    cache_hits: int = 0
    cache_total: int = 0

    @property
    def savings_ratio(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return self.tokens_saved / self.tokens_before

    @property
    def cache_hit_rate(self) -> float:
        if self.cache_total == 0:
            return 0.0
        return self.cache_hits / self.cache_total


class DashboardState:
    """Global dashboard state. Singleton via `get_state()`."""

    def __init__(self, history_path: Path | None = None, buffer_size: int = 500):
        self._lock = threading.RLock()
        self._buffer: deque[CompressionEvent] = deque(maxlen=buffer_size)
        self._session = _Aggregates()
        self._lifetime = _Aggregates()
        self._history_path = history_path
        self._subscribers: list[Any] = []
        self._started_at = time.time()
        if self._history_path and self._history_path.exists():
            self._load_history()

    def record_compression(
        self,
        *,
        model: str,
        tokens_before: int,
        tokens_after: int,
        transforms: list[str] | None = None,
        cache_hit_rate: float = 0.0,
        latency_ms: float = 0.0,
        profile: str = "default",
    ) -> CompressionEvent:
        transforms = transforms or []
        saved = max(0, tokens_before - tokens_after)
        event = CompressionEvent(
            timestamp=time.time(),
            model=model,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=saved,
            transforms=transforms,
            cache_hit_rate=cache_hit_rate,
            latency_ms=latency_ms,
            profile=profile,
        )
        with self._lock:
            self._buffer.append(event)
            self._apply(event, self._session)
            self._apply(event, self._lifetime)
            self._persist(event)
            self._broadcast(event)
        return event

    def _apply(self, event: CompressionEvent, agg: _Aggregates) -> None:
        agg.requests += 1
        agg.tokens_before += event.tokens_before
        agg.tokens_after += event.tokens_after
        agg.tokens_saved += event.tokens_saved
        if event.cache_hit_rate > 0:
            agg.cache_hits += int(event.cache_hit_rate * 100)
            agg.cache_total += 100

    def _persist(self, event: CompressionEvent) -> None:
        if not self._history_path:
            return
        try:
            with self._history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except OSError:
            pass

    def _load_history(self) -> None:
        assert self._history_path is not None
        try:
            for line in self._history_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                event = CompressionEvent(**data)
                self._apply(event, self._lifetime)
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    def subscribe(self) -> Any:
        import asyncio
        q: Any = asyncio.Queue(maxlen=64)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: Any) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _broadcast(self, event: CompressionEvent) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self._started_at, 1),
                "session": {
                    "requests": self._session.requests,
                    "tokens_before": self._session.tokens_before,
                    "tokens_after": self._session.tokens_after,
                    "tokens_saved": self._session.tokens_saved,
                    "savings_ratio": round(self._session.savings_ratio, 4),
                    "cache_hit_rate": round(self._session.cache_hit_rate, 4),
                },
                "lifetime": {
                    "requests": self._lifetime.requests,
                    "tokens_saved": self._lifetime.tokens_saved,
                    "savings_ratio": round(self._lifetime.savings_ratio, 4),
                    "cache_hit_rate": round(self._lifetime.cache_hit_rate, 4),
                },
                "recent": [e.to_dict() for e in list(self._buffer)[-25:]],
            }

    def recent(self, n: int = 25) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._buffer)[-n:]
            return [e.to_dict() for e in items]


_state: DashboardState | None = None
_state_lock = threading.Lock()


def get_state(history_path: Path | str | None = None) -> DashboardState:
    global _state
    if _state is None:
        with _state_lock:
            if _state is None:
                path = Path(history_path) if history_path else None
                _state = DashboardState(history_path=path)
    return _state
