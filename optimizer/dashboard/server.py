"""FastAPI server for the dashboard."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

from .state import get_state

logger = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).parent / "templates"

_server: Any = None
_thread: threading.Thread | None = None


def _build_app() -> Any:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="Token Optimizer Dashboard", docs_url=None, redoc_url=None)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> str:
        return (TEMPLATES_DIR / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def root() -> Any:
        from starlette.responses import RedirectResponse
        return RedirectResponse("/dashboard")

    @app.get("/api/stats")
    async def stats() -> JSONResponse:
        return JSONResponse(get_state().snapshot())

    @app.get("/api/requests")
    async def requests(n: int = 25) -> JSONResponse:
        return JSONResponse(get_state().recent(n))

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        state = get_state()
        queue = state.subscribe()

        async def event_generator():
            try:
                yield f"data: {json.dumps({'type': 'snapshot', 'payload': state.snapshot()})}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        payload = {"type": "event", "payload": event.to_dict()}
                        yield f"data: {json.dumps(payload)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                state.unsubscribe(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/reset")
    async def reset() -> JSONResponse:
        state = get_state()
        with state._lock:
            from .state import _Aggregates
            state._session = _Aggregates()
            state._buffer.clear()
        return JSONResponse({"ok": True})

    return app


def _run_server(host: str, port: int) -> None:
    import uvicorn
    app = _build_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    global _server
    _server = server
    server.run()


def start_dashboard(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the dashboard in a background thread. Non-blocking."""
    global _thread
    if _thread is not None and _thread.is_alive():
        logger.warning("Dashboard already running on %s:%s", host, port)
        return
    _thread = threading.Thread(
        target=_run_server, args=(host, port),
        name="optimizer-dashboard", daemon=True,
    )
    _thread.start()
    logger.info("Dashboard started → http://%s:%s/dashboard", host, port)


def stop_dashboard() -> None:
    global _server, _thread
    if _server is not None:
        _server.should_exit = True
    if _thread is not None:
        _thread.join(timeout=2.0)
    _server = None
    _thread = None
