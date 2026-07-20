"""`python -m optimizer` launcher — starts the dashboard + runs a demo."""
from __future__ import annotations

import argparse
import logging
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(prog="optimizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--demo", action="store_true", help="Run a demo compression")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from . import start_dashboard, compress

    print(f"starting dashboard → http://{args.host}:{args.port}/dashboard")
    start_dashboard(host=args.host, port=args.port)

    if args.demo:
        _run_demo()

    # Keep main thread alive so the daemon dashboard thread keeps running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


def _run_demo() -> None:
    from . import compress

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Read the file and fix the bug."},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me read the file."},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "main.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "def foo():\n    return 42\n" + ("# filler\n" * 2000)},
        ]},
        {"role": "assistant", "content": "I see the issue. Let me fix it."},
        {"role": "user", "content": "Yes please."},
    ]

    result = compress(messages, model="claude-sonnet-4-6", savings_profile="coding")
    print(f"\n[demo] model=claude-sonnet-4-6")
    print(f"[demo] tokens_before = {result.tokens_before}")
    print(f"[demo] tokens_after  = {result.tokens_after}")
    print(f"[demo] tokens_saved  = {result.tokens_saved}")
    print(f"[demo] ratio         = {result.compression_ratio:.1%}")
    print(f"[demo] transforms    = {result.transforms_applied}")
    print(f"[demo] latency       = {result.latency_ms:.1f} ms")


if __name__ == "__main__":
    sys.exit(main())
