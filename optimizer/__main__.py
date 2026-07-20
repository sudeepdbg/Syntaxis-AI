"""`python -m optimizer` launcher — starts the dashboard + runs a demo."""
from __future__ import annotations

import argparse
import logging
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(prog="optimizer")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the dashboard to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the dashboard to")
    parser.add_argument("--demo", action="store_true", help="Run a demo compression")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from optimizer import start_dashboard, compress

    print(f" Starting dashboard → http://{args.host}:{args.port}/dashboard")
    start_dashboard(host=args.host, port=args.port)

    if args.demo:
        _run_demo()

    # Keep main thread alive so the daemon dashboard thread keeps running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dashboard...")
    return 0


def _run_demo() -> None:
    from optimizer import compress

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Read the file and fix the bug."},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me read the file."},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "main.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "def foo():\n    return 42\n" + ("# filler line\n" * 2000)},
        ]},
        {"role": "assistant", "content": "I see the issue. Let me fix it."},
        {"role": "user", "content": "Yes please."},
    ]

    print("\n⚙️  Running demo compression...")
    result = compress(messages, model="claude-sonnet-4-6", savings_profile="coding")
    print(f"📊 Model          : claude-sonnet-4-6")
    print(f" Tokens Before  : {result.tokens_before:,}")
    print(f"📊 Tokens After   : {result.tokens_after:,}")
    print(f"💾 Tokens Saved   : {result.tokens_saved:,}")
    print(f"📉 Compression    : {result.compression_ratio:.1%}")
    print(f"⚡ Transforms     : {', '.join(result.transforms_applied)}")
    print(f"⏱️  Latency        : {result.latency_ms:.1f} ms")


if __name__ == "__main__":
    sys.exit(main())
