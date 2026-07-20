# Syntaxis-AI

# Token Optimizer

Cut LLM costs 50-90% without losing accuracy.

A Python library that compresses LLM messages before sending them to the API.
Works with Anthropic, OpenAI, Bedrock, Vertex, Gemini — any provider.

## Features

- **SmartCrusher** — lossless JSON/text compression (dedup arrays, truncate with head+tail, preserve errors)
- **CrossTurnDeduper** — collapse identical content across messages
- **CacheAligner** — stabilize prefix for provider KV cache hits (50-90% cost savings on agentic workloads)
- **LossyKompressor** — optional ML compression via `kompress-v2-base`
- **Relevance scoring** — BM25 + optional embeddings, so the right chunks get kept
- **Named profiles** — `coding`, `balanced`, `agent-90`, `general`
- **Live dashboard** — SSE-powered, terminal-flavored UI at `http://127.0.0.1:8765/dashboard`
- **OTel metrics** — optional OpenTelemetry integration
- **Inflation guard** — reverts automatically if compression makes things bigger
- **Air-gap ready** — `OPTIMIZER_OFFLINE=1` disables all outbound network

## Quick Start

```bash
# Install
pip install -e .

# Or with all extras
pip install -e ".[all]"

# Run the dashboard + demo
python -m optimizer --demo
# → http://127.0.0.1:8765/dashboard
```

## Usage

```python
from optimizer import compress

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Analyze this code..."},
    {"role": "assistant", "content": [{"type": "tool_use", ...}]},
    {"role": "user", "content": [{"type": "tool_result", "content": huge_output}]},
]

result = compress(messages, model="claude-sonnet-4-6", savings_profile="coding")

print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.0%})")
print(f"Transforms: {result.transforms_applied}")

# Send compressed messages to your LLM client
response = client.messages.create(
    model="claude-sonnet-4-6",
    messages=result.messages,
)
```

## Profiles

| Profile | Mode | Savings | Use case |
|---|---|---|---|
| `coding` (default) | cache | ~50% emergent | Coding agents (Claude Code, Cursor, Codex) |
| `balanced` | token | ~70% | General use |
| `agent-90` | token | ~90% | Aggressive (logs, search results) |
| `general` | token | ~60% | Chat, Q&A |

## Architecture

```
compress(messages)
    │
    ▼
┌──────────────────────────────────────────┐
│ Profile (coding/balanced/agent-90)       │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│ TransformPipeline                        │
│  1. SmartCrusher (lossless)              │
│  2. CrossTurnDeduper (optional)          │
│  3. LossyKompressor (optional, ML)       │
│  4. CacheAligner (provider breakpoints)  │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│ Inflation guard (revert if tokens grew)  │
└──────────────┬───────────────────────────┘
               ▼
     Compressed messages + metrics
```

## Development

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/token-optimizer.git
cd token-optimizer

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install with dev deps
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check optimizer/
mypy optimizer/
```

## License

MIT
