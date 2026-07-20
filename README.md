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
