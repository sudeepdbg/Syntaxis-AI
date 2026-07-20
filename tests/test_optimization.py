"""Tests for the core token optimization logic in Syntaxis-AI."""
import pytest
from optimizer import compress, CompressConfig
from optimizer.transforms.deduper import CrossTurnDeduper
from optimizer.relevance.bm25 import BM25Scorer


def test_compress_reduces_tokens_on_large_output():
    """Verify that large tool outputs are compressed and tokens are saved."""
    large_output = "def foo():\n    return 42\n" + ("# filler line\n" * 2000)

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Read main.py and find the bug."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "main.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": large_output}]},
    ]

    # FIX: The default CompressConfig protects the last 4 messages and skips user messages.
    # Since this test only has 4 messages, we must override these defaults to force compression.
    config = CompressConfig(
        compress_user_messages=True,  # Allow compressing the user/tool_result message
        protect_recent=0,             # Don't protect any messages from compression
        min_tokens_to_compress=100,   # Lower the threshold so it triggers
    )

    result = compress(messages, model="claude-sonnet-4-6", config=config)

    assert result.tokens_before > 0
    assert result.tokens_saved > 0, f"Expected tokens to be saved, but none were. Transforms: {result.transforms_applied}"
    assert result.compression_ratio > 0.0


def test_bm25_relevance_scoring():
    """Verify that BM25 scores relevant documents higher than irrelevant ones."""
    scorer = BM25Scorer()
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a high-level programming language used for machine learning.",
        "The weather today is sunny and warm.",
    ]
    
    scorer.fit(documents)
    query = "Python machine learning"
    scores = scorer.score_many(query, documents)
    
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    assert scores[1] > 0.0
