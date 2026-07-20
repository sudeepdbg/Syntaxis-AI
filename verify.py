"""Quick verification script for Syntaxis-AI."""
from optimizer import compress, CompressConfig, start_dashboard
import time

# 1. Start the dashboard in the background
print("Starting dashboard...")
start_dashboard()
time.sleep(2)

# 2. Create a massive fake tool output (2000 lines of filler)
huge_output = "def calculate():\n    return 42\n" + ("# useless filler comment\n" * 2000)

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Analyze this file."},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "main.py"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": huge_output}]},
]

# 3. Force compression by turning off default protections
# (Default settings protect the last 4 messages, which is why your previous test failed)
config = CompressConfig(
    compress_user_messages=True,
    protect_recent=0,
    min_tokens_to_compress=100,
)

print("\nRunning compression...")
result = compress(messages, model="claude-sonnet-4-6", config=config)

# 4. Print the results
print("\n" + "="*50)
print(f"Tokens Before : {result.tokens_before:,}")
print(f"Tokens After  : {result.tokens_after:,}")
print(f"Tokens Saved  : {result.tokens_saved:,}")
print(f"Savings       : {result.compression_ratio:.1%}")
print(f"Transforms    : {result.transforms_applied}")
print("="*50)

print("\nOpen your browser to: http://127.0.0.1:8765/dashboard")
print("You should see 1 request and ~8,000 tokens saved!")
print("\nPress Ctrl+C to stop the dashboard.")

# Keep the script running so the dashboard stays alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped.")
