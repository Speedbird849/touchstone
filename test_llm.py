import time
import ollama

def ask():
    t = time.time()
    resp = ollama.chat(
        model="qwen3.5:9b-mlx",
        messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
        think=False,
        keep_alive="30m",
    )
    print(resp["message"]["content"], f"({time.time() - t:.1f}s)")

ask()  # cold: includes model load
ask()  # warm: the real latency