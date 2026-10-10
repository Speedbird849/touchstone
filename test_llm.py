import os
import time
from openai import OpenAI

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b-mlx")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def ask():
    t = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
        temperature=0,
        reasoning_effort="none",
    )
    print(resp.choices[0].message.content.strip(), f"({time.time() - t:.2f}s)")


ask()  # cold: includes model load
ask()  # warm: the real latency