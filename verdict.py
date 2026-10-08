import json
import re
import ollama

MODEL = "qwen3.5:9b-mlx"  # your exact name
VALID = {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}

SYSTEM = """You are a fact-checker. Judge the claim using ONLY the evidence passages.
- SUPPORTED: a passage explicitly states what the claim says.
- REFUTED: a passage explicitly states something that cannot be true if the claim is true.
- NOT_ENOUGH_INFO: no passage addresses the claim's key detail. A claim being false in the real world does NOT make it REFUTED; the evidence itself must contradict it.
Never use outside knowledge. If you notice yourself relying on what you already know, the answer is NOT_ENOUGH_INFO.

Reply with ONLY a JSON object and no other text. Write the rationale FIRST, then the verdict:
{"rationale": "quote or point to what the evidence says about the claim's key detail", "evidence_ids": ["id", ...], "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO"}"""


def parse(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)  # grab the outermost {...}
    if not match:
        return None
    try:
        out = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return out if out.get("verdict") in VALID else None


def judge(claim, evidence, retries=1):
    context = "\n".join(f"[{p['id']}] {p['text']}" for p in evidence)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Claim: {claim}\n\nEvidence:\n{context}"},
    ]
    for _ in range(retries + 1):
        resp = ollama.chat(
            model=MODEL,
            messages=messages,
            think=False,
            keep_alive="30m",
            options={"temperature": 0},
        )
        out = parse(resp["message"]["content"])
        if out:
            return out
    return {"verdict": "PARSE_ERROR", "evidence_ids": [], "rationale": resp["message"]["content"]}

if __name__ == "__main__":
    from loader import load_passages
    from retriever import Retriever

    r = Retriever(load_passages())
    for claim in [
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower is in Berlin.",
        "Mount Everest is in Peru.",
        "Guido van Rossum won a Turing Award.",
    ]:
        print(claim)
        print(" ", judge(claim, r.search(claim, k=3)))