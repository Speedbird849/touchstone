import json
import re
import ollama

MODEL = "qwen3.5:4b-mlx"  # your exact name
VALID = {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}

SYSTEM = """You are a fact-checker. Judge the claim using ONLY the evidence passages.
- SUPPORTED: the passages state EVERYTHING the claim says. Every specific detail in the claim (year, number, name, place) must appear in a passage. If any detail is missing, the answer is NOT_ENOUGH_INFO.
- REFUTED: a passage explicitly states something that cannot be true if the claim is true.
- NOT_ENOUGH_INFO: no passage addresses the claim's key detail. A claim being false in the real world does NOT make it REFUTED; the evidence itself must contradict it.
Never use outside knowledge. If you notice yourself relying on what you already know, the answer is NOT_ENOUGH_INFO.

Reply with ONLY a JSON object and no other text. Write the rationale FIRST, then the verdict:
{"rationale": "quote or point to what the evidence says about the claim's key detail", "evidence_ids": ["id", ...], "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO"}"""


def parse(text):
    text_clean = text.strip()
    m = re.search(r"\{.*\}", text_clean, re.DOTALL)
    if m:
        candidate = m.group(0)
        for s in [candidate, candidate.replace(r'\"', '"')]:
            try:
                out = json.loads(s)
                if out.get("verdict") in VALID:
                    out["rationale"] = next((v for k, v in out.items() if k.startswith("rationa")), "")
                    out.setdefault("evidence_ids", [])
                    return out
            except json.JSONDecodeError:
                pass

    v = re.search(r'\\?"?verdict\\?"?\s*:\s*\\?"?(SUPPORTED|REFUTED|NOT_ENOUGH_INFO)\\?"?', text_clean)
    if v:
        return {"verdict": v.group(1), "evidence_ids": [], "rationale": text_clean}

    first_line = text_clean.split("\n")[0].strip()
    if first_line in VALID:
        return {"verdict": first_line, "evidence_ids": [], "rationale": text_clean[len(first_line):].strip()}

    return None

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
        stats = {"tokens": resp.eval_count or 0, "secs": (resp.eval_duration or 0) / 1e9}
        out = parse(resp["message"]["content"])
        if out:
            return {**out, "_stats": stats}
    return {"verdict": "PARSE_ERROR", "evidence_ids": [], "rationale": resp["message"]["content"], "_stats": stats}

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