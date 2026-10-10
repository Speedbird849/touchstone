import json
import re
import ollama

MODEL = "qwen3.5:4b-mlx"  # your exact name
VALID = {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}

SYSTEM = """You are a strict compliance and legal fact-checker. You judge claims strictly on what the evidence text explicitly states. Never use real-world common sense, outside knowledge, or assumptions.

Rules:
- SUPPORTED: The evidence passages explicitly state EVERY detail, entity, and relationship in the claim. If any specific detail is missing from the text, choose NOT_ENOUGH_INFO.
- REFUTED: An evidence passage contains an EXPLICIT, DIRECT textual contradiction on the exact same property (e.g., Claim: "directed by Peter Jackson" vs Evidence: "directed by Pablo Larraín"; Claim: "born in 1990" vs Evidence: "born in 1980").
- NOT_ENOUGH_INFO: The evidence does not contain proof for one or more details in the claim.
CRITICAL RULE ON OMISSIONS: Omission is NOT contradiction. If a passage mentions an entity, but never mentions a specific attribute claimed (e.g. their profession, hobby, nationality, or secondary status), you CANNOT assume mutual exclusivity. For example, if evidence says someone is a "musician" or "band", but does NOT explicitly state they are not a "lawyer", the detail about being a lawyer is simply unmentioned -> NOT_ENOUGH_INFO.

Reply with ONLY a JSON object and no other text. Write the rationale FIRST, then the verdict:
{"rationale": "cite what the text explicitly states or note what is unmentioned", "evidence_ids": ["id", ...], "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO"}"""


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