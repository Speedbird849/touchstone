from __future__ import annotations

from typing import Literal, Optional
import re
import ollama
from pydantic import BaseModel, Field, ValidationError

MODEL = "qwen3.5:4b-mlx"


class Verdict(BaseModel):
    rationale: str = Field(description="Cite what the text explicitly states or note what is unmentioned")
    evidence_ids: list[str] = Field(default_factory=list, description="IDs of evidence passages used")
    verdict: Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


SCHEMA = Verdict.model_json_schema()

SYSTEM = """You are a strict compliance and legal fact-checker. You judge claims strictly on what the evidence text explicitly states. Never use real-world common sense, outside knowledge, or assumptions.

Rules:
- SUPPORTED: The evidence passages explicitly state EVERY detail, entity, and relationship in the claim. If any specific detail is missing from the text, choose NOT_ENOUGH_INFO.
- REFUTED: An evidence passage contains an EXPLICIT, DIRECT textual contradiction on the exact same property (e.g., Claim: "directed by Peter Jackson" vs Evidence: "directed by Pablo Larraín"; Claim: "born in 1990" vs Evidence: "born in 1980").
- NOT_ENOUGH_INFO: The evidence does not contain proof for one or more details in the claim.
CRITICAL RULE ON OMISSIONS: Omission is NOT contradiction. If a passage mentions an entity, but never mentions a specific attribute claimed (e.g. their profession, hobby, nationality, or secondary status), you CANNOT assume mutual exclusivity. For example, if evidence says someone is a "musician" or "band", but does NOT explicitly state they are not a "lawyer", the detail about being a lawyer is simply unmentioned -> NOT_ENOUGH_INFO.

Reply with ONLY a JSON object and no other text:
{"rationale": "cite what the text explicitly states or note what is unmentioned", "evidence_ids": ["id", ...], "verdict": "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO"}"""


def parse(text: str) -> Optional[Verdict]:
    text_clean = text.strip()
    if text_clean.startswith("```"):
        text_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_clean, flags=re.MULTILINE).strip()

    for candidate in [text_clean, None]:
        if candidate is None:
            m = re.search(r"\{.*\}", text_clean, re.DOTALL)
            if not m:
                continue
            candidate = m.group(0)
        try:
            import json
            data = json.loads(candidate)
            if isinstance(data, dict):
                if "properties" in data and isinstance(data["properties"], dict):
                    data = data["properties"]
                return Verdict.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError):
            continue
    return None


def _query_ollama(messages):
    try:
        return ollama.chat(
            model=MODEL,
            messages=messages,
            format=SCHEMA,
            think=False,
            keep_alive="30m",
            options={"temperature": 0},
        )
    except ollama.ResponseError as e:
        # Ollama --mlx-engine runner returns 501 for format; fallback cleanly while keeping schema in prompt
        if e.status_code == 501:
            return ollama.chat(
                model=MODEL,
                messages=messages,
                think=False,
                keep_alive="30m",
                options={"temperature": 0},
            )
        raise


def judge(claim, evidence, retries=1, guardrail_engine=None):
    context = "\n".join(f"[{p['id']}] {p['text']}" for p in evidence)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Claim: {claim}\n\nEvidence:\n{context}"},
    ]
    for _ in range(retries + 1):
        resp = _query_ollama(messages)
        stats = {"tokens": resp.eval_count or 0, "secs": (resp.eval_duration or 0) / 1e9}
        out = parse(resp["message"]["content"])
        if out:
            res = {**out.model_dump(), "_stats": stats}
            if guardrail_engine is not None:
                return guardrail_engine.arbitrate(claim, evidence, res)
            return res
    fallback = {"verdict": "PARSE_ERROR", "evidence_ids": [], "rationale": resp["message"]["content"], "_stats": stats}
    if guardrail_engine is not None:
        return guardrail_engine.arbitrate(claim, evidence, fallback)
    return fallback

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