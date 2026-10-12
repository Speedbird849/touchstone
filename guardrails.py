"""
guardrails.py: Hybrid Guardrail Engine for strict fact-checking compliance.
Combines a discriminative NLI cross-encoder with deterministic citation validation.
"""

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

STOPWORDS = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "of", "and", "or", "but", "so", "that", "this", "these", "those", "it", "its"}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text, max_sentences=8):
    """Split a passage into sentence windows for focused NLI scoring.

    Falls back to the whole text when no sentence boundary is found.
    """
    parts = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    sents = [s for s in parts if len(re.findall(r"\w+", s)) >= 3]
    return (sents or [text.strip()])[:max_sentences]


class GuardrailEngine:
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                 nli_threshold=0.65, entail_floor=0.30, contra_cap=0.40,
                 use_sentence_window=True):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        # Hard verdict threshold (high-precision overrides) vs soft-gate band:
        # SUPPORTED is downgraded only if focused entailment < entail_floor
        # or contradiction > contra_cap. Fit on tune split — see calibrate_nli.py.
        self.nli_threshold = nli_threshold
        self.entail_floor = entail_floor
        self.contra_cap = contra_cap
        self.use_sentence_window = use_sentence_window

    def verify_nli(self, claim, passages, threshold=None):
        """
        Runs batched NLI cross-encoder inference across all candidate passages.

        Each passage is scored whole AND as sentence windows (when
        ``use_sentence_window`` is set): long paragraphs dilute cross-attention,
        so the focused per-sentence entailment signal is tracked alongside the
        paragraph signal. Both feed ``best_entailment`` / ``best_contradiction``
        as a max over topically aligned windows.
        Returns aggregated entailment, contradiction, and neutral scores.
        """
        if threshold is None:
            threshold = self.nli_threshold
        if not passages:
            return {
                "nli_verdict": "NOT_ENOUGH_INFO",
                "confidence": 1.0,
                "best_entailment": {"score": 0.0, "id": None},
                "best_contradiction": {"score": 0.0, "id": None},
                "sentence_contra_max": {"score": 0.0, "id": None},
                "primary_evidence_id": None,
                "passage_scores": [],
            }

        claim_words = set(re.findall(r"\b\w{3,}\b", claim.lower())) - STOPWORDS

        # Build batched items: (paragraph, sentence windows...) per passage
        items = []
        for p in passages:
            pid = p["id"]
            id_words = set(re.findall(r"\b\w{3,}\b", pid.replace("_", " ").lower()))
            items.append({"pid": pid, "text": p["text"], "kind": "para",
                          "idx": -1, "id_words": id_words})
            if self.use_sentence_window:
                for i, sent in enumerate(split_sentences(p.get("text", ""))):
                    items.append({"pid": pid, "text": sent, "kind": "sent",
                                  "idx": i, "id_words": id_words})

        # Micro-batched inference (re-tokenized per micro-batch for tight
        # padding): sentence windows + the joint-concat candidate inflate the
        # pair count ~6x; a single MPS batch OOMs.
        premises = [it["text"] for it in items]
        probs = []
        micro = 8
        with torch.no_grad():
            for s in range(0, len(items), micro):
                chunk_inputs = self.tokenizer(
                    premises[s:s + micro],
                    [claim] * min(micro, len(items) - s),
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**chunk_inputs).logits
                probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        if self.device == "mps":
            torch.mps.empty_cache()

        passage_scores = []
        max_entail = 0.0
        entail_src = None
        # Hard contradiction comes from whole-paragraph windows only: lone
        # sentences spike spurious contradiction on name variants (e.g. Hedda
        # Tesman vs Hedda Gabler) while the paragraph overall entails.
        # Sentence-level contradiction is still exposed via
        # ``sentence_contra_max`` for the soft gate.
        max_contra = 0.0
        contra_src = None
        sent_contra_max = 0.0
        sent_contra_src = None

        for p in passages:
            pid = p["id"]
            para_entry = sent_best_e = sent_best_c = None
            for it, prob in zip(items, probs):
                if it["pid"] != pid:
                    continue
                # DeBERTa index: 0: entailment, 1: neutral, 2: contradiction
                e_score, n_score, c_score = prob[0], prob[1], prob[2]
                # Topical overlap: window must share a content word with the
                # claim; sentence windows may also match via the parent page id
                # (entity names live there, e.g. Hedda_Gabler).
                w_text = (it["text"] + " " + " ".join(it["id_words"])).lower()
                w_words = set(re.findall(r"\b\w{3,}\b", w_text))
                is_topical = bool(claim_words & w_words) if claim_words else True
                entry = {
                    "kind": it["kind"], "idx": it["idx"],
                    "entailment": e_score, "neutral": n_score,
                    "contradiction": c_score, "topical": is_topical,
                    "text": it["text"][:160],
                }
                if it["kind"] == "para":
                    para_entry = entry
                else:
                    if sent_best_e is None or e_score > sent_best_e["entailment"]:
                        sent_best_e = entry
                    if sent_best_c is None or c_score > sent_best_c["contradiction"]:
                        sent_best_c = entry
                if is_topical:
                    if e_score > max_entail:
                        max_entail = e_score
                        entail_src = {"id": pid, "kind": it["kind"], "idx": it["idx"]}
                    if it["kind"] == "para":
                        if c_score > max_contra:
                            max_contra = c_score
                            contra_src = {"id": pid, "kind": it["kind"], "idx": it["idx"]}
                    else:
                        if c_score > sent_contra_max:
                            sent_contra_max = c_score
                            sent_contra_src = {"id": pid, "kind": it["kind"], "idx": it["idx"]}

            passage_scores.append({
                "id": pid,
                "entailment": para_entry["entailment"] if para_entry else 0.0,
                "neutral": para_entry["neutral"] if para_entry else 0.0,
                "contradiction": para_entry["contradiction"] if para_entry else 0.0,
                "topical": para_entry["topical"] if para_entry else False,
                "sent_best_entailment": (sent_best_e["entailment"] if sent_best_e else None),
                "sent_best_contra": (sent_best_c["contradiction"] if sent_best_c else None),
            })

        if max_contra >= threshold:
            nli_verdict = "REFUTED"
            conf = max_contra
            primary_id = contra_src["id"] if contra_src else None
        elif max_entail >= threshold:
            nli_verdict = "SUPPORTED"
            conf = max_entail
            primary_id = entail_src["id"] if entail_src else None
        else:
            nli_verdict = "NOT_ENOUGH_INFO"
            conf = 1.0 - max(max_entail, max_contra)
            primary_id = None

        return {
            "nli_verdict": nli_verdict,
            "confidence": conf,
            "best_entailment": {"score": max_entail, **(entail_src or {"id": None})},
            "best_contradiction": {"score": max_contra, **(contra_src or {"id": None})},
            "sentence_contra_max": {"score": sent_contra_max, **(sent_contra_src or {"id": None})},
            "primary_evidence_id": primary_id,
            "passage_scores": passage_scores,
        }

    def verify_citations(self, cited_ids, passages):
        """
        Deterministic check: ensures cited passage IDs actually exist in the retrieved context.
        """
        valid_ids = set()
        for p in passages:
            valid_ids.add(p["id"])
            for sid in p.get("sentence_ids", []):
                valid_ids.add(sid)

        cleaned_cited = [cid.strip("[]'\" ") for cid in cited_ids if cid]
        valid_citations = [cid for cid in cleaned_cited if cid in valid_ids]
        hallucinated_citations = [cid for cid in cleaned_cited if cid not in valid_ids]

        return {
            "valid": valid_citations,
            "hallucinated": hallucinated_citations,
            "has_hallucinations": len(hallucinated_citations) > 0,
        }

    def arbitrate(self, claim, passages, llm_result):
        """
        The Arbiter: Merges the LLM's draft rationale with the NLI verification.
        """
        llm_verdict = llm_result.get("verdict", "PARSE_ERROR")
        cited_ids = llm_result.get("evidence_ids", [])

        # 1. Check citations
        cite_check = self.verify_citations(cited_ids, passages)

        # 2. Select target passages for NLI
        # If LLM cited specific evidence passages for its verdict, verify those specific passages
        target_passages = passages
        if cite_check["valid"] and llm_verdict in ("SUPPORTED", "REFUTED"):
            valid_set = set(cite_check["valid"])
            matched = [
                p for p in passages
                if p["id"] in valid_set or any(sid in valid_set for sid in p.get("sentence_ids", []))
            ]
            if matched:
                target_passages = matched

        # If multiple passages were cited, evaluate joint/multi-hop evidence as well
        eval_candidates = list(target_passages)
        if len(target_passages) > 1:
            combined_candidate = {
                "id": "+".join(p["id"] for p in target_passages),
                "text": " ".join(p["text"] for p in target_passages),
                "sentence_ids": [sid for p in target_passages for sid in p.get("sentence_ids", [])],
            }
            eval_candidates.append(combined_candidate)

        # 3. Run NLI cross-encoder
        nli_check = self.verify_nli(claim, eval_candidates)
        nli_verdict = nli_check["nli_verdict"]

        # 4. Decision Matrix: NLI acts as strict compliance gatekeeper
        final_verdict = llm_verdict
        status = "AGREEMENT"

        if llm_verdict == nli_verdict:
            final_verdict = llm_verdict
            status = "AGREEMENT"
        elif llm_verdict == "REFUTED" and nli_verdict == "NOT_ENOUGH_INFO":
            # Guardrail intercepts overreaching refutation (e.g. unmentioned attribute)
            final_verdict = "NOT_ENOUGH_INFO"
            status = "OVERREACH_DOWNGRADED"
        elif llm_verdict == "SUPPORTED" and nli_verdict == "NOT_ENOUGH_INFO":
            # Soft gate: paragraph-level neutral with moderate focused
            # sentence entailment usually means attention dilution across the
            # 5-sentence chunk, not hallucination. Downgrade only on
            # negligible entailment or live contradiction (either window).
            e_score = nli_check["best_entailment"]["score"]
            c_score = max(nli_check["best_contradiction"]["score"],
                          nli_check.get("sentence_contra_max", {}).get("score", 0.0))
            if e_score < self.entail_floor or c_score > self.contra_cap:
                final_verdict = "NOT_ENOUGH_INFO"
                status = "HALLUCINATION_DOWNGRADED"
            else:
                final_verdict = "SUPPORTED"
                status = "SOFT_GATE_PRESERVED"
        elif llm_verdict == "SUPPORTED" and nli_verdict == "REFUTED" and nli_check["confidence"] >= 0.85:
            # Evidence directly contradicts the claim that the LLM mistakenly supported
            final_verdict = "REFUTED"
            status = "NLI_REFUTED_OVERRIDE"
        elif llm_verdict == "REFUTED" and nli_verdict == "SUPPORTED" and nli_check["confidence"] >= 0.85:
            # Evidence directly entails the claim that the LLM mistakenly refuted
            final_verdict = "SUPPORTED"
            status = "NLI_SUPPORT_OVERRIDE"
        else:
            # When LLM decided NOT_ENOUGH_INFO, trust LLM's careful omission analysis
            final_verdict = llm_verdict
            status = "AGREEMENT"

        return {
            "verdict": final_verdict,
            "rationale": llm_result.get("rationale", ""),
            "evidence_ids": cite_check["valid"] or ([nli_check["primary_evidence_id"]] if nli_check["primary_evidence_id"] else []),
            "guardrail_status": status,
            "nli_details": nli_check,
            "citation_details": cite_check,
            "_stats": llm_result.get("_stats", {"tokens": 0, "secs": 0.0}),
        }


if __name__ == "__main__":
    engine = GuardrailEngine()
    test_passages = [
        {"id": "Harold_Macmillan#p0", "text": "Harold Macmillan was a British Conservative politician who served as Prime Minister."},
        {"id": "Never_So_Good#p0", "text": "Never So Good is a play about Harold Macmillan, a 20th-century British politician."}
    ]
    test_claim = "Harold Macmillan was a swimmer."
    draft_llm = {
        "verdict": "REFUTED",
        "rationale": "He was a politician, which contradicts being a swimmer.",
        "evidence_ids": ["Harold_Macmillan#p0"],
    }
    arbitrated = engine.arbitrate(test_claim, test_passages, draft_llm)
    print("Claim:     ", test_claim)
    print("LLM wanted:", draft_llm["verdict"])
    print("Arbiter:   ", arbitrated["verdict"], f"({arbitrated['guardrail_status']})")
    print("NLI Probs: ", arbitrated["nli_details"]["passage_scores"][0])
