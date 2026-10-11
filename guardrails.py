"""
guardrails.py: Hybrid Guardrail Engine for strict fact-checking compliance.
Combines a discriminative NLI cross-encoder with deterministic citation validation.
"""

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class GuardrailEngine:
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def verify_nli(self, claim, passages, threshold=0.65):
        """
        Runs batched NLI cross-encoder inference across all candidate passages.
        Returns aggregated entailment, contradiction, and neutral scores.
        """
        if not passages:
            return {
                "nli_verdict": "NOT_ENOUGH_INFO",
                "confidence": 1.0,
                "best_entailment": 0.0,
                "best_contradiction": 0.0,
                "passage_scores": [],
            }

        pairs_p = [p["text"] for p in passages]
        pairs_h = [claim] * len(passages)

        inputs = self.tokenizer(
            pairs_p,
            pairs_h,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

        passage_scores = []
        max_entail = 0.0
        entail_pid = None
        max_contra = 0.0
        contra_pid = None

        STOPWORDS = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "of", "and", "or", "but", "so", "that", "this", "these", "those", "it", "its"}
        import re
        claim_words = set(re.findall(r"\b\w{3,}\b", claim.lower())) - STOPWORDS

        for p, prob in zip(passages, probs):
            # DeBERTa index: 0: entailment, 1: neutral, 2: contradiction
            e_score, n_score, c_score = prob[0], prob[1], prob[2]

            # Verify topical overlap: candidate passage must share at least one content word with the claim
            p_text = (p.get("text", "") + " " + p.get("id", "").replace("_", " ")).lower()
            p_words = set(re.findall(r"\b\w{3,}\b", p_text))
            is_topical = bool(claim_words & p_words) if claim_words else True

            passage_scores.append({
                "id": p["id"],
                "entailment": e_score,
                "neutral": n_score,
                "contradiction": c_score,
                "topical": is_topical,
            })
            if is_topical:
                if e_score > max_entail:
                    max_entail = e_score
                    entail_pid = p["id"]
                if c_score > max_contra:
                    max_contra = c_score
                    contra_pid = p["id"]

        if max_contra >= threshold:
            nli_verdict = "REFUTED"
            conf = max_contra
            primary_id = contra_pid
        elif max_entail >= threshold:
            nli_verdict = "SUPPORTED"
            conf = max_entail
            primary_id = entail_pid
        else:
            nli_verdict = "NOT_ENOUGH_INFO"
            conf = 1.0 - max(max_entail, max_contra)
            primary_id = None

        return {
            "nli_verdict": nli_verdict,
            "confidence": conf,
            "best_entailment": {"score": max_entail, "id": entail_pid},
            "best_contradiction": {"score": max_contra, "id": contra_pid},
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

        # 3. Run NLI cross-encoder
        nli_check = self.verify_nli(claim, target_passages)
        nli_verdict = nli_check["nli_verdict"]

        # 4. Decision Matrix
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
            # Guardrail intercepts hallucinated support
            final_verdict = "NOT_ENOUGH_INFO"
            status = "HALLUCINATION_DOWNGRADED"
        elif nli_verdict in ("SUPPORTED", "REFUTED") and nli_check["confidence"] >= 0.85:
            # High-confidence NLI override when LLM is uncertain
            final_verdict = nli_verdict
            status = "NLI_OVERRIDE"

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
