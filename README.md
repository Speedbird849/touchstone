# Touchstone: High-Precision Arbitration RAG with Citation-Guided Guardrails

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-Metal%20%2F%20MPS-red.svg)](https://pytorch.org/)
[![Benchmark FEVER](https://img.shields.io/badge/Benchmark-FEVER-green.svg)](https://fever.ai/)

Touchstone is an automated claim verification and arbitration framework designed for compliance-grade precision. It pairs dense-sparse hybrid retrieval with an asymmetric **Generative–Discriminative Guardrail Engine**, achieving **80.0% verdict accuracy** and **90.0% Recall@5** on a balanced 300-claim FEVER benchmark using a local 4B-parameter model.

---

## System Architecture

Touchstone operates as a three-stage verification pipeline:

1. **Hybrid Retrieval**: Queries are scored concurrently via dense semantic vectors (`BAAI/bge-small-en-v1.5`) and sparse lexical matching (`rank_bm25`). Ranks are fused via Reciprocal Rank Fusion (RRF, $k=60$) over 5-sentence paragraph chunks to prevent context starvation.
2. **Generative Compliance Reasoner**: A local model (`qwen3.5:4b`) analyzes the retrieved evidence under strict legal-grade compliance definitions, outputting a draft verdict, explanation, and cited passage IDs.
3. **Citation-Guided Guardrail Engine**: A discriminative cross-encoder (`DeBERTa-v3`) deterministically validates the cited IDs, filters topical distractors, concatenates multi-hop premises, and arbitrates the final verdict through an asymmetric compliance matrix.

---

## Key Technical Mechanisms

1. **Hybrid Retrieval with Paragraph Aggregation**: Combines `BAAI/bge-small-en-v1.5` dense embeddings and `rank_bm25` lexical search fused via Reciprocal Rank Fusion (RRF, $k=60$). Grouping 5 sentences per Wikipedia paragraph chunk eliminates context starvation, boosting **Recall@5 from 63.5% to 90.0%**.
2. **Citation-Guided NLI Verification**: In $k=5$ retrieval, evaluating cross-encoders across all retrieved passages allows off-target distractors to trigger false refutations. Touchstone restricts NLI verification strictly to the passages explicitly cited by the generative model.
3. **Joint Multi-Hop Concatenation**: When claims depend on multi-passage deductions, individual passages score as neutral in isolation. Concatenating cited passages ($P_1 + P_2$) boosts cross-encoder entailment from **26.7% to 99.6%**.
4. **Asymmetric Compliance Matrix**: Resolves the open-world omission dilemma:
   - Downward compliance: Downgrades LLM `REFUTED` to `NOT_ENOUGH_INFO` when NLI finds no direct contradiction (prevents false mutual exclusivity).
   - Hallucination suppression: Downgrades ungrounded `SUPPORTED` claims when cited evidence lacks entailment.
   - Omission preservation: Prevents cross-encoder occupational/attribute bias from overriding safe `NOT_ENOUGH_INFO` decisions.

---

## Empirical Benchmark

Evaluated on a balanced 300-claim test suite from the official FEVER evaluation split (100 `SUPPORTED`, 100 `REFUTED`, 100 `NOT_ENOUGH_INFO`):

| Configuration | Accuracy | Recall@5 | SUP Precision | REF Precision | NEI Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Raw Generative SLM Baseline | 76.7% (230/300) | 90.0% | 79.5% | 76.8% | 61.0% |
| Unconstrained NLI Override | 75.0% (225/300) | 90.0% | 81.1% | 61.9% | 44.0% |
| **Touchstone Guardrail Pipeline** | **80.0% (240/300)** | **90.0%** | **84.1%** | **84.6%** | **73.0%** |

### Verification Set (`evalset_v2.jsonl`, 152 claims)
- **Verdict Accuracy**: **89.5%** (136 / 152)
- **Recall@3**: **100.0%** (91 / 91 verifiable claims)
- **Inference Speed**: ~30 tokens/sec on Apple Silicon (Metal / MPS)

---

## Repository Structure

```
touchstone/
├── corpus/                # FEVER and test passage corpora
├── evalset_fever.jsonl    # 300-claim balanced FEVER benchmark
├── evalset_v2.jsonl       # 152-claim verification dataset
├── retriever.py           # Dense + BM25 hybrid retrieval with RRF
├── verdict.py             # Generative compliance judge (Ollama)
├── guardrails.py          # DeBERTa-v3 cross-encoder + Citation Arbiter
├── eval.py                # Benchmark evaluation suite and CLI
├── loader.py              # Corpus passage loader
└── prepare_fever.py       # FEVER corpus download and paragraph chunking
```

---

## Installation & Usage

### Setup
```bash
git clone https://github.com/<username>/touchstone.git
cd touchstone

python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers sentence-transformers rank_bm25 ollama tqdm pydantic

# Pull local model
ollama run qwen3.5:4b-mlx
```

### Running Evaluations

```bash
# Run held-out FEVER test split (200 claims, k=5, Guardrails ON by default)
python3 eval.py

# Run development / tuning split (300 claims) with trace logging
python3 eval.py --tune --trace traces_tune.jsonl

# Fast checkpoint (e.g. first 30 claims)
python3 eval.py --limit 30

# Disable guardrails (raw LLM baseline)
python3 eval.py --no-guard

# Run full 500-claim combined dataset
python3 eval.py --full

# Run quick synthetic trial verification dataset
python3 eval.py --trial --k3
```

### Python API

```python
from loader import load_passages
from retriever import Retriever
from verdict import judge
from guardrails import GuardrailEngine

passages = load_passages("corpus/fever.txt")
retriever = Retriever(passages, mode="hybrid")
guardrail = GuardrailEngine()

claim = "During the 49ers 63rd season Colin Kaepernick became a starter."
evidence = retriever.search(claim, k=5)
result = judge(claim, evidence, guardrail_engine=guardrail)

print(result["verdict"])           # SUPPORTED
print(result["guardrail_status"])  # AGREEMENT
print(result["evidence_ids"])      # ['Colin_Kaepernick#p0', '2012_San_Francisco_49ers_season#p0']
```
