import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from loader import load_passages


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, passages, model_name="BAAI/bge-small-en-v1.5", mode="hybrid"):
        self.passages = passages
        self.mode = mode

        # Dense index
        if mode in ("dense", "hybrid"):
            self.model = SentenceTransformer(model_name)
            self.vecs = self.model.encode(
                [p["text"] for p in passages], normalize_embeddings=True
            )

        # Lexical (BM25) index
        if mode in ("bm25", "hybrid"):
            self.corpus_tokens = [tokenize(p["text"]) for p in passages]
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search_dense(self, query, k=3):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.vecs, q)
        top = np.argsort(-scores)[:k]
        return [{**self.passages[i], "score": float(scores[i])} for i in top]

    def search_bm25(self, query, k=3):
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top = np.argsort(-scores)[:k]
        return [{**self.passages[i], "score": float(scores[i])} for i in top]

    def search_hybrid(self, query, k=3, rrf_k=60, candidate_pool=50):
        # 1. Top candidates from dense
        q = self.model.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.dot(self.vecs, q)
        dense_top = np.argsort(-dense_scores)[:candidate_pool]

        # 2. Top candidates from BM25
        q_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_top = np.argsort(-bm25_scores)[:candidate_pool]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = defaultdict(float)
        for rank, idx in enumerate(dense_top):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(bm25_top):
            rrf_scores[idx] += 1.0 / (rrf_k + rank + 1)

        fused_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:k]
        return [{**self.passages[i], "score": float(rrf_scores[i])} for i in fused_indices]

    def search(self, query, k=3):
        if self.mode == "dense":
            return self.search_dense(query, k=k)
        elif self.mode == "bm25":
            return self.search_bm25(query, k=k)
        else:
            return self.search_hybrid(query, k=k)


if __name__ == "__main__":
    r = Retriever(load_passages(), mode="hybrid")
    for claim in ["The Eiffel Tower is in Berlin.", "Who created Python?"]:
        print(claim)
        for hit in r.search(claim):
            print(f"  {hit['score']:.4f}  {hit['id']}  {hit['text'][:50]}")