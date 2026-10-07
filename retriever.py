import numpy as np
from sentence_transformers import SentenceTransformer
from loader import load_passages


class Retriever:
    def __init__(self, passages, model_name="BAAI/bge-small-en-v1.5"):
        self.passages = passages
        self.model = SentenceTransformer(model_name)
        self.vecs = self.model.encode(
            [p["text"] for p in passages], normalize_embeddings=True
        )

    def search(self, query, k=3):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.vecs @ q
        top = np.argsort(-scores)[:k]
        return [{**self.passages[i], "score": float(scores[i])} for i in top]


if __name__ == "__main__":
    r = Retriever(load_passages())
    for claim in ["The Eiffel Tower is in Berlin.", "Who created Python?"]:
        print(claim)
        for hit in r.search(claim):
            print(f"  {hit['score']:.3f}  {hit['id']}  {hit['text'][:50]}")