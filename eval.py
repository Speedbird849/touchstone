import json
from collections import Counter

from loader import load_passages
from retriever import Retriever
from verdict import judge


def run(path="evalset.jsonl", k=3):
    retriever = Retriever(load_passages())
    rows = [json.loads(line) for line in open(path) if line.strip()]

    correct = hits = n_gold = 0
    confusion = Counter()
    for row in rows:
        evidence = retriever.search(row["claim"], k=k)
        got = judge(row["claim"], evidence)["verdict"]

        confusion[(row["label"], got)] += 1
        correct += got == row["label"]
        if row["gold_ids"]:
            n_gold += 1
            hits += any(p["id"] in row["gold_ids"] for p in evidence)
        if got != row["label"]:
            print(f"MISS  want={row['label']}  got={got}  | {row['claim']}")

    print(f"\nverdict accuracy: {correct}/{len(rows)} = {correct / len(rows):.0%}")
    print(f"retrieval recall@{k}: {hits}/{n_gold} = {hits / max(n_gold, 1):.0%}")
    for (want, got), n in sorted(confusion.items()):
        print(f"  want {want:16} got {got:16} {n}")


if __name__ == "__main__":
    run()