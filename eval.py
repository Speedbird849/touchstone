import json
from collections import Counter
from unittest import result
from unittest import result

from loader import load_passages
from retriever import Retriever
from verdict import judge
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")


def run(path="evalset_v2.jsonl", k=3):
    retriever = Retriever(load_passages())
    rows = [json.loads(line) for line in open(path) if line.strip()]

    correct = hits = n_gold = done = 0
    toks = gen_secs = 0.0
    confusion = Counter()
    bar = tqdm(rows, unit="claim")
    for row in bar:
        evidence = retriever.search(row["claim"], k=k)
        result = judge(row["claim"], evidence)
        got = result["verdict"]

        done += 1
        toks += result["_stats"]["tokens"]
        gen_secs += result["_stats"]["secs"]
        confusion[(row["label"], got)] += 1
        correct += got == row["label"]
        if row["gold_ids"]:
            n_gold += 1
            hits += any(p["id"] in row["gold_ids"] for p in evidence)
        if got != row["label"]:
            tqdm.write(f"MISS  want={row['label']}  got={got}  | {row['claim']}\n      why: {result.get('rationale', result)}")
        bar.set_postfix(acc=f"{correct / done:.0%}", tok_s=f"{toks / max(gen_secs, 1e-9):.0f}")

    print(f"\nverdict accuracy: {correct}/{len(rows)} = {correct / len(rows):.0%}")
    print(f"retrieval recall@{k}: {hits}/{n_gold} = {hits / max(n_gold, 1):.0%}")
    print(f"generation speed: {toks / max(gen_secs, 1e-9):.0f} tokens/s ({int(toks)} tokens total)")
    for (want, got), n in sorted(confusion.items()):
        print(f"  want {want:16} got {got:16} {n}")


if __name__ == "__main__":
    run()