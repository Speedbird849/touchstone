import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import json
from collections import Counter

from loader import load_passages
from retriever import Retriever
from verdict import judge
from tqdm import tqdm


def run(eval_path="evalset_v2.jsonl", corpus_path="corpus/test.txt", k=3, mode="hybrid", use_guardrails=False, limit=None):
    retriever = Retriever(load_passages(corpus_path), mode=mode)
    guardrail_engine = None
    if use_guardrails:
        print("Initializing Hybrid Guardrail Engine (DeBERTa-v3 cross-encoder + Citation Validator)...")
        from guardrails import GuardrailEngine
        guardrail_engine = GuardrailEngine()

    rows = [json.loads(line) for line in open(eval_path) if line.strip()]
    if limit is not None:
        rows = rows[:limit]

    correct = hits = n_gold = done = 0
    toks = gen_secs = 0.0
    confusion = Counter()
    guardrail_actions = Counter()
    bar = tqdm(rows, unit="claim")
    for row in bar:
        evidence = retriever.search(row["claim"], k=k)
        result = judge(row["claim"], evidence, guardrail_engine=guardrail_engine)
        got = result["verdict"]

        done += 1
        toks += result["_stats"]["tokens"]
        gen_secs += result["_stats"]["secs"]
        confusion[(row["label"], got)] += 1
        if "guardrail_status" in result:
            guardrail_actions[result["guardrail_status"]] += 1
        correct += got == row["label"]
        if row["gold_ids"]:
            n_gold += 1
            hits += any(any(gid in p.get("sentence_ids", [p["id"]]) for gid in row["gold_ids"]) for p in evidence)
        if got != row["label"]:
            status_note = f" [{result.get('guardrail_status')}]" if "guardrail_status" in result else ""
            tqdm.write(f"MISS{status_note}  want={row['label']}  got={got}  | {row['claim']}\n      why: {result.get('rationale', result)}")
        bar.set_postfix(acc=f"{correct / done:.0%}", tok_s=f"{toks / max(gen_secs, 1e-9):.0f}")

    print(f"\nverdict accuracy: {correct}/{len(rows)} = {correct / len(rows):.0%}")
    print(f"retrieval recall@{k}: {hits}/{n_gold} = {hits / max(n_gold, 1):.0%}")
    print(f"generation speed: {toks / max(gen_secs, 1e-9):.0f} tokens/s ({int(toks)} tokens total)")
    for (want, got), n in sorted(confusion.items()):
        print(f"  want {want:16} got {got:16} {n}")

    if use_guardrails:
        print("\nguardrail interventions:")
        for action, cnt in guardrail_actions.most_common():
            print(f"  {action:25} {cnt}")


if __name__ == "__main__":
    import sys
    use_guardrails = "--guardrails" in sys.argv or "-g" in sys.argv
    limit = None
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])
    if "--fever" in sys.argv:
        k = 5 if "--k5" in sys.argv else 3
        mode = "dense" if "--dense" in sys.argv else "bm25" if "--bm25" in sys.argv else "hybrid"
        run(eval_path="evalset_fever.jsonl", corpus_path="corpus/fever.txt", k=k, mode=mode, use_guardrails=use_guardrails, limit=limit)
    else:
        run(use_guardrails=use_guardrails, limit=limit)