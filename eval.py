import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import json
from collections import Counter

from loader import load_passages
from retriever import Retriever
from verdict import judge
from tqdm import tqdm


def run(eval_path="evalset_fever_test.jsonl", corpus_path="corpus/fever.txt", k=5, mode="hybrid", use_guardrails=True, limit=None, trace_path=None):
    retriever = Retriever(load_passages(corpus_path), mode=mode)
    guardrail_engine = None
    if use_guardrails:
        print("Initializing Hybrid Guardrail Engine (DeBERTa-v3 cross-encoder + Citation Validator)...")
        from guardrails import GuardrailEngine
        guardrail_engine = GuardrailEngine()

    rows = [json.loads(line) for line in open(eval_path) if line.strip()]
    if limit is not None:
        rows = rows[:limit]

    if "fever_test" in eval_path:
        dataset_name = "FEVER-TEST (held-out)"
    elif "fever_tune" in eval_path:
        dataset_name = "FEVER-TUNE"
    elif "fever_500" in eval_path:
        dataset_name = "FEVER-500"
    elif "fever" in eval_path:
        dataset_name = "FEVER-LEGACY"
    elif "test" in eval_path or "v2" in eval_path:
        dataset_name = "TRIAL"
    else:
        dataset_name = eval_path

    guard_status = "ENABLED" if use_guardrails else "DISABLED"
    print(f"\n=== Evaluation: {dataset_name} ({len(rows)} claims) | k={k} | mode={mode} | guardrails={guard_status} ===\n")

    correct = hits = n_gold = done = 0
    toks = gen_secs = 0.0
    confusion = Counter()
    guardrail_actions = Counter()
    trace_rows = []
    trace_fh = open(trace_path, "w") if trace_path else None
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
        if trace_fh:
            nli = result.get("nli_details", {}) or {}
            be = nli.get("best_entailment", {}) or {}
            bc = nli.get("best_contradiction", {}) or {}
            sc = nli.get("sentence_contra_max", {}) or {}
            trace_fh.write(json.dumps({
                "claim": row["claim"],
                "label": row["label"],
                "got": got,
                "guardrail_status": result.get("guardrail_status"),
                "nli_hard": nli.get("nli_verdict"),
                "E": be.get("score") if isinstance(be, dict) else be,
                "C": bc.get("score") if isinstance(bc, dict) else bc,
                "sentC": sc.get("score") if isinstance(sc, dict) else sc,
                "rationale": result.get("rationale", ""),
            }) + "\n")
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

    if trace_fh:
        trace_fh.close()
        print(f"\ntraces -> {trace_path}")


if __name__ == "__main__":
    import sys
    import re

    # Defaults: Held-out FEVER test split (200 claims), k=5, guardrails=True
    eval_path = "evalset_fever_test.jsonl"
    corpus_path = "corpus/fever.txt"
    k = 5
    use_guardrails = True
    mode = "hybrid"
    limit = None
    trace_path = None

    # Dataset split flags
    if "--trial" in sys.argv:
        # Small synthetic trial corpus
        eval_path = "evalset_v2.jsonl"
        corpus_path = "corpus/test.txt"
    elif "--tune" in sys.argv:
        # 300-claim development / tuning split
        eval_path = "evalset_fever_tune.jsonl"
        corpus_path = "corpus/fever.txt"
    elif any(arg in sys.argv for arg in ("--full", "--500", "--fever-500")):
        # Full 500-claim FEVER dataset
        eval_path = "evalset_fever_500.jsonl"
        corpus_path = "corpus/fever.txt"
    elif "--legacy" in sys.argv:
        # Legacy 300-claim dataset
        eval_path = "evalset_fever.jsonl"
        corpus_path = "corpus/fever.txt"
    elif "--fever-test" in sys.argv:
        # Explicit flag for held-out test split (matches default)
        eval_path = "evalset_fever_test.jsonl"
        corpus_path = "corpus/fever.txt"

    # Arbitrary eval file: --eval <path>
    if "--eval" in sys.argv:
        i = sys.argv.index("--eval")
        if i < len(sys.argv) - 1:
            eval_path = sys.argv[i + 1]

    # Per-claim trace logging: --trace <path>
    if "--trace" in sys.argv:
        i = sys.argv.index("--trace")
        if i < len(sys.argv) - 1:
            trace_path = sys.argv[i + 1]

    # Disable guardrails
    if any(arg in sys.argv for arg in ("--no_guard", "--no-guard", "--no_guardrails", "--no-guardrails")):
        use_guardrails = False

    # Retrieval mode overrides
    if "--dense" in sys.argv:
        mode = "dense"
    elif "--bm25" in sys.argv:
        mode = "bm25"

    # Parse k and limit
    for i, arg in enumerate(sys.argv[1:], 1):
        m = re.match(r"^--?k(\d+)$", arg)
        if m:
            k = int(m.group(1))
        elif arg in ("-k", "--k") and i < len(sys.argv) - 1 and sys.argv[i + 1].isdigit():
            k = int(sys.argv[i + 1])
        elif arg == "--limit" and i < len(sys.argv) - 1 and sys.argv[i + 1].isdigit():
            limit = int(sys.argv[i + 1])

    run(
        eval_path=eval_path,
        corpus_path=corpus_path,
        k=k,
        mode=mode,
        use_guardrails=use_guardrails,
        limit=limit,
        trace_path=trace_path,
    )