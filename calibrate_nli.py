"""
calibrate_nli.py: NLI-only threshold calibration on the tune split (no LLM calls).

For each tune claim, scores the canonical evidence the guardrail would see:
  - SUPPORTED / REFUTED: gold passages (+ joint concat, mirroring arbitrate)
  - NOT_ENOUGH_INFO: top-3 hybrid-retrieved passages (what a hallucinated
    SUPPORTED would be checked against)

Then sweeps (entail_floor, contra_cap) for the soft gate:
  preserve SUPPORTED iff focused_E >= floor and focused_C <= cap.
Reports rescue rate on SUPPORTED vs false-keep rate on NEI, plus a
sentence-window on/off ablation.
"""
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import itertools
import json

from loader import load_passages
from retriever import Retriever
from guardrails import GuardrailEngine


def candidates_for(row, passages_by_sid, retriever):
    if row["label"] in ("SUPPORTED", "REFUTED") and row["gold_ids"]:
        cands = []
        for gid in row["gold_ids"]:
            p = passages_by_sid.get(gid)
            if p and all(p["id"] != c["id"] for c in cands):
                cands.append(p)
        cands = cands or retriever.search(row["claim"], k=3)
    else:
        cands = retriever.search(row["claim"], k=3)
    out = list(cands)
    if len(cands) > 1:
        out = out + [{
            "id": "+".join(p["id"] for p in cands),
            "text": " ".join(p["text"] for p in cands),
            "sentence_ids": [s for p in cands for s in p.get("sentence_ids", [])],
        }]
    return out


def collect(engine, rows, passages_by_sid, retriever):
    from tqdm import tqdm
    signals = []
    for row in tqdm(rows, unit="claim", desc=f"window={engine.use_sentence_window}"):
        cands = candidates_for(row, passages_by_sid, retriever)
        nli = engine.verify_nli(row["claim"], cands)
        signals.append({
            "claim": row["claim"],
            "label": row["label"],
            "hard": nli["nli_verdict"],
            "E": nli["best_entailment"]["score"],
            "C": nli["best_contradiction"]["score"],
        })
    return signals


def main(eval_path="evalset_fever_tune.jsonl", out_path="tune_nli_signals.json"):
    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    passages = load_passages("corpus/fever.txt")
    by_sid = {}
    for p in passages:
        by_sid.setdefault(p["id"], p)
        for sid in p.get("sentence_ids", []):
            by_sid.setdefault(sid, p)
    retriever = Retriever(passages, mode="hybrid")

    results = {}
    for window in (False, True):
        engine = GuardrailEngine(use_sentence_window=window)
        sigs = collect(engine, rows, by_sid, retriever)
        results[str(window)] = sigs
        sup = [s for s in sigs if s["label"] == "SUPPORTED"]
        ref = [s for s in sigs if s["label"] == "REFUTED"]
        nei = [s for s in sigs if s["label"] == "NOT_ENOUGH_INFO"]
        hard_sup = sum(s["hard"] == "SUPPORTED" for s in sup) / max(len(sup), 1)
        hard_nei_ok = sum(s["hard"] == "NOT_ENOUGH_INFO" for s in nei) / max(len(nei), 1)
        print(f"\n[window={window}] hard SUP recall: {hard_sup:.1%} | "
              f"hard NEI-neutral: {hard_nei_ok:.1%} | "
              f"REF hard: {sum(s['hard'] == 'REFUTED' for s in ref) / max(len(ref), 1):.1%}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsignals -> {out_path}")

    # Sweep soft-gate thresholds on windowed signals
    sigs = results["True"]
    sup = [s for s in sigs if s["label"] == "SUPPORTED" and s["hard"] == "NOT_ENOUGH_INFO"]
    nei = [s for s in sigs if s["label"] == "NOT_ENOUGH_INFO"]
    print(f"\nSUPPORTED with hard-NEI (rescuable): {len(sup)} | NEI total: {len(nei)}")
    print(f"{'floor':>6} {'cap':>5} {'sup_rescue':>10} {'nei_keep':>9}  margin")
    best = None
    for floor, cap in itertools.product(
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            [0.20, 0.30, 0.40, 0.50, 0.60]):
        rescue = sum(s["E"] >= floor and s["C"] <= cap for s in sup) / max(len(sup), 1)
        keep = sum(s["E"] >= floor and s["C"] <= cap for s in nei) / max(len(nei), 1)
        margin = rescue - keep
        flag = ""
        if best is None or margin > best[0]:
            best = (margin, floor, cap, rescue, keep)
            flag = "  <-- best"
        print(f"{floor:>6.2f} {cap:>5.2f} {rescue:>10.1%} {keep:>9.1%}  {margin:+.2f}{flag}")
    print(f"\nrecommended: entail_floor={best[1]:.2f} contra_cap={best[2]:.2f} "
          f"(rescue {best[3]:.0%}, false-keep {best[4]:.0%})")


if __name__ == "__main__":
    main()
