"""
prepare_fever.py: Downloads a validated subset of FEVER from HuggingFace
and formats it directly into our touchstone schema.
"""

import json
from collections import defaultdict
from pathlib import Path
from huggingface_hub import hf_hub_download

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
}

def prepare(num_per_class=100, num_distractor_passages=2500, out_dir="corpus"):
    print("Downloading/loading FEVER validation set...")
    path = hf_hub_download(
        repo_id="copenlu/fever_gold_evidence",
        filename="valid.jsonl",
        repo_type="dataset",
    )

    claims_by_label = defaultdict(list)
    passages = {}  # passage_id -> passage_text

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            label = LABEL_MAP.get(row.get("label"))
            if not label:
                continue

            claim_text = row["claim"].strip()
            evidence_list = row.get("evidence", [])

            # Collect gold passage IDs for this claim
            gold_ids = []
            for ev in evidence_list:
                if len(ev) >= 3 and ev[2].strip():
                    page, line_num, text = ev[0], ev[1], ev[2].strip()
                    p_id = f"{page}#{line_num}"
                    passages[p_id] = text
                    if label != "NOT_ENOUGH_INFO":
                        gold_ids.append(p_id)

            claims_by_label[label].append({
                "claim": claim_text,
                "label": label,
                "gold_ids": list(set(gold_ids)),
            })

    # Sample balanced claims
    selected_claims = []
    for label in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]:
        subset = claims_by_label[label][:num_per_class]
        selected_claims.extend(subset)
        print(f"Sampled {len(subset)} {label} claims")

    # Add extra distractor passages from other unused rows
    print(f"Initial gold & related passages: {len(passages)}")
    with open(path) as f:
        for line in f:
            if len(passages) >= (len(selected_claims) + num_distractor_passages):
                break
            row = json.loads(line)
            for ev in row.get("evidence", []):
                if len(ev) >= 3 and ev[2].strip():
                    p_id = f"{ev[0]}#{ev[1]}"
                    if p_id not in passages:
                        passages[p_id] = ev[2].strip()

    # Save passages to corpus/fever.txt
    corpus_file = Path(out_dir) / "fever.txt"
    corpus_file.parent.mkdir(exist_ok=True)
    with open(corpus_file, "w") as f:
        # Write each passage separated by blank lines (compatible with loader.py)
        # We format as "[id]\ntext" or let loader index by stem
        # To maintain exact passage IDs, write standard blocks
        for p_id, text in passages.items():
            f.write(f"[{p_id}] {text}\n\n")

    # Save claims to evalset_fever.jsonl
    eval_file = Path("evalset_fever.jsonl")
    with open(eval_file, "w") as f:
        for c in selected_claims:
            f.write(json.dumps(c) + "\n")

    print(f"Successfully written:")
    print(f"  - {len(passages)} passages to {corpus_file}")
    print(f"  - {len(selected_claims)} claims to {eval_file}")

if __name__ == "__main__":
    prepare()

