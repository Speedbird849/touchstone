"""
prepare_fever.py: Downloads a validated subset of FEVER from HuggingFace
and formats it into coherent multi-sentence paragraph chunks.
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

def prepare(num_per_class=100, max_sentences_per_chunk=5, out_dir="corpus"):
    print("Downloading/loading FEVER validation set...")
    path = hf_hub_download(
        repo_id="copenlu/fever_gold_evidence",
        filename="valid.jsonl",
        repo_type="dataset",
    )

    page_sentences = defaultdict(dict)  # page -> {line_num: (sent_id, text)}
    claims_by_label = defaultdict(list)

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            label = LABEL_MAP.get(row.get("label"))
            if not label:
                continue

            claim_text = row["claim"].strip()
            evidence_list = row.get("evidence", [])

            # Collect gold sentence IDs for this claim
            gold_ids = []
            for ev in evidence_list:
                if len(ev) >= 3 and ev[2].strip():
                    page, line_num, text = ev[0], ev[1], ev[2].strip()
                    p_id = f"{page}#{line_num}"
                    try:
                        l_int = int(line_num)
                    except ValueError:
                        l_int = line_num
                    page_sentences[page][l_int] = (p_id, text)
                    if label != "NOT_ENOUGH_INFO":
                        gold_ids.append(p_id)

            claims_by_label[label].append({
                "claim": claim_text,
                "label": label,
                "gold_ids": list(set(gold_ids)),
            })

    # Sample balanced claims (100 per class = 300 total)
    selected_claims = []
    for label in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]:
        subset = claims_by_label[label][:num_per_class]
        selected_claims.extend(subset)
        print(f"Sampled {len(subset)} {label} claims")

    # Group sentences per Wikipedia page into paragraph chunks
    chunks = []
    for page, sents in page_sentences.items():
        sorted_keys = sorted(sents.keys(), key=lambda x: int(x) if isinstance(x, int) else 0)
        for i in range(0, len(sorted_keys), max_sentences_per_chunk):
            batch_keys = sorted_keys[i:i + max_sentences_per_chunk]
            p_ids = [sents[k][0] for k in batch_keys]
            texts = [sents[k][1] for k in batch_keys]
            chunk_id = f"{page}#p{i // max_sentences_per_chunk}"
            chunk_text = " ".join(texts)
            chunks.append({
                "id": chunk_id,
                "sentence_ids": p_ids,
                "text": chunk_text
            })

    # Save chunks to corpus/fever.txt
    corpus_file = Path(out_dir) / "fever.txt"
    corpus_file.parent.mkdir(exist_ok=True)
    with open(corpus_file, "w") as f:
        for ch in chunks:
            sids_str = ",".join(ch["sentence_ids"])
            f.write(f"[{ch['id']} | sids: {sids_str}] {ch['text']}\n\n")

    # Save claims to evalset_fever.jsonl
    eval_file = Path("evalset_fever.jsonl")
    with open(eval_file, "w") as f:
        for c in selected_claims:
            f.write(json.dumps(c) + "\n")

    print(f"Successfully generated:")
    print(f"  - {len(chunks)} paragraph chunks to {corpus_file}")
    print(f"  - {len(selected_claims)} claims to {eval_file}")

if __name__ == "__main__":
    prepare()
