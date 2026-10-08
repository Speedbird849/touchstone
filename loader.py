from pathlib import Path

def load_passages(corpus_dir="corpus"):
    passages = []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        for i, chunk in enumerate(path.read_text().split("\n\n")):
            if chunk.strip():
                passages.append({"id": f"{path.stem}#{i}", "text": chunk.strip()})
    return passages

if __name__ == "__main__":
    for p in load_passages():
        print(p["id"], "->", p["text"][:100])