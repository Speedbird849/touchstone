from pathlib import Path

def load_passages(corpus_path="corpus"):
    path_obj = Path(corpus_path)
    files = [path_obj] if path_obj.is_file() else sorted(path_obj.glob("*.txt"))
    passages = []
    for path in files:
        for i, chunk in enumerate(path.read_text().split("\n\n")):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.startswith("[") and "]" in chunk:
                p_id, text = chunk[1:].split("]", 1)
                passages.append({"id": p_id.strip(), "text": text.strip()})
            else:
                passages.append({"id": f"{path.stem}#{i}", "text": chunk})
    return passages

if __name__ == "__main__":
    for p in load_passages():
        print(p["id"], "->", p["text"][:100])