from pathlib import Path
import re

_SID_TAIL = re.compile(r"#\d+$")


def _split_sids(sids_part):
    """Split a serialized sentence-id list.

    Canonical separator is ';'. Page titles may contain commas
    (e.g. ``Syracuse,_New_York#2``), so a legacy comma-separated list is
    reassembled: comma-tokens are accumulated until a ``#<digits>`` tail
    completes one sentence id.
    """
    sids_part = sids_part.replace("sids:", "")
    if ";" in sids_part:
        return [s.strip() for s in sids_part.split(";") if s.strip()]
    sids, buf = [], ""
    for tok in sids_part.split(","):
        tok = tok.strip()
        if not tok:
            continue
        buf = f"{buf},{tok}" if buf else tok
        if _SID_TAIL.search(buf):
            sids.append(buf)
            buf = ""
    if buf:
        sids.append(buf)
    return sids

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
                header, text = chunk[1:].split("]", 1)
                if "|" in header:
                    p_id, sids_part = header.split("|", 1)
                    p_id = p_id.strip()
                    sids = _split_sids(sids_part)
                else:
                    p_id = header.strip()
                    sids = [p_id]
                passages.append({"id": p_id, "sentence_ids": sids, "text": text.strip()})
            else:
                p_id = f"{path.stem}#{i}"
                passages.append({"id": p_id, "sentence_ids": [p_id], "text": chunk})
    return passages

if __name__ == "__main__":
    for p in load_passages():
        print(p["id"], "->", p["text"][:100])