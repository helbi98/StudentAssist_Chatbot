import os, re, json
from tqdm import tqdm
import nltk

nltk.download('punkt', quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

INPUT_DIR = "scraped_pages"
OUT_FILE = "chunks.jsonl"

def read_pages(input_dir):
    docs = []
    for fname in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().strip()
        if not data:
            continue
        url, *rest = data.split("\n\n", 1)
        text = rest[0] if rest else ""
        docs.append({"id": fname, "url": url, "text": text})
    return docs

def chunk_doc(text, max_sentences=8):
    sents = sent_tokenize(text)
    chunks, cur = [], []
    for s in sents:
        cur.append(s)
        if len(cur) >= max_sentences:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def main():
    docs = read_pages(INPUT_DIR)
    out = []
    for doc in docs:
        chunks = chunk_doc(doc["text"], max_sentences=6)
        for i, c in enumerate(chunks):
            out.append({
                "id": f"{doc['id']}_chunk_{i}",
                "source": doc["url"],
                "text": c
            })
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Wrote", len(out), "chunks to", OUT_FILE)

if __name__ == "__main__":
    main()
