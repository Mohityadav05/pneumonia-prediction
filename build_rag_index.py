"""
Builds a FAISS index over the docs in rag_docs/ for retrieval-augmented
generation. Run this once (or whenever docs change) before starting main.py.

pip install sentence-transformers faiss-cpu
"""

import os
import glob
import pickle
from sentence_transformers import SentenceTransformer
import faiss

DOCS_DIR = "rag_docs"
INDEX_OUT = "rag_index.faiss"
CHUNKS_OUT = "rag_chunks.pkl"
CHUNK_SIZE = 500     # characters per chunk
CHUNK_OVERLAP = 100

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good enough


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def main():
    all_chunks = []  # list of {"text": ..., "source": ...}
    for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for chunk in chunk_text(text):
            if chunk.strip():
                all_chunks.append({"text": chunk.strip(), "source": os.path.basename(path)})

    print(f"Built {len(all_chunks)} chunks from {DOCS_DIR}/")

    embeddings = embedder.encode([c["text"] for c in all_chunks], show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_OUT)
    with open(CHUNKS_OUT, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved index to {INDEX_OUT} and chunks to {CHUNKS_OUT}")


if __name__ == "__main__":
    main()
