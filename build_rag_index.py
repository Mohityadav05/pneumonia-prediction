"""
Builds a FAISS index over the docs in rag_docs/ for retrieval-augmented
generation. Run this once (or whenever docs change) before starting main.py.

pip install fastembed faiss-cpu

NOTE: this uses fastembed (ONNX-based) instead of sentence-transformers
(torch-based) so the embedding model stays small enough to run comfortably
on a free-tier host like Render's 512MB instance. main.py MUST use the same
embedder/model for queries, since the index is built in that model's vector
space -- if you change EMBED_MODEL here, re-run this script AND update
main.py's EMBED_MODEL to match, or search results will be meaningless.
"""

import os
import glob
import pickle
from fastembed import TextEmbedding
import faiss
import numpy as np

DOCS_DIR = "rag_docs"
INDEX_OUT = "rag_index.faiss"
CHUNKS_OUT = "rag_chunks.pkl"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100

# small, fast, ONNX-based (no torch) -- 384-dim, same dimensionality as the
# old all-MiniLM-L6-v2 model, but this one is loaded via onnxruntime so it
# doesn't drag in torch's much larger memory footprint.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

embedder = TextEmbedding(model_name=EMBED_MODEL)


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

    # fastembed's .embed() returns a generator of numpy arrays, one per input
    embeddings = list(embedder.embed([c["text"] for c in all_chunks]))
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_OUT)
    with open(CHUNKS_OUT, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved index to {INDEX_OUT} and chunks to {CHUNKS_OUT}")


if __name__ == "__main__":
    main()