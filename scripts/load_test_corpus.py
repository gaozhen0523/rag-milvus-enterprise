# scripts/load_test_corpus.py
from __future__ import annotations

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from libs.chunking.text_chunker import TextChunker
from libs.embedding.dummy import DummyEmbeddingModel
from libs.db.milvus_client import MilvusClientFactory


def load_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Load test corpus into Milvus")
    parser.add_argument(
        "--file",
        type=str,
        default="sample.txt",
        help="Path to text file (UTF-8)",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default="test_doc_1",
        help="Logical doc_id stored in Milvus",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    text = load_text_file(file_path)

    # 1) chunk
    chunker = TextChunker(strategy="sentence", size=500, overlap=50)
    chunks = chunker.chunk(text)
    if not chunks:
        print("⚠️ No chunks produced, abort.")
        return

    print(f"📄 Loaded text from {file_path}, chunks = {len(chunks)}")

    # 2) embedding
    dim = int(os.getenv("EMBEDDING_DIM", 768))
    model = DummyEmbeddingModel(dim=dim, normalize=True)

    chunk_texts: List[str] = [c.text for c in chunks]
    vectors = model.embed_batch(chunk_texts)
    vectors = np.asarray(vectors, dtype="float32")

    print(f"🧠 Embedded {len(chunks)} chunks, dim={dim}")

    # 3) insert into Milvus
    factory = MilvusClientFactory()
    collection = factory.get_or_create_collection(dim=dim)
    collection.load()

    doc_ids = [args.doc_id] * len(chunks)
    chunk_ids = list(range(len(chunks)))
    metas: List[Dict[str, Any]] = [
        {
            "source": "test_corpus",
            "received_at": datetime.utcnow().isoformat(),
            "text": chunk_texts[i],
            "user_meta": {
                "file": str(file_path),
                "chunk_id": chunk_ids[i],
            },
        }
        for i in range(len(chunks))
    ]

    data = [vectors, doc_ids, chunk_ids, metas]
    result = collection.insert(data)
    collection.flush()

    print(
        f"✅ Inserted {len(result.primary_keys)} rows into "
        f"'{collection.name}'. Total entities = {collection.num_entities}"
    )


if __name__ == "__main__":
    main()
