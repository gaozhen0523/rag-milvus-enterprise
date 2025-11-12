#services/embedding_worker/worker.py
import os
from typing import Optional, Dict, Any
from datetime import datetime

from libs.chunking.text_chunker import TextChunker
from libs.embedding.dummy import DummyEmbeddingModel
from libs.db.milvus_client import MilvusClientFactory
from libs.embedding.factory import get_embedding_model


def process_document(
    doc_id: str,
    text: str,
    chunk_params,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Day 6 核心：文本 → chunk → embedding → Milvus insert.

    Returns:
        int: 插入成功的 chunk 数量
    """
    # -----------------------------
    # 1) Chunk
    # -----------------------------
    chunker = TextChunker(
        strategy=chunk_params.strategy,
        size=chunk_params.size,
        overlap=chunk_params.overlap,
    )
    chunks = chunker.chunk(text)
    if len(chunks) == 0:
        return 0

    # -----------------------------
    # 2) Embedding
    # -----------------------------
    dim = int(os.getenv("EMBEDDING_DIM", 768))  # 与 rag_collection 一致
    model = get_embedding_model()

    chunk_texts = [c.text if hasattr(c, "text") else str(c) for c in chunks]
    vectors = model.embed_batch(chunk_texts)

    # -----------------------------
    # 3) Milvus insert
    # -----------------------------
    factory = MilvusClientFactory()
    collection = factory.get_or_create_collection(name="rag_collection", dim=dim)
    collection.load()

    doc_ids = [doc_id] * len(chunks)
    chunk_ids = list(range(len(chunks)))
    metas = [
        {
            "source": "api_ingest",
            "received_at": datetime.utcnow().isoformat(),
            "user_meta": metadata or {},
        }
        for _ in chunks
    ]

    # 列模式插入：必须与 schema 对齐
    data = [vectors, doc_ids, chunk_ids, metas]

    result = collection.insert(data)
    collection.flush()

    inserted_count = len(result.primary_keys)
    return inserted_count