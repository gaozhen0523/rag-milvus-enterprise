# services/embedding_worker/worker.py
import os
import time
from datetime import datetime
from typing import Any

from libs.chunking.text_chunker import TextChunker
from libs.db.milvus_client import MilvusClientFactory
from libs.embedding.factory import get_embedding_model

# 单次批量插入的最大 chunk 数量，可通过环境变量覆盖
DEFAULT_BATCH_SIZE = 2000
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", DEFAULT_BATCH_SIZE))


def process_document(
    doc_id: str,
    text: str,
    chunk_params,
    metadata: dict[str, Any] | None = None,
) -> int:
    """
    文本 → chunk → embedding（分批）→ Milvus insert（分批）

    Day 13 改造点：
    - 支持大文件：按 BATCH_SIZE 分批 embedding + insert
    - 每批输出结构化日志
    - 返回插入成功的 chunk 总数
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
    num_chunks = len(chunks)
    if num_chunks == 0:
        print(
            f"[INGEST] {datetime.utcnow().isoformat()} doc_id={doc_id} no chunks, skip."
        )
        return 0

    print(
        f"[INGEST] {datetime.utcnow().isoformat()} "
        f"doc_id={doc_id} total_chunks={num_chunks} "
        f"strategy={chunk_params.strategy} size={chunk_params.size} "
        f"overlap={chunk_params.overlap}"
    )

    # -----------------------------
    # 2) Embedding Model / Milvus 准备
    # -----------------------------
    model = get_embedding_model()

    # 优先以模型维度为准，避免 env 与模型不一致
    model_dim = getattr(model, "dim", None)
    if model_dim is None:
        model_dim = int(os.getenv("EMBEDDING_DIM", 768))

    factory = MilvusClientFactory()
    collection = factory.get_or_create_collection(
        name="rag_collection",
        dim=model_dim,
    )
    # 搜索阶段会再 load + 建索引，这里只确保 collection 存在即可
    collection.load()

    # 用一个固定的 received_at，保证同一批次一致
    received_at = datetime.utcnow().isoformat()
    total_inserted = 0

    # 计算批次数方便日志
    batch_size = max(1, BATCH_SIZE)
    total_batches = (num_chunks + batch_size - 1) // batch_size

    # -----------------------------
    # 3) 分批 embedding + insert
    # -----------------------------
    for batch_idx, start in enumerate(range(0, num_chunks, batch_size), start=1):
        end = min(start + batch_size, num_chunks)
        batch_chunks = chunks[start:end]

        # ---- 3.1 准备文本 ----
        batch_texts = [c.text if hasattr(c, "text") else str(c) for c in batch_chunks]
        batch_count = len(batch_texts)

        # ---- 3.2 批量 embedding ----
        t_embed_start = time.time()
        batch_vectors = model.embed_batch(batch_texts)
        t_embed_end = time.time()

        # ---- 3.3 构建 doc_id / chunk_id / meta ----
        batch_doc_ids = [doc_id] * batch_count
        # chunk_id 使用全局连续编号，方便 debug
        batch_chunk_ids = list(range(start, end))
        batch_metas = [
            {
                "source": "api_ingest",
                "received_at": received_at,
                "text": batch_texts[i],
                "user_meta": metadata or {},
            }
            for i in range(batch_count)
        ]

        data = [batch_vectors, batch_doc_ids, batch_chunk_ids, batch_metas]

        # ---- 3.4 插入 Milvus ----
        t_insert_start = time.time()
        result = collection.insert(data)
        t_insert_end = time.time()

        # 有些版本的 result 可能不带 primary_keys，兜底以 batch_count 计数
        batch_inserted = (
            len(getattr(result, "primary_keys", []))
            if result is not None and hasattr(result, "primary_keys")
            else batch_count
        )
        total_inserted += batch_inserted

        embed_ms = (t_embed_end - t_embed_start) * 1000.0
        insert_ms = (t_insert_end - t_insert_start) * 1000.0

        print(
            f"[INGEST] {datetime.utcnow().isoformat()} "
            f"doc_id={doc_id} batch={batch_idx}/{total_batches} "
            f"batch_size={batch_count} embed_ms={embed_ms:.2f} "
            f"insert_ms={insert_ms:.2f} "
            f"cumulative_inserted={total_inserted}"
        )

    # -----------------------------
    # 4) Flush 一次，确保持久化
    # -----------------------------
    flush_start = time.time()
    collection.flush()
    flush_end = time.time()
    print(
        f"[INGEST] {datetime.utcnow().isoformat()} "
        f"doc_id={doc_id} flush_ms={(flush_end - flush_start) * 1000.0:.2f} "
        f"total_inserted={total_inserted}"
    )

    return total_inserted


def process_document_incremental(
    doc_id: str,
    chunks,
    metadata: dict[str, Any] | None = None,
) -> int:
    """
    Day 25 新增：
    专用于 chunk 去重后的增量插入场景。
    与 process_document 不同：
      - 不负责 chunk（外层已 chunk）
      - 不负责全量分页，只插入传入的 dedup_chunks
    """
    num_chunks = len(chunks)
    if num_chunks == 0:
        print(
            f"[INGEST-INCR] {datetime.utcnow().isoformat()} "
            f"doc_id={doc_id} no new chunks, skip."
        )
        return 0

    # -----------------------------
    # Embedding Model / Milvus 准备
    # -----------------------------
    model = get_embedding_model()

    model_dim = getattr(model, "dim", None)
    if model_dim is None:
        model_dim = int(os.getenv("EMBEDDING_DIM", 768))

    factory = MilvusClientFactory()
    collection = factory.get_or_create_collection(
        name="rag_collection",
        dim=model_dim,
    )
    collection.load()

    received_at = datetime.utcnow().isoformat()

    # -----------------------------
    # 构建 embedding 输入
    # -----------------------------
    batch_texts = [c.text for c in chunks]
    batch_vectors = model.embed_batch(batch_texts)

    batch_doc_ids = [doc_id] * num_chunks
    # 使用 chunk 自带 chunk_id，避免编号错乱
    batch_chunk_ids = [c.chunk_id for c in chunks]
    batch_metas = [
        {
            "source": "api_ingest_incremental",
            "received_at": received_at,
            "text": c.text,
            "user_meta": metadata or {},
        }
        for c in chunks
    ]

    data = [batch_vectors, batch_doc_ids, batch_chunk_ids, batch_metas]

    # -----------------------------
    # 插入 Milvus
    # -----------------------------
    t_insert_start = time.time()
    result = collection.insert(data)
    t_insert_end = time.time()

    batch_inserted = (
        len(getattr(result, "primary_keys", []))
        if result is not None and hasattr(result, "primary_keys")
        else num_chunks
    )

    print(
        f"[INGEST-INCR] {datetime.utcnow().isoformat()} "
        f"doc_id={doc_id} new_chunks={num_chunks} "
        f"insert_ms={(t_insert_end - t_insert_start) * 1000.0:.2f} "
        f"inserted={batch_inserted}"
    )

    # Flush（小批次也 flush，保证可查询）
    collection.flush()

    return batch_inserted
