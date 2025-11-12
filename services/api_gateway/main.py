#services/api_gateway/main.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Literal, Dict, Any
import logging
import uuid
from datetime import datetime, timezone
import os, time
import numpy as np

from libs.db.milvus_client import MilvusClientFactory

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="RAG API Gateway", version="0.0.4")
logger = logging.getLogger("uvicorn")

# -----------------------------------------------------------------------------
# Health Check （原逻辑保留）
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    factory = MilvusClientFactory()
    return factory.health_status()

# -----------------------------------------------------------------------------
# Ingest 契约定义
# -----------------------------------------------------------------------------
class ChunkParams(BaseModel):
    strategy: Literal["char", "sentence"] = "sentence"
    size: int = Field(800, ge=1)
    overlap: int = Field(100, ge=0)

    def validate_logic(self) -> None:
        if self.overlap >= self.size:
            raise ValueError("overlap must be < size")

class IngestRequest(BaseModel):
    text: Optional[str] = Field(None, description="原始文本（可选；与 file_url 二选一）")
    file_url: Optional[HttpUrl] = Field(None, description="文件地址（可选；与 text 二选一）")
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    source_id: Optional[str] = None
    chunk: ChunkParams = Field(default_factory=ChunkParams)

    def ensure_payload(self) -> None:
        if not self.text and not self.file_url:
            raise ValueError("Either 'text' or 'file_url' must be provided.")
        self.chunk.validate_logic()

class IngestAck(BaseModel):
    task_id: str
    accepted_at: str
    payload_kind: Literal["text", "file_url"]
    chunk_params: ChunkParams
    preview_chunks: Optional[int] = None
    note: Optional[str] = None

# -----------------------------------------------------------------------------
# Ingest 接口
# -----------------------------------------------------------------------------
@app.post("/ingest", response_model=IngestAck)
def ingest(payload: IngestRequest, dry_run: bool = Query(True, description="仅校验/预览，不入库/不入队")):
    try:
        payload.ensure_payload()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    task_id = str(uuid.uuid4())
    kind: Literal["text", "file_url"] = "text" if payload.text else "file_url"

    # 打日志方便追踪
    logger.info(
        "INGEST_ACCEPTED task_id=%s kind=%s chunk={strategy:%s,size:%d,overlap:%d} source_id=%s",
        task_id,
        kind,
        payload.chunk.strategy,
        payload.chunk.size,
        payload.chunk.overlap,
        payload.source_id,
    )

    ack = IngestAck(
        task_id=task_id,
        accepted_at=datetime.now(tz=timezone.utc).isoformat(),
        payload_kind=kind,
        chunk_params=payload.chunk,
        note="Accepted and validated. Queueing will be implemented on Day 6.",
    )

    # dry_run 模式：直接调用 chunker 预估分片数
    if dry_run:
        if payload.text:
            try:
                from libs.chunking.text_chunker import TextChunker
                chunker = TextChunker(
                    strategy=payload.chunk.strategy,
                    size=payload.chunk.size,
                    overlap=payload.chunk.overlap,
                )
                chunks = chunker.chunk(payload.text, meta=payload.metadata)
                ack.preview_chunks = len(chunks)
                ack.note = "Dry run only. No Milvus insert."
            except Exception as e:
                logger.exception("dry_run chunk failed: %s", e)

        return ack
    # ---------------------------------------------------------------------
    # dry_run == False → 真正的 ingest 流程
    # ---------------------------------------------------------------------
    # 1) 获取文本（支持 file_url）
    if payload.text:
        text = payload.text
    else:
        import requests
        try:
            r = requests.get(payload.file_url, timeout=10)
            r.raise_for_status()
            text = r.text
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to download file_url: {e}")

    # 2) 调用 Worker 执行 chunk → embed → milvus insert
    try:
        from services.embedding_worker.worker import process_document
        inserted = process_document(
            doc_id=task_id,
            text=text,
            chunk_params=payload.chunk,
            metadata=payload.metadata,
        )
    except Exception as e:
        logger.exception("Ingest processing failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")

    ack.preview_chunks = inserted
    ack.note = f"Inserted {inserted} chunks into Milvus."
    return ack

# -----------------------------------------------------------------------------
# Query 接口：向量检索（统一 DummyEmbeddingModel）
# -----------------------------------------------------------------------------
@app.get("/query")
def query_endpoint(
    q: str = Query(..., description="查询文本"),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    执行向量检索：
    - 从 EMBEDDING_MODEL 环境变量读取模型类型（默认 dummy）
    - 使用 DummyEmbeddingModel 生成向量（与 worker 一致）
    - 调用 Milvus 搜索
    """
    from libs.embedding.factory import get_embedding_model
    model_name = os.getenv("EMBEDDING_MODEL", "dummy").lower()
    dim = int(os.getenv("EMBEDDING_DIM", 768))

    start_time = time.time()


    model = get_embedding_model()
    vec = model.embed_one(q)

    factory = MilvusClientFactory()
    results = factory.search_vectors(np.array(vec, dtype="float32"), top_k=top_k)

    latency_ms = round((time.time() - start_time) * 1000, 2)
    return {
        "query": q,
        "model": model_name,
        "embed_dim": dim,
        "latency_ms": latency_ms,
        "results": results,
        "note": f"Model={model_name}; replaceable with real model later",
    }