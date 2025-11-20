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
from services.retriever.vector_retriever import VectorRetriever
from services.retriever.hybrid_retriever import HybridRetriever

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="RAG API Gateway", version="0.0.4")
logger = logging.getLogger("uvicorn")

vector_retriever = VectorRetriever()
hybrid_retriever = HybridRetriever()

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
    hybrid: bool = Query(False, description="是否使用 hybrid 检索"),
    vector_k: int = 5,
    bm25_k: int = 5,
):
    """
    支持 hybrid 检索：
    - hybrid=false: 仅向量检索
    - hybrid=true: vector + BM25
    """

    # -----------------------------
    # 仅向量检索模式
    # -----------------------------

    if not hybrid:
        t0 = time.time()
        res = vector_retriever.search(q, top_k)
        t1 = time.time()
        raw_hits = res.get("results", [])

        formatted = []

        for hit in raw_hits:
            # 兼容 meta 中的 text
            text = hit.get("text")
            meta = hit.get("meta") or {}

            if not text and isinstance(meta, dict):
                text = meta.get("text") or meta.get("content")

            item = {
                "doc_id": hit.get("doc_id"),
                "chunk_id": hit.get("chunk_id"),
                "text": text,
                "score_vector": float(hit["score"]) if "score" in hit else None,
                "score_bm25": None,
                "rrf_score": None,
                "sources": ["vector"],
            }

            if "error" in hit:
                item["error"] = hit["error"]
            formatted.append(item)

        latency_ms = {
            "vector": round((t1 - t0) * 1000, 2),
            "total": round((t1 - t0) * 1000, 2),
        }


        return {
            "query": q,
            "hybrid": False,
            "top_k": top_k,
            "latency_ms": latency_ms,
            "results": formatted,
            }

    # -----------------------------
    # Hybrid 模式：vector + BM25 + RRF
    # -----------------------------
    res = hybrid_retriever.search(q, vector_k=vector_k, bm25_k=bm25_k)


    return {
        "query": q,
        "hybrid": True,
        "top_k": top_k,
        "vector_k": vector_k,
        "bm25_k": bm25_k,
        "latency_ms": res.get("latency_ms", {}),
        "results": res.get("fused_results", []),
        }