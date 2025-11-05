from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Literal, Dict, Any
import logging
import uuid
from datetime import datetime, timezone

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
    if dry_run and payload.text:
        try:
            from libs.chunking.text_chunker import TextChunker
            chunker = TextChunker(
                strategy=payload.chunk.strategy,
                size=payload.chunk.size,
                overlap=payload.chunk.overlap,
            )
            chunks = chunker.chunk(payload.text, meta=payload.metadata)
            ack.preview_chunks = len(chunks)
        except Exception as e:
            logger.exception("dry_run chunk failed: %s", e)

    return ack