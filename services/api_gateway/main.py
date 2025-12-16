# services/api_gateway/main.py
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request

# --- OpenTelemetry Tracing ---
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pydantic import BaseModel, Field, HttpUrl

from libs.caching.query_cache import query_cache
from libs.db.milvus_client import MilvusClientFactory
from libs.logging.query_logger import query_logger
from services.retriever.bm25_retriever import BM25Retriever
from services.retriever.hybrid_retriever import HybridRetriever
from services.retriever.vector_retriever import VectorRetriever

load_dotenv(override=False)
# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
# --- OpenTelemetry Tracer Init ---
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
otel_tracer = trace.get_tracer("rag-api-gateway")
app = FastAPI(title="RAG API Gateway", version="0.0.4")

# ---------------------------------------------------------------------
# API Key Authentication (Day 25)
# ---------------------------------------------------------------------
API_GATEWAY_TOKEN = os.getenv("API_GATEWAY_TOKEN")


def require_api_key(request: Request):
    if not API_GATEWAY_TOKEN:
        return  # 如果未配置 token，则默认关闭鉴权（方便本地）
    key = request.headers.get("X-API-Key")
    if key != API_GATEWAY_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API Key")


@app.middleware("http")
async def inject_trace_id(request, call_next):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


FastAPIInstrumentor.instrument_app(app)
logger = logging.getLogger("uvicorn")

vector_retriever = VectorRetriever()
hybrid_retriever = HybridRetriever()
bm25_retriever = BM25Retriever()


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
    text: str | None = Field(None, description="原始文本（可选；与 file_url 二选一）")
    file_url: HttpUrl | None = Field(
        None, description="文件地址（可选；与 text 二选一）"
    )
    metadata: dict[str, Any] | None = Field(default=None)
    source_id: str | None = None
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
    preview_chunks: int | None = None
    note: str | None = None


# -----------------------------------------------------------------------------
# Ingest 接口
# -----------------------------------------------------------------------------
@app.post("/ingest", response_model=IngestAck)
def ingest(
    payload: IngestRequest,
    request: Request,
    api_ok: None = Depends(require_api_key),
    dry_run: bool = Query(True, description="仅校验/预览，不入库/不入队"),
):
    try:
        payload.ensure_payload()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    task_id = str(uuid.uuid4())
    kind: Literal["text", "file_url"] = "text" if payload.text else "file_url"

    # 打日志方便追踪
    logger.info(
        "INGEST_ACCEPTED task_id=%s kind=%s "
        "chunk={strategy:%s,size:%d,overlap:%d} source_id=%s",
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
            raise HTTPException(
                status_code=502, detail=f"Failed to download file_url: {e}"
            ) from e

    # 2) 调用 Worker 执行 chunk → embed → milvus insert
    try:
        from libs.chunking.text_chunker import TextChunker

        chunker = TextChunker(
            strategy=payload.chunk.strategy,
            size=payload.chunk.size,
            overlap=payload.chunk.overlap,
        )
        chunks = chunker.chunk(text, meta=payload.metadata)

        import hashlib

        def h(s: str) -> str:
            return hashlib.md5(s.encode("utf-8")).hexdigest()

        cached_flags = []
        dedup_chunks = []
        for c in chunks:
            ck = f"chunk:{h(c.text)}"
            if query_cache.get(ck):
                cached_flags.append(True)
            else:
                cached_flags.append(False)
                dedup_chunks.append(c)
                query_cache.set(ck, True, ttl=24 * 3600)  # 24h 避免重复写入

        # 3) 调用 Worker 处理去重后的 chunks
        from services.embedding_worker.worker import process_document_incremental

        inserted = process_document_incremental(
            doc_id=task_id,
            chunks=dedup_chunks,
            metadata=payload.metadata,
        )
    except Exception as e:
        logger.exception("Ingest processing failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e

    ack.preview_chunks = inserted
    ack.note = f"Inserted {inserted} chunks into Milvus."
    return ack


# -----------------------------------------------------------------------------
# Query 接口：向量检索（统一 DummyEmbeddingModel）
# -----------------------------------------------------------------------------
@app.get("/query")
def query_endpoint(
    request: Request,
    api_ok: None = Depends(require_api_key),
    q: str = Query(..., description="查询文本"),
    top_k: int = Query(5, ge=1, le=20),
    hybrid: bool = Query(False, description="是否使用 hybrid 检索"),
    vector_k: int = 5,
    bm25_k: int = 5,
    rerank: bool = Query(False, description="是否启用 rerank（仅 hybrid 模式）"),
    page: int = Query(1, ge=1, le=1_000_000),
    page_size: int = Query(10, ge=1, le=50),
    debug: bool = Query(False, description="是否返回调试信息（仅 hybrid 模式生效）"),
):
    """
    支持 hybrid 检索：
    - hybrid=false: 仅向量检索（Milvus）
    - hybrid=true: vector + BM25 (+ RRF + 可选 Rerank)
    Day23: 增加 Milvus/Redis 故障降级能力：
      - Milvus 故障 -> 自动降级为 BM25-only
      - Redis 缓存不可用 -> 自动回退内存缓存，标记 redis_ok=False
    """

    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    t_start = time.time()

    # -----------------------------------------------------
    # 降级/健康状态标记
    # -----------------------------------------------------
    milvus_ok = True
    degraded = False
    degraded_mode: str | None = None
    degraded_reason: str | None = None

    # Redis 状态
    try:
        redis_ok = query_cache.is_redis_available()
    except Exception:
        redis_ok = True  # 没这个方法就当作 True

    # -----------------------------------------------------
    # 缓存处理
    #   - debug=True 时：完全绕过缓存（不读不写）
    #   - 降级结果不写入缓存（避免缓存住故障状态）
    # -----------------------------------------------------
    cache_key: str | None = None
    cached: dict[str, Any] | None = None

    if not debug and redis_ok:
        cache_key = query_cache.make_key(
            q,
            hybrid,
            top_k,
            vector_k,
            bm25_k,
            page,
            page_size,
            rerank,
        )
        cached = query_cache.get(cache_key)

    if cached:
        # 给缓存结果补充 trace_id / cache_hit / 健康信息
        cached["trace_id"] = trace_id
        cached["cache_hit"] = True
        cached.setdefault("degraded", False)
        cached.setdefault("degraded_mode", None)
        cached.setdefault("degraded_reason", None)
        cached.setdefault("milvus_ok", True)
        cached.setdefault("redis_ok", redis_ok)

        query_logger.log(
            {
                "trace_id": trace_id,
                "query": q,
                "hybrid": hybrid,
                "top_k": top_k,
                "latency_ms": 0,
                "result_count": len(cached.get("results", [])),
                "cache_hit": True,
                "degraded": cached.get("degraded", False),
                "milvus_ok": cached.get("milvus_ok"),
                "redis_ok": cached.get("redis_ok"),
                "degraded_reason": cached.get("degraded_reason"),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )

        return cached

    # -----------------------------------------------------
    # 缓存 miss → 执行真实检索
    # -----------------------------------------------------
    if not hybrid:
        # =================================================
        # 分支一：纯向量检索（带 Milvus 降级）
        # =================================================
        try:
            t0 = time.time()
            res = vector_retriever.search(q, top_k)
            t1 = time.time()

            raw_hits = res.get("results", [])
            formatted: list[dict[str, Any]] = []

            for hit in raw_hits:
                text = hit.get("text")
                meta = hit.get("meta") or {}
                if not text and isinstance(meta, dict):
                    text = meta.get("text") or meta.get("content")

                item: dict[str, Any] = {
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
                "total": round((t1 - t_start) * 1000, 2),
            }

            response: dict[str, Any] = {
                "trace_id": trace_id,
                "cache_hit": False,
                "query": q,
                "hybrid": False,
                "top_k": top_k,
                "latency_ms": latency_ms,
                "results": formatted,
            }

        except Exception as e:
            # -----------------------------
            # Milvus 故障 → 降级为 BM25-only
            # -----------------------------
            milvus_ok = False
            degraded = True
            degraded_mode = "bm25_only"
            degraded_reason = f"vector_search_failed: {e}"

            t_bm0 = time.time()
            bm25_hits = bm25_retriever.search(q, top_k)
            t_bm1 = time.time()

            formatted: list[dict[str, Any]] = []
            for hit in bm25_hits:
                formatted.append(
                    {
                        "doc_id": None,
                        "chunk_id": hit.get("chunk_id"),
                        "text": hit.get("text"),
                        "score_vector": None,
                        "score_bm25": float(hit["score"]) if "score" in hit else None,
                        "rrf_score": None,
                        "sources": ["bm25"],
                    }
                )

            latency_ms = {
                "vector": 0.0,
                "bm25": round((t_bm1 - t_bm0) * 1000, 2),
                "total": round((t_bm1 - t_start) * 1000, 2),
            }

            response = {
                "trace_id": trace_id,
                "cache_hit": False,
                "query": q,
                "hybrid": False,
                "top_k": top_k,
                "latency_ms": latency_ms,
                "results": formatted,
            }

    else:
        # =================================================
        # 分支二：Hybrid 检索（向量 + BM25 + RRF）
        # 若内部出现 Milvus 错误，同样降级为 BM25-only
        # =================================================
        try:
            res = hybrid_retriever.search(
                query=q,
                vector_k=vector_k,
                bm25_k=bm25_k,
                top_k=top_k,
                rerank=rerank,
                page=page,
                page_size=page_size,
                debug=debug,
            )

            response = {
                "trace_id": trace_id,
                "cache_hit": False,
                "query": q,
                "hybrid": True,
                "top_k": top_k,
                "vector_k": vector_k,
                "bm25_k": bm25_k,
                "rerank": rerank,
                "latency_ms": res.get("latency_ms", {}),
                "pagination": res.get("pagination"),
                "results": res.get("final_results") or res.get("fused_results", []),
            }

            if debug:
                response["debug"] = res.get("debug")

        except Exception as e:
            # -----------------------------
            # Hybrid 内部异常（通常是 Milvus）→ BM25-only
            # -----------------------------
            milvus_ok = False
            degraded = True
            degraded_mode = "bm25_only"
            degraded_reason = f"hybrid_search_failed: {e}"

            t_bm0 = time.time()
            bm25_hits = bm25_retriever.search(q, top_k)
            t_bm1 = time.time()

            formatted: list[dict[str, Any]] = []
            for hit in bm25_hits:
                formatted.append(
                    {
                        "doc_id": None,
                        "chunk_id": hit.get("chunk_id"),
                        "text": hit.get("text"),
                        "score_vector": None,
                        "score_bm25": float(hit["score"]) if "score" in hit else None,
                        "rrf_score": None,
                        "sources": ["bm25"],
                    }
                )

            latency_ms = {
                "bm25": round((t_bm1 - t_bm0) * 1000, 2),
                "total": round((t_bm1 - t_start) * 1000, 2),
            }

            response = {
                "trace_id": trace_id,
                "cache_hit": False,
                "query": q,
                "hybrid": True,
                "top_k": top_k,
                "vector_k": vector_k,
                "bm25_k": top_k,
                "rerank": False,
                "latency_ms": latency_ms,
                "pagination": {
                    "page": 1,
                    "page_size": len(formatted) or page_size,
                    "total": len(formatted),
                },
                "results": formatted,
            }

    # -----------------------------------------------------
    # 补充降级 & 健康元信息
    # -----------------------------------------------------
    response["degraded"] = degraded
    response["degraded_mode"] = degraded_mode
    response["degraded_reason"] = degraded_reason
    response["milvus_ok"] = milvus_ok
    response["redis_ok"] = redis_ok

    # -----------------------------------------------------
    # 写入缓存（debug=True / degraded=True 不写；空结果也不写）
    # -----------------------------------------------------
    if (
        not debug
        and cache_key
        and response.get("results")
        and not response.get("degraded", False)
    ):
        # Day 12 约定：短期缓存 30s
        query_cache.set(cache_key, response, ttl=30)

    # -----------------------------------------------------
    # 写入日志（文件 + SQLite）
    # -----------------------------------------------------
    query_logger.log(
        {
            "trace_id": trace_id,
            "query": q,
            "hybrid": hybrid,
            "top_k": top_k,
            "latency_ms": response.get("latency_ms", {}).get("total", None),
            "result_count": len(response.get("results", [])),
            "cache_hit": False,
            "degraded": degraded,
            "degraded_mode": degraded_mode,
            "degraded_reason": degraded_reason,
            "milvus_ok": milvus_ok,
            "redis_ok": redis_ok,
            "payload": response,
        }
    )

    return response


# test1
