
# RAG Milvus Enterprise — Interview Notes

---

## Page 1 — Architecture & Key Decisions

**Problem I wanted to solve**

- Build a realistic, production-style RAG system:
  - ingest arbitrary documents,
  - store embeddings in a vector DB,
  - support hybrid retrieval and reranking,
  - and run on AWS in a way that looks like what real teams ship.

**High-level architecture**

- FastAPI **API Gateway** with `/ingest`, `/query`, `/health`
- **Embedding Worker** doing chunk → embed → batch insert
- **Milvus** as the vector database
- **Hybrid Retriever** combining:
  - Milvus vector search,
  - BM25 full-text search,
  - RRF fusion,
  - optional embedding-based rerank
- **Redis cache** for short-term query results
- Terraform modules + ECS Fargate + ALB for deployment

**Key design decisions**

- Separate API Gateway and Embedding Worker:
  - keeps API latency low,
  - makes it easier to scale embedding independently.
- Use Milvus instead of a generic SQL DB:
  - optimized for ANN search,
  - supports IVF indexes and different metrics.
- Add BM25 and RRF:
  - covers keyword-heavy queries that vector search alone may miss.
- Use Redis-backed query cache:
  - production-style cross-instance cache,
  - not just an in-process LRU.

---

## Page 2 — Performance & Scalability

**Ingestion path**

- Chunking with `size` / `overlap` so the system can handle large docs (2000+ chunks).
- Batch insert into Milvus with a configurable `batch_size` to balance:
  - insert throughput,
  - memory usage,
  - and backpressure on the DB.

**Query performance**

- `/query` benchmark script wraps the API and measures:
  - P50, P95, and mean latency,
  - success rate.
- On my dev machine (Docker Milvus + local API) I consistently see:
  - P50 ≈ 600 ms,
  - P95 slightly higher (depends on corpus size),
  - with all calls successful for a small demo dataset.

**Scalability story**

- Horizontally scale the API Gateway and Embedding Worker separately on ECS.
- Milvus can be switched from standalone to distributed mode as traffic grows.
- Redis cache reduces load on Milvus for repeated queries.
- There’s a clear path to:
  - move chunking/embedding to a job queue,
  - add autoscaling policies based on QPS and tail latencies.

---

## Page 3 — Reliability, Fault Tolerance & What I'd Improve

**Fault tolerance**

- `/health` endpoint reports:
  - Milvus connectivity,
  - collection existence,
  - version info.
- If Milvus is unavailable, the design falls back to BM25-only retrieval instead of returning 500s.
- For file ingestion:
  - HTTP download errors are surfaced with clear error messages,
  - large documents are processed in batches to avoid memory spikes.

**Observability**

- Structured logging on API side:
  - includes `trace_id`, latency metrics, query parameters.
- Benchmark scripts act as lightweight synthetic monitoring.
- The design explicitly leaves room for OpenTelemetry tracing to connect:
  - API → Retriever → Milvus.

**What I would do next in a real team**

- Replace the dummy embedding model with a real model (e.g. BGE / OpenAI).
- Add proper rate limiting and API keys for multi-tenant use.
- Introduce full tracing and metrics dashboards in Grafana:
  - per-endpoint latency distributions,
  - error rates,
  - Milvus usage.
- Add update / delete APIs and background re-indexing for evolving content.
````
