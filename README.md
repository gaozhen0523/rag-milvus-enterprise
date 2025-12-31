# RAG Milvus Enterprise

Enterprise-grade Retrieval-Augmented Generation (RAG) system built on top of **Milvus**, **FastAPI**, and **Redis**, designed as a realistic portfolio project to showcase **AI Infra / backend / cloud** engineering skills.

- 🌐 **Tech stack**: Python · FastAPI · Milvus · BM25 · Redis · Docker · AWS ECS · Terraform
- 🎯 **Features**: Hybrid retrieval (Vector + BM25) · RRF fusion · Embedding-based rerank · Query cache · Batch ingestion
- ☁️ **Cloud-ready**: ECS Fargate, ALB, Terraform modules, GitHub Actions CI/CD
- 📈 **Observability**: Structured logging, query latency benchmark, tracing hooks

---
## Demo Video (Mandarin)
https://www.youtube.com/watch?v=PcnfdrZuJWQ
## 1. What this project does

This repo implements an end-to-end RAG system:

- **Ingestion pipeline**
  - Accept raw text or file URL via `POST /ingest`
  - Chunk into sentences / sliding windows
  - Generate embeddings (dummy model by default, easily swappable to OpenAI / BGE)
  - Batch insert into Milvus with timing breakdown (chunk / embed / insert / flush)
- **Query pipeline**
  - Embed query → Vector search on Milvus
  - BM25 full-text search over corpus
  - Reciprocal Rank Fusion (RRF) to combine signals
  - Optional re-ranking using an embedding model
  - Unified JSON response with scores, sources, and latency breakdown
- **Caching**
  - Short-lived query cache in Redis (TTL 30–60s), keyed by query + parameters
  - Debug mode bypasses cache for observability
- **Reliability**
  - Health check exposes Milvus status and collection metadata
  - Degradation path when Milvus is unavailable (fallback BM25-only)
- **Cloud**
  - Designed to run in Docker locally
  - Terraform modules for VPC, ECS, ECR, ALB (deployable to AWS)

---

## 2. Architecture

### 2.1 High-level diagram

```mermaid
graph TD
    A[Client] -->|POST /ingest| B[API Gateway]

    B --> C[Embedding Worker]
    C -->|Chunk → Embed → Batch Insert| D[(Milvus Vector DB)]

    A2[Client] -->|GET /query| E[API Gateway]
    E --> F[Hybrid Retriever]

    subgraph RetrievalPipeline
        F --> D
        F --> G[BM25 Index]
        F --> H[RRF Fusion]
        H --> I["Rerank (optional)"]
    end

    I --> J[JSON Response]
````

### 2.2 Components

* **API Gateway (FastAPI)**

  * `/ingest` — validate input, orchestrate ingest pipeline (dry-run or real insert)
  * `/query` — run vector / hybrid retrieval, optional rerank, apply caching
  * `/health` — Milvus connectivity + collection metadata
  * Structured logging with `trace_id`, latency, error info

* **Embedding Worker**

  * Uses `TextChunker` (char / sentence) with configurable `size` / `overlap`
  * Embedding via `BaseEmbeddingModel` implementation (default `DummyEmbeddingModel`)
  * Batch insert into Milvus with configurable `batch_size` (e.g., 2000)
  * Logs `chunk_ms / embed_ms / insert_ms / flush_ms` for large docs

* **Hybrid Retriever**

  * Vector search (Milvus IVF index)
  * BM25 search over text corpus
  * RRF fusion for rank-level merging
  * Optional embedding-based rerank with multiple features (cosine, BM25, vector, RRF)
  * Pagination and debug mode for inspection

* **Milvus Vector DB**

  * Schema (simplified):

    | field    | type                |
    | -------- | ------------------- |
    | id       | primary key         |
    | vector   | float32[dim]        |
    | doc_id   | string              |
    | chunk_id | int                 |
    | meta     | JSON (text, source) |

* **Redis Cache**

  * `query_hash → cached_result`
  * Reduces repeated hybrid queries with identical parameters
  * In-memory fallback for environments without Redis

---

## 3. Directory Structure

```text
services/
  api_gateway/       # FastAPI: /ingest, /query, /health
  embedding_worker/  # Chunk -> embed -> insert into Milvus
  retriever/         # Vector/BM25/Hybrid/RRF/Rerank

libs/
  chunking/          # TextChunker
  embedding/         # BaseEmbeddingModel, DummyEmbeddingModel, factory
  caching/           # Query cache (Redis + in-memory fallback)
  db/                # Milvus client factory
  logging/           # Structured logging helpers

docker/
  api_gateway.Dockerfile
  embedding_worker.Dockerfile
  retriever.Dockerfile

infra/
  terraform/
    modules/         # vpc, ecs_service, ecr, etc.
    envs/
      dev/
        main.tf
        variables.tf
        outputs.tf

scripts/
  init_collection.py
  load_demo_corpus.py
  bench_query.py

tests/
  test_chunking.py
  ...
```

---

## 4. Getting Started (Local)

### 4.1 Requirements

* Python 3.10+
* Docker / Docker Compose
* (Optional) Milvus CLI / UI for inspecting collections

### 4.2 Start Milvus

```bash
# In repo root
docker compose up -d

docker ps
# You should see milvus-standalone, milvus-etcd, milvus-minio all healthy
```

Optional probe:

```bash
curl http://localhost:9091/healthz
# {"status": "healthy"}
```

### 4.3 Set up Python env

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Make sure versions roughly match:

```bash
pip show pymilvus marshmallow
# pymilvus ~= 2.4.x
# marshmallow < 4.0.0 (for compatibility)
```

### 4.4 Initialize collection

```bash
python -m scripts.init_collection
```

Expected:

```text
✅ Connected to Milvus at 127.0.0.1:19530
✅ Created or loaded collection: rag_collection
✅ Index created and collection loaded
```

### 4.5 Run API Gateway

```bash
uvicorn services.api_gateway.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

---

## 5. Ingestion API

### 5.1 Endpoint

```http
POST /ingest?dry_run={true|false}
Content-Type: application/json
```

Body examples:

**(1) Text ingest**

```json
{
  "text": "Milvus is a vector database designed for scalable similarity search.",
  "chunk_strategy": "sentence",
  "chunk_size": 500,
  "chunk_overlap": 50,
  "meta": {
    "source": "demo",
    "title": "milvus_intro"
  }
}
```

**(2) File URL ingest**

```json
{
  "file_url": "https://example.com/sample.pdf",
  "chunk_strategy": "sentence",
  "chunk_size": 500,
  "chunk_overlap": 50
}
```

### 5.2 Behavior

* `dry_run = true`

  * Only validate input and run chunking
  * Returns preview: `task_id`, `preview_chunk_count`, chunk params
* `dry_run = false`

  * Full pipeline: chunk → embed → batch insert Milvus
  * Returns `inserted_chunks` and timing breakdown
* File download failures return a 4xx/5xx with error message
* Large docs (2000+ chunks) are supported via batch insert

---

## 6. Query API

### 6.1 Basic vector search

```bash
curl -X GET "http://localhost:8000/query?q=what+is+milvus&top_k=5"
```

### 6.2 Hybrid retrieval (vector + BM25 + RRF)

```bash
curl -X GET "http://localhost:8000/query" \
  -G --data-urlencode "q=semantic search" \
  --data-urlencode "hybrid=true" \
  --data-urlencode "top_k=5"
```

### 6.3 Hybrid + Rerank + Pagination

```bash
curl -G "http://localhost:8000/query" \
  --data-urlencode "q=vector database" \
  --data-urlencode "hybrid=true" \
  --data-urlencode "rerank=true" \
  --data-urlencode "top_k=20" \
  --data-urlencode "page=1" \
  --data-urlencode "page_size=5" \
  --data-urlencode "debug=true"
```

### 6.4 Response (simplified)

```json
{
  "query": "vector database",
  "hybrid": true,
  "rerank": true,
  "latency_ms": {
    "vector": 9.1,
    "bm25": 3.0,
    "fusion": 0.2,
    "rerank": 4.5,
    "total": 16.8
  },
  "pagination": {
    "page": 1,
    "page_size": 5,
    "total": 20
  },
  "results": [
    {
      "doc_id": "doc_123",
      "chunk_id": 0,
      "text": "...",
      "score_vector": 0.67,
      "score_bm25": 3.45,
      "rrf_score": 0.021,
      "rerank_score": 0.93,
      "sources": ["vector", "bm25"]
    }
  ],
  "debug": {
    "vector_results": [...],
    "bm25_results": [...],
    "fused_before_rerank": [...]
  }
}
```

---

## 7. Configuration

Most runtime options are controlled via `.env`:

* `MILVUS_HOST`, `MILVUS_PORT`
* `MILVUS_COLLECTION_NAME`
* `EMBEDDING_MODEL` (e.g. `dummy`, `sentence`, `openai`)
* `EMBEDDING_DIM`
* `EMBEDDING_METRIC` (`IP` or `L2`)
* `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`
* `API_KEY` (for lightweight authentication)
* `QUERY_CACHE_TTL_SECONDS`

---

## 8. Deploying to AWS (High level)

This repo contains Terraform modules to deploy the API and workers on **AWS ECS Fargate**:

* **Modules**

  * `infra/terraform/modules/vpc` — VPC, subnets, routing
  * `modules/ecs_service` — ECS service + task definition + security groups
  * `modules/ecr` — ECR repositories for Docker images
* **Environment**

  * `infra/terraform/envs/dev/main.tf` — wires VPC + ECS + ALB + ECR for a `dev` environment

You can:

1. Build and push Docker image to ECR (via GitHub Actions or local).
2. `terraform init && terraform apply` under `infra/terraform/envs/dev`.
3. Access the public ALB endpoint for `/health`, `/ingest`, `/query`.

CI/CD is configured to:

* Build and tag image on push to `main`
* Push to the corresponding ECR repo
* Run Terraform apply in app-level Terraform directory (with remote state in S3 + DynamoDB lock)

---

## 9. Known limitations

* Currently uses a **dummy embedding model** for simplicity.

  * Can be swapped out with OpenAI/Cohere/SentenceTransformers.
* No authentication / quota system beyond a simple API key and rate limiting stub.
* Observability is “good enough for demo”, but not as deep as a full production stack.

---

## 10. Future work

* Replace dummy embedding with a real model (e.g. BGE / OpenAI).
* Richer reranker with cross-encoder support.
* Incremental ingestion and update / delete APIs.
* Distributed Milvus cluster mode.
* Full OpenTelemetry tracing + metrics + dashboards.

````


