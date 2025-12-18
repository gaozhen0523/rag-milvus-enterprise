
# RAG Milvus Enterprise — Roadmap

This project is intentionally "production-shaped" but still a portfolio project. Below is how I would extend it over the next 2–3 months in a real team.

---

## 1. Embedding & Retrieval

- ✅ Current:
  - Dummy embedding model for demo.
  - Vector + BM25 + RRF + embedding-based rerank.

- 🔜 Next:
  - Integrate a real embedding provider (BGE / OpenAI / Cohere).
  - Add cross-encoder reranker for better ranking quality.
  - Support multiple collections / indexes per domain (FAQ, docs, code).

---

## 2. Ingestion & Data Management

- Add:
  - Update/delete APIs for documents and chunks.
  - Incremental ingestion for updated files (diff-based).
  - Async ingestion via a separate task queue, decoupling API from heavy work.
- Improve:
  - Backfill and re-index jobs with progress tracking.
  - Metadata-driven routing (e.g. per-tenant collections).

---

## 3. Observability & SLOs

- Integrate OpenTelemetry tracing:
  - API → Retriever → Milvus → Cache.
- Export metrics:
  - query latency P50/P95/P99,
  - error rates (per component),
  - cache hit ratio.
- Define SLOs:
  - e.g. `P95 query latency < 800 ms` for 99% of days.
- Add dashboards in Grafana and alerts in Alertmanager or similar.

---

## 4. Multi-Tenancy & Security

- API keys per tenant with rate limiting and quotas.
- Per-tenant isolation:
  - dedicated collections or logical segments,
  - tenant-aware logging and tracing.
- Basic RBAC for admin vs. normal users.

---

## 5. Cloud & Reliability

- Move from Milvus standalone → distributed / managed (e.g. Zilliz Cloud).
- Run in multiple availability zones.
- Add blue-green / canary deployment strategy in CI/CD.
- Chaos experiments:
  - Simulate Milvus / Redis downtime and verify graceful degradation.
