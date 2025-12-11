# services/retriever/bm25_retriever.py

from typing import Any

from opentelemetry import trace
from rank_bm25 import BM25Okapi

from libs.db.milvus_client import MilvusClientFactory

_tracer = trace.get_tracer("bm25-retriever")


class BM25Retriever:
    """
    使用 rank_bm25 实现本地 BM25 召回。
    目前的 corpus 来源：Milvus 中所有 chunks 的原文片段（meta["text"]）。
    （可在 Day 11/12 迁移到 Redis/ES）
    """

    def __init__(self):
        self.milvus = MilvusClientFactory()
        self._initialized = False
        self.corpus: list[str] = []
        self.bm25: BM25Okapi | None = None

    def _load_corpus(self):
        """
        读取所有 chunks 的 meta.text 字段作为 BM25 的 corpus。
        """
        if self._initialized:
            return
        try:
            rows = self.milvus.fetch_all_documents()
        except Exception as e:
            print(f"⚠️ BM25 corpus load failed (Milvus unavailable): {e}")
            self.bm25 = None
            self._initialized = True
            return
        self.corpus: list[str] = [
            row["meta"].get("text", "")
            for row in rows
            if row.get("meta") and "text" in row["meta"]
        ]

        if not self.corpus:
            print("⚠️ BM25 corpus is empty. Skipping BM25 initialization.")
            self.bm25 = None
            self._initialized = True
            return

        tokenized = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
        self._initialized = True

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        BM25: 返回 top_k chunk 文本 + score
        """
        with _tracer.start_as_current_span("bm25.search"):
            if not self._initialized:
                self._load_corpus()
            if self.bm25 is None:
                return []

            tokens = query.split()
            scores = self.bm25.get_scores(tokens)

            # 排序 (idx, score)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

            results = []
            for idx, score in ranked:
                results.append(
                    {
                        "chunk_id": idx,
                        "score": float(score),
                        "text": self.corpus[idx],
                    }
                )

            return results
