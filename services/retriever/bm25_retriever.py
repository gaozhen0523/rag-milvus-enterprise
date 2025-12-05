# services/retriever/bm25_retriever.py

from typing import Any

from rank_bm25 import BM25Okapi

from libs.db.milvus_client import MilvusClientFactory


class BM25Retriever:
    """
    使用 rank_bm25 实现本地 BM25 召回。
    目前的 corpus 来源：Milvus 中所有 chunks 的原文片段（meta["text"]）。
    （可在 Day 11/12 迁移到 Redis/ES）
    """

    def __init__(self):
        self.milvus = MilvusClientFactory()
        self._load_corpus()

    def _load_corpus(self):
        """
        读取所有 chunks 的 meta.text 字段作为 BM25 的 corpus。
        """
        rows = (
            self.milvus.fetch_all_documents()
        )  # 你需要提供一个 fetch 函数（我下面给你写）
        self.corpus: list[str] = [
            row["meta"].get("text", "")
            for row in rows
            if row.get("meta") and "text" in row["meta"]
        ]

        if not self.corpus:
            print("⚠️ BM25 corpus is empty. Skipping BM25 initialization.")
            self.bm25 = None
            return

        tokenized = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        BM25: 返回 top_k chunk 文本 + score
        """
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
