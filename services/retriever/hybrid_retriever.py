# services/retriever/hybrid_retriever.py

from typing import Dict, Any
import time

from services.retriever.vector_retriever import VectorRetriever
from services.retriever.bm25_retriever import BM25Retriever
from libs.retriever.rrf import rrf_fuse


class HybridRetriever:
    """
    Day 8: 双路检索 (Vector + BM25)
    Day 9: 将会加入 RRF 融合
    """

    def __init__(self):
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()
        # RRF 融合超参数（常用缺省为 60）
        self.rrf_k = 60

    def search(
        self,
        query: str,
        vector_k: int = 5,
        bm25_k: int = 5,
    ) -> Dict[str, Any]:
        t0 = time.time()
        vec_res = self.vector.search(query, vector_k)
        t1 = time.time()

        bm25_res = self.bm25.search(query, bm25_k)
        t2 = time.time()

        # RRF 融合（使用底层原始结果）
        fused_results = rrf_fuse(
            vector_results = vec_res.get("results", []),
            bm25_results = bm25_res or [],
            k = self.rrf_k,
        )
        t3 = time.time()

        latency_ms = {
            "vector": round((t1 - t0) * 1000, 2),
            "bm25": round((t2 - t1) * 1000, 2),
            "fusion": round((t3 - t2) * 1000, 2),
            "total": round((t3 - t0) * 1000, 2),
        }


        return {
            "query": query,
            "vector_results": vec_res.get("results", []),
            "bm25_results": bm25_res or [],
            "fused_results": fused_results,
            "latency_ms": latency_ms,
        }
def main():
    h = HybridRetriever()
    print(h.search("test document", vector_k=3, bm25_k=3))
if __name__ == "__main__":
    main()