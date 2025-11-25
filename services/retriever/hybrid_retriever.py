# services/retriever/hybrid_retriever.py

from typing import Dict, Any
import time

from services.retriever.vector_retriever import VectorRetriever
from services.retriever.bm25_retriever import BM25Retriever
from services.retriever.rerank import Reranker
from libs.retriever.rrf import rrf_fuse


class HybridRetriever:
    """
    Day 8: 双路检索 (Vector + BM25)
    Day 9: 将会加入 RRF 融合
    Day10: 可选 Rerank + 分页 + Debug
    """

    def __init__(self):
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()
        # RRF 融合超参数（常用缺省为 60）
        self.reranker = Reranker()
        self.rrf_k = 60

    def search(
            self,
            query: str,
            vector_k: int = 5,
            bm25_k: int = 5,
            top_k: int = 5,
            rerank: bool = False,
            page: int = 1,
            page_size: int = 10,
            debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Hybrid 检索：
        - vector 检索
        - BM25 检索
        - RRF 融合
        - 取前 top_k 作为候选池
        - 可选 Rerank
        - 分页
        """
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
        ) or []
        t3 = time.time()

        # 只对前 top_k 个做 rerank / 分页
        candidates = fused_results[:top_k] if top_k > 0 else fused_results
        total_candidates = len(candidates)

        if rerank and candidates:
            rerank_start = time.time()
            reranked_candidates = self.reranker.rerank(query, candidates)
            rerank_end = rerank_start + (time.time() - rerank_start)
        else:
            rerank_start = rerank_end = t3
            reranked_candidates = candidates

        # 分页（在候选池内部分页）
        start = (page - 1) * page_size
        end = start + page_size
        if start >= len(reranked_candidates):
            page_items = []
        else:
            page_items = reranked_candidates[start:end]

        # 构造延时信息
        latency_ms = {
            "vector": round((t1 - t0) * 1000, 2),
            "bm25": round((t2 - t1) * 1000, 2),
            "fusion": round((t3 - t2) * 1000, 2),
        }

        if rerank:
            latency_ms["rerank"] = round((rerank_end - rerank_start) * 1000, 2)
            latency_ms["total"] = round((rerank_end - t0) * 1000, 2)
        else:
            latency_ms["total"] = round((t3 - t0) * 1000, 2)

        pagination = {
            "page": page,
            "page_size": page_size,
            "total": total_candidates,
        }

        debug_payload: Dict[str, Any] = {}
        if debug:
            debug_payload = {
                "vector_results_raw": vec_res.get("results", []),
                "bm25_results_raw": bm25_res or [],
                "fused_results_full": fused_results,
                "candidates_before_rerank": candidates,
                "rerank_enabled": rerank,
            }

        return {
            "query": query,
            "vector_results": vec_res.get("results", []),
            "bm25_results": bm25_res or [],
            "fused_results": fused_results,
            "final_results": page_items,
            "latency_ms": latency_ms,
            "pagination": pagination,
            "rerank": rerank,
            "debug": debug_payload if debug else None,
        }

def main():
    h = HybridRetriever()
    h = HybridRetriever()
    print(
        h.search(
            "test document",
            vector_k=3,
            bm25_k=3,
            top_k=5,
            rerank=True,
            page=1,
            page_size=5,
            debug=True,
        )
    )
if __name__ == "__main__":
    main()