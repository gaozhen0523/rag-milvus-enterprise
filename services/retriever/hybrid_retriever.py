# services/retriever/hybrid_retriever.py

from typing import Dict, Any
from services.retriever.vector_retriever import VectorRetriever
from services.retriever.bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Day 8: 双路检索 (Vector + BM25)
    Day 9: 将会加入 RRF 融合
    """

    def __init__(self):
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()

    def search(
        self,
        query: str,
        vector_k: int = 5,
        bm25_k: int = 5,
    ) -> Dict[str, Any]:

        vec_res = self.vector.search(query, vector_k)
        bm25_res = self.bm25.search(query, bm25_k)

        # 粗合并：仅简单拼接（Day 9 会做 RRF）
        combined = []

        for hit in vec_res["results"]:
            combined.append({
                "source": "vector",
                "score": float(hit["score"]),
                "text": hit["meta"].get("text") if hit.get("meta") else None,
                "doc_id": hit.get("doc_id"),
                "chunk_id": hit.get("chunk_id"),
            })

        for hit in bm25_res:
            combined.append({
                "source": "bm25",
                "score": float(hit["score"]),
                "text": hit["text"],
                "chunk_id": hit["chunk_id"],
            })

        return {
            "query": query,
            "vector_results": vec_res["results"],
            "bm25_results": bm25_res,
            "combined": combined,
        }
def main():
    h = HybridRetriever()
    h.search("test document")
if __name__ == "__main__":
    main()