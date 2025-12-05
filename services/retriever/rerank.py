# services/retriever/rerank.py

from typing import Any

import numpy as np

from libs.embedding.factory import get_embedding_model


class Reranker:
    """
    基于 embedding 的轻量级 Rerank：
    - 使用 DummyEmbeddingModel（或实际 embedding 模型）对 query / chunk text 做 embed
    - 计算 cosine 相似度
    - 结合 BM25 / vector / RRF 的分数做归一化加权
    """

    def __init__(
        self,
        alpha: float = 1.0,  # 主权重：query-chunk 语义相似度
        beta: float = 0.2,  # 辅：BM25 分数
        gamma: float = 0.2,  # 辅：向量检索分数
        delta: float = 0.3,  # 辅：RRF 分数
    ) -> None:
        self.model = get_embedding_model()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype="float32")
        b = np.asarray(b, dtype="float32")
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(a.dot(b) / (na * nb))

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        """简单 min-max 归一化；None 视为 0。"""
        vals = [s for s in scores if s is not None]
        if not vals:
            return [0.0 for _ in scores]

        mn = min(vals)
        mx = max(vals)
        if abs(mx - mn) < 1e-9:
            # 所有值相等时统一给一个中间值，避免全 0
            return [0.5 if s is not None else 0.0 for s in scores]

        return [(s - mn) / (mx - mn) if s is not None else 0.0 for s in scores]

    def rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        对候选列表进行重排。
        期望 candidate 至少包含：
        - text 或 meta.text
        - score_vector / score_bm25 / rrf_score（若有则用于加权）
        """
        if not candidates:
            return []

        q_vec = self.model.embed_one(query)

        cos_sims: list[float] = []
        bm25_scores: list[float] = []
        vec_scores: list[float] = []
        rrf_scores: list[float] = []

        # 逐个 chunk 计算 embedding + cosine
        for c in candidates:
            text = c.get("text")
            if not text:
                meta = c.get("meta") or {}
                if isinstance(meta, dict):
                    text = meta.get("text") or meta.get("content") or ""

            if text:
                c_vec = self.model.embed_one(text)
                cos_val = self._cosine(q_vec, np.asarray(c_vec, dtype="float32"))
            else:
                cos_val = 0.0

            cos_sims.append(cos_val)

            bm25_scores.append(
                c.get("score_bm25")
                if c.get("score_bm25") is not None
                else c.get("bm25_score")
            )

            vec_scores.append(
                c.get("score_vector")
                if c.get("score_vector") is not None
                else (
                    c.get("vector_score")
                    if c.get("vector_score") is not None
                    else c.get("score")
                )  # 兜底：Milvus 原始 score
            )

            rrf_scores.append(c.get("rrf_score"))

        # 归一化各路分数
        cos_norm = self._normalize(cos_sims)
        bm25_norm = self._normalize(bm25_scores)
        vec_norm = self._normalize(vec_scores)
        rrf_norm = self._normalize(rrf_scores)

        reranked: list[dict[str, Any]] = []
        for c, cs, nb, nv, nr in zip(
            candidates,
            cos_norm,
            bm25_norm,
            vec_norm,
            rrf_norm,
            strict=True,
        ):
            score = self.alpha * cs + self.beta * nb + self.gamma * nv + self.delta * nr
            item = dict(c)
            item["rerank_score"] = float(score)
            item["score_rerank_cos"] = float(cs)
            reranked.append(item)

        # 按 rerank_score 降序排序
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked
