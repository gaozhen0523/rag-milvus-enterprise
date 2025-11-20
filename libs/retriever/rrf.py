# libs/retriever/rrf.py
from typing import List, Dict, Any


def _build_key(hit: Dict[str, Any], source: str, idx: int) -> str:
    """
    用 doc_id + chunk_id 作为主要 key；
    如果都没有，就退化为 source + idx，保证 key 唯一。
    """
    doc_id = hit.get("doc_id")
    chunk_id = hit.get("chunk_id")

    if chunk_id is not None:
        return f"{doc_id or ''}::{chunk_id}"
    # 兜底：BM25 可能只有 text/score/chunk_id，或者异常结构
    return f"{source}:{idx}"


def _extract_text_from_vector_hit(hit: Dict[str, Any]) -> str | None:
    """
    vector 命中通常来自 Milvus，text 可能藏在 meta 里。
    """
    # 有些情况下上层已经把 text 展开成 hit["text"]
    if "text" in hit and hit["text"]:
        return hit["text"]

    meta = hit.get("meta") or {}
    if isinstance(meta, dict):
        return meta.get("text") or meta.get("content")

    return None


def rrf_fuse(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion:
        RRF(d) = Σ_s 1 / (k + rank_s(d))

    输入：
        - vector_results: Milvus 搜索结果（list[dict]）
        - bm25_results: BM25 搜索结果（list[dict]）
    输出：
        - list[dict]，每个元素带：
            doc_id, chunk_id, text,
            score_vector, score_bm25,
            rrf_score, sources
    """
    fused: Dict[str, Dict[str, Any]] = {}

    def add_results(results: List[Dict[str, Any]], source: str, is_vector: bool) -> None:
        for rank, hit in enumerate(results, start=1):
            key = _build_key(hit, source, rank)

            if key not in fused:
                # 初始化一条融合结果
                doc_id = hit.get("doc_id")
                chunk_id = hit.get("chunk_id")

                if is_vector:
                    text = _extract_text_from_vector_hit(hit)
                else:
                    # BM25 一般直接有 text
                    text = hit.get("text")

                fused[key] = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "score_vector": None,
                    "score_bm25": None,
                    "rrf_score": 0.0,
                    "sources": [],
                }

            entry = fused[key]

            # 记录来源
            if source not in entry["sources"]:
                entry["sources"].append(source)

            # 记录原始 score
            raw_score = hit.get("score")
            if raw_score is not None:
                try:
                    raw_score = float(raw_score)
                except (TypeError, ValueError):
                    raw_score = None

            if is_vector and raw_score is not None:
                entry["score_vector"] = raw_score
            elif (not is_vector) and raw_score is not None:
                entry["score_bm25"] = raw_score

            # RRF 增量
            contribution = 1.0 / (k + rank)
            entry["rrf_score"] += contribution

    # 先加 vector 再加 bm25，这样优先用向量结果补全 doc_id/text 等字段
    add_results(vector_results or [], source="vector", is_vector=True)
    add_results(bm25_results or [], source="bm25", is_vector=False)

    # 转为 list 并按 rrf_score 降序排序
    fused_list = list(fused.values())
    fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_list
