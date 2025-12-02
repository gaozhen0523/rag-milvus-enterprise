# tests/test_rrf.py
from libs.retriever.rrf import rrf_fuse


def test_rrf_fusion_basic():
    vector_hits = [
        {"doc_id": "a", "score": 0.9},
        {"doc_id": "b", "score": 0.8},
    ]
    bm25_hits = [
        {"doc_id": "b", "score": 3.0},
        {"doc_id": "c", "score": 2.0},
    ]

    fused = rrf_fuse(
        vector_results=vector_hits,
        bm25_results=bm25_hits,
        k=60,
    )

    # ✅ 第一名应该是来自 vector rank=1 的 'a'
    assert fused[0]["doc_id"] == "a"

    # ✅ 不同来源/不同 rank 的 b 会产生两条 entry
    b_items = [x for x in fused if x["doc_id"] == "b"]
    assert len(b_items) == 2

    # ✅ 所有 doc_id 都应该出现
    ids = {x["doc_id"] for x in fused}
    assert ids == {"a", "b", "c"}
