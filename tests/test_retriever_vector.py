# tests/test_vector_retriever.py
from services.retriever.vector_retriever import VectorRetriever


def test_vector_retriever_runs():
    vr = VectorRetriever()
    res = vr.search("hello world", top_k=3)

    assert "results" in res
    assert isinstance(res["results"], list)
