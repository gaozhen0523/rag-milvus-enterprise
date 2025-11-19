# services/retriever/vector_retriever.py

import os
import numpy as np
from typing import List, Dict, Any
from libs.embedding.factory import get_embedding_model
from libs.db.milvus_client import MilvusClientFactory


class VectorRetriever:
    """
    封装向量检索逻辑：
    - 生成 query 向量
    - 调用 MilvusClientFactory.search_vectors()
    - 返回结果
    """

    def __init__(self):
        self.model_name = os.getenv("EMBEDDING_MODEL", "dummy").lower()
        self.dim = int(os.getenv("EMBEDDING_DIM", 768))
        self.model = get_embedding_model()
        self.factory = MilvusClientFactory()

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        vec = self.model.embed_one(query)
        results = self.factory.search_vectors(
            np.array(vec, dtype="float32"), top_k=top_k
        )
        return {
            "query": query,
            "model": self.model_name,
            "embed_dim": self.dim,
            "results": results,
        }