# libs/embedding/factory.py
import os
from libs.embedding.base import BaseEmbeddingModel


def get_embedding_model() -> BaseEmbeddingModel:
    """
    根据 .env 中的 EMBEDDING_MODEL 动态选择 embedding 模型。
    支持：dummy / sentence / openai
    """
    model_name = os.getenv("EMBEDDING_MODEL", "dummy").lower()
    dim = int(os.getenv("EMBEDDING_DIM", "768"))

    if model_name == "dummy":
        from libs.embedding.dummy import DummyEmbeddingModel
        return DummyEmbeddingModel(dim=dim, normalize=True)

    else:
        raise ValueError(f"Unsupported EMBEDDING_MODEL={model_name}")