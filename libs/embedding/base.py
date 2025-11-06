from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingModel(ABC):
    """抽象基类，定义统一嵌入接口"""

    @abstractmethod
    def embed_one(self, text: str) -> List[float]:
        """生成单条文本的 embedding 向量"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成 embedding 向量"""
        pass