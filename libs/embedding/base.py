# libs/embedding/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingModel(ABC):
    """
    所有嵌入模型的抽象基类。
    保证 /ingest 与 /query 共享统一接口。
    """

    def __init__(self, dim: int = 768, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize

    # -------------------------------------------------------------------------
    # 核心接口：单条文本 -> 向量
    # -------------------------------------------------------------------------
    @abstractmethod
    def embed_one(self, text: str) -> np.ndarray:
        """生成单条文本的向量"""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # 批量接口：多条文本 -> 向量列表
    # -------------------------------------------------------------------------
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """默认逐条调用 embed_one，可被子类重写为批量加速版本"""
        return [self.embed_one(t) for t in texts]

    # -------------------------------------------------------------------------
    # 归一化辅助函数
    # -------------------------------------------------------------------------
    def _normalize_vec(self, vec: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vec
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
