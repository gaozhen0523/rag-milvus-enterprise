import numpy as np
from libs.embedding.base import BaseEmbeddingModel

class DummyEmbeddingModel(BaseEmbeddingModel):
    """占位 embedding 模型，用随机向量模拟"""

    def __init__(self, dim: int = 384, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize

    def _rand_vec(self, seed: int):
        rng = np.random.default_rng(seed)
        v = rng.random(self.dim)
        if self.normalize:
            v /= np.linalg.norm(v)
        return v

    def embed_one(self, text: str):
        return self._rand_vec(abs(hash(text)) % (2**32)).tolist()

    def embed_batch(self, texts):
        return [self.embed_one(t) for t in texts]