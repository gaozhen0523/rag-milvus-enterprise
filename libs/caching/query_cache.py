import time
import hashlib
from threading import Lock
from typing import Dict, Any, Optional


class QueryCache:
    """
    Day 11 占位 Cache：
    - 使用 in-memory 字典存储
    - 支持 TTL
    - 接口与未来 Redis 版本完全一致
    """

    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.expire: Dict[str, float] = {}
        self.lock = Lock()

    # --------------------------------------------------------------
    # 生成 query 的稳定 hash（未来 Redis 版本保持一致）
    # --------------------------------------------------------------
    @staticmethod
    def make_key(q: str, hybrid: bool, top_k: int, vector_k: int, bm25_k: int):
        raw = f"{q}|{hybrid}|{top_k}|{vector_k}|{bm25_k}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------
    # get：命中返回 dict，miss 返回 None
    # --------------------------------------------------------------
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            # 过期
            if key in self.expire and time.time() > self.expire[key]:
                self.store.pop(key, None)
                self.expire.pop(key, None)
                return None

            return self.store.get(key)

    # --------------------------------------------------------------
    # set：写入缓存，支持 ttl（默认 60 秒）
    # --------------------------------------------------------------
    def set(self, key: str, value: Dict[str, Any], ttl: int = 60) -> None:
        with self.lock:
            self.store[key] = value
            self.expire[key] = time.time() + ttl


# 单例
query_cache = QueryCache()
