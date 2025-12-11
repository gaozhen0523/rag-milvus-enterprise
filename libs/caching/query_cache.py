# libs/caching/query_cache.py
import hashlib
import json
import logging
import os
import time
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore
except ImportError:  # redis 包不存在时，自动走内存模式
    redis = None


class QueryCache:
    """
    Query 级缓存（Day 12 版本）

    优先使用 Redis（适合 Docker / ECS 多实例）
    - 通过环境变量配置：
        REDIS_HOST=localhost
        REDIS_PORT=6379
        REDIS_DB=0
        REDIS_PASSWORD= 可选
    Redis 不可用时，自动回退到 in-memory 字典缓存。
    """

    def __init__(self) -> None:
        # in-memory 备用存储
        self._store: dict[str, Any] = {}
        self._expire: dict[str, float] = {}
        self._lock = Lock()

        self._use_redis = False
        self._redis_client: redis.Redis | None = None

        self._init_redis()

    def is_redis_available(self) -> bool:
        """
        用于对外暴露当前是否在使用 Redis 作为后端。
        - True  表示正在用 Redis
        - False 表示走 in-memory fallback
        """
        return bool(self._use_redis and self._redis_client is not None)

    # --------------------------------------------------------------
    # Redis 初始化（失败自动回退内存）
    # --------------------------------------------------------------
    def _init_redis(self) -> None:
        if redis is None:
            logger.warning(
                "redis package not installed, QueryCache will use in-memory store."
            )
            return

        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        password = os.getenv("REDIS_PASSWORD", None)

        try:
            client = redis.Redis(
                host=host, port=port, db=db, password=password, decode_responses=True
            )
            # 测试连接
            client.ping()
            self._redis_client = client
            self._use_redis = True
            logger.info(
                "QueryCache initialized with Redis at %s:%d db=%d", host, port, db
            )
        except Exception as e:
            logger.warning(
                "Failed to init Redis for QueryCache, fallback to in-memory. err=%s", e
            )
            self._use_redis = False
            self._redis_client = None

    # --------------------------------------------------------------
    # 生成 query 的稳定 key（sha256 + 归一化）
    # --------------------------------------------------------------
    @staticmethod
    def make_key(
        q: str,
        hybrid: bool,
        top_k: int,
        vector_k: int,
        bm25_k: int,
        page: int,
        page_size: int,
        rerank: bool,
    ) -> str:
        """
        Day 12 最终版 cache key:
        - query 文本（大小写敏感，为了保持用户输入语义）
        - hybrid 模式
        - top_k
        - vector_k
        - bm25_k
        - page / page_size
        - rerank
        """
        raw = f"{q}|{hybrid}|{top_k}|{vector_k}|{bm25_k}|{page}|{page_size}|{rerank}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        return f"qcache:{digest}"

    # --------------------------------------------------------------
    # get：命中返回 dict，miss 返回 None
    # --------------------------------------------------------------
    def get(self, key: str) -> dict[str, Any] | None:
        # Redis 模式
        if self._use_redis and self._redis_client is not None:
            try:
                val = self._redis_client.get(key)
                if val is None:
                    return None
                return json.loads(val)
            except Exception as e:
                logger.warning("QueryCache Redis get failed, key=%s, err=%s", key, e)
                return None

        # in-memory 回退
        with self._lock:
            # 过期检查
            if key in self._expire and time.time() > self._expire[key]:
                self._store.pop(key, None)
                self._expire.pop(key, None)
                return None
            return self._store.get(key)

    # --------------------------------------------------------------
    # set：写入缓存，支持 ttl（秒）
    # --------------------------------------------------------------
    def set(self, key: str, value: dict[str, Any], ttl: int = 60) -> None:
        if value is None:
            return

        # Redis 模式
        if self._use_redis and self._redis_client is not None:
            try:
                payload = json.dumps(value, ensure_ascii=False)
                # setex 自带过期时间
                self._redis_client.setex(key, ttl, payload)
                return
            except Exception as e:
                logger.warning(
                    "QueryCache Redis set failed, "
                    "key=%s, err=%s; fallback to in-memory",
                    key,
                    e,
                )
                # 失败后自动退回内存模式一次，但不关闭 Redis，以便后续恢复
                # 不修改 self._use_redis，让它有机会再次尝试

        # in-memory 模式
        with self._lock:
            self._store[key] = value
            self._expire[key] = time.time() + ttl


# 单例
query_cache = QueryCache()
