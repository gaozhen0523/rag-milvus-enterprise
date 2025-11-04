import os
from dotenv import load_dotenv
from pymilvus import connections

load_dotenv(override=False)

class MilvusClientFactory:
    """
    Milvus连接工厂：
    - 从环境变量读取 MILVUS_HOST / MILVUS_PORT
    - 自动建立或复用连接
    """
    def __init__(self, host=None, port=None):
        self.host = host or os.getenv("MILVUS_HOST", "127.0.0.1")
        self.port = port or os.getenv("MILVUS_PORT", "19530")

    def connect(self, alias: str = "default"):
        """连接Milvus，如果连接已存在则复用"""
        # 如果已连接，则直接返回 True
        if connections.has_connection(alias):
            print(f"🔁 Reusing existing Milvus connection ({alias})")
            return True
        # 否则建立新连接
        connections.connect(alias=alias, host=self.host, port=self.port)
        print(f"✅ Connected to Milvus at {self.host}:{self.port}")
        return True