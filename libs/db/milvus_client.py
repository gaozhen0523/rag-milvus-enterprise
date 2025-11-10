#libs/db/milvus_client.py
import os
from dotenv import load_dotenv
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
import numpy as np

load_dotenv(override=False)


class MilvusClientFactory:
    """
    Milvus连接工厂：
    - 从环境变量读取 MILVUS_HOST / MILVUS_PORT
    - 自动建立或复用连接
    - 提供 collection 初始化 / 索引 / 加载工具
    """

    def __init__(self, host=None, port=None):
        self.host = host or os.getenv("MILVUS_HOST", "127.0.0.1")
        self.port = port or os.getenv("MILVUS_PORT", "19530")

    # -------------------------------
    # 连接管理
    # -------------------------------
    def connect(self, alias: str = "default"):
        """连接Milvus，如果连接已存在则复用"""
        if connections.has_connection(alias):
            print(f"🔁 Reusing existing Milvus connection ({alias})")
            return True
        connections.connect(alias=alias, host=self.host, port=self.port)
        print(f"✅ Connected to Milvus at {self.host}:{self.port}")
        return True

    # -------------------------------
    # Collection 初始化
    # -------------------------------
    def get_or_create_collection(self, name="rag_collection", dim=768, alias="default"):
        """获取或创建 collection"""
        self.connect(alias)

        if utility.has_collection(name, using=alias):
            print(f"ℹ️ Collection '{name}' already exists.")
            return Collection(name=name, using=alias)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="meta", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description="RAG document chunks")
        collection = Collection(name=name, schema=schema, using=alias)
        print(f"✅ Created new collection: {name}")
        return collection

    # -------------------------------
    # 索引 + 加载
    # -------------------------------
    def ensure_index_and_load(
        self,
        collection: Collection,
        index_type="IVF_FLAT",
        metric_type="IP",
        nlist=128,
    ):
        """创建索引并加载到内存"""
        index_params = {
            "metric_type": metric_type,
            "index_type": index_type,
            "params": {"nlist": nlist},
        }
        collection.create_index(field_name="vector", index_params=index_params)
        collection.load()
        print(f"✅ Index created and collection loaded: {collection.name}")
        return index_params

    # -------------------------------
    # Demo 数据插入（用于初始化验证）
    # -------------------------------
    def insert_demo_data(self, collection: Collection, num_rows: int = 5, dim: int = 768):
        """插入一些随机向量进行验证"""
        import numpy as np
        vectors = np.random.random((num_rows, dim)).astype("float32").tolist()
        doc_ids = [f"doc_{i}" for i in range(num_rows)]
        chunk_ids = list(range(num_rows))
        metas = [{"source": "demo", "tags": ["init", "day3"]} for _ in range(num_rows)]

        # 列模式插入，顺序必须与 schema 中定义一致（除主键）
        data = [vectors, doc_ids, chunk_ids, metas]

        result = collection.insert(data)
        collection.flush()

        print(f"✅ Inserted {len(result.primary_keys)} demo rows into '{collection.name}'")
        print(f"Total entities now: {collection.num_entities}")
        return result

    # -------------------------------
    # 健康检查
    # -------------------------------
    def health_status(self):
        """返回 Milvus 连接与 Collection 状态"""
        try:
            self.connect()
            version = utility.get_server_version()
            has_col = utility.has_collection("rag_collection")
            return {
                "status": "ok",
                "version": version,
                "rag_collection": has_col,
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}
