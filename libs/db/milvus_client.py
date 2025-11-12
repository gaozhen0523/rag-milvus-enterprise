#libs/db/milvus_client.py
import os
from dotenv import load_dotenv
from typing import List, Optional
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

    def __init__(self, host=None, port=None, collection_name=None):
        self.host = host or os.getenv("MILVUS_HOST", "127.0.0.1")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.collection_name = collection_name or os.getenv("MILVUS_COLLECTION", "rag_collection")

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
    def get_or_create_collection(self, name=None, dim=768, alias="default"):
        name = name or self.collection_name
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
        metric_type="L2",
        nlist=128,
    ):
        """创建索引并加载到内存"""
        index_params = {
            "metric_type": metric_type,
            "index_type": index_type,
            "params": {"nlist": nlist},
        }
        # 如果已存在索引则跳过
        try:
            current_indexes = collection.indexes
            if current_indexes and len(current_indexes) > 0:
                print(f"ℹ️ Index already exists on '{collection.name}', skip create_index.")
            else:
                collection.create_index(field_name="vector", index_params=index_params)
        except Exception as e:
            # 某些版本/场景 collection.indexes 可能不可用，兜底创建
            try:
                collection.create_index(field_name="vector", index_params=index_params)
            except Exception as inner:
                print(f"⚠️ create_index skipped or failed: {inner}")

        collection.load()
        print(f"✅ Index ensured and collection loaded: {collection.name}")
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
            has_col = utility.has_collection(self.collection_name)
            return {
                "status": "ok",
                "version": version,
                "rag_collection": has_col,
                "collection": self.collection_name,
                "host": self.host,
                "port": self.port,
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def search_vectors(
            self,
            query_vector: np.ndarray,
            top_k: int = 5,
            collection_name: Optional[str] = None,
            metric_type: str = "L2",
            nprobe: int = 8,
            output_fields: Optional[List[str]] = None,
            alias: str = "default",
    ):
        """
        在指定 collection 上执行向量检索。
        返回：List[ {doc_id, chunk_id, score, meta?} ]
        """
        name = collection_name or self.collection_name
        self.connect(alias)
        col = Collection(name=name, using=alias)

        # 兼容：确保存储索引 metric 与搜索 metric 一致（若不一致 Milvus 也会按索引的 metric 来）
        search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}
        output_fields = output_fields or ["doc_id", "chunk_id", "meta"]

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.asarray(query_vector, dtype="float32")
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype("float32")

        # Milvus 要求二维数组：[ [dim], [dim], ... ]
        data = [query_vector.tolist()]

        try:
            res = col.search(
                data=data,
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields,
            )
        except Exception as e:
            print(f"❌ Milvus search error: {e}")
            return [{"error": str(e)}]

        hits = []
        # res[0] 是第一个查询向量的命中列表
        for hit in res[0]:
            item = {
                "score": hit.distance,
            }
            # 命中实体字段
            try:
                # 新版 PyMilvus 建议通过 entity.get()
                for f in output_fields:
                    item[f] = hit.entity.get(f)
            except Exception:
                # 旧版可能用 ._entity 或 .id 等，这里保持容错
                pass
            hits.append(item)

        return hits
