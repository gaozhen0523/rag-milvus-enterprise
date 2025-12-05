"""
验证Milvus连接是否成功：
  python -m scripts.load_demo_corpus
"""

from pymilvus import utility
from libs.db.milvus_client import MilvusClientFactory


def main():
    MilvusClientFactory().connect()
    print("✅ Connected to Milvus.")
    print("Server version:", utility.get_server_version())
    print("Existing collections:", utility.list_collections())


if __name__ == "__main__":
    main()
