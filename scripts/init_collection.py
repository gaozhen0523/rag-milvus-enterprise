from libs.db.milvus_client import MilvusClientFactory

def init_collection():
    factory = MilvusClientFactory()
    collection = factory.get_or_create_collection(name="rag_collection", dim=768)
    factory.ensure_index_and_load(collection, index_type="IVF_FLAT", metric_type="IP", nlist=128)
    factory.insert_demo_data(collection)
    print("✅ Collection initialization complete.")

if __name__ == "__main__":
    init_collection()