from fastapi import FastAPI
from libs.db.milvus_client import MilvusClientFactory

app = FastAPI()

@app.get("/health")
def health_check():
    factory = MilvusClientFactory()
    return factory.health_status()