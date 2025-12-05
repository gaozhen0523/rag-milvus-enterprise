# services/embedding_worker/main.py
import os
from datetime import datetime

from libs.chunking.text_chunker import TextChunker
from libs.embedding.dummy import DummyEmbeddingModel


def process_file(file_path: str):
    """读取文件 → 分块 → 生成 embedding → 打印统计"""
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    chunker = TextChunker(strategy="sentence", size=500, overlap=50)
    chunks = chunker.chunk(text)

    model = DummyEmbeddingModel(
        dim=int(os.getenv("EMBEDDING_DIM", 384)),
        normalize=True,
    )

    texts = [c.text if hasattr(c, "text") else str(c) for c in chunks]
    vectors = model.embed_batch(texts)

    print(f"[{datetime.now()}] Processed {len(chunks)} chunks, dim={model.dim}")
    return vectors


if __name__ == "__main__":
    test_file = os.getenv("TEST_DOC", "sample.txt")
    if not os.path.exists(test_file):
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("This is a demo text for embedding worker test. " * 5)
    process_file(test_file)
