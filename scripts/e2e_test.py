# scripts/e2e_test.py
import requests
import time
import uuid
import json

API_BASE = "http://localhost:8000"


def pretty(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)


def run_ingest():
    """
    Step 1: ingest test text
    """
    test_id = str(uuid.uuid4())

    test_text = (
        "This is a test sentence for E2E testing. "
        "The quick brown fox jumps over the lazy dog. "
        "Milvus vector search works correctly. "
    ) * 5  # 生成多几句

    payload = {
        "text": test_text,
        "chunk": {"strategy": "sentence", "size": 200, "overlap": 50},
        "source_id": test_id,
        "metadata": {"test_id": test_id},
    }

    print("\n[1] Sending /ingest request...")
    r = requests.post(f"{API_BASE}/ingest?dry_run=false", json=payload)
    if r.status_code != 200:
        print("❌ Ingest failed:", r.text)
        return None

    data = r.json()
    print("[INGEST RESPONSE]:")
    print(pretty(data))

    return test_id, test_text, data.get("preview_chunks")


def run_query(test_text):
    """
    Step 2: query to verify content inserted
    """
    query_word = "Milvus"  # 测试句子的关键字之一
    print("\n[2] Sending /query request:", query_word)

    r = requests.get(f"{API_BASE}/query", params={"q": query_word, "hybrid": True})
    if r.status_code != 200:
        print("❌ Query failed:", r.text)
        return None

    data = r.json()
    print("[QUERY RESPONSE]:")
    print(pretty(data))

    return data


def verify_result(query_resp, test_text):
    """
    Step 3: determine pass/fail
    """
    hits = query_resp.get("results", [])
    if not hits:
        print("❌ FAIL: no results returned")
        return False

    # 看 text 字段是否包含我们的 test_text 中的内容（任意 chunk 即可）
    matched = False
    for h in hits:
        t = h.get("text") or ""
        if isinstance(t, str) and "Milvus" in t:
            matched = True
            break

    if matched:
        print("\n✅ PASS: Query successfully hit inserted text")
        return True
    else:
        print("\n❌ FAIL: Query returned results but not the inserted text")
        return False


if __name__ == "__main__":
    print("\n===============================")
    print("   RAG E2E Smoke Test (Day 13) ")
    print("===============================")

    step1 = run_ingest()
    if not step1:
        exit(1)

    test_id, test_text, chunk_count = step1

    print(f"\nWaiting 0.5s for worker flush... (chunks={chunk_count})")
    time.sleep(0.5)

    step2 = run_query(test_text)
    if not step2:
        exit(1)

    verify_result(step2, test_text)

    print("\nDone.")
