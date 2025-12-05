#!/usr/bin/env python3
"""
Benchmark script for /query endpoint.

目标：
- 模拟多次查询，记录每次延迟（ms）
- 打印 P50 / P95
- 可快速验证 API 性能是否稳定
"""

import requests
import time
import statistics
import random
import os

# ---------------------------------------------------------------------
# 配置参数
# ---------------------------------------------------------------------
API_URL = os.getenv("BENCH_QUERY_URL", "http://localhost:8000/query")
N_RUNS = int(os.getenv("BENCH_RUNS", "10"))
TOP_K = int(os.getenv("BENCH_TOPK", "3"))

# 可以自行添加更多常见查询词
QUERIES = [
    "milvus vector database",
    "fastapi service gateway",
    "embedding worker pipeline",
    "retrieval augmented generation",
    "document search",
    "semantic similarity",
    "knowledge base",
    "aws ecs terraform",
]


# ---------------------------------------------------------------------
# 主执行逻辑
# ---------------------------------------------------------------------
def run_benchmark():
    latencies = []
    errors = 0

    print(f"🚀 Benchmarking {API_URL} ... ({N_RUNS} runs, top_k={TOP_K})\n")

    for i in range(N_RUNS):
        q = random.choice(QUERIES)
        try:
            t0 = time.time()
            resp = requests.get(API_URL, params={"q": q, "top_k": TOP_K}, timeout=10)
            elapsed = (time.time() - t0) * 1000
            latencies.append(elapsed)
            status = resp.status_code
            print(f"[{i+1:02d}] {q:<35} {elapsed:7.2f} ms (status={status})")
        except Exception as e:
            errors += 1
            print(f"[{i+1:02d}] ERROR: {e}")

    if not latencies:
        print("\n❌ No successful runs. Please check if API is running.")
        return

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies)) - 1]

    print("\n📊 Results:")
    print(f"  Runs: {N_RUNS - errors}/{N_RUNS} successful")
    print(f"  P50 : {p50:.2f} ms")
    print(f"  P95 : {p95:.2f} ms")
    print(f"  Mean: {statistics.mean(latencies):.2f} ms")

    if errors > 0:
        print(f"  ⚠️  {errors} errors during benchmark")


if __name__ == "__main__":
    run_benchmark()
