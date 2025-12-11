#!/usr/bin/env python3
"""
Benchmark: Hybrid retrieval latency (online ECS)

This script calls the /query endpoint with:
  - hybrid = true
  - rerank = false
  - vector_k = top_k
  - bm25_k  = top_k
  - top_k in [5, 10, 20]

For each (top_k), it runs N times and computes:
  - p50 / p95 / mean / std of total latency (ms),
    preferring server-side latency_ms["total"] when available.

Outputs:
  benchmarks/ai/results/hybrid_ivf_flat_topk_{k}.json
  benchmarks/ai/plots/hybrid_latency.png
"""

import os
import time
import json
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import requests

# ================================
# Config
# ================================

API_URL = os.getenv(
    "RAG_API_URL",
    "http://rag-api-gateway-alb-804929386.us-east-1.elb.amazonaws.com/query",
)

RUNS_PER_CASE = int(os.getenv("HYBRID_RUNS", "30"))
TOP_K_LIST = [5, 10, 20]

OUT_DIR = Path("benchmarks/ai")
RESULT_DIR = OUT_DIR / "results"
PLOT_DIR = OUT_DIR / "plots"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "milvus vector search",
    "aws ecs deployment",
    "hybrid retrieval",
    "semantic search system",
    "document embedding",
    "terraform vpc",
    "knowledge base",
    "vector similarity",
]

# ================================
# Helpers
# ================================


def p50(values):
    return statistics.median(values)


def p95(values):
    values_sorted = sorted(values)
    idx = max(0, int(len(values_sorted) * 0.95) - 1)
    return values_sorted[idx]


# ================================
# Single case
# ================================


def run_single_case(top_k: int):
    latencies = []
    errors = 0

    print(f"\n🚀 [Hybrid] top_k={top_k} ... running {RUNS_PER_CASE} times")

    for i in range(RUNS_PER_CASE):
        q = random.choice(QUERIES)

        params = {
            "q": q,
            "top_k": top_k,
            "hybrid": True,
            "vector_k": top_k,
            "bm25_k": top_k,
            "rerank": False,
            "page": 1,
            "page_size": top_k,
            "debug": False,
        }

        try:
            t0 = time.time()
            resp = requests.get(API_URL, params=params, timeout=10)
            wall_elapsed_ms = (time.time() - t0) * 1000.0

            if resp.status_code != 200:
                errors += 1
                print(
                    f"[{i+1:02d}] {q:<35} ERROR status={resp.status_code} "
                    f"({wall_elapsed_ms:7.2f} ms)"
                )
                continue

            data = resp.json()
            # 优先使用服务端的 total latency（毫秒）
            server_latency = None
            if isinstance(data, dict):
                latency_dict = data.get("latency_ms") or {}
                server_latency = latency_dict.get("total")

            latency_ms = (
                float(server_latency) if server_latency is not None else wall_elapsed_ms
            )
            latencies.append(latency_ms)

            print(
                f"[{i+1:02d}] {q:<35} {latency_ms:7.2f} ms "
                f"(status={resp.status_code})"
            )

        except Exception as e:
            errors += 1
            print(f"[{i+1:02d}] ERROR: {e}")

    if not latencies:
        return None

    result = {
        "top_k": top_k,
        "runs": RUNS_PER_CASE,
        "errors": errors,
        "p50_ms": round(p50(latencies), 2),
        "p95_ms": round(p95(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "std_ms": round(statistics.pstdev(latencies), 2),
        "raw_latencies_ms": latencies,
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
    }

    out_file = RESULT_DIR / f"hybrid_ivf_flat_topk_{top_k}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"✅ Saved hybrid result → {out_file}")
    return result


# ================================
# Main
# ================================


def main():
    all_results = []

    for k in TOP_K_LIST:
        res = run_single_case(k)
        if res:
            all_results.append(res)

    if not all_results:
        print("\n⚠️ No hybrid benchmark results collected. Check network / API.")
        return

    ks = [r["top_k"] for r in all_results]
    p50s = [r["p50_ms"] for r in all_results]
    p95s = [r["p95_ms"] for r in all_results]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, p50s, marker="o", label="P50")
    plt.plot(ks, p95s, marker="o", label="P95")
    plt.xlabel("top_k")
    plt.ylabel("Latency (ms)")
    plt.title("Hybrid Retrieval Latency vs top_k (IVF_FLAT)")
    plt.grid(True)
    plt.legend()

    out_path = PLOT_DIR / "hybrid_latency.png"
    plt.savefig(out_path, dpi=160)
    print(f"📈 Saved hybrid plot → {out_path}")

    print("\n🎉 Hybrid benchmark finished.")


if __name__ == "__main__":
    main()
