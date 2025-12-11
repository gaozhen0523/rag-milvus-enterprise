#!/usr/bin/env python3
"""
Benchmark: Vector-only retrieval latency (online ECS)

This script calls the /query endpoint with:
  - hybrid = false
  - top_k = 5 / 10 / 20
  - 30 runs each

Outputs:
  benchmarks/ai/results/vector_ivf_flat_topk_{k}.json
  benchmarks/ai/plots/vector_latency.png
"""

import os
import time
import json
import statistics
import random
import requests
import matplotlib.pyplot as plt
from pathlib import Path

# ================================
# Configurable environment
# ================================

API_URL = os.getenv(
    "RAG_API_URL",
    "http://rag-api-gateway-alb-804929386.us-east-1.elb.amazonaws.com/query",
)

RUNS_PER_CASE = int(os.getenv("RUNS", "30"))
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
# Helper
# ================================


def p50(values):
    return statistics.median(values)


def p95(values):
    values_sorted = sorted(values)
    idx = max(0, int(len(values_sorted) * 0.95) - 1)
    return values_sorted[idx]


# ================================
# Single benchmark run
# ================================


def run_single_case(top_k: int):
    latencies = []
    errors = 0

    print(f"\n🚀 [Vector] top_k={top_k} ... running {RUNS_PER_CASE} times")

    for i in range(RUNS_PER_CASE):
        q = random.choice(QUERIES)

        try:
            t0 = time.time()
            resp = requests.get(
                API_URL,
                params={"q": q, "top_k": top_k, "hybrid": False},
                timeout=10,
            )
            elapsed = (time.time() - t0) * 1000  # ms
            latencies.append(elapsed)

            print(f"[{i+1:02d}] {q:<35} {elapsed:7.2f} ms  status={resp.status_code}")

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
    }

    out_file = RESULT_DIR / f"vector_ivf_flat_topk_{top_k}.json"
    out_file.write_text(json.dumps(result, indent=2))

    print(f"✅ Saved result → {out_file}")
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
        print("\n⚠️ No benchmark results collected. Check network / API.")
        return

    # -------- Plot --------
    ks = [r["top_k"] for r in all_results]
    p50s = [r["p50_ms"] for r in all_results]
    p95s = [r["p95_ms"] for r in all_results]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, p50s, marker="o", label="P50")
    plt.plot(ks, p95s, marker="o", label="P95")
    plt.xlabel("top_k")
    plt.ylabel("Latency (ms)")
    plt.title("Vector Retrieval Latency vs top_k (IVF_FLAT)")
    plt.grid(True)
    plt.legend()

    out_path = PLOT_DIR / "vector_latency.png"
    plt.savefig(out_path, dpi=160)
    print(f"📈 Saved plot → {out_path}")

    print("\n🎉 Vector benchmark finished.")


if __name__ == "__main__":
    main()
