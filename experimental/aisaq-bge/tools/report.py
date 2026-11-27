"""Score an AiSAQ run against a uint32/float32 Big ANN ground-truth file."""
import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="OUTPUT from build/search")
    parser.add_argument("truth", type=Path)
    parser.add_argument("--window", type=Path, help="ANN_WINDOW, default OUTPUT.window")
    args = parser.parse_args()
    with args.truth.open("rb") as truth:
        rows, k = map(int, np.fromfile(truth, "<u4", 2))
        ids = np.fromfile(truth, "<u4", rows * k).reshape(rows, k)
        distances = np.fromfile(truth, "<f4", rows * k).reshape(rows, k)
        if k < 10 or truth.read(1):
            parser.error("Expected top-10 or deeper Big ANN truth")
    results = np.fromfile(args.prefix + "_ids.i64", "<i8").reshape(rows, 10)
    hits = sum(len(set(result).intersection(row[dist <= dist[9]]))
               for result, row, dist in zip(results, ids, distances))
    with open(args.prefix + "_queries.csv") as f:
        queries = list(csv.DictReader(f))
    if len(queries) != rows:
        parser.error("Query count differs from ground truth")
    window = args.window or Path(args.prefix + ".window")
    counters = np.loadtxt(window, dtype=np.int64, ndmin=2)
    if counters.shape[0] != 2:
        parser.error("Each run must have a fresh ANN_WINDOW file with two snapshots")
    delta = counters[1] - counters[0]
    graph_requests = sum(int(row["io_requests"]) for row in queries)
    cache_hits = sum(int(row["cache_hits"]) for row in queries)
    rerank_reads = sum(int(row["rerank_reads"]) for row in queries)
    print(f"Recall@10: {100 * hits / (10 * rows):.2f}%")
    print(f"Hot-cache hits / expanded nodes: {cache_hits}/{graph_requests + cache_hits}")
    print(f"Device KiB/query (includes rerank): {delta[3] * 0.5 / rows:.3f}")
    print(f"FP32 rerank KiB/query: {rerank_reads * 4 / rows:.3f}")


if __name__ == "__main__":
    main()
