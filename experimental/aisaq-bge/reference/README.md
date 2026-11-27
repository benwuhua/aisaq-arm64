# BGE-M3-10M benchmark reference

The reference runs used one ARM64 server and one NVMe SSD, with four or 32 query workers. All rows return ten results per query. The search-list size `L` was calibrated against a target of 95.11% Recall@10. The CSVs include the observed recall range because completion order can change the search path.

| Experiment | Inputs held fixed | Results |
| --- | --- | --- |
| Outstanding read window | Original 8 KiB AiSAQ-full records, degree 32, 64-byte PQ code per neighbor, same entry node. The batch control used no static cache. All rolling rows shared one 16 MiB static cache. | [Five rounds per configuration](read-window.csv) |
| Record layout | Same base vectors, edges, neighbor order, entry node and rolling policy. All rows used zero shared cache. Compact layouts reranked 16 candidates with exact FP32 distances. | [Three rounds per configuration](record-layout.csv) |

Compare rolling widths within the first table to isolate the window effect. Comparing its batch control with its rolling rows includes both scheduling and caching. In the second table, `base` denotes FP32 with 64-byte neighbor PQ codes. `fp48` and `fp56` store FP16 vectors with separately trained, full-dimensional 48- and 56-byte PQ codes. `sq64` stores SQ8 vectors and retains the original 64-byte navigation codes. The compact records all use five bytes per neighbor ID.

## Recreating the inputs

The data source is the [fixed revision of Upstash's English Wikipedia BGE-M3 embeddings](https://huggingface.co/datasets/Upstash/wikipedia-2024-06-bge-m3/tree/ce4e0ea49276d99975816b8bc85a85bf416b8b62/data/en). The experiment used its `embedding` column. To reproduce the subset, sort the 471 English Parquet files by name and use the two draws below in the indicated order. They require NumPy.

```python
import numpy as np

rng = np.random.default_rng(20261120)
shard_order = rng.permutation(471)
query_rows = np.sort(rng.choice(10_001_000, size=1_000, replace=False))
```

Read the first 10,001,000 embeddings in that shard order, preserving row order within each file. This consumes 101 shards and ends within the last shard. Convert each input row to float32 and divide by its L2 norm. Write the 1,000 selected rows as queries in ascending input order and the remaining 10,000,000 rows as the base. Both matrices have 1,024 columns. Each `.fbin` starts with two little-endian uint32 values (rows and columns) followed by row-major little-endian float32 values. Use these SHA-256 digests to check the split.

| Matrix | SHA-256 |
| --- | --- |
| Base | `b6a3e3d799120839e2ef70a4fcd71b3255ddc7651c8dbbf035031b359c7241b3` |
| Query | `3d49a4bed89b240e0d561e183b9c41aaa4e445e96bf276b24e6befa279d87112` |

Use exhaustive squared-L2 search against this base for the top-100 ground truth. The indexed graph, its codebooks, and the static cache list are separate benchmark inputs. Reuse a single full-inline AiSAQ graph across treatments. The original static cache held full node records and their lookup metadata for the first 2,656 nodes of a fixed descending in-degree ranking. Ties followed the archived ranking's order. Choosing a different graph or tie order can change hit rate.

## Timing and counters

1. Apply the memory limit to the process **before** graph initialization and static-cache loading. The reference Linux cgroup v1 had a 67,108,864-byte limit for both memory and memory-plus-swap and zero swappiness. The calling harness must set and check the limit, choose the same worker CPU set across treatments, and monitor the block device hosting the index. Index construction and PQ training run outside the search memory limit.
2. The search and exact vector rereads use aligned direct I/O. Warm up the first `max(32, 2 × workers)` queries, then time all 1,000 queries. The measured wall time includes query execution and exact reranking. Index construction, cache preload and result-file export are outside the window. Interleave configurations and keep unrelated traffic off the measured device.
3. For each query, check that graph reads submitted equal completions reaped and that peak outstanding reads are within the configured limit. Here outstanding means submitted and awaiting collection. The exact reranking stage starts after the graph reads drain, submitting at most two FP32 reads at once.
4. Report throughput as completed queries divided by wall-clock seconds. Report the median QPS, read volume and Recall@10 across runs. Pool query latencies across all repeats before taking P99 with the nearest-rank rule. Compute Recall@10 from unique returned IDs intersecting the ground-truth IDs within the tenth-neighbor distance, divided by ten. Cache hit percentage is total expanded-node cache hits divided by total graph reads plus those hits.
5. Device KiB/query comes from the change in the SSD's sectors-read counter over the timed window, multiplied by 512 and divided by 1,024 and by query count. Compare that result with the process storage-read counter and the requested graph and rerank bytes. The compact runs charge an additional 16 direct 4 KiB FP32 rereads per query, or 64 KiB/query. Total device traffic can differ slightly from the sum of the application byte counters.

A 4 KiB compact graph consumes one 4 KiB page per node. The reference reranker read the original vector from the first 4 KiB of each 8 KiB source record. If a deployment retains a separate dense FP32 vector pool for exact reranking, account for another 4 KiB per vector.
