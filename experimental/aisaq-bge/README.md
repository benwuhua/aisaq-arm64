# AiSAQ scheduling and disk layout on BGE-M3-10M

This standalone benchmark extends the full-inline AiSAQ search in Knowhere. It compares waiting for a batch with expanding a completed read and refilling its slot, and supports three experimental 4 KiB record formats. The query scheduler, cache, packer, and reranker used by the experiments live in this directory. The source shape and measured results are documented in [the benchmark reference](reference/README.md).

## Build on ARM64 Linux

Use GCC with C++17, ARMv8.2-A FP16/NEON, OpenMP and Linux libaio development headers. The runner uses the internal C++ ABI of Knowhere. The tested combination is this Knowhere v2.6.22 source tree, the matching ARM64 `libknowhere.so` and runtime libraries from the Milvus 2.6.26 distribution, and Milvus common v2.6.9 headers. Boost and nlohmann/json headers must also be available. `src/scheduler.cpp` includes the modified AiSAQ translation unit derived from v2.6.22's `thirdparty/DiskANN/src/pq_flash_aisaq_index.cpp`. Its executable definitions override that search implementation in the shared library. The package still supplies the builder, reader, PQ training and other Knowhere internals. The build creates standalone executables, leaving the Milvus service binary untouched.

From this directory, set the paths for your installation.

```bash
export KNOWHERE_LIB_DIR="$MILVUS_INSTALL/lib/milvus"
export EXTRA_INCLUDE_DIRS="$MILVUS_COMMON_SRC/include"
# Append :DIR entries for Boost and nlohmann/json if they are outside compiler search paths.
export LD_LIBRARY_PATH="$KNOWHERE_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./build.sh
```

`KNOWHERE_ROOT` defaults to the repository root. Override it if building this directory outside that checkout. `BUILD_DIR` defaults to `./build`. `build/search`, `build/pack` and `build/pq` are the generated executables. The `compat/` headers provide the minimal declarations and logging adapter used by the standalone harness.

## Native graph, cache and scheduler

Start from a little-endian FP32 BGE `.fbin` base and query split. Set `BASE`, `QUERIES`, `TRUTH`, `PREFIX` and `OUT` to the base matrix, query matrix, Big ANN ground truth, common native index prefix and a **new output prefix for each run**, respectively. Create their parent directories. The reference reused a pre-existing full-inline graph. To create a new AiSAQ graph with the builder:

```bash
export OPENBLAS_NUM_THREADS=1
./build/search build "$BASE" "$PREFIX" 32 32 24 0
```

The last arguments set 32 inline neighbor PQ codes, 32 build threads, a 24 GiB build budget and disabled record rearrangement. The builder uses degree 32, construction search list 100, L2, and 64-byte PQ codes. Retain all the generated index files under one prefix. Topology, entry and trained codebooks must match across the compared runs. A fresh build can produce a different graph. The 4 KiB packer expects the exact BGE record shape described below.

```bash
export ANN_GRAPH_FILE="${PREFIX}_disk.index"
export ANN_BLOCK_DEVICE=nvme0n1  # select the block device hosting the graph
export ANN_THREADS=4 ANN_WIDTH=2 ANN_SCHEDULE=batch RR_C=0
export ANN_WINDOW="${OUT}.window"
./tools/run-capped.sh ./build/search search "$QUERIES" "$PREFIX" 405 "$OUT" 0 0
python3 tools/report.py "$OUT" "$TRUTH"
```

The wrapper needs a writable memory cgroup with memory and swap accounting. It applies the 64 MiB limit before index initialization. Set `CGROUP_MEMORY_ROOT` for a delegated cgroup if necessary. Apply the same CPU affinity to each run. The two trailing zeros disable native PQ page and code caches. The full-inline FP32 graph supplies both the neighbor codes and the full vector for each expansion.

For the native cache treatment, create the raw ranked-ID list with `python3 tools/hot_ids.py "$ALAYALASER_CACHE_RANKING" "$HOT_IDS"`, where the input is the matching graph's ranked `.index_cache_ids` file. Set `ANN_HOT_IDS="$HOT_IDS"` to preload a shared 16 MiB arena holding 2,656 complete nodes plus lookup metadata. Unset `ANN_HOT_IDS` for zero static cache. The ranking belongs to the graph's ID space. The load happens before the measurement window.

Set `ANN_SCHEDULE=rolling` for completion-driven refill. The native sweeps used `ANN_WIDTH=2, 3, 4, 6` in separate runs and `ANN_THREADS=4` or `32`. Each width bounds graph reads outstanding per query. Set `ANN_SCHED_PROFILE=1` on a separate profiling run to collect a histogram of differences between userspace-observed completions. It does not timestamp SSD firmware completions. Choose `L` separately for each configuration to match the target recall.

## Repack the same graph and rerank

The source graph has a 4 KiB header and one 8 KiB on-disk slot per node. Its 6,276-byte payload holds 1,024 FP32 coordinates, a uint32 degree, 32 uint32 neighbor IDs and 32 inline 64-byte neighbor codes. The packer requires exactly 32 neighbors per source record. Each compact file has a 4 KiB header followed by one 4 KiB record per node. Edges and neighbor order are retained. The experimental IDs occupy five on-disk bytes, with the high byte zero. The prototype retains the upstream 32-bit node-ID limit.

For SQ8 with original 64-byte neighbor codes:

```bash
./build/pack 64 "${PREFIX}_disk.index" "${PREFIX}_pq_compressed.bin" "$SQ_GRAPH"
```

For FP16 and retrained 48- or 56-byte neighbor codes, reuse one raw little-endian uint32 file of training IDs for both widths. Use the original IDs for exact training-set provenance. To draw a new sample, use `./build/pq sample "$BASE" "$TRAIN_IDS"`. The sampling routine targets about 256,000 rows on the 10M base. Run the following for each width; select independent `PQ_PREFIX` and `PACKED_GRAPH` paths.

```bash
WIDTH=48
./build/pq train "$BASE" "$PQ_PREFIX" "$WIDTH" "$TRAIN_IDS"
./build/pq encode "$BASE" "$PQ_PREFIX" "$WIDTH"
./build/pack "$WIDTH" "${PREFIX}_disk.index" "${PQ_PREFIX}_pq_compressed.bin" "$PACKED_GRAPH"
```

Train and pack outside the search cgroup. Allocate space for a full compact graph (38.15 GiB for 10M rows), alternate PQ files if applicable, and the native graph retained for reranking. A completed pack is written to the named output by renaming a sibling `.partial` file. Choose a distinct output path per format.

To search a compact graph, keep the **native** index prefix on the command line. Its native artifacts still supply the graph geometry, metadata and original codebook for SQ8. The custom reader uses the packed graph for traversal. Exact reranking reads one aligned 4 KiB FP32 vector for each candidate from the original graph.

```bash
unset ANN_HOT_IDS
export ANN_SCHEDULE=rolling ANN_WIDTH=2 ANN_THREADS=4
export ANN_LAYOUT=fp48 ANN_LAYOUT_FILE="$PACKED_GRAPH"
export ANN_LAYOUT_PIVOTS="${PQ_PREFIX}_pq_pivots.bin"
export RR_C=16 RR_WIDTH=2 RR_OFFSET=4096 RR_STRIDE=8192
export RR_POOL="${PREFIX}_disk.index"
export ANN_WINDOW="${OUT}.window"
./tools/run-capped.sh ./build/search search "$QUERIES" "$PREFIX" "$L" "$OUT" 0 0
python3 tools/report.py "$OUT" "$TRUTH"
```

Use `fp56` and its own codebook for the 56-byte layout, or `sq64` and `ANN_LAYOUT_FILE="$SQ_GRAPH"` for the SQ8 layout. SQ8 uses its original 64-byte codebook. To return to the native format, unset `ANN_LAYOUT`, `ANN_LAYOUT_FILE`, and `ANN_LAYOUT_PIVOTS`, and restore `RR_C=0`.

## Reading a run

The program reports QPS and nearest-rank P50/P99 on standard output. It writes raw top-ten IDs and distances, `_queries.csv` with per-query latency, graph reads, cache hits and rerank reads, and `.schedule.csv` with actual submissions, completions and peak depth. The window file contains pre- and post-search block-device counters. `tools/report.py` requires NumPy and computes ties-aware Recall@10 and KiB/query, including rereads. Device counters include unrelated traffic on the same device, so keep it idle during measurement. Use [the reference configurations](reference/README.md) for the actual matched-recall depths and repeat counts.
