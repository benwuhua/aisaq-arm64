# AiSAQ scheduling and disk layout on BGE-M3-10M

This standalone benchmark extends the full-inline AiSAQ search in Knowhere. It compares waiting for a batch with expanding a completed read and refilling its slot, and supports three experimental 4 KiB record formats. The query scheduler, cache, packer, and reranker used by the experiments live in this directory. The source shape and measured results are documented in [the benchmark reference](reference/README.md).

## Build on ARM64 Linux

Use GCC with C++17, ARMv8.2-A FP16/NEON, OpenMP and Linux libaio development headers. Install liburing development files and pkg-config to enable the optional io_uring backend. The build selects `armv8.2-a+fp16+dotprod` when the host advertises both extensions. Set `AISAQ_MARCH` when cross-compiling or `AISAQ_URING=off` for a libaio-only build. The runner uses the internal C++ ABI of Knowhere. The tested combination is this Knowhere v2.6.22 source tree, the matching ARM64 `libknowhere.so` and runtime libraries from the Milvus 2.6.26 distribution, and Milvus common v2.6.9 headers. Boost and nlohmann/json headers must also be available. `src/scheduler.cpp` includes the modified AiSAQ translation unit derived from v2.6.22's `thirdparty/DiskANN/src/pq_flash_aisaq_index.cpp`. The native search specialization uses the package’s entry selection, PQ table and FP32 distance function. The compact reader loads the matching PQ table directly and uses a 64-bit search frontier. The package still supplies the builder, reader, PQ training and other Knowhere internals. The build creates standalone executables, leaving the Milvus service binary untouched.

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

For the native cache treatment, create the raw ranked-ID list with `python3 tools/hot_ids.py "$ALAYALASER_CACHE_RANKING" "$HOT_IDS"`, where the input is the matching graph's ranked `.index_cache_ids` file. Set `ANN_HOT_IDS="$HOT_IDS"` to preload a shared 16 MiB arena holding 2,656 complete nodes plus lookup metadata. Unset `ANN_HOT_IDS` for zero static cache. The ranking belongs to the graph's ID space. The load happens before the measurement window. For compact graphs, pass `--layout compact` to the ID extractor. The same 16 MiB arena holds 4,080 complete 4 KiB pages and a 64 KiB lookup table. Shorter lists populate the supplied prefix. The ranking starts with a little-endian uint64 count. By default the extractor reads and writes 4-byte IDs. For a ranking with 8-byte IDs, pass `--layout compact --id-bytes 8` and set `ANN_HOT_ID_BYTES=8` for search. The extractor preserves the selected IDs and their order.

Set `ANN_SCHEDULE=rolling` for completion-driven refill. The native sweeps used `ANN_WIDTH=2, 3, 4, 6` in separate runs and `ANN_THREADS=4` or `32`. Each width bounds graph reads outstanding per query. `ANN_WIDTH` accepts 1 through 256 and allocates that many aligned slots per active worker. Memory consumption increases with width. Select `ANN_IO_BACKEND=libaio` (default) or `uring` to use the same scheduler with a different completion backend. Rolling mode checks for additional visible completions even when it already has a ready node. With io_uring this check reads the mapped completion ring. Ordinary submission and blocking waits still use the kernel. `ANN_SQPOLL=1` enables kernel submission polling for io_uring and consumes additional CPU. Each active query worker owns a graph ring, and each enabled rerank worker owns another ring. With 32 workers this can create up to 32 submission-polling threads for graph-only search or 64 with reranking. For a backend comparison use `ANN_SQPOLL=0` on the same CPU set. Measure SQPOLL separately and record the pollers’ allowed CPUs and CPU time as well as the query workers’ affinity. Set `ANN_SCHED_PROFILE=1` on a separate batch run at width 2 to collect a histogram of differences between userspace-observed completions. It does not timestamp SSD firmware completions. Choose `L` separately for each configuration to match the target recall.

## Repack the same graph and rerank

The source graph has a 4 KiB header and one 8 KiB on-disk slot per node. Its 6,276-byte payload holds 1,024 FP32 coordinates, a uint32 degree, 32 uint32 neighbor IDs and 32 inline 64-byte neighbor codes. The packer preserves the actual outdegree from 0 through 32. Unused edge and PQ slots are zeroed. Each compact file starts with a 4 KiB header and stores one 4 KiB record per node. Neighbor order is preserved. SQ8 and FP16 coordinates are scored in NEON registers with FP32 accumulation.

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

For compact search the positional prefix supplies the default SQ64 PQ pivot filename. With `ANN_LAYOUT_PIVOTS` set, the runner only needs the packed graph and the corresponding Knowhere pivot files. The following example uses the original native graph as an FP32 rerank pool.

```bash
# To enable the compact hot cache, export ANN_HOT_IDS="$COMPACT_HOT_IDS"
export ANN_SCHEDULE=rolling ANN_WIDTH=2 ANN_THREADS=4
export ANN_LAYOUT=fp48 ANN_LAYOUT_FILE="$PACKED_GRAPH"
export ANN_LAYOUT_PIVOTS="${PQ_PREFIX}_pq_pivots.bin"
export RR_C=16 RR_WIDTH=2 RR_SCHEDULE=rolling RR_FORMAT=fp32 RR_OFFSET=4096 RR_STRIDE=8192
export RR_POOL="${PREFIX}_disk.index"
export ANN_WINDOW="${OUT}.window"
./tools/run-capped.sh ./build/search search "$QUERIES" "$PREFIX" "$L" "$OUT" 0 0
python3 tools/report.py "$OUT" "$TRUTH"
```

Use `fp56` and its own codebook for the 56-byte layout, or `sq64` and `ANN_LAYOUT_FILE="$SQ_GRAPH"` for the SQ8 layout. SQ8 uses its original 64-byte codebook. To return to the native format, unset `ANN_LAYOUT`, `ANN_LAYOUT_FILE`, `ANN_LAYOUT_PIVOTS` and `ANN_HOT_IDS`, and restore `RR_C=0`.

## Compact graph IDs and interchange

The new little-endian `BGELAY02` header stores PQ width at byte 8 (uint32), row count at 16, entry ID at 24 and first ID at 32 (each uint64). The entry PQ code starts at byte 64. IDs span a contiguous interval beginning at the first ID. A node's file offset is `4096 * (1 + id - first)`. Each record starts with its five-byte ID, tag `0x4b`, degree and PQ width. The vector starts at byte 16. Next come 32 five-byte neighbor slots, then 32 fixed-width code slots. Both node and neighbor IDs can use all 40 bits. Existing `BGELAY01` files remain readable.

To import adjacency produced by a wide-ID graph builder, use the following command.

```bash
./build/pack wide 64 "$FIRST_ID" "$ENTRY_ID" rows.fp32 edges.u64 codes.raw graph.4k
```

All three streams are headerless and in row order. Each vector is 1,024 little-endian FP32 values. Each adjacency record is a uint32 degree and 32 little-endian uint64 **absolute** neighbor IDs. Each PQ row is exactly the specified width in bytes. Row count comes from the 64-bit vector file size. The packer validates these streams before publishing a new output. The bundled native Knowhere graph builder and its PQ-compressed input format still use 32-bit IDs. A ten-billion-row compact output requires 40.96 TB and a filesystem that supports a single file of that size.

## Rerank pools and scheduling

`RR_C` is the maximum number of graph candidates read per query. Values from 1 through L are supported, and zero disables reranking. Only returned candidates are read and ranked. If fewer than ten exist, remaining IDs are −1 with infinite distance. `RR_WIDTH` bounds simultaneous rerank reads independently of the graph window. `RR_SCHEDULE=rolling` (default) scores each harvested completion and refills free slots. Use `batch` to wait for an entire group. Both policies start after traversal has finalized the candidate list.

For an FP16 rerank pool, first materialize the complete row-major file.

```bash
./build/pack fp16-pool "$BASE" "$HALF_POOL"
export RR_POOL="$HALF_POOL" RR_FORMAT=fp16 RR_C=16 RR_WIDTH=2
unset RR_OFFSET RR_STRIDE
```

`pack fp16-pool-raw` accepts headerless FP32 input and derives a 64-bit row count. The FP16 file contains a 4 KiB header, two 2,048-byte rows per page, and end padding for an odd row count. `RR_FIRST_ID` sets the logical ID of pool row zero. The runner probes O_DIRECT at a 2 KiB length and offset before timing. With 2 KiB support it reads individual rows. On a 4 KiB-only filesystem it reads the containing page and selects the proper half. FP16 distance uses FP32 accumulation against the stored FP16 coordinates. Use `RR_FORMAT=fp32` for original-vector distances.

## Reading a run

The program reports QPS and nearest-rank P50/P99 on standard output. It writes raw top-ten IDs and distances, `_queries.csv` with per-query latency, graph reads, cache hits and rerank reads, and `.schedule.csv` with actual submissions, completions and peak depth. The per-query rerank columns include submitted bytes and peak depth. In the scheduler CSV, `getevents_calls` counts calls to the backend completion collector, including userspace CQ polls. `submit_calls` counts calls into the submission API. Use a separate syscall trace to measure actual kernel entries. The window file contains pre- and post-search block-device counters. `tools/report.py` requires NumPy and computes ties-aware Recall@10 and KiB/query, including rereads. Device counters include unrelated traffic on the same device, so keep it idle during measurement. Use [the reference configurations](reference/README.md) for the actual matched-recall depths and repeat counts.


## Regression checks on ARM64 Linux

Use the configured build and runtime paths for these checks.

```bash
python3 tests/test_hot_ids.py
g++ -std=c++17 -O2 -Isrc -march=armv8.2-a -DAISAQ_HAVE_URING tests/rerank.cpp -luring -laio -o build/test-rerank
for backend in libaio uring; do
    python3 tests/regression.py build/pack build/search "${PREFIX}_pq_pivots.bin" \
        --backend "$backend" --rerank-test build/test-rerank
done
```

Tests use short synthetic graphs with variable degree, boundary-crossing IDs, legacy records, cache hits and misses, and successive empty and short result lists. Add `--pivots48 FILE --pivots56 FILE` for the FP16 graph variants. `--work /tmp --large-offset` also follows edges across 16 TiB and to row 9,999,999,999 of a sparse 40.96 TB file. Use a filesystem with a sufficient maximum file offset for this fixture. To compare fused and materialized L2 kernels, build `tests/kernel_bench.cpp` with the same ARM target and pin it to a single core.
