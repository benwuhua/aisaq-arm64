"""Integration tests for compact files, full-width IDs, direct I/O and short search results."""

import argparse
import csv
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pack", "search", "pivots"):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    parser.add_argument("--backend", choices=("libaio", "uring"), default="libaio")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--expect-half-read", type=int, choices=(2048, 4096))
    parser.add_argument(
        "--large-offset",
        action="store_true",
        help="use a sparse 40.96 TB fixture on a filesystem supporting this logical size",
    )
    for k in (48, 56):
        parser.add_argument(f"--pivots{k}", type=lambda p: Path(p).resolve())
    parser.add_argument("--rerank-test", type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=args.work) as tmp:
        root = Path(tmp)
        env = dict(os.environ, ANN_IO_BACKEND=args.backend, OMP_NUM_THREADS="2")

        def run(*cmd, overrides=None, success=True):
            rc = subprocess.run(
                [str(x) for x in cmd],
                cwd=root,
                env=env | (overrides or {}),
                capture_output=True,
                text=True,
            )
            if (rc.returncode == 0) != success:
                raise AssertionError(f"{cmd}: {rc.returncode}: {rc.stderr[-2000:]}")

        rows = 13
        first = 2**32 - 2
        vectors = b"".join(struct.pack("<f", i) * 1024 for i in range(rows))
        (root / "raw").write_bytes(vectors)
        (root / "fbin").write_bytes(struct.pack("<II", rows, 1024) + vectors)
        (root / "pq").write_bytes(bytes(rows * 64))
        (root / "q").write_bytes(struct.pack("<II", 1, 1024) + struct.pack("<f", 3.25) * 1024)

        def adjacency(base):
            links = [base + i for i in range(1, rows)]
            records = [struct.pack("<I32Q", len(links), *(links + [2**64 - 1] * (32 - len(links))))]
            records.extend(struct.pack("<I32Q", 0, *([2**64 - 1] * 32)) for _ in links)
            return b"".join(records)

        last_base = 2**40 - rows
        for base, graph in [(first, "wide"), (last_base, "last"), (0, "small")]:
            (root / "edges").write_bytes(adjacency(base))
            run(args.pack, "wide", 64, base, base, "raw", "edges", "pq", graph)
        raw = (root / "wide").read_bytes()
        assert int.from_bytes(raw[4096:4101], "little") == first
        assert int.from_bytes(raw[4096 + 1040 + 5 : 4096 + 1040 + 10], "little") == 2**32
        assert raw[4096 + 1040 + 12 * 5 : 4096 + 1200] == bytes((32 - 12) * 5)
        assert not any(raw[8192 + 1040 : 12288])  # second record has degree zero
        run(args.pack, "fp16-pool", "fbin", "pool")
        run(args.pack, "fp16-pool-raw", "raw", "pool-raw")
        assert (root / "pool").read_bytes() == (root / "pool-raw").read_bytes()
        assert (root / "pool").stat().st_size == 4096 + 7 * 4096
        if args.rerank_test:
            for schedule in ("batch", "rolling"):
                for fmt, pool in [("fp16", "pool"), ("fp32", "raw")]:
                    run(
                        args.rerank_test,
                        pool,
                        overrides={"RR_SCHEDULE": schedule, "RR_FORMAT": fmt, "RR_FIRST_ID": "0"},
                    )
        # Older files have a uint32 ID, two-byte record tag and no ID base.
        legacy = bytearray((root / "small").read_bytes())
        legacy[:64] = bytes(64)
        struct.pack_into("<8sIII", legacy, 0, b"BGELAY01", 64, rows, 0)
        for i in range(rows):
            legacy[(1 + i) * 4096 + 4] = 0x35
        (root / "legacy").write_bytes(legacy)
        # Unused IDs and inline codes in a native source can contain garbage.
        native = bytearray(4096 + 8192 * rows)
        struct.pack_into("<5Q", native, 8, rows, 1024, 0, 6276, 0)
        for i in range(rows):
            start = 4096 + 8192 * i
            native[start : start + 4096] = vectors[i * 4096 : (i + 1) * 4096]
            native[start + 4096 : start + 6276] = b"\xff" * 2180
            degree = 12 if i == 0 else 0
            struct.pack_into("<I", native, start + 4096, degree)
            for j in range(degree):
                struct.pack_into("<I", native, start + 4100 + 4 * j, j + 1)
            native[start + 4228 : start + 4228 + degree * 64] = bytes(degree * 64)
        (root / "native").write_bytes(native)
        (root / "codes.bin").write_bytes(struct.pack("<II", rows, 64) + bytes(rows * 64))
        run(args.pack, 64, "native", "codes.bin", "converted")
        assert (root / "converted").read_bytes() == (root / "small").read_bytes()
        for bad_degree, bad_edge in [(33, 1), (1, rows)]:
            struct.pack_into("<II", native, 8192, bad_degree, bad_edge)
            (root / "invalid").write_bytes(native)
            run(args.pack, 64, "invalid", "codes.bin", "bad", success=False)
            assert not (root / "bad").exists() and not (root / "bad.partial").exists()
        # Search, cutoff, ranking, byte counters and completion accounting via the public runner.
        for i, (graph, base, hot, cap, mode) in enumerate(
            [
                ("wide", first, True, 16, "rolling"),
                ("last", last_base, True, 16, "rolling"),
                ("wide", first, False, 3, "batch"),
                ("legacy", 0, False, 16, "batch"),
                ("converted", 0, True, 16, "rolling"),
            ]
        ):
            out = root / f"r{i}"
            (root / "hot").write_bytes(struct.pack("<2Q", base, base + 3))
            cfg = {
                "ANN_LAYOUT": "sq64",
                "ANN_LAYOUT_FILE": str(root / graph),
                "ANN_LAYOUT_PIVOTS": str(args.pivots),
                "ANN_HOT_IDS": str(root / "hot") if hot else "",
                "ANN_HOT_ID_BYTES": "8",
                "ANN_WIDTH": "9",
                "ANN_THREADS": "1",
                "ANN_SCHEDULE": mode,
                "ANN_WINDOW": str(out) + ".window",
                "RR_WIDTH": "4",
                "RR_C": str(cap),
                "RR_POOL": str(root / "pool"),
                "RR_FORMAT": "fp16",
                "RR_FIRST_ID": str(base),
                "RR_SCHEDULE": mode,
                "RR_OFFSET": "0",
                "RR_STRIDE": "0",
            }
            run(args.search, "search", "q", "unused", 24, out, 0, 0, overrides=cfg)
            got = struct.unpack("<10q", Path(str(out) + "_ids.i64").read_bytes())
            ds = struct.unpack("<10f", Path(str(out) + "_distances.f32").read_bytes())
            expected = sorted(range(rows), key=lambda j: (abs(j - 3.25), j))[: min(cap, 10)]
            assert list(got) == [base + j for j in expected] + [-1] * (10 - len(expected))
            assert all(math.isinf(v) for v in ds[len(expected) :])
            with open(str(out) + "_queries.csv") as f:
                s = next(csv.DictReader(f))
            with open(str(out) + ".schedule.csv") as f:
                sched = next(csv.DictReader(f))
            assert int(s["cache_hits"]) == (2 if hot else 0)
            assert int(s["io_requests"]) + int(s["cache_hits"]) == rows
            assert int(s["rerank_reads"]) == min(cap, rows) and int(s["rerank_peak"]) <= 4
            read_size = int(s["rerank_bytes"]) // min(cap, rows)
            assert read_size in (2048, 4096) and int(s["rerank_bytes"]) == read_size * min(cap, rows)
            if args.expect_half_read:
                assert read_size == args.expect_half_read
            assert int(sched["submitted"]) == int(sched["reaped"]) and 6 < int(sched["peak_depth"]) <= 9
        if args.large_offset:
            # Only the three reachable test records and header are allocated. These reads exercise
            # the real 64-bit file offset, not just the nonzero logical ID base.
            total = 10_000_000_000
            chain = [0, 2**32 + 3, total - 1]
            sparse = root / "large"
            with sparse.open("wb") as f:
                f.truncate((total + 1) * 4096)
                header = bytearray(raw[:4096])
                struct.pack_into("<QQQ", header, 16, total, 0, 0)
                f.seek(0)
                f.write(header)
                for j, node_id in enumerate(chain):
                    record = bytearray(4096)
                    record[:5] = node_id.to_bytes(5, "little")
                    record[5:8] = bytes([0x4B, int(j + 1 < len(chain)), 64])
                    struct.pack_into("<ff", record, 8, float(j), 0.0)
                    if j + 1 < len(chain):
                        record[1040:1045] = chain[j + 1].to_bytes(5, "little")
                    f.seek((node_id + 1) * 4096)
                    f.write(record)
            out = root / "large-result"
            cfg.update(
                ANN_LAYOUT="sq64",
                ANN_LAYOUT_FILE=str(sparse),
                ANN_LAYOUT_PIVOTS=str(args.pivots),
                ANN_HOT_IDS="",
                ANN_WINDOW=str(out) + ".window",
                RR_C="0",
            )
            run(args.search, "search", "q", "unused", 24, out, 0, 0, overrides=cfg)
            got = struct.unpack("<10q", Path(str(out) + "_ids.i64").read_bytes())
            assert got == tuple(reversed(chain)) + (-1,) * 7
        for k in (48, 56):
            pivots = getattr(args, f"pivots{k}")
            if not pivots:
                continue
            (root / "edges").write_bytes(adjacency(first))
            (root / "pq-alt").write_bytes(bytes(k * rows))
            run(args.pack, "wide", k, first, first, "raw", "edges", "pq-alt", f"fp{k}")
            out = root / f"fp{k}-result"
            (root / "hot").write_bytes(struct.pack("<2Q", first, first + 3))
            cfg.update(
                ANN_LAYOUT=f"fp{k}",
                ANN_LAYOUT_FILE=str(root / f"fp{k}"),
                ANN_LAYOUT_PIVOTS=str(pivots),
                ANN_HOT_IDS=str(root / "hot"),
                ANN_HOT_ID_BYTES="8",
                ANN_WINDOW=str(out) + ".window",
                RR_C="0",
            )
            run(args.search, "search", "q", "unused", 24, out, 0, 0, overrides=cfg)
            got = struct.unpack("<10q", Path(str(out) + "_ids.i64").read_bytes())
            assert list(got) == [
                first + j for j in sorted(range(rows), key=lambda j: abs(j - 3.25))[:10]
            ]
    print(
        f"{args.backend}: format, high-ID traversal, hot/cold parity, native conversion and rerank passed"
    )


if __name__ == "__main__":
    main()
