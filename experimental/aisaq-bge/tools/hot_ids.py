"""Extract the shared cache prefix from a count-prefixed, little-endian ranked ID file."""
import argparse
import os
import struct
from pathlib import Path

CAPACITY = {"native": 2656, "compact": 4080}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranking", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layout", choices=CAPACITY, default="native")
    parser.add_argument("--id-bytes", type=int, choices=(4, 8), default=4,
                        help="width of IDs in both the ranking and the raw output (default: 4)")
    args = parser.parse_args()
    item = struct.Struct("<I" if args.id_bytes == 4 else "<Q")
    with args.ranking.open("rb") as source:
        header = source.read(8)
        if len(header) != 8:
            parser.error("Truncated ranking count")
        total, = struct.unpack("<Q", header)
        if not total or os.fstat(source.fileno()).st_size != 8 + item.size * total:
            parser.error(f"Expected a uint64 count followed by {args.id_bytes}-byte IDs")
        count = min(total, CAPACITY[args.layout])
        payload = source.read(item.size * count)
    if len(payload) != item.size * count:
        parser.error("Truncated hot prefix")
    ids = {value for (value,) in item.iter_unpack(payload)}
    if len(ids) != count:
        parser.error("Duplicate node ID in hot prefix")
    limit = 1 << (40 if args.layout == "compact" else 32)
    if max(ids) >= limit:
        parser.error(f"A selected node ID exceeds the {args.layout} format")
    with args.output.open("xb") as output:
        output.write(payload)


if __name__ == "__main__":
    main()
