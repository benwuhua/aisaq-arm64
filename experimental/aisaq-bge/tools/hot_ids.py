"""Extract the shared 16 MiB cache prefix from AlayaLaser's ranked cache-ID file."""
import argparse
import struct
from pathlib import Path

COUNT = 2656


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranking", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.ranking.open("rb") as source:
        header = source.read(8)
        if len(header) != 8:
            parser.error("Truncated AlayaLaser count")
        total, = struct.unpack("<Q", header)
        if total < COUNT or args.ranking.stat().st_size != 8 + 4 * total:
            parser.error("Expected a 64-bit count followed by little-endian uint32 node IDs")
        payload = source.read(4 * COUNT)
    if len(set(struct.unpack(f"<{COUNT}I", payload))) != COUNT:
        parser.error("Duplicate node ID in hot prefix")
    with args.output.open("xb") as output:
        output.write(payload)


if __name__ == "__main__":
    main()
