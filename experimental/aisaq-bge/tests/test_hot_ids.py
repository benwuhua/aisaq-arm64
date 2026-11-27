"""Check the on-disk ranking and raw hot-list contracts without a graph or SSD."""

import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "hot_ids.py"


class HotListTests(unittest.TestCase):
    def extract(self, data, *args, expected_error=None):
        with tempfile.TemporaryDirectory() as directory:
            ranking = Path(directory) / "ranking"
            output = Path(directory) / "hot"
            ranking.write_bytes(data)
            run = subprocess.run(
                [sys.executable, TOOL, ranking, output, *args],
                capture_output=True,
                text=True,
            )
            if expected_error:
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(expected_error, run.stderr)
                self.assertFalse(output.exists())
            else:
                self.assertEqual(run.returncode, 0, run.stderr)
                return output.read_bytes()

    def test_default_native_prefix(self):
        ids = list(reversed(range(2711)))
        data = struct.pack(f"<Q{len(ids)}I", len(ids), *ids)
        self.assertEqual(self.extract(data), data[8 : 8 + 4 * 2656])

    def test_wide_prefix_preserves_high_bits_and_order(self):
        ids = [1, 2**32 + 1, 2**40 - 1, *range(2, 4100)]
        data = struct.pack(f"<Q{len(ids)}Q", len(ids), *ids)
        self.assertEqual(
            self.extract(data, "--layout", "compact", "--id-bytes", "8"), data[8 : 8 + 8 * 4080]
        )

    def test_invalid_wide_rankings(self):
        args = ["--layout", "compact", "--id-bytes", "8"]
        self.extract(struct.pack("<Q2Q", 2, 1, 1), *args, expected_error="Duplicate")
        self.extract(struct.pack("<QQ", 1, 2**40), *args, expected_error="exceeds")
        self.extract(struct.pack("<QI", 1, 23), *args, expected_error="8-byte")
        self.extract(struct.pack("<QQ", 1, 2**32), "--id-bytes", "8", expected_error="native")


if __name__ == "__main__":
    unittest.main()
