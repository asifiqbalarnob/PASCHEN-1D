from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.normalize_lxcat_ion_data import parse_lxcat_file, source_family


class LXCatNormalizationTests(unittest.TestCase):
    def test_group_comments_stay_attached_to_their_data_blocks(self) -> None:
        raw = """DATABASE: Viehland database
PERMLINK: www.lxcat.net/Viehland
HOW TO REFERENCE: Example citation.
COMMENT: Raw diffusion Data from Example Laboratory.
SPECIES: D3^+ / D2
PROCESS: Diffusion x gas density (NDz)
PARAM.: Tgas = 300 K
UPDATED: 2026-01-01
COLUMNS: Reduced electric field (Td) | Diffusion x gas density (1E20/ms)
-----------------------------
5.0 5.0
10.0 6.0
-----------------------------
COMMENT: Raw mobility Data from Example Laboratory.
SPECIES: D3^+ / D2
PROCESS: Mobility (Ko)
PARAM.: Tgas = 300 K
UPDATED: 2026-01-01
COLUMNS: Reduced electric field (Td) | Mobility (cm2/Vs)
-----------------------------
5.0 8.0
10.0 8.1
-----------------------------
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.txt"
            path.write_text(raw, encoding="utf-8")
            diffusion, mobility = parse_lxcat_file(path, root)

        self.assertIn("diffusion", diffusion.group_comment)
        self.assertIn("mobility", mobility.group_comment)
        self.assertEqual(
            source_family(diffusion.group_comment),
            source_family(mobility.group_comment),
        )


if __name__ == "__main__":
    unittest.main()
