from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_paths import (
    resolve_electron_swarm_data_file,
    resolve_ion_swarm_data_file,
)


class SwarmDataPathTests(unittest.TestCase):
    def test_basename_and_nested_relative_path_resolve_inside_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "release" / "tables"
            nested.mkdir(parents=True)
            table = nested / "unique_table.csv"
            table.write_text("test", encoding="utf-8")

            with patch("data_paths.ION_SWARM_DATA_DIR", root):
                self.assertEqual(resolve_ion_swarm_data_file(table.name), table.resolve())
                self.assertEqual(
                    resolve_ion_swarm_data_file("release/tables/unique_table.csv"),
                    table.resolve(),
                )

    def test_absolute_and_parent_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("data_paths.ELECTRON_SWARM_DATA_DIR", root):
                with self.assertRaisesRegex(ValueError, "relative"):
                    resolve_electron_swarm_data_file(root / "table.dat")
                with self.assertRaisesRegex(ValueError, "cannot leave"):
                    resolve_electron_swarm_data_file("../table.dat")

    def test_duplicate_basename_requires_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subfolder in ("one", "two"):
                target = root / subfolder
                target.mkdir()
                (target / "duplicate.csv").write_text("test", encoding="utf-8")

            with patch("data_paths.ION_SWARM_DATA_DIR", root):
                with self.assertRaisesRegex(ValueError, "Ambiguous"):
                    resolve_ion_swarm_data_file("duplicate.csv")


if __name__ == "__main__":
    unittest.main()
