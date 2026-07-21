from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data_paths
from config import SimulationConfig
from electron_transport import REQUIRED_SECTIONS, load_electron_swarm_table
from ion_transport import load_ion_transport_table
from tools.build_release_archive import included_files
from validation import validate_simulation_config


class ReleaseValidationTests(unittest.TestCase):
    def assert_invalid(self, mutation, message: str) -> None:
        cfg = SimulationConfig()
        mutation(cfg)
        with self.assertRaisesRegex(ValueError, message):
            validate_simulation_config(cfg)

    def test_invalid_public_selectors_and_numerics_fail_fast(self) -> None:
        cases = (
            (lambda c: setattr(c.numerics, "hotloop_backend", "gpu"), "hotloop_backend"),
            (lambda c: setattr(c.waveform, "waveform_type", "triangle"), "waveform_type"),
            (lambda c: setattr(c.circuit, "circuit_type", "R"), "circuit_type"),
            (lambda c: setattr(c.circuit, "circuit_time_scheme", "rk4"), "circuit_time_scheme"),
            (lambda c: setattr(c.numerics, "max_substeps", 0), "max_substeps"),
            (lambda c: setattr(c.numerics, "bc_poisson_picard_max_iter", 0), "max_iter"),
            (lambda c: setattr(c.plasma_state, "n0", -1.0), "plasma_state.n0"),
            (lambda c: setattr(c.plasma, "impact_ionization_model", "magic"), "impact_ionization_model"),
            (lambda c: setattr(c.plasma, "recombination_model", "three_body"), "recombination_model"),
            (lambda c: setattr(c.emission, "electrode_material_mode", "automatic"), "electrode_material_mode"),
            (lambda c: setattr(c.boundary, "anode_ion_boundary", "reflecting"), "anode_ion_boundary"),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                self.assert_invalid(mutation, message)

    def test_electron_table_gas_mismatch_is_rejected(self) -> None:
        cfg = SimulationConfig()
        cfg.plasma_state.gas = "nitrogen"
        cfg.plasma.electron_kinetics_model = "local_field_approximation"
        cfg.local_field_approximation.electron_transport_source = (
            "swarm_data_table_interpolation"
        )
        cfg.local_field_approximation.electron_swarm_data_path = (
            "ar_swarm_output_full_EoverN.dat"
        )
        with self.assertRaisesRegex(ValueError, "gas mismatch"):
            validate_simulation_config(cfg)

    def test_modified_electron_table_fails_manifest_checksum(self) -> None:
        source_root = data_paths.ELECTRON_SWARM_DATA_DIR
        source_manifest = json.loads((source_root / "manifest.json").read_text())
        record = next(
            item
            for item in source_manifest["tables"]
            if item["filename"].endswith("/ar_swarm_output_full_EoverN.dat")
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / record["filename"]
            table.parent.mkdir(parents=True)
            shutil.copyfile(source_root / record["filename"], table)
            table.write_text(table.read_text() + "\n# corruption\n", encoding="utf-8")
            manifest = {
                "format": source_manifest["format"],
                "table_count": 1,
                "required_sections": list(REQUIRED_SECTIONS),
                "tables": [record],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("data_paths.ELECTRON_SWARM_DATA_DIR", root):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    load_electron_swarm_table(record["filename"])

    def test_electron_manifest_check_is_portable_without_bolsig_source(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            missing_generation_manifest = Path(directory) / "not-present.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(project_root / "tools" / "build_electron_manifest.py"),
                    "--check",
                    "--generation-manifest",
                    str(missing_generation_manifest),
                ],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    def test_release_archive_excludes_generated_packaging_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "keep.py"
            source.write_text("pass\n", encoding="utf-8")
            generated = root / "paschen_1d.egg-info" / "SOURCES.txt"
            generated.parent.mkdir()
            generated.write_text("generated\n", encoding="utf-8")
            self.assertEqual(included_files(root), [source])

    def test_modified_bundled_ion_table_fails_manifest_checksum(self) -> None:
        normalized = data_paths.ION_SWARM_DATA_DIR / "normalized_lxcat_2026-07-21"
        manifest = json.loads((normalized / "manifest.json").read_text())
        record = next(
            item
            for item in manifest["records"]
            if item.get("dataset_id") == "lxcat_606acd7d3f0d4373cf14"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_root = root / "normalized_lxcat_2026-07-21"
            table = target_root / record["normalized_file"]
            table.parent.mkdir(parents=True)
            shutil.copyfile(normalized / record["normalized_file"], table)
            table.write_text(table.read_text() + "\n", encoding="utf-8")
            (target_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "paschen-1d-ion-transport-v1",
                        "schema_version": 1,
                        "records": [record],
                        "compatible_mobility_diffusion_pairs": [],
                    }
                ),
                encoding="utf-8",
            )
            with patch("data_paths.ION_SWARM_DATA_DIR", root):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    load_ion_transport_table(table)


if __name__ == "__main__":
    unittest.main()
