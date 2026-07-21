from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from config import SimulationConfig
from physics import (
    build_ion_diffusion_profile,
    build_ion_mobility_profile,
    build_swarm_interpolation_cache,
    build_transport_reference_state,
)
from ion_transport import (
    IonTableInterpolator,
    canonical_neutral,
    load_ion_transport_table,
    validate_table_identity,
)


def write_table(path: Path, *, quantity: str = "reduced_mobility") -> None:
    metadata = {
        "format": "paschen-1d-ion-transport-v1",
        "dataset_id": "test_dataset",
        "quantity": quantity,
        "reduced_quantity": (
            "K0_times_N0"
            if quantity == "reduced_mobility"
            else "N_times_D_longitudinal"
        ),
        "units": "1/(m V s)" if quantity == "reduced_mobility" else "1/(m s)",
        "ion": "D3 +",
        "neutral": "D2",
        "gas_temperature_K": 300.0,
        "database": "test",
        "process": "test process",
        "parameters": "Tgas = 300 K",
        "group_comment": "Synthetic test data",
        "source_family": "synthetic test data",
        "permalink": "https://example.invalid/test",
        "reference": "Synthetic test reference",
        "source_file": "raw/test.txt",
        "source_file_sha256": "0" * 64,
    }
    lines = [f"# {key}: {json.dumps(value)}\n" for key, value in metadata.items()]
    lines.extend(
        [
            "reduced_electric_field_Td,reduced_transport_SI\n",
            "1.0,1.0e24\n",
            "10.0,1.0e25\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


class IonTransportTests(unittest.TestCase):
    def test_reference_state_has_no_argon_nitrogen_gate(self) -> None:
        cfg = SimulationConfig()
        cfg.plasma_state.gas = "deuterium"
        state = build_transport_reference_state(cfg)
        self.assertGreater(float(state.neutral_density), 0.0)

    def test_aliases_and_strict_identity(self) -> None:
        self.assertEqual(canonical_neutral("deuterium"), "d2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mobility.csv"
            write_table(path)
            table = load_ion_transport_table(path)
            validate_table_identity(
                table,
                expected_quantity="reduced_mobility",
                configured_ion="D3+",
                configured_neutral="deuterium",
                gas_temperature_K=300.0,
                temperature_tolerance_K=1.0,
            )
            with self.assertRaisesRegex(ValueError, "neutral"):
                validate_table_identity(
                    table,
                    expected_quantity="reduced_mobility",
                    configured_ion="D3+",
                    configured_neutral="argon",
                    gas_temperature_K=300.0,
                    temperature_tolerance_K=1.0,
                )

    def test_reduced_table_interpolation_and_density_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mobility.csv"
            write_table(path)
            table = load_ion_transport_table(path)
            interp = IonTableInterpolator.from_table(table, "clip")
            density = 1.0e20
            fields = np.asarray([0.1, 1.0e-3])  # 1 Td and below-range -> clipped
            result = interp.evaluate(fields, density)
            np.testing.assert_allclose(result, [1.0e4, 1.0e4], rtol=1e-6)

    def test_error_range_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mobility.csv"
            write_table(path)
            interp = IonTableInterpolator.from_table(
                load_ion_transport_table(path), "error"
            )
            with self.assertRaisesRegex(ValueError, "outside dataset"):
                interp.evaluate(np.asarray([0.0]), 1.0e20)

    def test_runtime_cache_builds_table_ion_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mobility_path = root / "mobility.csv"
            diffusion_path = root / "diffusion.csv"
            write_table(mobility_path)
            write_table(diffusion_path, quantity="reduced_longitudinal_diffusion")
            cfg = SimulationConfig()
            cfg.plasma_state.gas = "deuterium"
            cfg.plasma.ion_kinetics_model = "local_field_ion_kinetics"
            cfg.ion_transport.positive_ion = "D3+"
            cfg.ion_transport.mobility_source_mode = "swarm_data_table_interpolation"
            cfg.ion_transport.diffusion_source_mode = "swarm_data_table_interpolation"
            cfg.ion_transport.mobility_table_path = mobility_path.name
            cfg.ion_transport.diffusion_table_path = diffusion_path.name
            manifest_dir = root / "normalized_lxcat_2026-07-21"
            manifest_dir.mkdir()
            (manifest_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "paschen-1d-ion-transport-v1",
                        "schema_version": 1,
                        "records": [],
                        "compatible_mobility_diffusion_pairs": [
                            {
                                "pair_id": "synthetic_pair",
                                "mobility_dataset_id": "test_dataset",
                                "diffusion_dataset_id": "test_dataset",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch("data_paths.ION_SWARM_DATA_DIR", root):
                cache = build_swarm_interpolation_cache(cfg)
            x = np.linspace(0.0, 1.0, 3)
            field = np.full(3, 0.1)
            density = 1.0e20
            mobility = build_ion_mobility_profile(
                cfg, x, field, density, cache
            )
            diffusion = build_ion_diffusion_profile(
                cfg, x, field, density, mobility, cache
            )
            np.testing.assert_allclose(mobility, 1.0e4, rtol=1e-6)
            np.testing.assert_allclose(diffusion, 1.0e4, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
