from __future__ import annotations

from dataclasses import fields, is_dataclass
import importlib
import unittest
from pathlib import Path

import config
from config_loader import load_simulation_case
from paschen_1d import validate_simulation_config


class ConfigurationTests(unittest.TestCase):
    @staticmethod
    def _module_dataclass_schema(module: object) -> dict[str, tuple[tuple[str, str], ...]]:
        return {
            name: tuple(
                (
                    item.name,
                    str(item.type).replace(f"{module.__name__}.", ""),
                )
                for item in fields(value)
            )
            for name, value in vars(module).items()
            if isinstance(value, type)
            and value.__module__ == module.__name__
            and is_dataclass(value)
        }

    def test_case_modules_are_complete_standalone_configurations(self) -> None:
        expected = {
            "config_case_argon_photoemission_discharge": ("argon", "Ar+"),
            "config_case_nitrogen_pulsed_discharge": ("nitrogen", "N2+"),
            "config_case_argon_dc_discharge": ("argon", "Ar+"),
            "config_case_deuterium_pulsed_discharge": ("deuterium", "D3+"),
            "config_case_helium_photoemission_discharge": ("helium", "4He+"),
        }
        reference_schema = self._module_dataclass_schema(config)
        for module_name, (gas, ion) in expected.items():
            module = importlib.import_module(module_name)
            case_type, _ = load_simulation_case(module_name)
            self.assertIs(case_type, module.SimulationConfig)
            self.assertIsNot(case_type, config.SimulationConfig)
            self.assertEqual(
                self._module_dataclass_schema(module),
                reference_schema,
                msg=f"{module_name} configuration schema differs from config.py",
            )
            case = case_type()
            self.assertEqual(case.plasma_state.gas, gas)
            self.assertEqual(case.ion_transport.positive_ion, ion)
            validate_simulation_config(case)

    def test_deuterium_case_selects_electron_and_ion_tables(self) -> None:
        case_type, _ = load_simulation_case(
            "config_case_deuterium_pulsed_discharge"
        )
        case = case_type()
        self.assertEqual(case.plasma.electron_kinetics_model, "local_field_approximation")
        self.assertEqual(
            case.local_field_approximation.electron_transport_source,
            "swarm_data_table_interpolation",
        )
        self.assertEqual(case.plasma.ion_kinetics_model, "local_field_ion_kinetics")
        self.assertEqual(
            case.ion_transport.mobility_source_mode,
            "swarm_data_table_interpolation",
        )
        self.assertEqual(
            case.ion_transport.diffusion_source_mode,
            "swarm_data_table_interpolation",
        )
        self.assertEqual(
            Path(case.local_field_approximation.electron_swarm_data_path).name,
            case.local_field_approximation.electron_swarm_data_path,
        )
        self.assertEqual(
            Path(case.ion_transport.mobility_table_path).name,
            case.ion_transport.mobility_table_path,
        )
        self.assertEqual(
            Path(case.ion_transport.diffusion_table_path).name,
            case.ion_transport.diffusion_table_path,
        )

    def test_table_mode_requires_paths(self) -> None:
        case = config.SimulationConfig()
        case.plasma.ion_kinetics_model = "local_field_ion_kinetics"
        case.ion_transport.mobility_source_mode = "swarm_data_table_interpolation"
        with self.assertRaisesRegex(ValueError, "mobility_table_path"):
            validate_simulation_config(case)


if __name__ == "__main__":
    unittest.main()
