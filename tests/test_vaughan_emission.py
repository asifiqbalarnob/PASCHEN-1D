from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from electron_transport import SECTION_MEAN_ENERGY, load_electron_swarm_table
from paschen_1d import run_simulation
from physical_constants import e, kB, m_e
from physics import (
    SwarmRuntimeInterpolationCache,
    build_swarm_interpolation_cache,
    build_transport_reference_state,
    compute_anode_electron_induced_yield,
    compute_electron_impact_energy_proxy_eV,
    compute_vaughan_secondary_electron_yield,
)
from release_test_utils import make_miniature_config


class VaughanEmissionTests(unittest.TestCase):
    """Regression tests for the optional Vaughan anode-emission path."""

    def test_default_effective_temperature_mode_is_fixed(self) -> None:
        cfg = make_miniature_config(run_name="vaughan_default_temperature_mode")
        self.assertEqual(cfg.emission.vaughan_effective_temperature_mode, "fixed")

    def test_impact_energy_proxy_combines_directed_and_thermal_energy(self) -> None:
        incident_flux = 2.0e20
        density_inner = 1.0e15
        T_e_eV = 1.25

        expected = (
            (m_e / (2.0 * e)) * (incident_flux / density_inner) ** 2
            + 2.0 * T_e_eV
        )
        actual = compute_electron_impact_energy_proxy_eV(
            incident_electron_flux=incident_flux,
            electron_density_inner=density_inner,
            T_e_eV=T_e_eV,
        )

        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=0.0)

    def test_vaughan_piecewise_energy_branches_match_equations(self) -> None:
        Emax0_eV = 100.0
        dmax0 = 2.0
        for normalized_energy, exponent in ((0.5, 0.56), (2.0, 0.25)):
            with self.subTest(normalized_energy=normalized_energy):
                actual = compute_vaughan_secondary_electron_yield(
                    impact_energy_eV=normalized_energy * Emax0_eV,
                    Emax0_eV=Emax0_eV,
                    dmax0=dmax0,
                    ks=0.0,
                    z=0.0,
                    E0_eV=0.0,
                )
                expected = dmax0 * (
                    normalized_energy * np.exp(1.0 - normalized_energy)
                ) ** exponent
                np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=0.0)

        normalized_energy = 4.0
        actual = compute_vaughan_secondary_electron_yield(
            impact_energy_eV=normalized_energy * Emax0_eV,
            Emax0_eV=Emax0_eV,
            dmax0=dmax0,
            ks=0.0,
            z=0.0,
            E0_eV=0.0,
        )
        expected = dmax0 * 1.125 / (normalized_energy ** 0.35)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=0.0)

    def test_constant_yield_path_is_unchanged(self) -> None:
        actual = compute_anode_electron_induced_yield(
            incident_electron_flux=3.0e20,
            electron_density_inner=2.0e15,
            T_e_eV=1.0,
            constant_yield=0.37,
            use_vaughan_sey=False,
        )
        self.assertEqual(actual, 0.37)

    def test_vaughan_selection_uses_energy_dependent_yield(self) -> None:
        parameters = {
            "incident_electron_flux": 2.0e20,
            "electron_density_inner": 1.0e15,
            "T_e_eV": 1.0,
            "constant_yield": 0.0,
            "use_vaughan_sey": True,
            "vaughan_Emax0_eV": 100.0,
            "vaughan_dmax0": 2.0,
            "vaughan_ks": 0.0,
            "vaughan_z": 0.0,
            "vaughan_E0": 0.0,
        }
        impact_energy = compute_electron_impact_energy_proxy_eV(
            incident_electron_flux=parameters["incident_electron_flux"],
            electron_density_inner=parameters["electron_density_inner"],
            T_e_eV=parameters["T_e_eV"],
        )
        expected = compute_vaughan_secondary_electron_yield(
            impact_energy_eV=impact_energy,
            Emax0_eV=parameters["vaughan_Emax0_eV"],
            dmax0=parameters["vaughan_dmax0"],
            ks=parameters["vaughan_ks"],
            z=parameters["vaughan_z"],
            E0_eV=parameters["vaughan_E0"],
        )

        actual = compute_anode_electron_induced_yield(**parameters)

        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=0.0)
        self.assertGreater(actual, 0.0)

    def test_local_field_temperature_interpolates_bolsig_mean_energy(self) -> None:
        cfg = make_miniature_config(run_name="vaughan_local_temperature")
        cfg.emission.use_vaughan_sey = True
        cfg.emission.vaughan_effective_temperature_mode = (
            "local_field_approximation"
        )
        cfg.electron_swarm_data.out_of_range_policy = "error"
        transport = build_transport_reference_state(cfg)
        cache = build_swarm_interpolation_cache(cfg)
        table = load_electron_swarm_table(
            cfg.local_field_approximation.electron_swarm_data_path
        )
        mean_energy = table.section(SECTION_MEAN_ENERGY)
        table_index = len(mean_energy.reduced_field_Td) // 2
        local_field = (
            mean_energy.reduced_field_Td[table_index]
            * float(transport.neutral_density)
            / 1.0e21
        )

        actual = cache.vaughan_effective_temperature_eV_from_field(
            E_anode=float(local_field),
            neutral_density=float(transport.neutral_density),
        )
        expected = (2.0 / 3.0) * mean_energy.reduced_values_SI[table_index]

        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)
        with self.assertRaisesRegex(ValueError, "outside its electron table"):
            cache.vaughan_effective_temperature_eV_from_field(
                E_anode=0.0,
                neutral_density=float(transport.neutral_density),
            )

    def test_solver_routes_enabled_vaughan_model_to_boundary_flux(self) -> None:
        cfg = make_miniature_config(run_name="vaughan_boundary_flux")
        cfg.plasma_state.n0 = 1.0e12
        cfg.boundary.anode_electron_boundary = "electron_emission"
        cfg.emission.anode_electron_induced_yield = 0.0
        cfg.emission.use_vaughan_sey = True

        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            with contextlib.redirect_stdout(io.StringIO()), patch(
                "paschen_1d.tqdm", lambda iterable, **kwargs: iterable
            ), patch(
                "paschen_1d.compute_anode_electron_induced_yield",
                wraps=compute_anode_electron_induced_yield,
            ) as yield_model:
                state = run_simulation(cfg)

        self.assertGreater(yield_model.call_count, 0)
        self.assertTrue(
            all(call.kwargs["use_vaughan_sey"] for call in yield_model.call_args_list)
        )
        expected_fixed_temperature_eV = kB * cfg.plasma_state.T_e / e
        self.assertTrue(
            all(
                call.kwargs["T_e_eV"] == expected_fixed_temperature_eV
                for call in yield_model.call_args_list
            )
        )
        self.assertTrue(np.all(np.isfinite(state.ne_final)))
        self.assertTrue(np.all(np.isfinite(state.ni_final)))
        self.assertTrue(np.all(np.isfinite(state.phi_final)))
        self.assertTrue(np.all(state.ne_final >= 0.0))
        self.assertTrue(np.all(state.ni_final >= 0.0))

    def test_solver_uses_local_field_temperature_at_anode(self) -> None:
        cfg = make_miniature_config(run_name="vaughan_local_temperature_solver")
        cfg.plasma_state.n0 = 1.0e12
        cfg.boundary.anode_electron_boundary = "electron_emission"
        cfg.emission.anode_electron_induced_yield = 0.0
        cfg.emission.use_vaughan_sey = True
        cfg.emission.vaughan_effective_temperature_mode = (
            "local_field_approximation"
        )
        marker_temperature_eV = 2.75

        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            with contextlib.redirect_stdout(io.StringIO()), patch(
                "paschen_1d.tqdm", lambda iterable, **kwargs: iterable
            ), patch.object(
                SwarmRuntimeInterpolationCache,
                "vaughan_effective_temperature_eV_from_field",
                return_value=marker_temperature_eV,
            ) as temperature_model, patch(
                "paschen_1d.compute_anode_electron_induced_yield",
                wraps=compute_anode_electron_induced_yield,
            ) as yield_model:
                run_simulation(cfg)

        self.assertGreater(temperature_model.call_count, 0)
        self.assertTrue(
            all(
                isinstance(call.kwargs["E_anode"], float)
                for call in temperature_model.call_args_list
            )
        )
        self.assertTrue(
            all(
                call.kwargs["T_e_eV"] == marker_temperature_eV
                for call in yield_model.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
