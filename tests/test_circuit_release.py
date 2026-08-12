from __future__ import annotations

import unittest

import numpy as np

from circuit import step_circuit
from circuit_implicit_euler import step_circuit_implicit_euler
from circuit_mna import step_circuit_mna


TOPOLOGIES = (
    "dielectric_plasma",
    "R0_Cp",
    "R0_Cp_Rm",
    "R0_Rm_Cext",
    "R0_Cs_Cp",
    "R0_Cs_Cp_Rm",
    "R0_Cs_Ls_Cp",
    "R0_Cs_Ls_Cp_Rm",
    "R0_Cs_Ls_Cp_Lp",
    "R0_Cs_Ls_Cp_Lp_Rm_Cext",
)


def circuit_arguments(circuit_type: str) -> dict:
    return {
        "circuit_type": circuit_type,
        "V_app_func": lambda t: 100.0,
        "t": 0.0,
        "dt": 1.0e-12,
        "V_gap_prev": 10.0,
        "Gamma_i": np.zeros(5),
        "Gamma_e": np.zeros(5),
        "dx": 2.5e-4,
        "A": 1.0e-4,
        "L": 1.0e-3,
        "l": 1.0e-4,
        "eps_r": 4.0,
        "R0": 1.0e5,
        "C_s": 1.0e-9,
        "C_p": 1.0e-10,
        "R_m": 2.0e5,
        "L_s": 1.0e-6,
        "L_p": 2.0e-6,
        "V_d_prev": 0.0,
        "V_n_prev": 10.0,
        "V_Cs_prev": 0.0,
        "I_s_prev": 0.0,
        "I_Lp_prev": 0.0,
        "C_ext": 1.0e-11,
    }


class CircuitReleaseTests(unittest.TestCase):
    def assert_finite_step(self, stepper, topology: str) -> None:
        result = stepper(**circuit_arguments(topology))
        self.assertEqual(len(result), 7)
        for value in result:
            if value is not None:
                self.assertTrue(np.isfinite(value), (topology, result))

    def test_every_named_reduced_topology_explicit(self) -> None:
        for topology in TOPOLOGIES:
            with self.subTest(topology=topology):
                self.assert_finite_step(step_circuit, topology)

    def test_every_named_reduced_topology_implicit(self) -> None:
        for topology in TOPOLOGIES:
            with self.subTest(topology=topology):
                self.assert_finite_step(step_circuit_implicit_euler, topology)

    def test_unified_mna_topology(self) -> None:
        self.assert_finite_step(step_circuit_mna, "R0_Cs_Ls_Cp_Lp_Rm_Cext")

    def test_removed_aliases_are_errors(self) -> None:
        for topology in ("none", "R", "R0_Cs_Ls_Cp_Lp_Rm", "automatic"):
            with self.subTest(topology=topology):
                with self.assertRaisesRegex(ValueError, "unsupported|supports"):
                    step_circuit(**circuit_arguments(topology))


if __name__ == "__main__":
    unittest.main()
