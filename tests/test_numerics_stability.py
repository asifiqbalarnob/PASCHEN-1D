from __future__ import annotations

import unittest

import numpy as np

from numerics import compute_diffusion_cfl
from numerics_jit import compute_diffusion_cfl_numba, is_numba_available


class NumericsStabilityTests(unittest.TestCase):
    """Regression tests for explicit transport-stability diagnostics."""

    def test_diffusion_cfl_uses_largest_absolute_diffusion_profile_value(self) -> None:
        D_e = np.asarray([0.1, 0.3, -0.2], dtype=np.float32)
        D_i = np.asarray([0.05, 0.7, 0.1], dtype=np.float32)
        dt = 2.0e-9
        dx = 1.0e-4

        expected = float(
            max(float(np.max(np.abs(D_e))), float(np.max(np.abs(D_i))))
            * dt
            / (dx * dx)
        )

        actual = compute_diffusion_cfl(D_e, D_i, dt, dx)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-7, atol=0.0)

    @unittest.skipUnless(is_numba_available(), "Numba is not installed")
    def test_numba_diffusion_cfl_matches_numpy_utility(self) -> None:
        D_e = np.asarray([0.1, 0.3, -0.2], dtype=np.float32)
        D_i = np.asarray([0.05, 0.7, 0.1], dtype=np.float32)
        dt = 2.0e-9
        dx = 1.0e-4

        expected = compute_diffusion_cfl(D_e, D_i, dt, dx)
        actual = compute_diffusion_cfl_numba(D_e, D_i, dt, dx)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-7, atol=0.0)


if __name__ == "__main__":
    unittest.main()
