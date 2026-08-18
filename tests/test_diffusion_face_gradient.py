from __future__ import annotations

import unittest

import numpy as np

import numerics_jit
from numerics import (
    create_linear_rk4_workspace,
    kt_flux_update,
    kt_flux_update_linear_reuse,
)


def _quadratic_diffusion_problem() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    x = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    n = x**2
    zeros = np.zeros_like(x)
    diffusion = np.ones_like(x)
    return n, zeros, diffusion, float(x[1] - x[0])


def _assert_quadratic_diffusion_rhs(rhs: np.ndarray) -> None:
    np.testing.assert_allclose(
        rhs[1:-1],
        2.0,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


class DiffusionFaceGradientTests(unittest.TestCase):
    """Manufactured-solution checks for the physical face gradient."""

    def test_quadratic_manufactured_diffusion_rhs_public_path(self) -> None:
        n, zeros, diffusion, dx = _quadratic_diffusion_problem()

        def zero_flux(n_face: np.ndarray, field_face: np.ndarray) -> np.ndarray:
            return np.zeros_like(n_face)

        rhs = kt_flux_update(
            n=n,
            f=zero_flux,
            df=zero_flux,
            E=zeros,
            D=diffusion,
            S=zeros,
            dx=dx,
            kt_limiter_theta=1.01,
        )

        _assert_quadratic_diffusion_rhs(rhs)

    def test_quadratic_manufactured_diffusion_rhs_reuse_path(self) -> None:
        n, zeros, diffusion, dx = _quadratic_diffusion_problem()
        rhs = np.empty_like(n)
        workspace = create_linear_rk4_workspace(len(n), dtype=np.float64)

        kt_flux_update_linear_reuse(
            n=n,
            u=zeros,
            D=diffusion,
            S=zeros,
            dx=dx,
            kt_limiter_theta=1.01,
            adv_coeff=1.0,
            rhs_out=rhs,
            ws=workspace,
        )

        _assert_quadratic_diffusion_rhs(rhs)

    def _run_numba_path(self, *, parallel: bool) -> np.ndarray:
        n, zeros, diffusion, dx = _quadratic_diffusion_problem()
        rhs = np.empty_like(n)
        workspace = numerics_jit.create_numba_linear_rk4_workspace(
            len(n), dtype=np.float64
        )
        kernel = (
            numerics_jit._kt_rhs_linear_parallel
            if parallel
            else numerics_jit._kt_rhs_linear_serial
        )
        kernel(
            n,
            zeros,
            diffusion,
            zeros,
            dx,
            1.01,
            1.0,
            rhs,
            workspace["slope"],
            workspace["nL_p"],
            workspace["nR_p"],
            workspace["nL_m"],
            workspace["nR_m"],
            workspace["u_face_p"],
            workspace["u_face_m"],
            workspace["a_p"],
            workspace["a_m"],
            workspace["H_p"],
            workspace["H_m"],
            workspace["grad_p"],
            workspace["grad_m"],
            workspace["D_p"],
            workspace["D_m"],
            workspace["Fd_p"],
            workspace["Fd_m"],
            0.0,
            0.0,
            False,
            False,
        )
        return rhs

    @unittest.skipUnless(
        numerics_jit.is_numba_available(), "Numba is not installed"
    )
    def test_quadratic_manufactured_diffusion_rhs_numba_serial(self) -> None:
        _assert_quadratic_diffusion_rhs(self._run_numba_path(parallel=False))

    @unittest.skipUnless(
        numerics_jit.is_numba_available(), "Numba is not installed"
    )
    def test_quadratic_manufactured_diffusion_rhs_numba_parallel(self) -> None:
        _assert_quadratic_diffusion_rhs(self._run_numba_path(parallel=True))


if __name__ == "__main__":
    unittest.main()
