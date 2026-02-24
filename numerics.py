"""
numerics.py

Numerical kernels used by PASCHEN-1D.

Included components:
1) 1D Poisson solve with Dirichlet boundaries (banded Laplacian).
2) Minmod slope limiter.
3) KT central-upwind drift flux + explicit diffusion + source RHS.
4) RK4 density update wrapper.
5) Boundary-condition helpers for electron/ion densities.
6) Drift-based CFL diagnostic.

All routines assume a uniform 1D spatial grid.
"""

import numpy as np
from typing import Callable
from scipy.linalg import solve_banded

from physical_constants import e, eps0
from physics import (
    boundary_zero_density,
    boundary_electron_emission_density,
    boundary_cathode_ion_implicit_drift_density,
    boundary_anode_electron_implicit_drift_density,
)


# ============================================================
# Poisson solver (banded Laplacian with Dirichlet BCs)
# ============================================================


def build_poisson_tridiag_interior(Nx: int, dtype=np.float64) -> np.ndarray:
    """
    Build banded interior matrix for 1D Laplacian with Dirichlet endpoints.

    We discretize the Laplacian on a uniform grid as:

        (phi_{j+1} - 2 phi_j + phi_{j-1}) / dx² ≈ d²phi/dx²

    For `scipy.linalg.solve_banded` with (l, u) = (1, 1), the banded
    matrix layout is:

        ab[0, 1:]  = upper diagonal  (1)
        ab[1, :]   = main  diagonal  (-2)
        ab[2, :-1] = lower diagonal  (1)

    so the interior system (Nx-2 unknowns) reads:

        A_int phi_int = b_int,

    with phi_0 and phi_{Nx-1} enforced separately through b_int.

    Parameters
    ----------
    Nx : int
        Total number of grid points in x (must be >= 3).
    dtype : data-type, optional
        Data type for the banded matrix. Default is np.float64.

    Returns
    -------
    ab : np.ndarray
        Banded matrix representation with shape (3, Nx - 2).
    """
    Nint = Nx - 2
    if Nint <= 0:
        raise ValueError("Nx must be >= 3.")
    ab = np.zeros((3, Nint), dtype=dtype)
    ab[0, 1:] = 1.0
    ab[1, :] = -2.0
    ab[2, :-1] = 1.0
    return ab


def poisson_1d_dirichlet_interior(
    n_i: np.ndarray,
    n_e: np.ndarray,
    dx: float,
    phi_left: float,
    phi_right: float,
    ab_int: np.ndarray,
    phi_out: np.ndarray,
    E_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the 1D Poisson equation

        d²phi/dx² = -(e/eps0) (n_i - n_e)

    on a grid j = 0..Nx-1 with Dirichlet boundary conditions:

        phi(0)      = phi_left
        phi(Nx - 1) = phi_right.

    The solution and derived electric field are written in-place to
    `phi_out` and `E_out`.

    Parameters
    ----------
    n_i, n_e : np.ndarray
        Ion and electron number densities [m⁻³], shape (Nx,).
    dx : float
        Grid spacing [m].
    phi_left, phi_right : float
        Dirichlet boundary potentials at x = 0 and x = L [V].
    ab_int : np.ndarray
        Banded Laplacian for interior nodes, as produced by
        `build_poisson_tridiag_interior`, shape (3, Nx - 2).
    phi_out : np.ndarray
        Output array for potential phi(x), shape (Nx,). Will be overwritten.
    E_out : np.ndarray
        Output array for electric field E(x) = -dphi/dx, shape (Nx,).
        Will be overwritten.

    Returns
    -------
    phi_out, E_out : np.ndarray
        References to the same arrays passed in (for convenience).

    Notes
    -----
    * The banded solver is applied only on interior nodes (1..Nx-2).
    * Boundary values are set directly from phi_left, phi_right.
    * E is computed via second-order central differences in the interior
      and first-order one-sided differences at the boundaries.
    """
    # RHS for interior equations. Multiply by dx^2 so the discrete operator
    # remains [-2, 1, 1] without extra dx scaling inside A.
    b = -(e / eps0) * (n_i.astype(np.float64) - n_e.astype(np.float64)) * (dx * dx)
    b_int = b[1:-1].copy()
    b_int[0] -= phi_left
    b_int[-1] -= phi_right

    # Solve for interior phi using banded solver
    phi_int = solve_banded(
        (1, 1),
        ab_int,
        b_int,
        overwrite_ab=False,
        overwrite_b=False,
        check_finite=False,
    )

    # Assemble full potential.
    phi_out[0] = float(phi_left)
    phi_out[-1] = float(phi_right)
    phi_out[1:-1] = phi_int.astype(phi_out.dtype, copy=False)

    # Electric field E = -dphi/dx.
    inv_dx = 1.0 / dx
    # Interior: second-order central
    E_out[1:-1] = -(phi_out[2:] - phi_out[:-2]) * (0.5 * inv_dx)
    # Boundaries: one-sided first-order
    E_out[0] = -(phi_out[1] - phi_out[0]) * inv_dx
    E_out[-1] = -(phi_out[-1] - phi_out[-2]) * inv_dx

    return phi_out, E_out


# ============================================================
# Minmod limiter
# ============================================================


def minmod(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Vectorized three-argument minmod slope limiter.

    For each component i:

        minmod(a_i, b_i, c_i) =
            min(a_i, b_i, c_i)  if all > 0,
            max(a_i, b_i, c_i)  if all < 0,
            0                   otherwise.

    Parameters
    ----------
    a, b, c : np.ndarray
        Arrays of the same shape containing candidate slopes.

    Returns
    -------
    np.ndarray
        Limited slopes with the same shape as the inputs.

    Notes
    -----
    * This function is used to compute limited gradients for the KT
      central-upwind scheme.
    * TVD-style limiter to suppress spurious oscillations near steep gradients.
    """
    result = np.zeros_like(a)

    pos_mask = (a > 0) & (b > 0) & (c > 0)
    neg_mask = (a < 0) & (b < 0) & (c < 0)

    result[pos_mask] = np.minimum(np.minimum(a[pos_mask], b[pos_mask]), c[pos_mask])
    result[neg_mask] = np.maximum(np.maximum(a[neg_mask], b[neg_mask]), c[neg_mask])

    return result


# ============================================================
# Kurganov-Tadmor flux update with diffusion and sources
# ============================================================


def kt_flux_update(
    n: np.ndarray,
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    df: Callable[[np.ndarray, np.ndarray], np.ndarray],
    E: np.ndarray,
    D: float | np.ndarray,
    S: np.ndarray,
    dx: float,
    kt_limiter_theta: float = 1.1,
) -> np.ndarray:
    """
    Compute finite-volume RHS for scalar transport with drift, diffusion, source.

        dn/dt = - (H_{j+1/2} - H_{j-1/2}) / dx
                - (F_d{j+1/2} - F_d{j-1/2}) / dx
                + S_j,

    using the Kurganov-Tadmor (KT) central-upwind flux for drift
    and an explicit diffusion flux.

    Parameters
    ----------
    n : np.ndarray
        Cell-centered number density, shape (Nx,).
    f : callable
        Drift/convection flux function:

            f(n_face, E_face) -> Gamma

        where Gamma is the particle flux at a face.
    df : callable
        Derivative of f with respect to n, i.e. df/dn, used to estimate
        local wave speeds |a_{j±1/2}|.
    E : np.ndarray
        Cell-centered electric field [V/m], shape (Nx,).
    D : float or np.ndarray
        Diffusion coefficient [m²/s], scalar or cell-centered array
        of shape (Nx,).
    S : np.ndarray
        Cell-centered source term (e.g. ionization, recombination),
        shape (Nx,).
    dx : float
        Grid spacing [m].
    kt_limiter_theta : float, optional
        KT slope-limiter parameter (theta >= 1). Larger values reduce
        limiting; default is 1.1.

    Returns
    -------
    rhs : np.ndarray
        Time derivative dn/dt at each cell center, shape (Nx,).

    Notes
    -----
    * Boundary cells j=0 and j=N-1 are left with rhs=0; enforce BCs externally.
    * Limiter strength is controlled by kt_limiter_theta.
    """
    rhs = np.zeros_like(n)

    # Limited slopes (cell-centered).
    # theta >= 1. Larger theta gives less limiting.
    theta = kt_limiter_theta
    dn_bwd = (n[1:-1] - n[:-2]) / dx
    dn_central = 0.5 * (n[2:] - n[:-2]) / dx
    dn_fwd = (n[2:] - n[1:-1]) / dx

    slope = np.zeros_like(n)
    slope[1:-1] = minmod(theta * dn_bwd, dn_central, theta * dn_fwd)

    # Reconstruct left/right states at faces.
    # face j+1/2 uses cells j (left) and j+1 (right)
    nL_p = n[1:-1] + 0.5 * dx * slope[1:-1]  # left state at j+1/2
    nR_p = n[2:] - 0.5 * dx * slope[2:]      # right state at j+1/2

    # face j-1/2 uses cells j-1 (left) and j (right)
    nL_m = n[:-2] + 0.5 * dx * slope[:-2]    # left state at j-1/2
    nR_m = n[1:-1] - 0.5 * dx * slope[1:-1]  # right state at j-1/2

    # E-field at faces (neighbor average).
    Ei_p = 0.5 * (E[1:-1] + E[2:])   # at j+1/2
    Ei_m = 0.5 * (E[1:-1] + E[:-2])  # at j-1/2

    # Local speeds a_{j+/-1/2} = max |df/dn| from left/right states.
    a_p = np.maximum(np.abs(df(nL_p, Ei_p)), np.abs(df(nR_p, Ei_p)))  # j+1/2
    a_m = np.maximum(np.abs(df(nL_m, Ei_m)), np.abs(df(nR_m, Ei_m)))  # j-1/2

    # KT convective fluxes at faces.
    H_p = 0.5 * (f(nL_p, Ei_p) + f(nR_p, Ei_p)) - 0.5 * a_p * (nR_p - nL_p)
    H_m = 0.5 * (f(nL_m, Ei_m) + f(nR_m, Ei_m)) - 0.5 * a_m * (nR_m - nL_m)

    # Diffusion fluxes at faces: F_d = -D_face * (dn/dx)_face.
    grad_p = (nR_p - nL_p) / dx
    grad_m = (nR_m - nL_m) / dx

    if np.isscalar(D):
        D_p = D
        D_m = D
    else:
        # Arithmetic average to face centers.
        D_p = 0.5 * (D[1:-1] + D[2:])
        D_m = 0.5 * (D[:-2] + D[1:-1])

    Fd_p = -D_p * grad_p
    Fd_m = -D_m * grad_m

    # Finite-volume update on interior cells.
    rhs[1:-1] = -(H_p - H_m) / dx - (Fd_p - Fd_m) / dx + S[1:-1]

    # Boundaries: rhs[0], rhs[-1] remain zero; BCs enforced elsewhere.
    return rhs


def rk4_step(
    n: np.ndarray,
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    df: Callable[[np.ndarray, np.ndarray], np.ndarray],
    E: np.ndarray,
    D: float | np.ndarray,
    S: np.ndarray,
    dx: float,
    dt: float,
    kt_limiter_theta: float = 1.1,
) -> np.ndarray:
    """
    Advance density `n` by one time step with RK4 using KT+diffusion+source RHS.

    Parameters
    ----------
    n : np.ndarray
        Current cell-centered density, shape (Nx,).
    f, df : callable
        Flux and its derivative with respect to n (see `kt_flux_update`).
    E : np.ndarray
        Electric field [V/m], shape (Nx,).
    D : float or np.ndarray
        Diffusion coefficient [m²/s], scalar or array of shape (Nx,).
    S : np.ndarray
        Source term, shape (Nx,).
    dx : float
        Grid spacing [m].
    dt : float
        Time step [s].
    kt_limiter_theta : float, optional
        KT slope-limiter parameter passed to `kt_flux_update`.

    Returns
    -------
    n_new : np.ndarray
        Updated density after one RK4 step, shape (Nx,). Negative
        values are projected to zero via:

            n_new = max(n_new, 0).

    Notes
    -----
    * Boundary conditions are not applied inside this routine.
    """
    k1 = kt_flux_update(n,               f, df, E, D, S, dx, kt_limiter_theta)
    k2 = kt_flux_update(n + 0.5 * dt * k1, f, df, E, D, S, dx, kt_limiter_theta)
    k3 = kt_flux_update(n + 0.5 * dt * k2, f, df, E, D, S, dx, kt_limiter_theta)
    k4 = kt_flux_update(n +       dt * k3, f, df, E, D, S, dx, kt_limiter_theta)

    n_new = n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Enforce non-negativity after combining RK stages.
    np.maximum(n_new, 0.0, out=n_new)
    return n_new


# ============================================================
# Boundary conditions for n_e and n_i
# ============================================================


def set_boundary_condition_implicit(
    ne_next: np.ndarray,
    ni_next: np.ndarray,
    ne_curr: np.ndarray,
    ni_curr: np.ndarray,
    phi_curr: np.ndarray,
    gamma: float,
    mu_i: float,
    mu_e: float,
    dx: float,
    dt: float,
    Gamma_ext_anode: float = 0.0,
    Gamma_ext_cathode: float = 0.0,
    anode_ion_boundary: str = "zero_density",
    anode_electron_boundary: str = "implicit_drift_closure",
    cathode_ion_boundary: str = "implicit_drift_closure",
    cathode_electron_boundary: str = "electron_emission",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mode-driven boundary-condition update for anode/cathode and both species.

    Geometry / labels:
        x = 0   → anode
        x = L   → cathode

    Supported per-boundary modes:
        - "zero_density"
        - "electron_emission" (electrons only)
        - "implicit_drift_closure" (implemented for anode electrons and cathode ions)

    Parameters
    ----------
    ne_next, ni_next : np.ndarray
        Electron/ion densities at the next time step, shape (Nx,).
        These arrays are modified in-place at the boundaries.
    ne_curr, ni_curr : np.ndarray
        Electron/ion densities at the current time step, shape (Nx,).
    phi_curr : np.ndarray
        Electric potential at the current time step, shape (Nx,).
    gamma : float
        Ion-induced secondary electron yield coefficient.
    mu_i, mu_e : float
        Ion/electron mobilities [m²/(V·s)].
    dx : float
        Spatial step [m].
    dt : float
        Time step [s].
    Gamma_ext_anode : float, optional
        Externally driven electron-emission number flux magnitude at the anode
        [m^-2 s^-1] used by Eq. (11a)-style closure. Default is 0.
    Gamma_ext_cathode : float, optional
        Externally driven electron-emission number flux magnitude at the cathode
        [m^-2 s^-1] used by Eq. (11a)-style closure. Default is 0.

    Returns
    -------
    ne_next, ni_next : np.ndarray
        Updated next-step densities with BCs enforced and clipped to be
        non-negative.
    """
    # -----------------------
    # Ion boundary conditions
    # -----------------------
    if anode_ion_boundary == "zero_density":
        ni_next[0] = boundary_zero_density()
    elif anode_ion_boundary == "implicit_drift_closure":
        raise ValueError(
            "Unsupported boundary combination: anode_ion_boundary='implicit_drift_closure'. "
            "Current implementation supports implicit ion drift closure only at the cathode."
        )
    elif anode_ion_boundary == "electron_emission":
        raise ValueError("Invalid ion boundary mode: 'electron_emission'.")
    else:
        raise ValueError(f"Unknown anode_ion_boundary mode: {anode_ion_boundary}")

    if cathode_ion_boundary == "implicit_drift_closure":
        ni_next[-1] = boundary_cathode_ion_implicit_drift_density(
            ni_curr_right=float(ni_curr[-1]),
            ni_next_inner=float(ni_next[-2]),
            phi_right=float(phi_curr[-1]),
            phi_inner=float(phi_curr[-2]),
            phi_inner2=float(phi_curr[-3]),
            gamma=gamma,
            mu_i=mu_i,
            dx=dx,
            dt=dt,
        )
    elif cathode_ion_boundary == "zero_density":
        ni_next[-1] = boundary_zero_density()
    elif cathode_ion_boundary == "electron_emission":
        raise ValueError("Invalid ion boundary mode: 'electron_emission'.")
    else:
        raise ValueError(f"Unknown cathode_ion_boundary mode: {cathode_ion_boundary}")

    # ----------------------------
    # Electron boundary conditions
    # ----------------------------
    if anode_electron_boundary == "implicit_drift_closure":
        ne_next[0] = boundary_anode_electron_implicit_drift_density(
            ne_curr_left=float(ne_curr[0]),
            ne_next_inner=float(ne_next[1]),
            phi_left=float(phi_curr[0]),
            phi_inner=float(phi_curr[1]),
            phi_inner2=float(phi_curr[2]),
            mu_e=mu_e,
            dx=dx,
            dt=dt,
        )
    elif anode_electron_boundary == "zero_density":
        ne_next[0] = boundary_zero_density()
    elif anode_electron_boundary == "electron_emission":
        ne_next[0] = boundary_electron_emission_density(
            boundary_side="anode",
            gamma=gamma,
            ni_boundary=float(ni_next[0]),
            mu_i=mu_i,
            mu_e=mu_e,
            phi_boundary=float(phi_curr[0]),
            phi_inner=float(phi_curr[1]),
            dx=dx,
            Gamma_ext=Gamma_ext_anode,
        )
    else:
        raise ValueError(
            f"Unknown anode_electron_boundary mode: {anode_electron_boundary}"
        )

    if cathode_electron_boundary == "electron_emission":
        ne_next[-1] = boundary_electron_emission_density(
            boundary_side="cathode",
            gamma=gamma,
            ni_boundary=float(ni_next[-1]),
            mu_i=mu_i,
            mu_e=mu_e,
            phi_boundary=float(phi_curr[-1]),
            phi_inner=float(phi_curr[-2]),
            dx=dx,
            Gamma_ext=Gamma_ext_cathode,
        )
    elif cathode_electron_boundary == "zero_density":
        ne_next[-1] = boundary_zero_density()
    elif cathode_electron_boundary == "implicit_drift_closure":
        raise ValueError(
            "Unsupported boundary combination: cathode_electron_boundary='implicit_drift_closure'. "
            "Current implementation supports implicit electron drift closure only at the anode."
        )
    else:
        raise ValueError(
            f"Unknown cathode_electron_boundary mode: {cathode_electron_boundary}"
        )

    return np.clip(ne_next, 0.0, None), np.clip(ni_next, 0.0, None)


# ============================================================
# CFL diagnostic
# ============================================================


def CFL_test(
    mu_e: float,
    mu_i: float,
    E_next: np.ndarray,
    dt: float,
    dx: float,
    time: np.ndarray,
    n_idx: int,
) -> float:
    """
    Compute drift-based CFL number and print a warning if CFL > 1.

    We define:

        CFL = a_max * dt / dx,

    where

        a_max = max_x max(|mu_e E|, |mu_i E|).

    Parameters
    ----------
    mu_e, mu_i : float
        Electron and ion mobilities [m²/(V·s)].
    E_next : np.ndarray
        Electric field at the next time step [V/m], shape (Nx,).
    dt : float
        Time step [s].
    dx : float
        Spatial step [m].
    time : np.ndarray
        Time array, shape (Nt,).
    n_idx : int
        Current time-step index (n). The function will reference
        time[n_idx + 1] when printing warnings, if in range.

    Returns
    -------
    C : float
        CFL number for the current step.

    Notes
    -----
    * This diagnostic uses only the drift term; diffusion/source impose
      separate constraints.
    * If CFL > 1, you are in a regime where an explicit drift update
      would be unstable; however, the KT + RK4 combo may still be
      more tolerant than a simple upwind scheme.
    """
    # Characteristic speeds at each cell.
    a_e = np.abs(mu_e * E_next)
    a_i = np.abs(mu_i * E_next)
    a_max = max(a_e.max(), a_i.max())

    C = a_max * dt / dx

    # Optional diagnostic print (with bounds guard).
    if C > 1.0:
        if (n_idx + 1) < len(time):
            t_str = f"{time[n_idx + 1]:.4e}"
        else:
            t_str = "unknown"
        print(f"CFL condition violated at time {t_str}, CFL = {C:.3f}")

    return C
