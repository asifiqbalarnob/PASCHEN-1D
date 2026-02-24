"""
physics.py

Physics helper routines for PASCHEN-1D.

This module provides:
1) Applied-voltage waveform builders.
2) Gas-dependent transport/recombination coefficients.
3) Initial-state construction for phi, E, n_e, and n_i.
4) Townsend ionization coefficient evaluation from E/p fits.

Conventions:
- SI units are used unless noted otherwise.
- Pressure inputs for empirical ionization/transport fits remain in Torr.
"""

from typing import Callable

import numpy as np

from physical_constants import kB, e
from config import SimulationConfig, TransportCoeffs


# ============================================================
# Applied-voltage waveforms
# ============================================================

def make_voltage_waveform(cfg: SimulationConfig) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build the applied-voltage function V_app(t) from cfg.waveform_type.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation configuration specifying waveform_type and the associated
        parameters (V_peak, tV_start, tV_end, tau, t_peak, f_rf, etc.).

    Returns
    -------
    V_app_func : callable
        Function V_app(t_array) -> V(t_array) [V].
        Accepts scalar or array-like time input [s] and returns a NumPy
        array with matching shape.

    Notes
    -----
    Supported waveform types (cfg.waveform_type):

    - "gaussian":
        V(t) = V_peak * exp(-((t - t_peak) / tau)^2)

    - "dc":
        V(t) = V_peak  (constant in time)

    - "step":
        V(t) = V_peak for t in [tV_start, tV_end]
               ≈ 0     otherwise
        A small floor (min_V) is used outside the step to avoid exactly
        zero voltage in this implementation.

    - "rf":
        V(t) = V_dc + V_peak * sin(2π f_rf t + phi_rf)
    """
    if cfg.waveform_type == "gaussian":
        def V_app_func(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t)
            return cfg.V_peak * np.exp(-((t - cfg.t_peak) / cfg.tau) ** 2)

    elif cfg.waveform_type == "dc":
        def V_app_func(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t)
            return cfg.V_peak * np.ones_like(t)

    elif cfg.waveform_type == "step":
        # Small nonzero floor to avoid exactly zero applied voltage
        # (can help prevent degeneracies in some models).
        min_V = 1e-15

        def V_app_func(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t)
            return (
                cfg.V_peak * ((t >= cfg.tV_start) & (t <= cfg.tV_end)) +
                min_V      * ((t < cfg.tV_start) | (t > cfg.tV_end))
            )

    # --- pure RF (optionally with DC bias) ---
    elif cfg.waveform_type == "rf":
        omega_rf = 2.0 * np.pi * cfg.f_rf
        V0       = cfg.V_dc
        Vrf      = cfg.V_peak
        phi_rf   = cfg.phi_rf

        def V_app_func(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t)
            return V0 + Vrf * np.sin(omega_rf * t + phi_rf)

    else:
        raise ValueError(f"Unknown waveform_type: {cfg.waveform_type}")

    return V_app_func


# ============================================================
# Transport and recombination coefficients
# ============================================================

def set_transport_coefficients(cfg: SimulationConfig) -> TransportCoeffs:
    """
    Compute transport and recombination coefficients for the selected gas.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation configuration containing:
        - gas          : gas name ("argon", "nitrogen", ...)
        - p_Torr       : pressure [Torr]
        - T_i, T_e     : ion/electron temperatures [K]

    Returns
    -------
    coeffs : TransportCoeffs
        Dataclass with:
        - mu_e, mu_i : electron/ion mobilities [m²/(V·s)]
        - D_e, D_i   : diffusion coefficients [m²/s]
        - beta       : volume recombination coefficient [m³/s]
        - pr         : reduced pressure p * (T_ref / T_i) (dimensionless)
        - T_e_eV     : electron temperature in eV (for diagnostics)
        - T_i_eV     : ion temperature in eV (for diagnostics)

    Notes
    -----
    * Pressure is supplied in Torr, so the empirical fits take the
      usual form mu ∝ 1 / p_Torr.

    * For argon, D_e is currently set equal to mu_e by choice
      (empirical placeholder, not Einstein relation).

    * For nitrogen, D_e and D_i are computed using the Einstein relation:

        D = mu (k_B T / e)

      with T in K and k_B, e in SI units.

    * T_e_eV and T_i_eV are fixed diagnostics values in the current model.
    """
    p_Torr = cfg.p_Torr
    T_i    = cfg.T_i
    T_e    = cfg.T_e

    # Reduced pressure (Surzhikov-style scaling), typical form:
    #   pr = p * (T_ref / T_i)
    # with T_ref ≈ 300 K
    pr      = p_Torr * 300.0 / T_i
    T_e_eV  = 1.0
    T_i_eV  = 0.0258

    gas = cfg.gas.lower()

    if gas == "argon":
        # Empirical scalings: mu ∝ 1 / p_Torr
        mu_e_val = 29.3 / p_Torr        # [m²/(V·s)]
        mu_i_val = 1.5e-1 / p_Torr      # [m²/(V·s)]

        # D_e is set equal to mu_e here (not Einstein), kept as an
        # empirical placeholder choice in the present implementation.
        D_e_val  = 29.3 / p_Torr        # [m²/s]
        D_i_val  = 0.006 / p_Torr       # [m²/s]

        beta_val = 2e-13                # [m³/s], volume recombination

    elif gas == "nitrogen":
        # Example parameters; can be replaced with updated fits later.
        mu_e_val = 30.4 / p_Torr        # [m²/(V·s)]
        mu_i_val = 2.09e-1 / p_Torr     # [m²/(V·s)]

        # Einstein relations: T in Kelvin
        D_e_val = mu_e_val * kB * T_e / e   # [m²/s]
        D_i_val = mu_i_val * kB * T_i / e   # [m²/s]

        beta_val = 2e-13                # [m³/s]

    else:
        raise NotImplementedError(f"Gas '{cfg.gas}' not implemented yet.")

    return TransportCoeffs(
        mu_e=np.float32(mu_e_val),
        mu_i=np.float32(mu_i_val),
        D_e=np.float32(D_e_val),
        D_i=np.float32(D_i_val),
        beta=np.float32(beta_val),
        pr=pr,
        T_e_eV=T_e_eV,
        T_i_eV=T_i_eV,
    )


# ============================================================
# Initial conditions
# ============================================================

def build_initial_conditions(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    V_app_func: Callable[[float], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Construct initial phi(x), E(x), n_e(x), n_i(x), and initial V_gap(0).

    Current initialization:
    - Uniform plasma: n_e(x) = n_i(x) = n0.
    - Zero potential: phi(x) = 0.
    - Zero electric field: E(x) = 0.

    Parameters
    ----------
    cfg : SimulationConfig
        Provides geometry (L, l, eps_r) and initial density n0.
    x_array : np.ndarray
        Spatial grid points [m], shape (Nx,).
    V_app_func : callable
        Applied voltage function V_app(t). Only V_app(0.0) is used here.

    Returns
    -------
    phi0 : np.ndarray
        Initial potential profile [V], shape (Nx,).
    E0 : np.ndarray
        Initial electric field profile [V/m], shape (Nx,).
    ne0 : np.ndarray
        Initial electron density [m⁻³], shape (Nx,).
    ni0 : np.ndarray
        Initial ion density [m⁻³], shape (Nx,).
    V0 : float
        Initial applied/gap voltage, V_app(0) [V].

    Notes
    -----
    The commented block below shows an alternative linear potential
    profile across an effective gas+dielectric length (Adamovic
    convention). It is intentionally left inactive.
    """
    Nx    = x_array.size
    n0    = cfg.n0

    # Initial gap voltage = applied voltage at t = 0
    V0 = float(V_app_func(0.0))

    # Active initialization: start from phi = 0, E = 0.
    phi0 = np.zeros(Nx, dtype=np.float32)
    E0   = np.zeros(Nx, dtype=np.float32)

    # Alternative initialization (inactive): linear phi across
    # gas + dielectric effective length.
    # effective_length = L + 2.0 * l / eps_r
    # phi0 = ((V0 / effective_length) * (L - x_array)).astype(np.float32)
    # dx   = x_array[1] - x_array[0]
    # E0   = (-np.gradient(phi0.astype(np.float64), dx)).astype(np.float32)

    # Uniform initial plasma
    ne0 = np.full(Nx, np.float32(n0), dtype=np.float32)
    ni0 = np.full(Nx, np.float32(n0), dtype=np.float32)

    return phi0, E0, ne0, ni0, V0


# ============================================================
# Boundary-physics closures
# ============================================================

def boundary_zero_density() -> float:
    """
    Return the zero-density closure value for a boundary state.
    """
    return 0.0


def boundary_electron_emission_density(
    boundary_side: str,
    gamma: float,
    ni_boundary: float,
    mu_i: float,
    mu_e: float,
    phi_boundary: float,
    phi_inner: float,
    dx: float,
    Gamma_ext: float = 0.0,
) -> float:
    """
    Electron-emission boundary closure in flux form, converted to density.

    Flux target follows the Eq. (11a)-style form with side-aware sign:

        Gamma_e,b = s * [Gamma_ext + gamma * Gamma_i,incident,b]

    where:
      - s = -1 for cathode (emitted electrons move toward -x),
      - s = +1 for anode   (emitted electrons move toward +x).

    The incident-ion flux magnitude is computed from local drift ion flux
    at the boundary using side-aware incidence:
      - cathode: max(Gamma_i, 0)
      - anode  : max(-Gamma_i, 0)

    The target electron flux is converted to boundary density through:

        Gamma_e = -mu_e * n_e * E.

    Parameters
    ----------
    boundary_side : str
        "anode" or "cathode".
    gamma : float
        Secondary electron emission coefficient.
    ni_boundary : float
        Ion density at the boundary cell.
    mu_i, mu_e : float
        Ion and electron mobilities.
    phi_boundary, phi_inner : float
        Potential at boundary node and nearest interior node.
    dx : float
        Grid spacing.
    Gamma_ext : float, optional
        External emission number flux magnitude [m^-2 s^-1].
    """
    if boundary_side not in ("anode", "cathode"):
        raise ValueError(f"Unknown boundary_side: {boundary_side}")

    # Local boundary electric field from one-sided potential gradient.
    E_boundary = -(phi_boundary - phi_inner) / dx

    # Drift ion flux and side-aware incident component magnitude.
    Gamma_i_drift = mu_i * ni_boundary * E_boundary
    if boundary_side == "cathode":
        Gamma_i_incident = max(Gamma_i_drift, 0.0)
        emission_flux_sign = -1.0
    else:
        Gamma_i_incident = max(-Gamma_i_drift, 0.0)
        emission_flux_sign = 1.0

    Gamma_e_target = emission_flux_sign * (Gamma_ext + gamma * Gamma_i_incident)

    # Convert target electron flux to density using drift closure.
    coeff = -mu_e * E_boundary  # Gamma_e = coeff * n_e
    if abs(coeff) <= 1e-20:
        return 0.0

    return max(Gamma_e_target / coeff, 0.0)


def boundary_cathode_ion_implicit_drift_density(
    ni_curr_right: float,
    ni_next_inner: float,
    phi_right: float,
    phi_inner: float,
    phi_inner2: float,
    gamma: float,
    mu_i: float,
    dx: float,
    dt: float,
) -> float:
    """
    Cathode-side implicit ion drift closure used by PASCHEN-1D.
    """
    mu_i_eff = (1.0 + gamma) * mu_i
    Ci = (mu_i_eff * dt) / dx

    dphi_R = (phi_right - phi_inner) / dx
    dphi_L = (phi_inner - phi_inner2) / dx

    den = 1.0 - Ci * dphi_R
    if den < 1e-12:
        den = 1e-12

    rhs_i = ni_curr_right - Ci * ni_next_inner * dphi_L
    return max(rhs_i / den, 0.0)


def boundary_anode_electron_implicit_drift_density(
    ne_curr_left: float,
    ne_next_inner: float,
    phi_left: float,
    phi_inner: float,
    phi_inner2: float,
    mu_e: float,
    dx: float,
    dt: float,
) -> float:
    """
    Anode-side implicit electron drift closure used by PASCHEN-1D.
    """
    Ce = (mu_e * dt) / dx
    dphi_01 = (phi_inner - phi_left) / dx
    dphi_12 = (phi_inner2 - phi_inner) / dx

    den = 1.0 - Ce * dphi_01
    if den < 1e-12:
        den = 1e-12

    rhs_e = ne_curr_left - Ce * ne_next_inner * dphi_12
    return max(rhs_e / den, 0.0)


# ============================================================
# Townsend ionization coefficient alpha(E)
# ============================================================

def compute_townsend_alpha(
    E_column: np.ndarray,
    p_Torr: float,
    pr: float,
    gas: str = "argon",
) -> np.ndarray:
    """
    Compute the Townsend ionization coefficient alpha [1/m] for a given
    electric-field distribution.

    Parameters
    ----------
    E_column : np.ndarray
        Local electric field array [V/m], shape (Nx,).
    p_Torr : float
        Gas pressure [Torr].
    pr : float
        Reduced pressure (dimensionless), typically p_Torr * T_ref / T_i.
        Currently not used in the active expression, but kept for API
        compatibility and potential future use.
    gas : str, optional
        Gas species ("argon" or "nitrogen" at present). Controls A, B
        parameters in the empirical alpha(E/p) fit.

    Returns
    -------
    alpha_column : np.ndarray
        Townsend ionization coefficient alpha(x) [1/m], shape (Nx,).

    Model
    -----
    Uses the standard exponential fit:

        alpha/p = A * exp(-B * p / E)

    with pressure p in Torr and E in V/m. Rearranged:

        alpha = p * A * exp(-B * p / E)

    where A and B are gas-dependent empirical constants. In typical LTP
    tabulations, A has units [1/(m*Torr)] and B has units [V/(m*Torr)].

    A small floor on |E| is introduced to avoid numerical issues for
    very weak fields (E → 0).
    """
    gas = gas.lower()

    # Gas-dependent A, B fits.
    if gas == "argon":
        A = 1150.0
        B = 17600.0
    elif gas == "nitrogen":
        A = 1180.0
        B = 34200.0
    else:
        raise NotImplementedError(f"Townsend alpha not implemented for gas '{gas}'")

    # Floor the magnitude of E to avoid division by extremely small values.
    # E_floor is chosen as (B * p) / floor_factor so that E/p never drops
    # too far below ~B / floor_factor.
    floor_factor = 20.0
    E_floor = B * p_Torr / floor_factor

    # Ensure double precision for exponent and apply |E|.
    Eabs = np.maximum(np.abs(E_column).astype(np.float64), E_floor)

    # alpha ~ p * A * exp[-B / (E/p)] written directly as:
    #   alpha = p * A * exp(-B * p / |E|)
    alpha_column = p_Torr * A * np.exp(-B * p_Torr / Eabs)
#     alpha_column = C * p_Torr * np.exp( -D * np.sqrt(p_Torr / Eabs) ) # More accurate for inert gases (Raizer) 

    return alpha_column
