"""
circuit.py

Lumped external-circuit coupling for PASCHEN-1D.

This module advances external circuit states and returns the updated plasma-gap
voltage `V_gap`, which then feeds the electrostatic solve in the plasma model.

Core coupling quantities:
- Plasma transport current from fluxes:
      I_transport = (A e / L) * integral(Gamma_i - Gamma_e) dx
- Geometric gas-gap capacitance:
      C_gap = eps0 * A / L
- Dielectric mapping coefficients:
      alpha_d = 1 + 2l/(eps_r L), beta_d = 2el/(eps0 eps_r L)

Supported topologies:
- "dielectric_plasma" / "none"
- "R0_Cp"
- "R0_Cp_Rm"
- "R0_Cs_Cp"
- "R0_Cs_Ls_Cp"
- "R0_Cs_Cp_Rm"
- "R0_Cs_Ls_Cp_Rm"
- "R0_Cs_Ls_Cp_Lp"
- "R0_Cs_Ls_Cp_Lp_Rm"

`step_circuit(...)` is the unified entry point. Internal update routines use
explicit Euler time stepping by design.
"""

from typing import Callable, Optional

import numpy as np

from physical_constants import e, eps0


# ============================================================
# Utility: plasma transport current from flux profiles
# ============================================================


def _compute_transport_current(
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
) -> float:
    """
    Compute plasma-branch transport current from ion/electron fluxes.

    Parameters
    ----------
    Gamma_i : np.ndarray
        Ion flux profile Gamma_i(x) [m⁻² s⁻¹] over the 1D grid (len = Nx).
    Gamma_e : np.ndarray
        Electron flux profile Gamma_e(x) [m⁻² s⁻¹] over the 1D grid (len = Nx).
    dx : float
        Spatial grid spacing [m].
    A : float
        Cross-sectional area of the discharge [m²].
    L : float
        Electrode separation (plasma gap length) [m].

    Returns
    -------
    I_transport : float
        Net transport current [A] flowing through the plasma branch.

    Notes
    -----
    Define the flux difference

        ΔGamma(x) = Gamma_i(x) - Gamma_e(x),

    and approximate its integral with the composite trapezoidal rule:

        ∫₀ᴸ ΔGamma(x) dx ≈ 0.5 * dx * [ΔGamma₀ + ΔGamma_{N-1} + 2 Σ ΔGamma_j (j = 1..N-2)]

    Then, with the 1D current-density relation j = e(Gamma_i - Gamma_e),

        I_transport = (A e / L) ∫₀ᴸ (Gamma_i - Gamma_e) dx

    which is exactly what is returned here.
    """
    flux_diff = Gamma_i - Gamma_e

    # Trapezoidal rule for ∫(Gamma_i - Gamma_e) dx
    integral_flux = 0.5 * dx * (
        flux_diff[0] + flux_diff[-1] + 2.0 * np.add.reduce(flux_diff[1:-1])
    )

    return (A * e / L) * integral_flux


# ============================================================
# Case A: dielectric-only (Adamovic), no external R/C/L
# ============================================================


def update_circuit_no_external_component(
    V_app_func: Callable[[float], np.ndarray | float],
    t: float,
    dt: float,
    V_gap_prev: float,
    V_d_prev: float,
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float = 0.0,
    C_p: float = 0.0,
    dielectric: Optional[bool] = None,
) -> tuple[float, float, float]:
    """
    Update V_gap with Adamovic dielectric relation only (no external R/C/L).

    Topology (conceptual)
    ---------------------
        Vs(t) ---[dielectric + plasma + dielectric]--- ground

    Coupling is handled only through the dynamic relation between source voltage
    and plasma-gap voltage with dielectric layers of thickness `l` and
    permittivity `eps_r`.

    Parameters
    ----------
    V_app_func : callable
        Function V_app_func(t) → Vs(t), returning the applied/source voltage [V].
    t : float
        Current time [s].
    dt : float
        Time step [s].
    V_gap_prev : float
        Plasma gap voltage at the previous time step, V_gap(t) [V].
    Gamma_i, Gamma_e : np.ndarray
        Ion and electron flux profiles [m⁻² s⁻¹].
    dx : float
        Spatial grid spacing [m].
    A : float
        Cross-sectional area [m²].
    L : float
        Plasma gap length [m].
    l : float
        Dielectric thickness near each electrode [m]. Must be > 0.
    eps_r : float
        Relative permittivity of the dielectric (dimensionless).
    R0 : float, optional
        Unused; retained for interface compatibility.
    C_p : float, optional
        Unused; retained for interface compatibility.
    dielectric : bool, optional
        Unused flag; retained for backward compatibility.

    Returns
    -------
    V_gap_new : float
        Updated plasma gap voltage at t + dt [V].
    I_tot : float
        Total current delivered to the plasma branch, including conduction
        and displacement contributions [A].
    """
    # Adamovic convention requires nonzero dielectric thickness.
    assert l > 0.0, "dielectric_plasma topology (Adamovic) requires l > 0."

    # Common pieces: conduction current and gap capacitance.
    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)
    C_gap = eps0 * A / L  # bare geometric capacitance of the gas gap

    # Recover the flux integral Φ from I_transport:
    # I_transport = (A e / L) * Φ  ⇒  Φ = I_transport * L / (A e)
    integral_flux = I_transport * L / (A * e)

    # Dielectric mapping coefficients.
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)

    # Approximate dV_app/dt with centered finite difference:
    # dV_app/dt ≈ [ V_app(t + dt) - V_app(t - dt) ] / (2 dt)
    dV_app_dt = (V_app_func(t + dt) - V_app_func(t - dt)) / (2.0 * dt)

    # Dielectric-voltage dynamics from Eq. (23)-compatible mapping.
    dV_d_dt = beta_d * integral_flux
    # Differential mapping (Eq. 19): alpha_d * dV_gap/dt = dV_s/dt - dV_d/dt
    dV_gap_dt = (dV_app_dt - dV_d_dt) / alpha_d

    # Explicit Euler update.
    V_gap_new = V_gap_prev + dt * dV_gap_dt
    V_d_new = V_d_prev + dt * dV_d_dt

    # Total current in the plasma branch (displacement + conduction)
    C_used = C_gap  # effective capacitance used here
    I_tot = I_transport + C_used * dV_gap_dt

    return V_gap_new, I_tot, V_d_new


# ============================================================
# Case B: R0 + Cp (+ optional dielectric)
# ============================================================


def update_circuit_R0_Cp(
    V_app_func: Callable[[float], float],
    t: float,
    dt: float,
    V_n_prev: float,
    V_gap_prev: float,
    V_d_prev: float,
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float,
    C_p: float,
) -> tuple[float, float, float, float]:
    """
    Update the circuit for the R0–Cp–plasma (+ dielectric) topology.

    Topology
    --------
        Vs(t) -- R0 -- (node V_n) --+-- Cp -- ground
                                       |
                                       +-- [dielectric + plasma + dielectric] -- ground

    - If l > 0: dielectric + plasma + dielectric (Adamovic convention).
    - If l = 0: reduces to R0 feeding a bare plasma gap (C_gap).
    - C_p = 0: reduces to R0 directly feeding the plasma branch.

    V_gap is always defined as the plasma voltage (across the gas gap).

    Parameters
    ----------
    V_app_func : callable
        V_app_func(t) → Vs(t), applied source voltage [V].
    t : float
        Current time [s].
    dt : float
        Time step [s].
    V_n_prev : float
        Node voltage at previous time step, V_n(t) [V].
    V_gap_prev : float
        Plasma gap voltage at previous time step, V_gap(t) [V].
    Gamma_i, Gamma_e : np.ndarray
        Ion and electron flux profiles [m⁻² s⁻¹].
    dx : float
        Spatial grid spacing [m].
    A : float
        Cross-sectional area [m²].
    L : float
        Plasma gap length [m].
    l : float
        Dielectric thickness [m].
    eps_r : float
        Relative permittivity of dielectric.
    R0 : float
        Series resistor [Ω].
    C_p : float
        Shunt (parallel) capacitor at the node [F].

    Returns
    -------
    V_n_new : float
        Updated node voltage at t + dt [V].
    V_gap_new : float
        Updated plasma gap voltage at t + dt [V].
    I_dis : float
        Discharge (plasma-branch) current at t + dt [A].
    """
    assert R0 > 0.0, "R0 must be > 0 for R0_Cp"

    # Geometric gap capacitance (bare gas gap)
    C_gap = eps0 * A / L

    # Conduction current from plasma
    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)

    # Recover Φ such that I_transport = (A e / L) * Φ
    Phi = I_transport * L / (A * e)

    # Dielectric mapping coefficients
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)

    Vs = float(V_app_func(t))

    dV_d_dt = beta_d * Phi

    # Denominator in dV_gap/dt expression.
    denom = C_p * alpha_d + C_gap
    if denom == 0.0:
        # Degenerate case: no gap-voltage update.
        dV_gap_dt = 0.0
    else:
        # Numerator from node KCL + dielectric mapping.
        num = (Vs - V_n_prev) / R0 - C_p * dV_d_dt - I_transport
        dV_gap_dt = num / denom

    # dV_n/dt from mapping: V_n = alpha_d * V_gap + V_d.
    dV_n_dt = alpha_d * dV_gap_dt + dV_d_dt

    # Explicit Euler updates
    V_gap_new = V_gap_prev + dt * dV_gap_dt
    V_d_new = V_d_prev + dt * dV_d_dt
    V_n_new = V_n_prev + dt * dV_n_dt

    # Discharge (plasma branch) current = conduction + capacitive
    I_dis = I_transport + C_gap * dV_gap_dt

    return V_n_new, V_gap_new, I_dis, V_d_new


# ============================================================
# Case C: R0 + Cp + Rm (+ optional dielectric)
# ============================================================


def update_circuit_R0_Cp_Rm(
    V_app_func: Callable[[float], float],
    t: float,
    dt: float,
    V_n_prev: float,
    V_gap_prev: float,  # V_plasma at time t
    V_d_prev: float,
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float,
    C_p: float,
    R_m: float,
) -> tuple[float, float, float, float]:
    """
    Dielectric-aware R0-Cp-Rm circuit with Adamovic convention.

    Topology
    --------
        Vs(t) -- R0 -- (node V_n) --+-- Cp -- ground
                                       |
                                       +-- Rm -- [dielectric + plasma + dielectric] -- ground

    V_gap is always the plasma voltage V_pl.

    - If l > 0: includes dielectric via dielectric mapping coefficients (alpha_d, beta_d).
    - If l = 0: smoothly reduces to bare electrodes (alpha_d → 1, beta_d → 0).

    Parameters
    ----------
    V_app_func : callable
        V_app_func(t) → Vs(t) [V].
    t : float
        Current time [s].
    dt : float
        Time step [s].
    V_n_prev : float
        Node voltage at previous time step [V].
    V_gap_prev : float
        Plasma gap voltage at previous time step [V].
    Gamma_i, Gamma_e : np.ndarray
        Ion and electron flux profiles [m⁻² s⁻¹].
    dx : float
        Spatial step [m].
    A : float
        Cross-sectional area [m²].
    L : float
        Plasma gap length [m].
    l : float
        Dielectric thickness [m].
    eps_r : float
        Relative permittivity of dielectric.
    R0 : float
        Series resistor [Ω].
    C_p : float
        Node shunt capacitor [F].
    R_m : float
        Matching / measurement resistor in series with plasma [Ω].

    Returns
    -------
    V_n_new : float
        Updated node voltage [V].
    V_gap_new : float
        Updated plasma gap voltage [V].
    I_plasma : float
        Plasma-branch (discharge) current [A].
    """
    # Basic checks
    assert R0 > 0.0, "R0 must be > 0 for R0_Cp_Rm circuit."
    assert C_p > 0.0, "C_p must be > 0 for R0_Cp_Rm circuit."
    assert R_m > 0.0, "R_m must be > 0 for R0_Cp_Rm circuit."

    # Geometric gap capacitance (bare gas gap)
    C_gap = eps0 * A / L

    # Conduction current from plasma
    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)

    # Flux integral Φ from I_transport: I_transport = (A e / L) * Φ
    Phi = I_transport * L / (A * e)

    # Dielectric mapping coefficients
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)  # → 1 when l = 0
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)  # → 0 when l = 0

    Vs = float(V_app_func(t))  # source voltage at time t

    dV_d_dt = beta_d * Phi

    # dV_gap/dt from Eq. (23):
    # (V_n - (alpha_d * V_gap + V_d)) / R_m = I_transport + C_gap * dV_gap/dt
    dV_gap_dt = (
        (V_n_prev - alpha_d * V_gap_prev - V_d_prev) / R_m - I_transport
    ) / C_gap

    # --- dV_n/dt from node KCL ---
    # (Vs - V_n)/R0 = C_p dV_n/dt + I_transport + C_gap dV_gap/dt
    # ⇒ dV_n/dt = [ (Vs - V_n)/R0 - I_transport - C_gap dV_gap/dt ] / C_p
    dV_n_dt = (
        (Vs - V_n_prev) / R0 - I_transport - C_gap * dV_gap_dt
    ) / C_p

    # Explicit Euler updates
    V_gap_new = V_gap_prev + dt * dV_gap_dt
    V_d_new = V_d_prev + dt * dV_d_dt
    V_n_new = V_n_prev + dt * dV_n_dt

    # Plasma-branch (discharge) current
    I_plasma = I_transport + C_gap * dV_gap_dt

    return V_n_new, V_gap_new, I_plasma, V_d_new


# ============================================================
# Case D: R0 + Cs + Cp (no Rm, + optional dielectric)
# ============================================================


def update_circuit_R0_Cs_Cp(
    V_app_func: Callable[[float], float],
    t: float,
    dt: float,
    V_n_prev: float,
    V_gap_prev: float,  # plasma voltage at time t
    V_d_prev: float,
    V_Cs_prev: float,  # series capacitor voltage at time t
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float,
    C_s: float,
    C_p: float,
) -> tuple[float, float, float, float, float]:
    """
    Dielectric-aware R0-Cs-Cp circuit (no Rm).
    """
    # Safety checks
    assert R0 > 0.0, "R0 must be > 0 for R0_Cs_Cp circuit."
    assert C_s > 0.0, "C_s must be > 0 for R0_Cs_Cp circuit."
    assert C_p > 0.0, "C_p must be > 0 for R0_Cs_Cp circuit."

    # 1) Source voltage and series-branch current through R0 and Cs
    Vs = float(V_app_func(t))
    I_s = (Vs - V_n_prev - V_Cs_prev) / R0  # [A]

    # 2) Update series capacitor voltage
    V_Cs_new = V_Cs_prev + dt * (I_s / C_s)

    # 3) Plasma conduction current
    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)

    # 4) Geometric gap capacitance
    C_gap = eps0 * A / L  # [F]

    # 5) Flux integral Φ from I_transport: I_transport = (A e / L) * Φ
    Phi = I_transport * L / (A * e)

    # 6) Dielectric mapping coefficients
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)

    dV_d_dt = beta_d * Phi

    # 7) Node KCL + mapping derivative:
    #    I_s = C_p*dV_n/dt + I_transport + C_gap*dV_gap/dt
    #    dV_n/dt = alpha_d*dV_gap/dt + dV_d/dt
    dV_gap_dt = (I_s - I_transport - C_p * dV_d_dt) / (C_p * alpha_d + C_gap)

    # 8) Mapping derivative for node voltage
    dV_n_dt = alpha_d * dV_gap_dt + dV_d_dt

    # 9) Explicit Euler updates
    V_gap_new = V_gap_prev + dt * dV_gap_dt
    V_d_new = V_d_prev + dt * dV_d_dt
    V_n_new = V_n_prev + dt * dV_n_dt

    # 10) Plasma-branch (discharge) current
    I_pl = I_transport + C_gap * dV_gap_dt

    return V_n_new, V_gap_new, I_pl, V_Cs_new, V_d_new


# ============================================================
# Case E: R0 + Cs + Cp + Rm (+ optional dielectric)
# ============================================================


def update_circuit_R0_Cs_Cp_Rm(
    V_app_func: Callable[[float], float],
    t: float,
    dt: float,
    V_n_prev: float,
    V_gap_prev: float,  # plasma voltage at time t
    V_d_prev: float,
    V_Cs_prev: float,  # series capacitor voltage at time t
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float,
    C_s: float,
    C_p: float,
    R_m: float,
) -> tuple[float, float, float, float, float]:
    """
    Dielectric-aware R0-Cs-Cp-Rm circuit with Adamovic convention.

    Topology
    --------
        Vs(t) -- R0 -- Cs -- (node V_n) --+-- Cp -- ground
                                             |
                                             +-- Rm -- [dielectric + plasma + dielectric] -- ground

    V_gap is always the plasma voltage V_pl.

    - If l > 0: includes dielectric via dielectric mapping coefficients (alpha_d, beta_d).
    - If l = 0: smoothly reduces to bare electrodes (alpha_d → 1, beta_d → 0).

    Requires: R0 > 0, C_s > 0, C_p > 0, R_m > 0.

    Parameters
    ----------
    V_app_func : callable
        V_app_func(t) → Vs(t) [V].
    t : float
        Current time [s].
    dt : float
        Time step [s].
    V_n_prev : float
        Node voltage at previous step [V].
    V_gap_prev : float
        Plasma gap voltage at previous step [V].
    V_Cs_prev : float
        Series capacitor Cs voltage at previous step [V].
    Gamma_i, Gamma_e : np.ndarray
        Ion and electron flux profiles [m⁻² s⁻¹].
    dx : float
        Spatial step [m].
    A : float
        Cross-sectional area [m²].
    L : float
        Plasma gap length [m].
    l : float
        Dielectric thickness [m].
    eps_r : float
        Relative permittivity of the dielectric.
    R0 : float
        Series resistor [Ω].
    C_s : float
        Series capacitor [F].
    C_p : float
        Node shunt capacitor [F].
    R_m : float
        Resistive branch in series with plasma [Ω].

    Returns
    -------
    V_n_new : float
        Updated node voltage [V].
    V_gap_new : float
        Updated plasma gap voltage [V].
    I_pl : float
        Plasma-branch current [A].
    V_Cs_new : float
        Updated voltage across Cs [V].
    """
    # Safety checks
    assert R0 > 0.0, "R0 must be > 0 for R0_Cs_Cp_Rm circuit."
    assert C_s > 0.0, "C_s must be > 0 for R0_Cs_Cp_Rm circuit."
    assert C_p > 0.0, "C_p must be > 0 for R0_Cs_Cp_Rm circuit."
    assert R_m > 0.0, "R_m must be > 0 for R0_Cs_Cp_Rm circuit."

    # 1) Source voltage and series-branch current through R0 and Cs
    Vs = float(V_app_func(t))
    I_s = (Vs - V_n_prev - V_Cs_prev) / R0  # [A]

    # 2) Update series capacitor voltage
    V_Cs_new = V_Cs_prev + dt * (I_s / C_s)

    # 3) Plasma conduction current
    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)

    # 4) Geometric gap capacitance
    C_gap = eps0 * A / L  # [F]

    # 5) Flux integral Φ from I_transport: I_transport = (A e / L) * Φ
    Phi = I_transport * L / (A * e)

    # 6) Dielectric mapping coefficients
    #    For l → 0: alpha_d → 1, beta_d → 0 → bare electrodes
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)

    dV_d_dt = beta_d * Phi

    # 7) dV_gap/dt from Eq. (23):
    #    (V_n - (alpha_d * V_gap + V_d)) / R_m = I_transport + C_gap * dV_gap/dt
    dV_gap_dt = (
        (V_n_prev - alpha_d * V_gap_prev - V_d_prev) / R_m - I_transport
    ) / C_gap

    # 8) dV_n/dt from node KCL:
    #    I_s = C_p dV_n/dt + I_transport + C_gap dV_gap/dt
    #    ⇒ dV_n/dt = [ I_s - I_transport - C_gap dV_gap/dt ] / C_p
    dV_n_dt = (I_s - I_transport - C_gap * dV_gap_dt) / C_p

    # 9) Explicit Euler updates
    V_gap_new = V_gap_prev + dt * dV_gap_dt
    V_d_new = V_d_prev + dt * dV_d_dt
    V_n_new = V_n_prev + dt * dV_n_dt

    # 10) Plasma-branch (discharge) current
    I_pl = I_transport + C_gap * dV_gap_dt

    return V_n_new, V_gap_new, I_pl, V_Cs_new, V_d_new


# ============================================================
# Case E: R0 + Cs + Ls + Cp + Rm (+ optional dielectric)
# ============================================================


def update_circuit_R0_Cs_Ls_Cp_Rm(
    V_app_func: Callable[[float], float],
    t: float,
    dt: float,
    V_n_prev: float,
    V_gap_prev: float,  # plasma voltage at time t
    V_d_prev: float,
    V_Cs_prev: float,  # series capacitor voltage at time t
    I_s_prev: float,  # series-branch current at time t
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float,
    C_s: float,
    L_s: float,
    C_p: float,
    R_m: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Dielectric-aware R0-Cs-Ls-Cp-Rm circuit with Adamovic convention.

    Topology
    --------
        Vs(t) -- R0 -- Cs -- Ls -- (node V_n) --+-- Cp -- ground
                                                  |
                                                  +-- Rm -- [dielectric + plasma + dielectric] -- ground

    State variables
    ---------------
    V_gap : plasma voltage
    V_n   : node voltage
    V_Cs   : series capacitor voltage
    I_s    : series-branch current (R0–Cs–Ls)

    Parameters
    ----------
    V_app_func : callable
        Applied source waveform Vs(t) [V].
    t : float
        Current time [s].
    dt : float
        Time step [s].
    V_n_prev, V_gap_prev, V_Cs_prev, I_s_prev : float
        Previous-step state values for node voltage, plasma gap voltage,
        series-capacitor voltage, and series-branch current.
    Gamma_i, Gamma_e : np.ndarray
        Ion/electron flux profiles [m^-2 s^-1].
    dx : float
        Spatial step [m].
    A : float
        Discharge area [m^2].
    L : float
        Gap length [m].
    l : float
        Dielectric thickness [m].
    eps_r : float
        Relative permittivity of dielectric.
    R0, C_s, L_s, C_p, R_m : float
        Circuit component values (SI units).

    Returns
    -------
    V_n_new : float
        Updated node voltage [V].
    V_gap_new : float
        Updated plasma gap voltage [V].
    I_pl : float
        Plasma-branch (discharge) current [A].
    V_Cs_new : float
        Updated Cs voltage [V].
    I_s_new : float
        Updated series branch current [A].
    """
    # Safety checks
    assert R0 > 0.0, "R0 must be > 0 for R0_Cs_Ls_Cp_Rm circuit."
    assert C_s > 0.0, "C_s must be > 0 for R0_Cs_Ls_Cp_Rm circuit."
    assert L_s > 0.0, "L_s must be > 0 for R0_Cs_Ls_Cp_Rm circuit."
    assert C_p > 0.0, "C_p must be > 0 for R0_Cs_Ls_Cp_Rm circuit."
    assert R_m > 0.0, "R_m must be > 0 for R0_Cs_Ls_Cp_Rm circuit."

    # 1) Source voltage
    Vs = float(V_app_func(t))

    # 2) Plasma conduction current
    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)

    # 3) Geometric gap capacitance
    C_gap = eps0 * A / L  # [F]

    # 4) Flux integral Φ from I_transport: I_transport = (A e / L) * Φ
    Phi = I_transport * L / (A * e)

    # 5) Dielectric mapping coefficients
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)  # → 1 when l = 0
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)  # → 0 when l = 0

    dV_d_dt = beta_d * Phi

    # 6) dV_gap/dt from Eq. (23):
    #    (V_n - (alpha_d * V_gap + V_d)) / R_m = I_transport + C_gap * dV_gap/dt
    dV_gap_dt = (
        (V_n_prev - alpha_d * V_gap_prev - V_d_prev) / R_m - I_transport
    ) / C_gap

    # 7) Node KCL:
    #    I_s = C_p dV_n/dt + I_transport + C_gap dV_gap/dt
    dV_n_dt = (I_s_prev - I_transport - C_gap * dV_gap_dt) / C_p

    # 8) Inductor dynamic via KVL on R0–Cs–Ls–node branch:
    #    Vs - R0 I_s - V_Cs - V_Ls - V_n = 0,  V_Ls = L_s dI_s/dt
    dI_s_dt = (Vs - R0 * I_s_prev - V_Cs_prev - V_n_prev) / L_s

    # 9) Series capacitor Cs: dV_Cs/dt = I_s / C_s
    dV_Cs_dt = I_s_prev / C_s

    # Explicit Euler updates
    V_gap_new = V_gap_prev + dt * dV_gap_dt
    V_d_new = V_d_prev + dt * dV_d_dt
    V_n_new = V_n_prev + dt * dV_n_dt
    I_s_new = I_s_prev + dt * dI_s_dt
    V_Cs_new = V_Cs_prev + dt * dV_Cs_dt

    # Plasma-branch current
    I_pl = I_transport + C_gap * dV_gap_dt

    return V_n_new, V_gap_new, I_pl, V_Cs_new, I_s_new, V_d_new


# ============================================================
# Case F: R0 + Cs + Ls + Cp (no Rm, + optional dielectric)
# ============================================================


def update_circuit_R0_Cs_Ls_Cp(
    V_app_func: Callable[[float], float],
    t: float,
    dt: float,
    V_n_prev: float,
    V_gap_prev: float,  # plasma voltage at time t
    V_d_prev: float,
    V_Cs_prev: float,  # series capacitor voltage at time t
    I_s_prev: float,  # series-branch current at time t
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float,
    C_s: float,
    L_s: float,
    C_p: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Dielectric-aware R0-Cs-Ls-Cp circuit (no Rm).
    """
    # Safety checks
    assert R0 > 0.0, "R0 must be > 0 for R0_Cs_Ls_Cp circuit."
    assert C_s > 0.0, "C_s must be > 0 for R0_Cs_Ls_Cp circuit."
    assert L_s > 0.0, "L_s must be > 0 for R0_Cs_Ls_Cp circuit."
    assert C_p > 0.0, "C_p must be > 0 for R0_Cs_Ls_Cp circuit."

    # 1) Source voltage
    Vs = float(V_app_func(t))

    # 2) Plasma conduction current
    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)

    # 3) Geometric gap capacitance
    C_gap = eps0 * A / L  # [F]

    # 4) Flux integral Φ from I_transport
    Phi = I_transport * L / (A * e)

    # 5) Dielectric mapping coefficients
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)

    dV_d_dt = beta_d * Phi

    # 6) Node KCL + mapping derivative:
    #    I_s = C_p*dV_n/dt + I_transport + C_gap*dV_gap/dt
    #    dV_n/dt = alpha_d*dV_gap/dt + dV_d/dt
    dV_gap_dt = (I_s_prev - I_transport - C_p * dV_d_dt) / (C_p * alpha_d + C_gap)
    dV_n_dt = alpha_d * dV_gap_dt + dV_d_dt

    # 7) Series branch KVL (R0-Cs-Ls-node)
    dI_s_dt = (Vs - R0 * I_s_prev - V_Cs_prev - V_n_prev) / L_s

    # 8) Series capacitor Cs dynamic
    dV_Cs_dt = I_s_prev / C_s

    # Explicit Euler updates
    V_gap_new = V_gap_prev + dt * dV_gap_dt
    V_d_new = V_d_prev + dt * dV_d_dt
    V_n_new = V_n_prev + dt * dV_n_dt
    I_s_new = I_s_prev + dt * dI_s_dt
    V_Cs_new = V_Cs_prev + dt * dV_Cs_dt

    # Plasma-branch current
    I_pl = I_transport + C_gap * dV_gap_dt

    return V_n_new, V_gap_new, I_pl, V_Cs_new, I_s_new, V_d_new


# ============================================================
# Case G: R0 + Cs + Ls + Cp + Lp (no Rm, + optional dielectric)
# ============================================================


def update_circuit_R0_Cs_Ls_Cp_Lp(
    V_app_func: Callable[[float], float],
    t: float,
    dt: float,
    V_n_prev: float,
    V_gap_prev: float,  # plasma voltage at time t
    V_d_prev: float,  # dielectric mapping voltage at time t
    V_Cs_prev: float,  # series capacitor voltage at time t
    I_s_prev: float,  # series-branch current at time t
    I_Lp_prev: float,  # shunt inductor current at time t
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float,
    C_s: float,
    L_s: float,
    C_p: float,
    L_p: float,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Dielectric-aware R0-Cs-Ls-(Cp || Lp || plasma) circuit (no Rm).

    Topology
    --------
        Vs(t) -- R0 -- Cs -- Ls -- (node V_n) --+-- Cp -- ground
                                                  +-- Lp -- ground
                                                  +-- [dielectric + plasma + dielectric] -- ground

    State variables
    ---------------
    V_gap : plasma voltage
    V_d   : dielectric mapping voltage
    V_n   : node voltage
    V_Cs  : series capacitor voltage (Cs)
    I_s   : series-branch current (R0–Cs–Ls)
    I_Lp  : shunt inductor current through L_p
    """
    # Safety checks
    assert R0 > 0.0, "R0 must be > 0 for R0_Cs_Ls_Cp_Lp circuit."
    assert C_s > 0.0, "C_s must be > 0 for R0_Cs_Ls_Cp_Lp circuit."
    assert L_s > 0.0, "L_s must be > 0 for R0_Cs_Ls_Cp_Lp circuit."
    assert C_p > 0.0, "C_p must be > 0 for R0_Cs_Ls_Cp_Lp circuit."
    assert L_p > 0.0, "L_p must be > 0 for R0_Cs_Ls_Cp_Lp circuit."

    # 1) Source voltage
    Vs = float(V_app_func(t))

    # 2) Plasma conduction current
    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)

    # 3) Geometric gap capacitance
    C_gap = eps0 * A / L  # [F]

    # 4) Flux integral Φ from I_transport: I_transport = (A e / L) * Φ
    Phi = I_transport * L / (A * e)

    # 5) Dielectric mapping coefficients
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)  # → 1 when l = 0
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)  # → 0 when l = 0

    # 6) Dielectric mapping ODE
    dV_d_dt = beta_d * Phi

    # 7) Node KCL + mapping derivative:
    #    I_s = C_p*dV_n/dt + I_Lp + I_transport + C_gap*dV_gap/dt
    #    dV_n/dt = alpha_d*dV_gap/dt + dV_d/dt
    #    => dV_gap/dt = [I_s - I_Lp - I_transport - C_p*dV_d/dt] / (C_p*alpha_d + C_gap)
    denom = C_p * alpha_d + C_gap
    dV_gap_dt = (I_s_prev - I_Lp_prev - I_transport - C_p * dV_d_dt) / denom

    # 8) Mapping derivative for node voltage
    dV_n_dt = alpha_d * dV_gap_dt + dV_d_dt

    # 9) Series branch (R0–Cs–Ls) via KVL:
    #    Vs - R0 I_s - V_Cs - V_Ls - V_n = 0,  V_Ls = L_s dI_s/dt
    dI_s_dt = (Vs - R0 * I_s_prev - V_Cs_prev - V_n_prev) / L_s

    # 10) Series capacitor Cs: dV_Cs/dt = I_s / C_s
    dV_Cs_dt = I_s_prev / C_s

    # 11) Parallel inductor L_p at node: V_n = L_p dI_Lp/dt
    dI_Lp_dt = V_n_prev / L_p

    # Explicit Euler updates
    V_gap_new = V_gap_prev + dt * dV_gap_dt
    V_d_new = V_d_prev + dt * dV_d_dt
    V_n_new = V_n_prev + dt * dV_n_dt
    I_s_new = I_s_prev + dt * dI_s_dt
    V_Cs_new = V_Cs_prev + dt * dV_Cs_dt
    I_Lp_new = I_Lp_prev + dt * dI_Lp_dt

    # Plasma-branch current
    I_pl = I_transport + C_gap * dV_gap_dt

    return V_n_new, V_gap_new, I_pl, V_Cs_new, I_s_new, I_Lp_new, V_d_new


# ============================================================
# Case G: R0 + Cs + Ls + Cp + Lp + Rm (+ optional dielectric)
# ============================================================


def update_circuit_R0_Cs_Ls_Cp_Lp_Rm(
    V_app_func: Callable[[float], float],
    t: float,
    dt: float,
    V_n_prev: float,
    V_gap_prev: float,  # plasma voltage at time t
    V_d_prev: float,
    V_Cs_prev: float,  # series capacitor voltage at time t
    I_s_prev: float,  # series-branch current at time t
    I_Lp_prev: float,  # shunt inductor current at time t
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float,
    C_s: float,
    L_s: float,
    C_p: float,
    L_p: float,
    R_m: float,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Dielectric-aware R0-Cs-Ls-(Cp || Lp || Rm+plasma) circuit with Adamovic convention.

    Topology
    --------
        Vs(t) -- R0 -- Cs -- Ls -- (node V_n) --+-- Cp -- ground
                                                  +-- Lp -- ground
                                                  +-- Rm -- [dielectric + plasma + dielectric] -- ground

    State variables
    ---------------
    V_gap : plasma voltage
    V_n   : node voltage
    V_Cs   : series capacitor voltage (Cs)
    I_s    : series-branch current (R0–Cs–Ls)
    I_Lp   : shunt inductor current through L_p

    Parameters
    ----------
    V_app_func : callable
        Applied source waveform Vs(t) [V].
    t : float
        Current time [s].
    dt : float
        Time step [s].
    V_n_prev, V_gap_prev, V_Cs_prev, I_s_prev, I_Lp_prev : float
        Previous-step state values for node voltage, plasma gap voltage,
        series-capacitor voltage, series-branch current, and shunt-inductor current.
    Gamma_i, Gamma_e : np.ndarray
        Ion/electron flux profiles [m^-2 s^-1].
    dx : float
        Spatial step [m].
    A : float
        Discharge area [m^2].
    L : float
        Gap length [m].
    l : float
        Dielectric thickness [m].
    eps_r : float
        Relative permittivity of dielectric.
    R0, C_s, L_s, C_p, L_p, R_m : float
        Circuit component values (SI units).

    Returns
    -------
    V_n_new : float
        Updated node voltage [V].
    V_gap_new : float
        Updated plasma gap voltage [V].
    I_pl : float
        Plasma-branch current [A].
    V_Cs_new : float
        Updated Cs voltage [V].
    I_s_new : float
        Updated series branch current [A].
    I_Lp_new : float
        Updated shunt inductor current [A].
    """
    # Safety checks
    assert R0 > 0.0, "R0 must be > 0 for R0_Cs_Ls_Cp_Lp_Rm circuit."
    assert C_s > 0.0, "C_s must be > 0 for R0_Cs_Ls_Cp_Lp_Rm circuit."
    assert L_s > 0.0, "L_s must be > 0 for R0_Cs_Ls_Cp_Lp_Rm circuit."
    assert C_p > 0.0, "C_p must be > 0 for R0_Cs_Ls_Cp_Lp_Rm circuit."
    assert L_p > 0.0, "L_p must be > 0 for R0_Cs_Ls_Cp_Lp_Rm circuit."
    assert R_m > 0.0, "R_m must be > 0 for R0_Cs_Ls_Cp_Lp_Rm circuit."

    # 1) Source voltage
    Vs = float(V_app_func(t))

    # 2) Plasma conduction current
    I_transport = _compute_transport_current(Gamma_i, Gamma_e, dx, A, L)

    # 3) Geometric gap capacitance
    C_gap = eps0 * A / L  # [F]

    # 4) Flux integral Φ from I_transport: I_transport = (A e / L) * Φ
    Phi = I_transport * L / (A * e)

    # 5) Dielectric mapping coefficients
    alpha_d = 1.0 + 2.0 * l / (eps_r * L)  # → 1 when l = 0
    beta_d = (2.0 * e * l) / (eps0 * eps_r * L)  # → 0 when l = 0

    dV_d_dt = beta_d * Phi

    # 6) Gap-voltage dynamics from Eq. (23):
    #    (V_n - (alpha_d * V_gap + V_d)) / R_m = I_transport + C_gap * dV_gap/dt
    dV_gap_dt = (
        (V_n_prev - alpha_d * V_gap_prev - V_d_prev) / R_m - I_transport
    ) / C_gap

    # 7) Node KCL with Cp, Lp, and plasma branch:
    #    I_s = I_Cp + I_Lp + I_plasma
    #        = C_p dV_n/dt + I_Lp_prev + I_transport + C_gap dV_gap/dt
    dV_n_dt = (
        I_s_prev - I_Lp_prev - I_transport - C_gap * dV_gap_dt
    ) / C_p

    # 8) Series branch (R0–Cs–Ls) via KVL:
    #    Vs - R0 I_s - V_Cs - V_Ls - V_n = 0,  V_Ls = L_s dI_s/dt
    dI_s_dt = (Vs - R0 * I_s_prev - V_Cs_prev - V_n_prev) / L_s

    # 9) Series capacitor Cs: dV_Cs/dt = I_s / C_s
    dV_Cs_dt = I_s_prev / C_s

    # 10) Parallel inductor L_p at node:
    #     V_n = L_p dI_Lp/dt
    dI_Lp_dt = V_n_prev / L_p

    # Explicit Euler updates (can be upgraded to RK if needed)
    V_gap_new = V_gap_prev + dt * dV_gap_dt
    V_d_new = V_d_prev + dt * dV_d_dt
    V_n_new = V_n_prev + dt * dV_n_dt
    I_s_new = I_s_prev + dt * dI_s_dt
    V_Cs_new = V_Cs_prev + dt * dV_Cs_dt
    I_Lp_new = I_Lp_prev + dt * dI_Lp_dt

    # Plasma-branch current
    I_pl = I_transport + C_gap * dV_gap_dt

    return V_n_new, V_gap_new, I_pl, V_Cs_new, I_s_new, I_Lp_new, V_d_new


# ============================================================
# Unified interface: single-step circuit advance
# ============================================================


def step_circuit(
    circuit_type: str,
    V_app_func: Callable[[float], float],
    t: float,
    dt: float,
    V_gap_prev: float,
    Gamma_i: np.ndarray,
    Gamma_e: np.ndarray,
    dx: float,
    A: float,
    L: float,
    l: float,
    eps_r: float,
    R0: float,
    C_s: float,
    C_p: float,
    R_m: float,
    L_s: float,
    L_p: float,
    V_d_prev: Optional[float],
    V_n_prev: Optional[float],
    V_Cs_prev: Optional[float],
    I_s_prev: Optional[float],
    I_Lp_prev: Optional[float],
) -> tuple[
    float,
    float,
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """
    Advance the external circuit by one time step in a unified way.

    Public entry point for all supported external-circuit topologies.
    Based on `circuit_type`, the function dispatches to the appropriate
    update routine and returns a consistent set of outputs.

    Parameters
    ----------
    circuit_type : str
        Name of the circuit topology to use. Valid values include:

            "R0_Cs_Ls_Cp_Lp"
            "R0_Cs_Ls_Cp_Lp_Rm"
            "R0_Cs_Ls_Cp_Rm"
            "R0_Cs_Cp_Rm"
            "R0_Cs_Ls_Cp"
            "R0_Cs_Cp"
            "R0_Cp_Rm"
            "R0_Cp" or "R"
            "dielectric_plasma" or "none"

        If none of these match, an auto-detect based on the parameter
        values (R0, C_s, C_p, R_m, l) is attempted, otherwise an error
        is raised.
    V_app_func : callable
        V_app_func(t) → Vs(t) [V], applied source voltage.
    t : float
        Current time [s].
    dt : float
        Time step [s].
    V_gap_prev : float
        Plasma gap voltage at previous step [V].
    Gamma_i, Gamma_e : np.ndarray
        Ion and electron flux profiles [m⁻² s⁻¹].
    dx : float
        Spatial step [m].
    A : float
        Cross-sectional area [m²].
    L : float
        Plasma gap length [m].
    l : float
        Dielectric thickness [m].
    eps_r : float
        Relative permittivity of dielectric.
    R0 : float
        Series resistor [Ω].
    C_s : float
        Series capacitor [F].
    C_p : float
        Node shunt capacitor [F].
    R_m : float
        Matching / measurement resistor [Ω].
    L_s : float
        Series inductor [H].
    L_p : float
        Shunt inductor at the node [H].
    V_d_prev : float or None
        Dielectric mapping voltage state at previous step [V], if applicable.
    V_n_prev : float or None
        Node voltage at previous step [V], if required by the chosen topology.
    V_Cs_prev : float or None
        Series capacitor Cs voltage at previous step [V], if required.
    I_s_prev : float or None
        Series branch current at previous step [A], if required.
    I_Lp_prev : float or None
        Shunt inductor current at previous step [A], if required.

    Returns
    -------
    V_gap_new : float
        Updated plasma gap voltage [V].
    I_discharge : float
        Plasma-branch current (conduction + displacement) [A].
    V_d_new : float or None
        Updated dielectric mapping voltage state [V], if applicable.
    V_n_new : float or None
        Updated node voltage [V], if applicable; otherwise None.
    V_Cs_new : float or None
        Updated Cs voltage [V], if applicable; otherwise None.
    I_s_new : float or None
        Updated series branch current [A], if applicable; otherwise None.
    I_Lp_new : float or None
        Updated shunt inductor current [A], if applicable; otherwise None.

    Notes
    -----
    Return order is fixed:
        (V_gap_new, I_discharge, V_d_new, V_n_new, V_Cs_new, I_s_new, I_Lp_new)

    For topologies that do not use one or more circuit state variables, the
    corresponding return values are None.
    """
    # ------------------------------------------------------------
    # 0) R0–Cs–Ls–Cp–Lp + dielectric-aware plasma branch (no Rm)
    # ------------------------------------------------------------
    if circuit_type == "R0_Cs_Ls_Cp_Lp":
        if any(v is None for v in (V_n_prev, V_Cs_prev, I_s_prev, I_Lp_prev)):
            raise ValueError(
                "R0_Cs_Ls_Cp_Lp requires V_n_prev, V_Cs_prev, I_s_prev, I_Lp_prev."
            )

        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_n_new, V_gap_new, I_pl, V_Cs_new, I_s_new, I_Lp_new, V_d_new = (
            update_circuit_R0_Cs_Ls_Cp_Lp(
                V_app_func=V_app_func,
                t=t,
                dt=dt,
                V_n_prev=V_n_prev,
                V_gap_prev=V_gap_prev,
                V_d_prev=V_d_prev,
                V_Cs_prev=V_Cs_prev,
                I_s_prev=I_s_prev,
                I_Lp_prev=I_Lp_prev,
                Gamma_i=Gamma_i,
                Gamma_e=Gamma_e,
                dx=dx,
                A=A,
                L=L,
                l=l,
                eps_r=eps_r,
                R0=R0,
                C_s=C_s,
                L_s=L_s,
                C_p=C_p,
                L_p=L_p,
            )
        )
        return V_gap_new, I_pl, V_d_new, V_n_new, V_Cs_new, I_s_new, I_Lp_new

    # ------------------------------------------------------------
    # 1) R0–Cs–Ls–Cp–Lp–Rm + dielectric-aware plasma branch
    # ------------------------------------------------------------
    if circuit_type == "R0_Cs_Ls_Cp_Lp_Rm":
        if R_m <= 0.0:
            # Automatic reduction to no-Rm topology.
            circuit_type = "R0_Cs_Ls_Cp_Lp"
        else:
            if any(v is None for v in (V_n_prev, V_Cs_prev, I_s_prev, I_Lp_prev)):
                raise ValueError(
                    "R0_Cs_Ls_Cp_Lp_Rm requires V_n_prev, V_Cs_prev, I_s_prev, I_Lp_prev."
                )

            if V_d_prev is None:
                raise ValueError("Dielectric mapping requires V_d_prev.")

            V_n_new, V_gap_new, I_pl, V_Cs_new, I_s_new, I_Lp_new, V_d_new = (
                update_circuit_R0_Cs_Ls_Cp_Lp_Rm(
                    V_app_func=V_app_func,
                    t=t,
                    dt=dt,
                    V_n_prev=V_n_prev,
                    V_gap_prev=V_gap_prev,
                    V_d_prev=V_d_prev,
                    V_Cs_prev=V_Cs_prev,
                    I_s_prev=I_s_prev,
                    I_Lp_prev=I_Lp_prev,
                    Gamma_i=Gamma_i,
                    Gamma_e=Gamma_e,
                    dx=dx,
                    A=A,
                    L=L,
                    l=l,
                    eps_r=eps_r,
                    R0=R0,
                    C_s=C_s,
                    L_s=L_s,
                    C_p=C_p,
                    L_p=L_p,
                    R_m=R_m,
                )
            )
            return V_gap_new, I_pl, V_d_new, V_n_new, V_Cs_new, I_s_new, I_Lp_new

    # ------------------------------------------------------------
    # 2) R0–Cs–Ls–Cp–Rm + dielectric-aware plasma branch
    # ------------------------------------------------------------
    if circuit_type == "R0_Cs_Ls_Cp_Rm":
        if R_m <= 0.0:
            # Automatic reduction to no-Rm topology.
            circuit_type = "R0_Cs_Ls_Cp"
        else:
            if V_n_prev is None or V_Cs_prev is None or I_s_prev is None:
                raise ValueError(
                    "R0_Cs_Ls_Cp_Rm requires V_n_prev, V_Cs_prev, and I_s_prev."
                )

            if V_d_prev is None:
                raise ValueError("Dielectric mapping requires V_d_prev.")

            V_n_new, V_gap_new, I_pl, V_Cs_new, I_s_new, V_d_new = (
                update_circuit_R0_Cs_Ls_Cp_Rm(
                    V_app_func=V_app_func,
                    t=t,
                    dt=dt,
                    V_n_prev=V_n_prev,
                    V_gap_prev=V_gap_prev,
                    V_d_prev=V_d_prev,
                    V_Cs_prev=V_Cs_prev,
                    I_s_prev=I_s_prev,
                    Gamma_i=Gamma_i,
                    Gamma_e=Gamma_e,
                    dx=dx,
                    A=A,
                    L=L,
                    l=l,
                    eps_r=eps_r,
                    R0=R0,
                    C_s=C_s,
                    L_s=L_s,
                    C_p=C_p,
                    R_m=R_m,
                )
            )
            return V_gap_new, I_pl, V_d_new, V_n_new, V_Cs_new, I_s_new, None

    # ------------------------------------------------------------
    # 3) R0–Cs–Ls–Cp + dielectric-aware plasma branch (no Rm)
    # ------------------------------------------------------------
    if circuit_type == "R0_Cs_Ls_Cp":
        if V_n_prev is None or V_Cs_prev is None or I_s_prev is None:
            raise ValueError(
                "R0_Cs_Ls_Cp requires V_n_prev, V_Cs_prev, and I_s_prev."
            )

        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_n_new, V_gap_new, I_pl, V_Cs_new, I_s_new, V_d_new = (
            update_circuit_R0_Cs_Ls_Cp(
                V_app_func=V_app_func,
                t=t,
                dt=dt,
                V_n_prev=V_n_prev,
                V_gap_prev=V_gap_prev,
                V_d_prev=V_d_prev,
                V_Cs_prev=V_Cs_prev,
                I_s_prev=I_s_prev,
                Gamma_i=Gamma_i,
                Gamma_e=Gamma_e,
                dx=dx,
                A=A,
                L=L,
                l=l,
                eps_r=eps_r,
                R0=R0,
                C_s=C_s,
                L_s=L_s,
                C_p=C_p,
            )
        )
        return V_gap_new, I_pl, V_d_new, V_n_new, V_Cs_new, I_s_new, None

    # ------------------------------------------------------------
    # 4) R0–Cs–Cp + dielectric-aware plasma branch (no Rm)
    # ------------------------------------------------------------
    if circuit_type == "R0_Cs_Cp":
        if V_n_prev is None or V_Cs_prev is None:
            raise ValueError("R0_Cs_Cp circuit requires V_n_prev and V_Cs_prev.")

        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_n_new, V_gap_new, I_pl, V_Cs_new, V_d_new = update_circuit_R0_Cs_Cp(
            V_app_func=V_app_func,
            t=t,
            dt=dt,
            V_n_prev=V_n_prev,
            V_gap_prev=V_gap_prev,
            V_d_prev=V_d_prev,
            V_Cs_prev=V_Cs_prev,
            Gamma_i=Gamma_i,
            Gamma_e=Gamma_e,
            dx=dx,
            A=A,
            L=L,
            l=l,
            eps_r=eps_r,
            R0=R0,
            C_s=C_s,
            C_p=C_p,
        )
        return V_gap_new, I_pl, V_d_new, V_n_new, V_Cs_new, None, None

    # ------------------------------------------------------------
    # 5) Full R0–Cs–Cp–Rm + dielectric-aware plasma branch
    # ------------------------------------------------------------
    if circuit_type == "R0_Cs_Cp_Rm":
        if R_m <= 0.0:
            # Automatic reduction to no-Rm topology.
            circuit_type = "R0_Cs_Cp"
        else:
            if V_n_prev is None or V_Cs_prev is None:
                raise ValueError(
                    "R0_Cs_Cp_Rm circuit requires V_n_prev and V_Cs_prev."
                )

            if V_d_prev is None:
                raise ValueError("Dielectric mapping requires V_d_prev.")

            V_n_new, V_gap_new, I_pl, V_Cs_new, V_d_new = update_circuit_R0_Cs_Cp_Rm(
                V_app_func=V_app_func,
                t=t,
                dt=dt,
                V_n_prev=V_n_prev,
                V_gap_prev=V_gap_prev,
                V_d_prev=V_d_prev,
                V_Cs_prev=V_Cs_prev,
                Gamma_i=Gamma_i,
                Gamma_e=Gamma_e,
                dx=dx,
                A=A,
                L=L,
                l=l,
                eps_r=eps_r,
                R0=R0,
                C_s=C_s,
                C_p=C_p,
                R_m=R_m,
            )
            return V_gap_new, I_pl, V_d_new, V_n_new, V_Cs_new, None, None

    # ------------------------------------------------------------
    # 6) R0–Cp–Rm + dielectric-aware plasma branch
    # ------------------------------------------------------------
    if circuit_type == "R0_Cp_Rm":
        if R_m <= 0.0:
            # Automatic reduction to no-Rm topology.
            circuit_type = "R0_Cp"
        else:
            if V_n_prev is None:
                raise ValueError("R0_Cp_Rm circuit requires V_n_prev.")

            if V_d_prev is None:
                raise ValueError("Dielectric mapping requires V_d_prev.")

            V_n_new, V_gap_new, I_pl, V_d_new = update_circuit_R0_Cp_Rm(
                V_app_func=V_app_func,
                t=t,
                dt=dt,
                V_n_prev=V_n_prev,
                V_gap_prev=V_gap_prev,
                V_d_prev=V_d_prev,
                Gamma_i=Gamma_i,
                Gamma_e=Gamma_e,
                dx=dx,
                A=A,
                L=L,
                l=l,
                eps_r=eps_r,
                R0=R0,
                C_p=C_p,
                R_m=R_m,
            )
            return V_gap_new, I_pl, V_d_new, V_n_new, None, None, None

    # ------------------------------------------------------------
    # 7) Simple R0–Cp (+ optional dielectric) topology
    # ------------------------------------------------------------
    if circuit_type in ("R0_Cp", "R"):
        if V_n_prev is None:
            raise ValueError("R0_Cp circuit requires V_n_prev.")

        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_n_new, V_gap_new, I_pl, V_d_new = update_circuit_R0_Cp(
            V_app_func=V_app_func,
            t=t,
            dt=dt,
            V_n_prev=V_n_prev,
            V_gap_prev=V_gap_prev,
            V_d_prev=V_d_prev,
            Gamma_i=Gamma_i,
            Gamma_e=Gamma_e,
            dx=dx,
            A=A,
            L=L,
            l=l,
            eps_r=eps_r,
            R0=R0,
            C_p=C_p,
        )
        return V_gap_new, I_pl, V_d_new, V_n_new, None, None, None

    # ------------------------------------------------------------
    # 8) Plasma-only Adamovic topology (no explicit external elements)
    # ------------------------------------------------------------
    if circuit_type in ("none", "dielectric_plasma"):
        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_gap_new, I_pl, V_d_new = update_circuit_no_external_component(
            V_app_func=V_app_func,
            t=t,
            dt=dt,
            V_gap_prev=V_gap_prev,
            V_d_prev=V_d_prev,
            Gamma_i=Gamma_i,
            Gamma_e=Gamma_e,
            dx=dx,
            A=A,
            L=L,
            l=l,
            eps_r=eps_r,
        )
        return V_gap_new, I_pl, V_d_new, None, None, None, None

    # ------------------------------------------------------------
    # 9) Fallback/auto-detect based on parameter values
    # ------------------------------------------------------------
    # If circuit_type does not match a known string, infer topology from
    # nonzero component values.
    if R_m == 0.0 and C_s > 0.0 and C_p > 0.0 and L_s > 0.0 and L_p > 0.0:
        if any(v is None for v in (V_n_prev, V_Cs_prev, I_s_prev, I_Lp_prev)):
            raise ValueError("Auto R0_Cs_Ls_Cp_Lp requires V_n_prev, V_Cs_prev, I_s_prev, and I_Lp_prev.")
        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_n_new, V_gap_new, I_pl, V_Cs_new, I_s_new, I_Lp_new, V_d_new = (
            update_circuit_R0_Cs_Ls_Cp_Lp(
                V_app_func=V_app_func,
                t=t,
                dt=dt,
                V_n_prev=V_n_prev,
                V_gap_prev=V_gap_prev,
                V_d_prev=V_d_prev,
                V_Cs_prev=V_Cs_prev,
                I_s_prev=I_s_prev,
                I_Lp_prev=I_Lp_prev,
                Gamma_i=Gamma_i,
                Gamma_e=Gamma_e,
                dx=dx,
                A=A,
                L=L,
                l=l,
                eps_r=eps_r,
                R0=R0,
                C_s=C_s,
                L_s=L_s,
                C_p=C_p,
                L_p=L_p,
            )
        )
        return V_gap_new, I_pl, V_d_new, V_n_new, V_Cs_new, I_s_new, I_Lp_new

    if R_m == 0.0 and C_s > 0.0 and C_p > 0.0 and L_s > 0.0:
        if V_n_prev is None or V_Cs_prev is None or I_s_prev is None:
            raise ValueError("Auto R0_Cs_Ls_Cp requires V_n_prev, V_Cs_prev, and I_s_prev.")
        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_n_new, V_gap_new, I_pl, V_Cs_new, I_s_new, V_d_new = update_circuit_R0_Cs_Ls_Cp(
            V_app_func=V_app_func,
            t=t,
            dt=dt,
            V_n_prev=V_n_prev,
            V_gap_prev=V_gap_prev,
            V_d_prev=V_d_prev,
            V_Cs_prev=V_Cs_prev,
            I_s_prev=I_s_prev,
            Gamma_i=Gamma_i,
            Gamma_e=Gamma_e,
            dx=dx,
            A=A,
            L=L,
            l=l,
            eps_r=eps_r,
            R0=R0,
            C_s=C_s,
            L_s=L_s,
            C_p=C_p,
        )
        return V_gap_new, I_pl, V_d_new, V_n_new, V_Cs_new, I_s_new, None

    if R_m == 0.0 and C_s > 0.0 and C_p > 0.0:
        if V_n_prev is None or V_Cs_prev is None:
            raise ValueError("Auto R0_Cs_Cp requires V_n_prev and V_Cs_prev.")
        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_n_new, V_gap_new, I_pl, V_Cs_new, V_d_new = update_circuit_R0_Cs_Cp(
            V_app_func=V_app_func,
            t=t,
            dt=dt,
            V_n_prev=V_n_prev,
            V_gap_prev=V_gap_prev,
            V_d_prev=V_d_prev,
            V_Cs_prev=V_Cs_prev,
            Gamma_i=Gamma_i,
            Gamma_e=Gamma_e,
            dx=dx,
            A=A,
            L=L,
            l=l,
            eps_r=eps_r,
            R0=R0,
            C_s=C_s,
            C_p=C_p,
        )
        return V_gap_new, I_pl, V_d_new, V_n_new, V_Cs_new, None, None

    if R_m > 0.0 and C_s > 0.0 and C_p > 0.0:
        if V_n_prev is None or V_Cs_prev is None:
            raise ValueError("Auto R0_Cs_Cp_Rm requires V_n_prev and V_Cs_prev.")
        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_n_new, V_gap_new, I_pl, V_Cs_new, V_d_new = update_circuit_R0_Cs_Cp_Rm(
            V_app_func=V_app_func,
            t=t,
            dt=dt,
            V_n_prev=V_n_prev,
            V_gap_prev=V_gap_prev,
            V_d_prev=V_d_prev,
            V_Cs_prev=V_Cs_prev,
            Gamma_i=Gamma_i,
            Gamma_e=Gamma_e,
            dx=dx,
            A=A,
            L=L,
            l=l,
            eps_r=eps_r,
            R0=R0,
            C_s=C_s,
            C_p=C_p,
            R_m=R_m,
        )
        return V_gap_new, I_pl, V_d_new, V_n_new, V_Cs_new, None, None

    if R_m > 0.0 and C_p > 0.0 and C_s == 0.0:
        if V_n_prev is None:
            raise ValueError("Auto R0_Cp_Rm requires V_n_prev.")
        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_n_new, V_gap_new, I_pl, V_d_new = update_circuit_R0_Cp_Rm(
            V_app_func=V_app_func,
            t=t,
            dt=dt,
            V_n_prev=V_n_prev,
            V_gap_prev=V_gap_prev,
            V_d_prev=V_d_prev,
            Gamma_i=Gamma_i,
            Gamma_e=Gamma_e,
            dx=dx,
            A=A,
            L=L,
            l=l,
            eps_r=eps_r,
            R0=R0,
            C_p=C_p,
            R_m=R_m,
        )
        return V_gap_new, I_pl, V_d_new, V_n_new, None, None, None

    if R0 > 0.0:
        if V_n_prev is None:
            raise ValueError("Auto R0_Cp requires V_n_prev.")
        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_n_new, V_gap_new, I_pl, V_d_new = update_circuit_R0_Cp(
            V_app_func=V_app_func,
            t=t,
            dt=dt,
            V_n_prev=V_n_prev,
            V_gap_prev=V_gap_prev,
            V_d_prev=V_d_prev,
            Gamma_i=Gamma_i,
            Gamma_e=Gamma_e,
            dx=dx,
            A=A,
            L=L,
            l=l,
            eps_r=eps_r,
            R0=R0,
            C_p=C_p,
        )
        return V_gap_new, I_pl, V_d_new, V_n_new, None, None, None

    if l > 0.0:
        # Fallback: if we have a dielectric but no explicit R, C, L, revert
        # to the Adamovic-only plasma topology.
        if V_d_prev is None:
            raise ValueError("Dielectric mapping requires V_d_prev.")

        V_gap_new, I_pl, V_d_new = update_circuit_no_external_component(
            V_app_func=V_app_func,
            t=t,
            dt=dt,
            V_gap_prev=V_gap_prev,
            V_d_prev=V_d_prev,
            Gamma_i=Gamma_i,
            Gamma_e=Gamma_e,
            dx=dx,
            A=A,
            L=L,
            l=l,
            eps_r=eps_r,
        )
        return V_gap_new, I_pl, V_d_new, None, None, None, None

    # If we reach this point, we do not know how to interpret the parameters.
    raise ValueError(f"Unknown or unsupported circuit_type: {circuit_type}")
