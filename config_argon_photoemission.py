"""
config.py

Configuration data structures for the 1D pulsed-discharge
drift-diffusion-Poisson solver (PASCHEN-1D).

This module is meant to be the *single source of truth* for all high-level
simulation parameters:

    - Geometry and gas/plasma properties
    - External circuit topology and component values
    - Applied voltage waveform (step, Gaussian, DC, RF)
    - Time/space discretization and numerical scheme
    - Emission model parameters
    - Basic containers for output (SimulationState) and transport data
      (TransportCoeffs)

The idea is that a user can configure and run many different physical scenarios
by modifying only this file (or constructing `SimulationConfig` instances in
Python) without touching the core numerics.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases for clarity / safety
# ---------------------------------------------------------------------------

#: Allowed types of applied voltage waveform.
VoltageWaveform = Literal["step", "gaussian", "dc", "rf"]

#: Boundary condition mode per electrode/species.
BoundaryMode = Literal["zero_density", "electron_emission", "implicit_drift_closure"]

#: Allowed circuit time-integration schemes.
CircuitTimeScheme = Literal["explicit_euler", "implicit_euler"]

ElectrodeMaterialMode = Literal["shared", "separate"]

#: Allowed external circuit topologies (see circuit.py for details).
CircuitType = Literal[
    "dielectric_plasma",      # Adamovic dielectric + plasma only (no explicit R/C/L)
    "R0_Cp",                  # Vs -- R0 -- (node) -- [Cp || plasma]
    "R0_Cp_Rm",               # Vs -- R0 -- (node) -- [Cp || (Rm + plasma)]
    "R0_Cs_Cp",               # Vs -- R0 -- Cs -- (node) -- [Cp || plasma]
    "R0_Cs_Cp_Rm",            # Vs -- R0 -- Cs -- (node) -- [Cp || (Rm + plasma)]
    "R0_Cs_Ls_Cp",            # Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || plasma]
    "R0_Cs_Ls_Cp_Rm",         # Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || (Rm + plasma)]
    "R0_Cs_Ls_Cp_Lp",         # Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || Lp || plasma]
    "R0_Cs_Ls_Cp_Lp_Rm",      # Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || Lp || (Rm + plasma)]
]



TemporalDiagnosticQuantity = Literal[
    "V_app",
    "V_gap",
    "I_discharge",
    "cfl",
    "particle_inventory",
]

SpatialDiagnosticQuantity = Literal[
    "ne",
    "ni",
    "phi",
    "E",
    "Gamma_i",
    "Gamma_e",
    "townsend_alpha",
    "nu_i",
    "S",
    "S_i",
    "S_e",
]


@dataclass
class TemporalDiagnosticsConfig:
    """
    Post-run time-series diagnostics.

    Users select which quantities to plot and can optionally limit
    the plotted time window to [t_start, t_end]. This does not affect
    solver output; it only affects plotting.
    """
    enabled: bool = True
    quantities: tuple[TemporalDiagnosticQuantity, ...] = (
        "V_app",
        "V_gap",
        "I_discharge",
        "cfl",
        "particle_inventory",
    )
    # Optional grouped plotting.
    # Example:
    plot_groups = (("V_app", "V_gap"), ("I_discharge",), ("particle_inventory",),)
    # If None, each quantity in `quantities` is plotted separately.
#     plot_groups: tuple[tuple[TemporalDiagnosticQuantity, ...], ...] | None = None
    # None means full simulation range.
    # Example: t_start = 0.5e-6 (do not use quotes).
    t_start: float | None = None
    # Example: t_end = 1.0e-6
    t_end: float | None = None
    # If provided, figures are saved as "<prefix>_<quantity>.pdf".
    savepath_prefix: str | None = None


@dataclass
class SpatialDiagnosticsConfig:
    """
    Post-run spatial-profile diagnostics.

    Users select quantities and requested sample times. Requested times
    are mapped to the nearest saved snapshot time unless they request
    the final time for fields that are available exactly in memory
    (ne, ni, phi, E).
    """
    enabled: bool = True
    quantities: tuple[SpatialDiagnosticQuantity, ...] = ("ne", "E")
    
    # Optional grouped plotting.
    # Example:
    #   plot_groups = (("ne", "ni"), ("phi",), ("E",),)
    # If None, each quantity in `quantities` is plotted separately.
    plot_groups: tuple[tuple[SpatialDiagnosticQuantity, ...], ...] | None = None
    # None means final time only.
    # Example tuple syntax:
#     t_samples = (80e-9, 86e-9, 87e-9, 88e-9, 89e-9, 90e-9)
    # A single time still needs a trailing comma:
    #   t_samples = (0.5e-6,)
    t_samples: tuple[float, ...] | None = None
    x_unit: Literal["m", "cm", "mm"] = "cm"
    # If provided, figures are saved as "<prefix>_<quantity>.pdf".
    savepath_prefix: str | None = None


@dataclass
class DiagnosticsConfig:
    """
    Grouped diagnostics settings to keep SimulationConfig compact.

    Quick usage:
    - `temporal.quantities`: choose from V_app, V_gap, I_discharge, cfl, particle_inventory.
    - `temporal.plot_groups`: optional grouped time-series overlays.
      Example: `(("V_app", "V_gap"), ("I_discharge",),)`
    - `temporal.t_start/t_end`: optional plotting window; None uses full run.
    - `spatial.quantities`: choose fields/profiles to plot.
    - `spatial.plot_groups`: optional grouped spatial overlays.
      Example: `(("ne", "ni"), ("phi",), ("E",),)`
    - `spatial.t_samples`: requested times [s]; None plots final-time snapshot.
    - `*_savepath_prefix`: optional output prefix for saved figures.
    """
    temporal: TemporalDiagnosticsConfig = field(default_factory=TemporalDiagnosticsConfig)
    spatial: SpatialDiagnosticsConfig = field(default_factory=SpatialDiagnosticsConfig)

# ---------------------------------------------------------------------------
# High-level configuration for a single simulation run
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """
    High-level configuration for a 1D pulsed discharge simulation.

    This is the primary place where a user tweaks parameters without touching
    the numerical backend. It is conceptually grouped as:

        - Geometry and electrode / dielectric setup
        - Plasma / gas properties
        - External circuit topology and component values
        - Applied-voltage waveform parameters (step, Gaussian, DC, RF)
        - Time/space discretization and numerics
        - Emission parameters
        - Output / logging controls

    Most fields have physically meaningful SI units unless otherwise specified.
    """

    # ----------------------------------------------------------------------
    # Run identification / labeling
    # ----------------------------------------------------------------------
    #: Optional label used in file naming, plots, etc.
    run_name: str = "argon_photoemission"

    # ----------------------------------------------------------------------
    # Geometry and electrodes
    # ----------------------------------------------------------------------
    #: Gap length [m] between electrodes (plasma domain length).
    L: float = 0.35e-2
    #: Electrode area [m²] (used in current / capacitance calculations).
    A: float = 22e-3 * 13e-3
    #: Dielectric thickness near each electrode [m]. l = 0 → bare electrodes.
    l: float = 0.0
    #: Relative permittivity of the dielectric (dimensionless).
    eps_r: float = 4.3

    # ----------------------------------------------------------------------
    # Plasma / gas parameters
    # ----------------------------------------------------------------------
    #: Gas species name (currently used for lookup of transport, alpha(E/p), etc.).
    gas: str = "argon"  # e.g. "argon", "nitrogen" (extend as needed)
    #: Gas pressure [Torr].
    p_Torr: float = 2.88
    #: Electron temperature [K] for initial conditions / Einstein relation, etc.
    T_e: float = 11600.0
    #: Ion temperature [K] (typically room temperature).
    T_i: float = 300.0
    #: Ion-induced secondary electron yield (SEY) coefficient gamma.
    gamma: float = 0.1
    #: Initial uniform plasma density [m⁻³] for both electrons and ions.
    n0: float = 1e14

    # ----------------------------------------------------------------------
    # External circuit configuration
    # ----------------------------------------------------------------------
    # circuit_type options (see circuit.py for the exact ODEs):
    #
    #   "dielectric_plasma"
    #       dielectric + plasma (Adamovic relation), no explicit R0 / Cp.
    #
    #   "R0_Cp"
    #       Vs -- R0 -- (node) -- [Cp || (dielectric + plasma + dielectric)]
    #
    #   "R0_Cp_Rm"
    #       Vs -- R0 -- (node) -- [Cp || (Rm + dielectric + plasma + dielectric)]
    #
    #   "R0_Cs_Cp"
    #       Vs -- R0 -- Cs -- (node) -- [Cp || (dielectric + plasma + dielectric)]
    #
    #   "R0_Cs_Cp_Rm"
    #       Vs -- R0 -- Cs -- (node) -- [Cp || (Rm + dielectric + plasma + dielectric)]
    #
    #   "R0_Cs_Ls_Cp"
    #       Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || (dielectric + plasma + dielectric)]
    #
    #   "R0_Cs_Ls_Cp_Rm"
    #       Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || (Rm + dielectric + plasma + dielectric)]
    #
    #   "R0_Cs_Ls_Cp_Lp"
    #       Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || Lp || (dielectric + plasma + dielectric)]
    #
    #   "R0_Cs_Ls_Cp_Lp_Rm"
    #       Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || Lp || (Rm + dielectric + plasma + dielectric)]
    #
    # Notes:
    #   - l = 0 ⇒ bare electrodes (no dielectric) in all topologies.
    #   - For "R0_Cp", C_p may be 0 (pure R0 + plasma branch).
    #   - For no-Rm topologies with Cs/Ls, require C_p > 0.
    #   - For "*_Rm" topologies, require R_m > 0 and C_p > 0.
    #   - For "R0_Cs_Cp", require C_s > 0.
    #   - For "R0_Cs_Ls_Cp", require C_s > 0 and L_s > 0.
    #   - For "R0_Cs_Cp_Rm", require C_s > 0.
    #   - For "R0_Cs_Ls_Cp_Rm", require C_s > 0 and L_s > 0.
    #   - For "R0_Cs_Ls_Cp_Lp", require C_s > 0, L_s > 0, and L_p > 0.
    #   - For "R0_Cs_Ls_Cp_Lp_Rm", require C_s > 0, L_s > 0, and L_p > 0.
    #
    # circuit_time_scheme options:
    #   "explicit_euler"  -> existing explicit stepper in circuit.py
    #   "implicit_euler"  -> implicit stepper in circuit_implicit_euler.py
    #                         (recommended for stiff circuit parameter sets)
    
    #: Selected external circuit topology.
    circuit_type: CircuitType = "R0_Cp_Rm"
    #: Time integrator for external circuit ODEs.
    circuit_time_scheme: CircuitTimeScheme = "explicit_euler"

    #: Series resistance [Ω] between source and plasma circuit.
    R0: float = 76.4
    #: Series capacitor in source branch [F].
    C_s: float = 0.0
    #: Series inductor in source branch [H].
    L_s: float = 0.0
    #: Parallel capacitor at the node (in parallel with plasma branch) [F].
    C_p: float = 206e-12
    #: Parallel inductor at the node [H].
    L_p: float = 0.0
    #: Series resistor in plasma branch [Ω] (e.g. measurement / matching).
    R_m: float = 1e6

    # ----------------------------------------------------------------------
    # Global time horizon
    # ----------------------------------------------------------------------
    #: Total simulation time [s].
    T_total: float = 20e-6

    # ----------------------------------------------------------------------
    # Applied voltage waveform parameters
    # ----------------------------------------------------------------------
    #: Type of applied waveform ("step", "gaussian", "dc", "rf").
    waveform_type: VoltageWaveform = "step"

    # --- Step/DC waveform parameters ---
    #: Peak or DC amplitude [V] (interpreted depending on waveform_type).
    V_peak: float = 130.0
    #: Step ON time [s] (for "step" / pulsed waveforms).
    tV_start: float = 0e-6
    #: Step OFF time [s]. Default = full simulation window.
    tV_end: float = T_total

    # --- Gaussian waveform parameters ---
    #: Gaussian FWHM / width [s] (for "gaussian" waveform).
    tau: float = 15e-9
    #: Center time [s] of Gaussian pulse.
    t_peak: float = 100e-9

    # --- RF waveform parameters ---
    #: RF driving frequency [Hz].
    f_rf: float = 13.56e6
    #: Optional DC bias [V] added to RF.
    V_dc: float = 0.0
    #: RF phase [rad] at t = 0.
    phi_rf: float = 0.0

    # ----------------------------------------------------------------------
    # Time discretization
    # ----------------------------------------------------------------------
    #: Number of time steps over T_total (dt = T_total / Nt).
    Nt: int = 2_000_000

    # ----------------------------------------------------------------------
    # Space discretization
    # ----------------------------------------------------------------------
    #: Number of spatial grid points in the 1D domain [0, L].
    Nx: int = 200

    # ----------------------------------------------------------------------
    # Numerics
    # ----------------------------------------------------------------------
    #: KT slope-limiter parameter (theta >= 1; larger means less limiting).
    kt_limiter_theta: float = 1.1

    # ----------------------------------------------------------------------
    # Boundary conditions
    # ----------------------------------------------------------------------
    #: Ion BC mode at anode (x=0).
    anode_ion_boundary: BoundaryMode = "zero_density"
    #: Electron BC mode at anode (x=0).
    anode_electron_boundary: BoundaryMode = "implicit_drift_closure"
    #: Ion BC mode at cathode (x=L).
    cathode_ion_boundary: BoundaryMode = "implicit_drift_closure"
    #: Electron BC mode at cathode (x=L).
    cathode_electron_boundary: BoundaryMode = "electron_emission"

    # ----------------------------------------------------------------------
    # Source-term controls (for controlled tests / ablation studies)
    # ----------------------------------------------------------------------
    #: Master switch for volumetric source terms in continuity equations.
    #: If False, both ionization and recombination are disabled regardless
    #: of the individual toggles below.
    enable_volume_sources: bool = True
    #: Enable Townsend ionization contribution (+nu_i * n_e).
    enable_ionization_source: bool = True
    #: Enable recombination contribution (-beta * n_i * n_e).
    enable_recombination_sink: bool = True
    #: Include emission-induced electron flux in circuit-current coupling.
    enable_emission_in_circuit_current: bool = True
    # ----------------------------------------------------------------------
    # Output / logging controls
    # ----------------------------------------------------------------------
    #: Save data every `save_every` time steps.
    save_every: int = 2000
    #: If True, write intermediate sampled fields (Gamma_i, Gamma_e, etc.).
    log_intermediate: bool = True
    #: Print startup run summary (geometry, numerics, circuit, diagnostics).
    print_run_summary: bool = True
    #: Print non-fatal config consistency warnings at startup.
    warn_on_config_mismatch: bool = True
    #: Diagnostics menu for post-run plots.
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    # ============================
    # Emission models
    # ============================

    #: Master switch for externally driven emission models.
    #: If False, emission model current density is disabled globally.
    enable_external_emission: bool = True
    #: Electrode parameter mapping mode for emission/material properties.
    #: - "shared": both electrodes use shared_* parameters.
    #: - "separate": anode_* and cathode_* parameters are used independently.
    electrode_material_mode: ElectrodeMaterialMode = "shared"
    #: Enable externally driven emission contribution at anode (x=0).
    enable_anode_external_emission: bool = False
    #: Enable externally driven emission contribution at cathode (x=L).
    enable_cathode_external_emission: bool = True

    # Per-electrode, per-model external emission toggles.
    # Multiple enabled models on the same electrode are summed.
    # Anode (x=0)
    anode_enable_constant_J_emission: bool = False
    anode_enable_fn_emission: bool = False
    anode_enable_mg_emission: bool = False
    anode_enable_rd_emission: bool = False
    anode_enable_quantum_pulse_emission: bool = False
    # Cathode (x=L)
    cathode_enable_constant_J_emission: bool = False
    cathode_enable_fn_emission: bool = False
    cathode_enable_mg_emission: bool = True
    cathode_enable_rd_emission: bool = False
    cathode_enable_quantum_pulse_emission: bool = True
        
        
    # --- Shared electrode material/emission parameters ---
    # Used when electrode_material_mode = "shared".
    shared_fn_work_function_eV: float = 4.5
    shared_fn_field_scale_factor: float = 1.0
    shared_mg_work_function_eV: float = 4.5
    shared_mg_field_scale_factor: float = 1.0
    shared_mg_f_clip_min: float = 1.0e-9
    shared_mg_f_clip_max: float = 0.99
    shared_rd_A_R: float = 1.2e6
    shared_rd_emitter_K: float = 300.0
    shared_rd_work_function_eV: float = 4.1
    shared_emission_T: float = 300.0
    shared_emission_W_eV: float = 4.1
    shared_emission_Ef_eV: float = 11.7

    # --- Shared constant-J controls ---
    shared_emission_J_const: float = 1.0e5
    shared_emission_t_start: float = 9.5e-6
    shared_emission_t_end: float = 10.5e-6

    # --- Shared quantum-pulse controls ---
    shared_emission_epsilon0_eV: float = 12.0
    shared_emission_k_ph: int = 14
    shared_laser_t0: float = 10e-6
    shared_emission_lambda_m: float = 230e-9
    shared_laser_U_J: float = 150e-6
    shared_laser_tau_p_s: float = 30e-12
    shared_laser_theta_deg: float = 19.0
    shared_laser_t_window_ps: float = 200.0
    shared_emission_dt_ps: float = 2.0
    shared_emission_eps_points: int = 40
    shared_emission_wt_points: int = 200
    shared_laser_wx_m: float = 8.3e-3
    shared_laser_wy_m: float = 3.0e-3

    # --- Per-electrode material/emission parameters ---
    # Used when electrode_material_mode = "separate".
    # Anode
    anode_fn_work_function_eV: float = 4.5
    anode_fn_field_scale_factor: float = 1.0
    anode_mg_work_function_eV: float = 4.5
    anode_mg_field_scale_factor: float = 1.0
    anode_mg_f_clip_min: float = 1.0e-9
    anode_mg_f_clip_max: float = 0.99
    anode_rd_A_R: float = 1.2e6
    anode_rd_emitter_K: float = 300.0
    anode_rd_work_function_eV: float = 4.1
    anode_emission_T: float = 300.0
    anode_emission_W_eV: float = 4.1
    anode_emission_Ef_eV: float = 11.7
    # Cathode
    cathode_fn_work_function_eV: float = 4.5
    cathode_fn_field_scale_factor: float = 1.0
    cathode_mg_work_function_eV: float = 4.5
    cathode_mg_field_scale_factor: float = 1.0
    cathode_mg_f_clip_min: float = 1.0e-9
    cathode_mg_f_clip_max: float = 0.99
    cathode_rd_A_R: float = 1.2e6
    cathode_rd_emitter_K: float = 300.0
    cathode_rd_work_function_eV: float = 4.1
    cathode_emission_T: float = 300.0
    cathode_emission_W_eV: float = 4.1
    cathode_emission_Ef_eV: float = 11.7

    # --- Per-electrode constant-J controls (used in "separate" mode) ---
    anode_emission_J_const: float = 1.0e5
    anode_emission_t_start: float = 9.5e-6
    anode_emission_t_end: float = 10.5e-6
    cathode_emission_J_const: float = 1.0e5
    cathode_emission_t_start: float = 9.5e-6
    cathode_emission_t_end: float = 10.5e-6

    # --- Per-electrode quantum-pulse controls (used in "separate" mode) ---
    anode_emission_epsilon0_eV: float = 12.0
    anode_emission_k_ph: int = 14
    anode_laser_t0: float = 10e-6
    anode_emission_lambda_m: float = 230e-9
    anode_laser_U_J: float = 150e-6
    anode_laser_tau_p_s: float = 30e-12
    anode_laser_theta_deg: float = 19.0
    anode_laser_t_window_ps: float = 200.0
    anode_emission_dt_ps: float = 2.0
    anode_emission_eps_points: int = 40
    anode_emission_wt_points: int = 200
    anode_laser_wx_m: float = 8.3e-3
    anode_laser_wy_m: float = 3.0e-3

    cathode_emission_epsilon0_eV: float = 12.0
    cathode_emission_k_ph: int = 14
    cathode_laser_t0: float = 10e-6
    cathode_emission_lambda_m: float = 230e-9
    cathode_laser_U_J: float = 150e-6
    cathode_laser_tau_p_s: float = 30e-12
    cathode_laser_theta_deg: float = 19.0
    cathode_laser_t_window_ps: float = 200.0
    cathode_emission_dt_ps: float = 2.0
    cathode_emission_eps_points: int = 40
    cathode_emission_wt_points: int = 200
    cathode_laser_wx_m: float = 8.3e-3
    cathode_laser_wy_m: float = 3.0e-3

# ---------------------------------------------------------------------------
# Containers for outputs and transport coefficients
# ---------------------------------------------------------------------------


@dataclass
class SimulationState:
    """
    Container for key outputs of a 1D pulsed-discharge run.

    This is typically populated *after* a simulation completes and can be used
    for post-processing, plotting, or saving to disk.

    Attributes
    ----------
    cfg : SimulationConfig
        The configuration used to generate this simulation state.
    time : np.ndarray
        1D array of time points [s], shape (Nt,).
    x : np.ndarray
        1D spatial grid [m] in the domain [0, L], shape (Nx,).
    V_gap : np.ndarray
        Time trace of plasma gap voltage [V], shape (Nt,).
    I_discharge : np.ndarray
        Time trace of discharge (plasma-branch) current [A], shape (Nt,).
    c_cfl : np.ndarray
        Time trace (or final snapshot) of CFL-like stability metric
        (dimensionless), shape (Nt,) or (Nt, Nx) depending on usage.
    ne_final : np.ndarray
        Electron density at final time step [m⁻³], shape (Nx,).
    ni_final : np.ndarray
        Ion density at final time step [m⁻³], shape (Nx,).
    phi_final : np.ndarray
        Electrostatic potential at final time [V], shape (Nx,).
    E_final : np.ndarray
        Electric field at final time [V/m], shape (Nx,).
    """
    cfg: SimulationConfig
    time: np.ndarray
    x: np.ndarray
    V_gap: np.ndarray
    I_discharge: np.ndarray
    c_cfl: np.ndarray
    ne_final: np.ndarray
    ni_final: np.ndarray
    phi_final: np.ndarray
    E_final: np.ndarray


@dataclass
class TransportCoeffs:
    """
    Container for basic transport and reaction coefficients.

    These are typically computed from the chosen gas, pressure, and
    temperature(s) and then used throughout the drift–diffusion update.

    Attributes
    ----------
    mu_e : float
        Electron mobility [m²/(V·s)].
    mu_i : float
        Ion mobility [m²/(V·s)].
    D_e : float
        Electron diffusion coefficient [m²/s].
    D_i : float
        Ion diffusion coefficient [m²/s].
    beta : float
        Effective recombination or attachment coefficient [m³/s] (if used).
    pr : float
        Pressure-related scaling parameter (e.g. p·r or other combination),
        kept generic as "pr" for now.
    T_e_eV : float
        Electron temperature [eV].
    T_i_eV : float
        Ion temperature [eV].
    """
    mu_e: float
    mu_i: float
    D_e: float
    D_i: float
    beta: float
    pr: float
    T_e_eV: float
    T_i_eV: float
