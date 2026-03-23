"""
config_argon_photoemission.py

Configuration schema for PASCHEN-1D.

This module is the single source of truth for user-facing simulation knobs.
Parameters are grouped into focused dataclasses so users can quickly find and
edit related controls in this order:

    1. run
    2. numerics
    3. geometry
    4. plasma_state
    5. plasma (mode selector only)
    6. user_defined_electron_kinetics
    7. local_field_approximation
    8. townsend_coefficient
    9. ionization_frequency_source
    10. recombination
    11. waveform
    12. boundary
    13. circuit
    14. emission
    15. output
    16. diagnostics

All runtime modules should access grouped fields explicitly
(`cfg.geometry.*`, `cfg.waveform.*`, etc.).
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases for clarity / safety
# ---------------------------------------------------------------------------

VoltageWaveform = Literal["step", "gaussian", "dc", "rf"]
BoundaryMode = Literal["zero_density", "electron_emission", "implicit_drift_closure"]
CircuitTimeScheme = Literal["explicit_euler", "implicit_euler"]
ElectrodeMaterialMode = Literal["shared", "separate"]
# Transport source selector used by electron-transport knobs below:
# - "user_defined_equation":
#     * electron mobility        -> physics.py: compute_user_defined_electron_mobility(...)
#     * ion mobility             -> physics.py: compute_user_defined_ion_mobility(...)
#     * electron diffusion       -> physics.py: compute_user_defined_electron_diffusion(...)
#     * ion diffusion            -> physics.py: compute_user_defined_ion_diffusion(...)
# - "swarm_data_table_interpolation":
#     table-based interpolation in the corresponding transport build_*_profile functions.
TransportSourceMode = Literal["user_defined_equation", "swarm_data_table_interpolation"]
HotloopBackend = Literal["numpy", "numba"]
ElectronKineticsModel = Literal[
    "user_defined_electron_kinetics",
    "local_field_approximation",
]
IonKineticsModel = Literal["user_defined_ion_kinetics"]
ImpactIonizationModel = Literal[
    "from_townsend_alpha",
    "from_ionization_frequency",
]
RecombinationModel = Literal["user_defined_constant_coefficient"]
AdaptiveSubstepOverflowPolicy = Literal["warn_and_cap", "error"]

CircuitType = Literal[
    "dielectric_plasma",
    "R0_Cp",
    "R0_Cp_Rm",
    "R0_Cs_Cp",
    "R0_Cs_Cp_Rm",
    "R0_Cs_Ls_Cp",
    "R0_Cs_Ls_Cp_Rm",
    "R0_Cs_Ls_Cp_Lp",
    "R0_Cs_Ls_Cp_Lp_Rm",
]

TemporalDiagnosticQuantity = Literal[
    "V_app",
    "V_gap",
    "I_discharge",
    "cfl",
    "picard_iterations",
    "adaptive_substeps",
    "adaptive_dt_sub",
    "adaptive_cfl_est",
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
    "S_ion",
    "S",
    "mu_e",
    "D_e",
]

AveragedSpatialMode = Literal["time_window", "last_n_cycles"]


# ---------------------------------------------------------------------------
# Core grouped simulation configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Run identification and naming."""

    # Label used to create output folder and metadata tags.
    run_name: str = "argon_cathode_photoemission_quantum_model"
    # Total simulation time [s].
    T_total: float = 20e-6

@dataclass
class NumericsConfig:
    """Grid and numerical-method controls."""

    # Number of time steps over run.T_total (dt = run.T_total / Nt).
    Nt: int = 2_000_000
    # Number of spatial grid points over [0, L].
    Nx: int = 200
    # Kurganov-Tadmor slope limiter parameter (theta >= 1).
    kt_limiter_theta: float = 1.1
    # Backend for density-update hot loops (KT+RK4):
    # - "numpy": vectorized NumPy path with reusable RK4 workspaces
    # - "numba": JIT-compiled linear KT+RK4 kernel (falls back to NumPy if unavailable)
    hotloop_backend: HotloopBackend = "numba"
    # Enable parallel Numba kernel variant when hotloop_backend="numba".
    # Practical note for 1D PASCHEN-1D runs:
    # - For typical grid sizes (Nx in the hundreds to low-thousands),
    #   thread-launch/synchronization overhead usually makes this slower.
    # - Set True only for very large Nx where kernel work per step is high.
    # If unsupported at runtime, code falls back to serial Numba kernel.
    numba_parallel: bool = False
    
    # Enable adaptive substepping inside each macro time step.
    # When enabled, each macro step dt is split into n_sub substeps so the
    # estimated drift CFL per substep is kept near/below target_cfl_substep.
    use_adaptive_substepping: bool = False
    # Target drift CFL for each substep when adaptive substepping is enabled.
    target_cfl_substep: float = 0.5
    # Hard cap on the number of substeps allowed per macro step.
    max_substeps: int = 64
    # Behavior when required substeps exceed max_substeps:
    # - "warn_and_cap": warn and run with n_sub=max_substeps
    # - "error": raise RuntimeError and stop
    adaptive_substep_overflow_policy: AdaptiveSubstepOverflowPolicy = "warn_and_cap"
    # Warning cadence (in macro steps) to avoid printing an overflow warning
    # every step in long runs.
    adaptive_substep_warn_every: int = 1000

    # BC+Poisson fixed-point controls used for the per-substep density/field closure.
    # Picard loop exits when:
    #   iter >= bc_poisson_picard_min_iter and max(|phi_new-phi_old|) < bc_poisson_picard_tol
    bc_poisson_picard_min_iter: int = 1
    bc_poisson_picard_max_iter: int = 10
    bc_poisson_picard_tol: float = 1.0e-6

@dataclass
class GeometryConfig:
    """Geometry and dielectric/electrode properties."""

    # Gap length [m] between electrodes.
    L: float = 0.35e-2
    # Effective electrode area [m^2].
    A: float = 22e-3 * 13e-3
    # Dielectric thickness [m] adjacent to each electrode (0 = bare electrodes).
    l: float = 0.0
    # Relative permittivity of dielectric.
    eps_r: float = 4.3

@dataclass
class PlasmaStateConfig:
    """Gas/plasma state variables shared by all electron-kinetics modes."""

    # Gas species identifier used by user-defined closures and table checks.
    # Example values currently supported by default equations: "argon", "nitrogen".
    # Additional gases can be added by editing user-defined closures in physics.py
    # or by providing compatible swarm-data files and matching gas labels.
    gas: str = "argon"
    # Gas pressure [Torr].
    p_Torr: float = 2.88
    # Electron temperature [K] used by user-defined closures/initialization.
    T_e: float = 11600.0
    # Ion (and gas-closure) temperature [K].
    T_i: float = 300.0
    # Initial uniform electron/ion density [m^-3].
    n0: float = 1e14

@dataclass
class PlasmaConfig:
    """Top-level plasma-physics model selectors."""

    # Electron kinetics model:
    # - "user_defined_electron_kinetics":
    #      electron transport from user equations in physics.py
    # - "local_field_approximation":
    #      electron transport from local E/N (table or user equation)
    electron_kinetics_model: ElectronKineticsModel = "user_defined_electron_kinetics"

    # Ion kinetics model (current implementation supports user-defined only).
    ion_kinetics_model: IonKineticsModel = "user_defined_ion_kinetics"

    # Impact-ionization model:
    # - "from_townsend_alpha":
    #      nu_i = alpha * |u_e|, with alpha source from TownsendCoefficientConfig
    # - "from_ionization_frequency":
    #      nu_i source from IonizationFrequencySourceConfig
    impact_ionization_model: ImpactIonizationModel = "from_townsend_alpha"

    # Recombination model (current implementation uses a single user-defined
    # constant coefficient from RecombinationConfig).
    recombination_model: RecombinationModel = "user_defined_constant_coefficient"


@dataclass
class UserDefinedElectronKineticsConfig:
    """
    Controls for the user-defined electron-kinetics mode.

    This mode currently uses user-defined electron transport equations in
    ``physics.py`` and requires no extra per-mode knobs.
    """
    pass


@dataclass
class TownsendCoefficientConfig:
    """Controls for Townsend-alpha sourcing in alpha-based ionization mode."""

    # Townsend-alpha source mode:
    # - "user_defined_equation":
    #      alpha from physics.py -> compute_user_defined_townsend_alpha(...)
    # - "interpolate_from_e_over_n_table":
    #      alpha/N(E/N) table interpolation, then alpha=(alpha/N)*N
    townsend_alpha_source_mode: Literal[
        "user_defined_equation",
        "interpolate_from_e_over_n_table",
    ] = "user_defined_equation"
    # Optional dedicated alpha/N table path.
    # If None, default table path is reused:
    # - E/N axis -> local_field_approximation.electron_swarm_data_path
    townsend_alpha_swarm_data_path: str = "ar_swarm_output_full_EoverN.dat"
        

@dataclass
class IonizationFrequencySourceConfig:
    """Controls for direct nu_i sourcing in impact-ionization mode."""

    # Direct ionization-frequency source mode:
    # - "user_defined_equation":
    #      nu_i from physics.py -> compute_user_defined_ionization_frequency(...)
    # - "interpolate_from_e_over_n_table":
    #      nu_i/N(E/N) table interpolation, then nu_i=(nu_i/N)*N
    ionization_frequency_source_mode: Literal[
        "user_defined_equation",
        "interpolate_from_e_over_n_table",
    ] = "interpolate_from_e_over_n_table"
    # Optional dedicated nu_i/N table path.
    # If None, default table path is reused:
    # - E/N axis -> local_field_approximation.electron_swarm_data_path
    ionization_frequency_swarm_data_path: str = "ar_swarm_output_full_EoverN.dat"


@dataclass
class RecombinationConfig:
    """Controls for volumetric electron-ion recombination."""

    # Constant recombination coefficient beta [m^3/s].
    recombination_coefficient: float = 2.0e-13
        
        
@dataclass
class LocalFieldApproximationConfig:
    """
    Controls for local-field approximation (LFA) mode.

    LFA uses local E/N to evaluate electron transport coefficients.
    """

    # Electron transport source for LFA.
    # If set to "user_defined_equation", define equations in:
    #   physics.py -> compute_user_defined_electron_mobility(...)
    #   physics.py -> compute_user_defined_electron_diffusion(...)
    # If set to "swarm_data_table_interpolation", interpolation is handled in:
    #   physics.py -> build_electron_mobility_profile(...)
    #   physics.py -> build_electron_diffusion_profile(...)
    electron_transport_source: TransportSourceMode = "swarm_data_table_interpolation"
    # E/N-axis electron swarm-data source.
    electron_swarm_data_path: str = "ar_swarm_output_full_EoverN.dat"

@dataclass
class WaveformConfig:
    """Applied-voltage waveform settings."""

    # Waveform type and parameters.
    # waveform_type options: "step", "gaussian", "dc", "rf".
    waveform_type: VoltageWaveform = "step"
    # Peak/drive amplitude [V] (interpretation depends on waveform_type).
    V_peak: float = 130.0
    # Step ON time [s].
    tV_start: float = 0e-6
    # Step-waveform OFF time [s]. This is intentionally separate from run.T_total.
    tV_end: float = 20e-6
    # Gaussian width [s].
    tau: float = 15e-9
    # Gaussian peak-center time [s].
    t_peak: float = 100e-9
    # RF frequency [Hz].
    f_rf: float = 13.56e6
    # RF DC offset [V].
    V_dc: float = 0.0
    # RF phase [rad] at t=0.
    phi_rf: float = 0.0

@dataclass
class BoundaryConfig:
    """Boundary-condition and volumetric source toggles."""

    # Species boundary mode per electrode side.
    anode_ion_boundary: BoundaryMode = "zero_density"
    anode_electron_boundary: BoundaryMode = "implicit_drift_closure"
    cathode_ion_boundary: BoundaryMode = "implicit_drift_closure"
    cathode_electron_boundary: BoundaryMode = "electron_emission"

    # Volumetric source toggles used in continuity equations.
    # If enable_volume_sources=False, ionization/recombination terms are ignored.
    enable_volume_sources: bool = True
    enable_ionization_source: bool = True
    enable_recombination_sink: bool = True

@dataclass
class CircuitConfig:
    """External-circuit topology and lumped-element values."""

    # circuit_type options (see circuit.py for branch equations):
    #
    #   "dielectric_plasma"
    #       dielectric + plasma mapping, no explicit R0/C/L branch elements.
    #
    #   "R0_Cp"
    #       Vs -- R0 -- (node) -- [Cp || plasma]
    #
    #   "R0_Cp_Rm"
    #       Vs -- R0 -- (node) -- [Cp || (Rm + plasma)]
    #
    #   "R0_Cs_Cp"
    #       Vs -- R0 -- Cs -- (node) -- [Cp || plasma]
    #
    #   "R0_Cs_Cp_Rm"
    #       Vs -- R0 -- Cs -- (node) -- [Cp || (Rm + plasma)]
    #
    #   "R0_Cs_Ls_Cp"
    #       Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || plasma]
    #
    #   "R0_Cs_Ls_Cp_Rm"
    #       Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || (Rm + plasma)]
    #
    #   "R0_Cs_Ls_Cp_Lp"
    #       Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || Lp || plasma]
    #
    #   "R0_Cs_Ls_Cp_Lp_Rm"
    #       Vs -- R0 -- Cs -- Ls -- (node) -- [Cp || Lp || (Rm + plasma)]
    #
    # circuit_time_scheme:
    #   "explicit_euler" -> explicit ODE step in circuit.py
    #   "implicit_euler" -> implicit ODE step in circuit_implicit_euler.py
    #                       (recommended for stiff parameter sets)

    circuit_type: CircuitType = "R0_Cp_Rm"
    circuit_time_scheme: CircuitTimeScheme = "explicit_euler"
    # Series drive resistance [ohm].
    R0: float = 76.4
    # Series drive capacitor [F].
    C_s: float = 0.0
    # Series drive inductor [H].
    L_s: float = 0.0
    # Node shunt capacitor [F].
    C_p: float = 206e-12
    # Node shunt inductor [H].
    L_p: float = 0.0
    # Optional series plasma-branch resistor [ohm].
    R_m: float = 1e6

@dataclass
class EmissionConfig:
    """All surface-emission controls, yields, modes, and per-electrode parameters."""

    # Secondary electron emission yields.
    gamma: float = 0.1
    anode_electron_induced_yield: float = 0.0

    # Anode electron-induced SEE model (Vaughan).
    use_vaughan_sey: bool = False
    vaughan_Emax0_eV: float = 400.0
    vaughan_dmax0: float = 3.2
    vaughan_ks: float = 1.0
    vaughan_z: float = 0.0
    vaughan_E0: float = 0.0

    # Emission-current coupling and master toggles.
    enable_emission_in_circuit_current: bool = True
    enable_external_emission: bool = True
    electrode_material_mode: ElectrodeMaterialMode = "shared"
    enable_anode_external_emission: bool = False
    enable_cathode_external_emission: bool = True

    # Per-electrode mechanism toggles.
    # Multiple enabled mechanisms on the same electrode are summed.
    anode_enable_constant_J_emission: bool = False
    anode_enable_fn_emission: bool = False
    anode_enable_mg_emission: bool = False
    anode_enable_rd_emission: bool = False
    anode_enable_quantum_pulse_emission: bool = False

    cathode_enable_constant_J_emission: bool = False
    cathode_enable_fn_emission: bool = False
    cathode_enable_mg_emission: bool = False
    cathode_enable_rd_emission: bool = False
    cathode_enable_quantum_pulse_emission: bool = True

    # Shared parameters (used when electrode_material_mode="shared").
    # One parameter set applies to both anode and cathode.
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

    shared_emission_J_const: float = 1.0e5
    shared_emission_t_start: float = 9.5e-6
    shared_emission_t_end: float = 10.5e-6

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

    # Per-electrode parameters (used when electrode_material_mode="separate").
    # Anode and cathode can have different material/emission parameters.
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

    anode_emission_J_const: float = 1.0e5
    anode_emission_t_start: float = 9.5e-6
    anode_emission_t_end: float = 10.5e-6
    cathode_emission_J_const: float = 1.0e5
    cathode_emission_t_start: float = 9.5e-6
    cathode_emission_t_end: float = 10.5e-6

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

@dataclass
class OutputConfig:
    """Runtime output and logging controls."""

    # Save sampled arrays every `save_every` time steps.
    save_every: int = max(1, NumericsConfig().Nt // 5000)
    # Save intermediate sampled fields (fluxes, source terms, etc.).
    log_intermediate: bool = True
    # Print resolved run summary at startup.
    print_run_summary: bool = True
    # Print non-fatal startup consistency warnings.
    warn_on_config_mismatch: bool = True

# ---------------------------------------------------------------------------
# Diagnostics configuration
# ---------------------------------------------------------------------------

@dataclass
class TemporalDiagnosticsConfig:
    """
    Post-run time-series diagnostics.

    `quantities` selects what to plot. Optional `plot_groups` lets users
    overlay multiple quantities on the same figure, for example:
      (("V_app", "V_gap"), ("I_discharge",),)
    """

    enabled: bool = True
    quantities: tuple[TemporalDiagnosticQuantity, ...] = (
        "V_app",
        "V_gap",
        "I_discharge",
        "cfl",
        "particle_inventory",
    )
    plot_groups: tuple[tuple[TemporalDiagnosticQuantity, ...], ...] | None = None
    # If None, use full simulation time window.
    t_start: float | None = None
    # If None, use full simulation time window.
    t_end: float | None = None
    # Optional prefix for saving figures to files.
    savepath_prefix: str | None = None

@dataclass
class SpatialDiagnosticsConfig:
    """
    Post-run spatial diagnostics at selected times.

    `t_samples=None` means final-time only.
    """

    enabled: bool = True
    quantities: tuple[SpatialDiagnosticQuantity, ...] = ("ne", "E")
    plot_groups: tuple[tuple[SpatialDiagnosticQuantity, ...], ...] | None = None
    # Tuple of sample times [s]. For a single item, use trailing comma: (0.5e-6,).
    t_samples: tuple[float, ...] | None = None
    # Unit used for x-axis in plots.
    x_unit: Literal["m", "cm", "mm"] = "cm"
    # Optional prefix for saving figures to files.
    savepath_prefix: str | None = None

@dataclass
class AveragedSpatialDiagnosticsConfig:
    """
    Post-run time-averaged spatial diagnostics.

    Two averaging modes are supported:
    - "time_window": average over [t_avg_start, t_avg_end]
    - "last_n_cycles": average over the last N_cycle_avg RF cycles
    """

    enabled: bool = False
    quantities: tuple[SpatialDiagnosticQuantity, ...] = ("ne", "ni", "phi", "E")
    plot_groups: tuple[tuple[SpatialDiagnosticQuantity, ...], ...] | None = None
    mode: AveragedSpatialMode = "time_window"
    # Used by mode="time_window". None -> full saved range.
    t_avg_start: float | None = None
    # Used by mode="time_window". None -> full saved range.
    t_avg_end: float | None = None
    # Used by mode="last_n_cycles". Must be > 0.
    N_cycle_avg: int = 1
    x_unit: Literal["m", "cm", "mm"] = "cm"
    savepath_prefix: str | None = None

@dataclass
class DiagnosticsConfig:
    """
    Top-level diagnostics menu.

    Keeps all post-run plotting controls grouped under:
      diagnostics.temporal
      diagnostics.spatial
      diagnostics.averaged_spatial
    """

    temporal: TemporalDiagnosticsConfig = field(default_factory=TemporalDiagnosticsConfig)
    spatial: SpatialDiagnosticsConfig = field(default_factory=SpatialDiagnosticsConfig)
    averaged_spatial: AveragedSpatialDiagnosticsConfig = field(
        default_factory=AveragedSpatialDiagnosticsConfig
    )


# ---------------------------------------------------------------------------
# High-level configuration for a single simulation run
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Top-level configuration composed of grouped dataclasses."""

    run: RunConfig = field(default_factory=RunConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    plasma_state: PlasmaStateConfig = field(default_factory=PlasmaStateConfig)
    plasma: PlasmaConfig = field(default_factory=PlasmaConfig)
    user_defined_electron_kinetics: UserDefinedElectronKineticsConfig = field(
        default_factory=UserDefinedElectronKineticsConfig
    )
    local_field_approximation: LocalFieldApproximationConfig = field(
        default_factory=LocalFieldApproximationConfig
    )
    townsend_coefficient: TownsendCoefficientConfig = field(
        default_factory=TownsendCoefficientConfig
    )
    ionization_frequency_source: IonizationFrequencySourceConfig = field(
        default_factory=IonizationFrequencySourceConfig
    )
    recombination: RecombinationConfig = field(default_factory=RecombinationConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)





# ---------------------------------------------------------------------------
# Containers for outputs and transport coefficients
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    """Container for key outputs of a completed PASCHEN-1D run."""

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
    mu_e_final: np.ndarray | None = None
    D_e_final: np.ndarray | None = None
    picard_iterations: np.ndarray | None = None
    adaptive_substeps: np.ndarray | None = None
    adaptive_dt_sub: np.ndarray | None = None
    adaptive_cfl_est: np.ndarray | None = None

@dataclass
class TransportCoeffs:
    """Container for baseline transport and reaction coefficients."""

    mu_e: float
    mu_i: float
    D_e: float
    D_i: float
    beta: float
    neutral_density: float
    pr: float
    T_e_eV: float
    T_i_eV: float
