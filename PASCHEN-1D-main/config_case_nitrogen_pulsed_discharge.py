"""
config_case_nitrogen_pulsed_discharge.py

Standalone complete configuration for the nitrogen nanosecond pulsed discharge.

This file is intentionally importable as ordinary Python, but it is organized
with ``# %% [markdown]`` and ``# %%`` cell markers so editors such as VS Code,
Spyder, and Jupytext can display it as a notebook-like guided configuration
document.

The numerical defaults define a nitrogen nanosecond pulsed discharge in a
dielectric-coated gap:

    V_app(t) -- dielectric layers -- plasma gap

The case uses a 20 kV Gaussian drive over 150 ns, 60 Torr nitrogen, and a
1.0e13 m^-3 initial plasma density. External emission is disabled and
ion-induced secondary emission uses gamma = 0.3. Sweep scripts should override
individual fields of ``SimulationConfig`` rather than putting loops inside this
config file.

This module is a complete list of user-facing simulation knobs. Parameters are
grouped into focused dataclasses so users can quickly find and edit related
controls in this order:

    1. run
    2. numerics
    3. geometry
    4. plasma_state
    5. plasma (mode selector only)
    6. user_defined_electron_kinetics
    7. local_field_approximation
    8. ion_transport
    9. townsend_coefficient
    10. ionization_frequency_source
    11. recombination
    12. waveform
    13. boundary
    14. circuit
    15. emission
    16. output
    17. diagnostics

All runtime modules should access grouped fields explicitly
(`cfg.geometry.*`, `cfg.waveform.*`, etc.).

Every ``config_case_*.py`` module carries the same complete type-alias and
dataclass structure with case-tuned defaults. When the public configuration
schema changes, update all configuration modules together; the release tests
verify that their dataclass field structures remain identical.
"""

# %% [markdown]
# # PASCHEN-1D Configuration Guide
#
# This case is loaded with
# `CONFIG_MODULE = "config_case_nitrogen_pulsed_discharge"` in
# `run_paschen_1d.ipynb`.
#
# **How to use this file**
#
# 1. Edit fields inside the dataclasses below.
# 2. Keep units in SI unless a field explicitly says otherwise, such as
#    `p_Torr` in Torr or laser wavelength in meters.
# 3. Keep `run.run_name` unique for each run; it becomes the output folder.
# 4. For quick one-off changes, override fields in `run_paschen_1d.ipynb`
#    after `cfg = SimulationConfig()`.
# 5. For reusable cases, copy a complete configuration module and tune its
#    dataclass defaults. The bundled `config_case_*.py` files are standalone
#    examples with the full configuration structure.
#
# **Important convention**
#
# PASCHEN-1D solves a one-dimensional plasma gap on `x in [0, L]`. The code
# stores all dimensional quantities in SI internally. Comments next to each
# field give the expected units and the physical meaning of that knob.
#
# **Default physics case**
#
# The defaults reproduce the nitrogen nanosecond pulsed-discharge example: a
# 20 kV Gaussian drive, 60 Torr nitrogen, dielectric-coated electrodes, no
# external emission, and ion-induced secondary emission with `gamma = 0.3`.

# %%
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# %% [markdown]
# ## Type Aliases
#
# The aliases below restrict user-facing string knobs to supported values.
# They do not run physics by themselves; they make valid options explicit and
# allow static checkers/editors to catch typos earlier.
#
# If a new model or circuit topology is added to the solver, its public string
# should be added here and then implemented in the corresponding backend module.

# %%
# ---------------------------------------------------------------------------
# Type aliases for clarity / safety
# ---------------------------------------------------------------------------

VoltageWaveform = Literal["step", "gaussian", "dc", "rf", "table", "tabulated", "measured_table"]
BoundaryMode = Literal["zero_density", "electron_emission", "implicit_drift_closure"]
CircuitTimeScheme = Literal["explicit_euler", "implicit_euler", "mna"]
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
IonMobilitySourceMode = Literal[
    "user_defined_equation",
    "swarm_data_table_interpolation",
]
IonDiffusionSourceMode = Literal[
    "user_defined_equation",
    "swarm_data_table_interpolation",
    "einstein_relation",
]
IonTableOutOfRangePolicy = Literal["clip", "error"]
ElectronTableOutOfRangePolicy = Literal["clip", "error"]
HotloopBackend = Literal["numpy", "numba"]
ElectronKineticsModel = Literal[
    "user_defined_electron_kinetics",
    "local_field_approximation",
]
IonKineticsModel = Literal[
    "user_defined_ion_kinetics",
    "local_field_ion_kinetics",
]
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
    "R0_Rm_Cext",
    "R0_Cs_Cp",
    "R0_Cs_Cp_Rm",
    "R0_Cs_Ls_Cp",
    "R0_Cs_Ls_Cp_Rm",
    "R0_Cs_Ls_Cp_Lp",
    "R0_Cs_Ls_Cp_Lp_Rm_Cext",
]

TemporalDiagnosticQuantity = Literal[
    "V_app",
    "V_node",
    "V_source",
    "V_gap",
    "I_discharge",
    "I_transport_plasma",
    "I_transport_circuit",
    "I_emission_circuit",
    "I_emission_area",
    "I_displacement_gap",
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
    "mu_i",
    "D_i",
]

AveragedSpatialMode = Literal["time_window", "last_n_cycles"]


# %% [markdown]
# ## Core Grouped Configuration
#
# The dataclasses below are the actual user-editable control blocks.
#
# The final `SimulationConfig` class simply gathers these blocks into one
# object. During a run, the solver reads fields using explicit paths such as:
#
# - `cfg.run.T_total`
# - `cfg.numerics.Nx`
# - `cfg.geometry.L`
# - `cfg.plasma_state.p_Torr`
# - `cfg.circuit.R_m`
# - `cfg.emission.gamma`
#
# This grouping is meant to make the configuration easier to scan and safer to
# modify than a long flat list of variables.

# ---------------------------------------------------------------------------
# Core grouped simulation configuration
# ---------------------------------------------------------------------------


# %% [markdown]
# ### RunConfig
#
# High-level run identity and final time.
#
# - `run_name` creates the output folder. Use a new name for every sweep point
#   or manually delete/rename old output folders before reusing a name.
# - `T_total` is the physical end time of the simulation. The macro time step is
#   `dt = T_total / (Nt - 1)`, where `Nt` is set in `NumericsConfig`.

# %%
@dataclass
class RunConfig:
    """Run identification and naming."""

    # Label used to create output folder and metadata tags.
    run_name: str = "nitrogen_pulsed_discharge"
    # Total simulation time [s].
    T_total: float = 150e-9


# %% [markdown]
# ### NumericsConfig
#
# Grid, time stepping, hot-loop backend, adaptive substepping, and the
# boundary-condition/Poisson fixed-point iteration controls.
#
# **Most common edits**
#
# - Increase `Nx` for finer spatial resolution.
# - Increase `Nt` or enable/adapt `use_adaptive_substepping` for better time
#   resolution and stability.
# - Use `hotloop_backend = "numba"` for long production runs if Numba is
#   installed; use `"numpy"` for maximum portability.
#
# **Stability note**
#
# Drift-diffusion plasma simulations can be stiff. If a run warns about CFL
# violation or adaptive substep overflow, reduce `dt` by increasing `Nt`, or
# lower `target_cfl_substep`, or increase `max_substeps` cautiously.

# %%
@dataclass
class NumericsConfig:
    """Grid and numerical-method controls."""

    # Number of time nodes over run.T_total (dt = run.T_total / (Nt - 1)).
    Nt: int = 150_000
    # Number of spatial grid points over [0, L].
    Nx: int = 1_000
    # Kurganov-Tadmor slope limiter parameter (theta >= 1).
    kt_limiter_theta: float = 1.01
    # Backend for density-update hot loops (KT+RK4). Options:
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
    # Behavior when required substeps exceed max_substeps. Options:
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


# %% [markdown]
# ### GeometryConfig
#
# Geometry defines the one-dimensional plasma gap and the area used to convert
# current density to total current.
#
# - `L` is the electrode spacing.
# - `A` is the effective area represented by the 1D model.
# - `l` and `eps_r` activate dielectric-coated-electrode physics. With
#   `l = 0`, the electrodes are treated as bare metal and the dielectric
#   mapping reduces to the ordinary plasma gap.

# %%
@dataclass
class GeometryConfig:
    """Geometry and dielectric/electrode properties."""

    # Gap length [m] between electrodes.
    L: float = 0.01
    # Effective electrode area [m^2].
    A: float = 0.001
    # Dielectric thickness [m] adjacent to each electrode (0 = bare electrodes).
    l: float = 0.00175
    # Relative permittivity of dielectric.
    eps_r: float = 4.3


# %% [markdown]
# ### PlasmaStateConfig
#
# Gas identity, pressure, temperatures, and initial density.
#
# - `gas` selects user-defined closure behavior in `physics.py`.
# - `p_Torr` controls neutral density and the reduced field scale.
# - `T_e` and `T_i` are used by user-defined transport/ionization closures and
#   initialization. They do not make PASCHEN-1D solve a separate energy equation.
# - `n0` seeds the initial electron and ion density uniformly. For
#   photoemission-triggered cases, `n0 = 0` can be used when the laser pulse is
#   intended to provide the seed charge.

# %%
@dataclass
class PlasmaStateConfig:
    """Gas/plasma state variables shared by all electron-kinetics modes."""

    # Neutral gas identifier used by closures and strict table identity checks.
    # The built-in empirical equations cover argon and nitrogen. Other gases
    # use compatible electron and ion transport tables.
    gas: str = "nitrogen"
    # Gas pressure [Torr].
    p_Torr: float = 60.0
    # Electron temperature [K] used by user-defined closures/initialization.
    T_e: float = 11600.0
    # Ion (and gas-closure) temperature [K].
    T_i: float = 300.0
    # Initial uniform electron/ion density [m^-3].
    n0: float = 1.0e13


# %% [markdown]
# ### PlasmaConfig
#
# This block chooses which physics closures are active. It points the solver to
# either user-defined formulas or local-field/table-based closures configured
# in the sections below.
#
# **Recommended editing pattern**
#
# - First choose the high-level model string here.
# - Then edit the matching detailed block, such as
#   `LocalFieldApproximationConfig`, `TownsendCoefficientConfig`, or
#   `IonizationFrequencySourceConfig`.
#
# Changing a model selector without checking its detailed block can silently
# switch the run to an unintended table or formula source.

# %%
@dataclass
class PlasmaConfig:
    """Top-level plasma-physics model selectors."""

    # Electron kinetics model. Options:
    # - "user_defined_electron_kinetics":
    #      electron transport from user equations in physics.py
    # - "local_field_approximation":
    #      electron transport from local E/N (table or user equation)
    electron_kinetics_model: ElectronKineticsModel = "user_defined_electron_kinetics"

    # Ion kinetics model. Options:
    # - "user_defined_ion_kinetics": empirical hooks in physics.py
    # - "local_field_ion_kinetics": E/N tables and/or Einstein diffusion
    ion_kinetics_model: IonKineticsModel = "user_defined_ion_kinetics"

    # Impact-ionization model. Options:
    # - "from_townsend_alpha":
    #      nu_i = alpha * |u_e|, with alpha source from TownsendCoefficientConfig
    # - "from_ionization_frequency":
    #      nu_i source from IonizationFrequencySourceConfig
    impact_ionization_model: ImpactIonizationModel = "from_townsend_alpha"

    # Recombination model. Options:
    # - "user_defined_constant_coefficient": use the constant coefficient
    #   from RecombinationConfig (the only currently supported option)
    recombination_model: RecombinationModel = "user_defined_constant_coefficient"


# %%
@dataclass
class UserDefinedElectronKineticsConfig:
    """
    Controls for the user-defined electron-kinetics mode.

    This mode currently uses user-defined electron transport equations in
    ``physics.py`` and requires no extra per-mode knobs.
    """
    pass


# %% [markdown]
# ### TownsendCoefficientConfig
#
# Used when `plasma.impact_ionization_model = "from_townsend_alpha"`.
#
# In this mode, PASCHEN-1D evaluates an effective ionization frequency as
# `nu_i = alpha * |u_e|`. The Townsend coefficient `alpha` can come from:
#
# - a user-defined formula in `physics.py`, or
# - an E/N swarm-data table.
#
# Use the table option when you want the ionization model to follow BOLSIG+ or
# another swarm solver rather than the built-in empirical expression.

# %%
@dataclass
class TownsendCoefficientConfig:
    """Controls for Townsend-alpha sourcing in alpha-based ionization mode."""

    # Townsend-alpha source mode. Options:
    # - "user_defined_equation":
    #      alpha from physics.py -> compute_user_defined_townsend_alpha(...)
    # - "interpolate_from_e_over_n_table":
    #      alpha/N(E/N) table interpolation, then alpha=(alpha/N)*N
    townsend_alpha_source_mode: Literal[
        "user_defined_equation",
        "interpolate_from_e_over_n_table",
    ] = "user_defined_equation"
    # Optional dedicated alpha/N filename (or path relative to
    # electron_swarm_data). If None, the default electron table is reused:
    # - E/N axis -> local_field_approximation.electron_swarm_data_path
    townsend_alpha_swarm_data_path: str = "n2_swarm_output_full_EoverN.dat"


# %% [markdown]
# ### IonizationFrequencySourceConfig
#
# Used when `plasma.impact_ionization_model = "from_ionization_frequency"`.
#
# This bypasses the `alpha * |u_e|` conversion and directly supplies `nu_i`.
# This can be useful when a swarm table or reduced model already provides an
# ionization frequency as a function of local E/N.

# %%
@dataclass
class IonizationFrequencySourceConfig:
    """Controls for direct nu_i sourcing in impact-ionization mode."""

    # Direct ionization-frequency source mode. Options:
    # - "user_defined_equation":
    #      nu_i from physics.py -> compute_user_defined_ionization_frequency(...)
    # - "interpolate_from_e_over_n_table":
    #      nu_i/N(E/N) table interpolation, then nu_i=(nu_i/N)*N
    ionization_frequency_source_mode: Literal[
        "user_defined_equation",
        "interpolate_from_e_over_n_table",
    ] = "interpolate_from_e_over_n_table"
    # Optional dedicated nu_i/N filename (or path relative to
    # electron_swarm_data). If None, the default electron table is reused:
    # - E/N axis -> local_field_approximation.electron_swarm_data_path
    ionization_frequency_swarm_data_path: str = "n2_swarm_output_full_EoverN.dat"


# %% [markdown]
# ### RecombinationConfig
#
# Controls the volumetric loss term `S_rec = beta * n_e * n_i` when
# `boundary.enable_recombination_sink = True`.
#
# The coefficient is in `m^3/s`. It is intentionally exposed because
# recombination can strongly affect the current-decay tail in transient
# discharge simulations.

# %%
@dataclass
class RecombinationConfig:
    """Controls for volumetric electron-ion recombination."""

    # Constant recombination coefficient beta [m^3/s].
    recombination_coefficient: float = 2.0e-13


# %% [markdown]
# ### LocalFieldApproximationConfig
#
# Used when `plasma.electron_kinetics_model = "local_field_approximation"`.
#
# Local-field approximation means the electron transport coefficients are
# evaluated from the instantaneous local reduced electric field `E/N`.
# This is a fluid-model approximation; it does not solve a nonlocal electron
# energy equation.
#
# The table path should point to a compatible swarm-output file containing the
# E/N-dependent coefficients required by `physics.py`.

# %%
@dataclass
class LocalFieldApproximationConfig:
    """
    Controls for local-field approximation (LFA) mode.

    LFA uses local E/N to evaluate electron transport coefficients.
    """

    # Electron transport source for LFA. Options:
    # - "user_defined_equation": define equations in:
    #   physics.py -> compute_user_defined_electron_mobility(...)
    #   physics.py -> compute_user_defined_electron_diffusion(...)
    # - "swarm_data_table_interpolation": interpolate the configured table in:
    #   physics.py -> build_electron_mobility_profile(...)
    #   physics.py -> build_electron_diffusion_profile(...)
    electron_transport_source: TransportSourceMode = "user_defined_equation"
    # Filename or path relative to the bundled electron_swarm_data directory.
    # Do not include an absolute machine-specific path.
    electron_swarm_data_path: str = "n2_swarm_output_full_EoverN.dat"


# %%
@dataclass
class ElectronSwarmDataConfig:
    """Validation and interpolation policy shared by all electron tables."""

    # Behavior when local E/N leaves a selected BOLSIG+ table range. Options:
    # - "clip": use the nearest endpoint value outside the table range
    # - "error": stop the run when any requested E/N is outside the range
    out_of_range_policy: ElectronTableOutOfRangePolicy = "clip"
    # Maximum difference between BOLSIG+ gas temperature and plasma_state.T_i.
    gas_temperature_tolerance_K: float = 5.0


# %% [markdown]
# ### IonTransportConfig
#
# Ion transport is identified by an ion/neutral pair. The neutral is always
# `plasma_state.gas`; `positive_ion` selects the transported positive ion.
# Table files are normalized LXCat exports containing embedded identity,
# temperature, source, citation, and checksum provenance. PASCHEN-1D validates
# those fields before allocating the simulation arrays.

# %%
@dataclass
class IonTransportConfig:
    """Positive-ion mobility and longitudinal-diffusion configuration."""

    # Positive ion transported by the single-ion fluid equation.
    positive_ion: str = "N2+"
    # Mobility source used by local_field_ion_kinetics. Options:
    # - "user_defined_equation": use the ion-mobility equation in physics.py
    # - "swarm_data_table_interpolation": interpolate mobility_table_path
    mobility_source_mode: IonMobilitySourceMode = "user_defined_equation"
    # Diffusion source used by local_field_ion_kinetics. Options:
    # - "user_defined_equation": use the ion-diffusion equation in physics.py
    # - "swarm_data_table_interpolation": interpolate diffusion_table_path
    # - "einstein_relation": compute D_i = mu_i*kB*T_i/e using the active
    #   mobility profile
    diffusion_source_mode: IonDiffusionSourceMode = "user_defined_equation"
    # Filenames or paths relative to the bundled ion_swarm_data directory. A
    # value is required for each selected table mode. Basenames are sufficient
    # for the bundled library because normalized table filenames are unique.
    mobility_table_path: str | None = None
    diffusion_table_path: str | None = None
    # Local E/N behavior outside the selected ion table's range. Options:
    # - "clip": use the nearest endpoint value outside the table range
    # - "error": stop the run when any requested E/N is outside the range
    out_of_range_policy: IonTableOutOfRangePolicy = "clip"
    # Maximum accepted difference between table Tgas and plasma_state.T_i.
    gas_temperature_tolerance_K: float = 5.0


# %% [markdown]
# ### WaveformConfig
#
# Controls the externally applied voltage function `V_app(t)`.
#
# - `step`: on/off voltage pulse using `V_peak`, `tV_start`, and `tV_end`.
# - `dc`: constant applied voltage.
# - `gaussian`: Gaussian voltage pulse using `V_peak`, `t_peak`, and `tau`.
# - `rf`: sinusoidal drive using `f_rf`, `V_dc`, and `phi_rf`.
# - `table` / `tabulated` / `measured_table`: interpolate a voltage waveform
#   from a file with user-selected columns and scale/offset factors.
#
# The source waveform is not necessarily equal to the plasma-gap voltage if an
# external circuit is active.

# %%
@dataclass
class WaveformConfig:
    """Applied-voltage waveform settings."""

    # Waveform type. Options:
    # - "step": rectangular on/off pulse
    # - "gaussian": Gaussian pulse
    # - "dc": constant voltage
    # - "rf": sinusoidal voltage
    # - "table", "tabulated", or "measured_table": file interpolation
    waveform_type: VoltageWaveform = "gaussian"
    # Peak/drive amplitude [V] (interpretation depends on waveform_type).
    V_peak: float = 20_000.0
    # Step ON time [s].
    tV_start: float = 0e-6
    # Step-waveform OFF time [s]. This is intentionally separate from run.T_total.
    tV_end: float = 91e-9
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
    # Tabulated waveform controls used when waveform_type is "table",
    # "tabulated", or "measured_table".  Columns are zero-based.
    table_path: str = "voltage_waveform.csv"
    table_time_column: int = 0
    table_voltage_column: int = 1
    table_time_scale: float = 1.0
    table_time_offset: float = 0.0
    table_voltage_scale: float = 1.0
    table_voltage_offset: float = 0.0


# %% [markdown]
# ### BoundaryConfig
#
# Controls particle boundary closure modes and whether volumetric source terms
# are active.
#
# Boundary modes:
#
# - `zero_density`: impose zero density at the boundary node.
# - `implicit_drift_closure`: infer the boundary density from the drift flux
#   closure and local field direction.
# - `electron_emission`: use the dedicated emission-aware electron boundary
#   closure. This is the correct mode for a surface that emits electrons.
#
# The source toggles are useful for controls:
#
# - turn off ionization to isolate photoemission/circuit response,
# - turn off recombination to test pure avalanche growth and transport,
# - turn off all volume sources for vacuum-like transport tests.

# %%
@dataclass
class BoundaryConfig:
    """Boundary-condition and volumetric source toggles."""

    # Species boundary mode used by each of the four fields below. Options:
    # - "zero_density": impose zero density at the boundary node
    # - "electron_emission": use the emission-aware electron closure
    # - "implicit_drift_closure": infer density from the drift-flux closure
    # Options: "zero_density", "electron_emission", "implicit_drift_closure".
    anode_ion_boundary: BoundaryMode = "zero_density"
    # Options: "zero_density", "electron_emission", "implicit_drift_closure".
    anode_electron_boundary: BoundaryMode = "implicit_drift_closure"
    # Options: "zero_density", "electron_emission", "implicit_drift_closure".
    cathode_ion_boundary: BoundaryMode = "implicit_drift_closure"
    # Options: "zero_density", "electron_emission", "implicit_drift_closure".
    cathode_electron_boundary: BoundaryMode = "electron_emission"

    # Volumetric source toggles used in continuity equations.
    # If enable_volume_sources=False, ionization/recombination terms are ignored.
    enable_volume_sources: bool = True
    enable_ionization_source: bool = True
    enable_recombination_sink: bool = True


# %% [markdown]
# ### CircuitConfig
#
# External circuit model coupled to the plasma gap.
#
# The reduced topology used by the default case is:
#
# ```text
# V_app(t) -- R0 -- V_node -- R_m -- V_gap
#                                      |
#                                     C_ext
#                                      |
#                                    ground
# ```
#
# Here `R0` can represent source-side sag or pulser/feedthrough resistance,
# `R_m` is the measurement/load resistor, and `C_ext` is the external
# load-side stray capacitance. The geometric gas-gap capacitance is handled
# separately by PASCHEN-1D.
#
# **Reduced topologies vs MNA**
#
# Reduced topology strings use hand-written explicit/implicit circuit ODEs.
# If the exact topology you want is not available, use the maximum topology
# `R0_Cs_Ls_Cp_Lp_Rm_Cext` with `circuit_time_scheme = "mna"`. In MNA mode,
# elements can be removed using neutral values:
#
# - `R0 = 0` removes source resistance as a short.
# - `C_s = np.inf` removes the series capacitor as a short.
# - `L_s = 0` removes the series inductor as a short.
# - `C_p = 0` removes the node shunt capacitor as an open.
# - `L_p = np.inf` removes the node shunt inductor as an open.
# - `R_m = 0` removes the measurement resistor as a short.
# - `C_ext = 0` removes the load-side capacitance as an open.

# %%
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
    #   "R0_Rm_Cext"
    #       Vs -- R0 -- V_node -- Rm -- [Cext || plasma]
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
    #   "R0_Cs_Ls_Cp_Lp_Rm_Cext"
    #       Vs -- R0 -- Cs -- Ls -- V_node -- Rm -- [Cext || plasma],
    #       with Cp and Lp shunting V_node to ground.
    #
    # circuit_time_scheme:
    #   "explicit_euler" -> explicit ODE step in circuit.py
    #   "implicit_euler" -> implicit ODE step in circuit_implicit_euler.py
    #                       (recommended for stiff parameter sets)
    #   "mna" -> unified backward-Euler modified-nodal-analysis backend in circuit_mna.py
    #
    # Usage guidance:
    # - If you choose a specific reduced topology such as "R0_Rm_Cext",
    #   "R0_Cp", or "R0_Cs_Cp_Rm", keep the listed elements in that topology
    #   as physical finite values expected by that topology.
    # - If you need a custom circuit that is not available as a named reduced
    #   topology, use the maximum topology "R0_Cs_Ls_Cp_Lp_Rm_Cext" with
    #   circuit_time_scheme="mna". In that mode, elements can be removed with
    #   neutral values: R0=0, C_s=np.inf, L_s=0, C_p=0, L_p=np.inf, R_m=0,
    #   and C_ext=0.

    # Circuit topology options: "dielectric_plasma", "R0_Cp", "R0_Cp_Rm",
    # "R0_Rm_Cext", "R0_Cs_Cp", "R0_Cs_Cp_Rm", "R0_Cs_Ls_Cp",
    # "R0_Cs_Ls_Cp_Rm", "R0_Cs_Ls_Cp_Lp", and
    # "R0_Cs_Ls_Cp_Lp_Rm_Cext". See the diagrams above for element layout.
    circuit_type: CircuitType = "dielectric_plasma"
    # Circuit time-integration options: "explicit_euler", "implicit_euler",
    # and "mna". See the backend descriptions above.
    circuit_time_scheme: CircuitTimeScheme = "explicit_euler"
    # Source-side effective resistance [ohm].  In the default reduced-source
    # example this lumps the HV pulser plus source-side cable/feedthrough sag.
    R0: float = 0.0
    # Series drive capacitor [F].
    C_s: float = 0.0
    # Series drive inductor [H].
    L_s: float = 0.0
    # Source-side node shunt capacitor [F].
    C_p: float = 0.0
    # Node shunt inductor [H].
    L_p: float = 0.0
    # Optional series plasma-branch resistor [ohm].
    R_m: float = 0.0
    # External load-side capacitance [F] from V_gap to ground.
    # The geometric gas-gap capacitance is handled separately by PASCHEN-1D.
    C_ext: float = 0.0


# %% [markdown]
# ### EmissionConfig
#
# Surface-emission physics for anode and cathode.
#
# **Secondary emission**
#
# - `gamma` is the cathode ion-induced secondary electron emission coefficient.
# - `anode_electron_induced_yield` is the anode electron-induced emission yield.
# - `use_vaughan_sey` activates the Vaughan-style electron-induced SEE model
#   for the anode branch.
#
# **External emission mechanisms**
#
# Each electrode can independently enable:
#
# - `constant_J`: prescribed current-density emission over a time window,
# - `fn`: Fowler-Nordheim field emission,
# - `mg`: Murphy-Good field emission,
# - `rd`: Richardson-Dushman thermionic emission,
# - `quantum_pulse`: pulsed photoemission from the quantum model.
#
# Multiple mechanisms on the same electrode are summed.
#
# **Material mode**
#
# - `electrode_material_mode = "shared"` uses the `shared_*` parameters for
#   both electrodes.
# - `electrode_material_mode = "separate"` uses `anode_*` and `cathode_*`
#   parameters independently.
#
# **Photoemission area scaling**
#
# The quantum photoemission model computes a physical laser/emission spot. The
# backend automatically converts the spot-emitted current density into the
# represented 1D electrode-area current using the laser widths
# `*_laser_wx_m`, `*_laser_wy_m`, and `geometry.A`. Users should set the laser
# spot dimensions physically rather than adding an extra empirical scaling knob.
#
# **Default case**
#
# External emission is disabled for this nitrogen pulsed-discharge case. The
# retained laser/material fields are inactive unless the corresponding
# emission toggles are deliberately enabled.

# %%
@dataclass
class EmissionConfig:
    """All surface-emission controls, yields, modes, and per-electrode parameters."""

    # Secondary electron emission yields.
    gamma: float = 0.3
    anode_electron_induced_yield: float = 0.0

    # Anode electron-induced SEE model (Vaughan).
    use_vaughan_sey: bool = False
    vaughan_Emax0_eV: float = 400.0
    vaughan_dmax0: float = 3.2
    vaughan_ks: float = 1.0
    vaughan_z: float = 0.0
    vaughan_E0: float = 0.0

    # External-emission controls.
    enable_external_emission: bool = False
    # Electrode material parameter mode. Options:
    # - "shared": use the shared_* parameters for both electrodes
    # - "separate": use the anode_* and cathode_* parameters independently
    electrode_material_mode: ElectrodeMaterialMode = "shared"
    enable_anode_external_emission: bool = False
    enable_cathode_external_emission: bool = False

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
    cathode_enable_quantum_pulse_emission: bool = False

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
    # Laser energy for the Q ~= 210 pC comparison case. The emitted charge is
    # computed by the quantum-photoemission backend; no target-charge
    # normalization is applied at runtime.
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


# %% [markdown]
# ### OutputConfig
#
# Controls what the solver writes during the run and how much status
# information is printed.
#
# - `save_every` controls temporal sampling cadence for saved arrays.
# - `log_intermediate` enables additional sampled spatial fields such as fluxes
#   and source terms, which are needed by many diagnostics notebooks.
# - `print_run_summary` is useful for traceability because it prints the
#   resolved case settings at the start of each run.

# %%
@dataclass
class OutputConfig:
    """Runtime output and logging controls."""

    # Save sampled arrays every `save_every` time steps.
    save_every: int = max(1, NumericsConfig().Nt // 5000)
    # Save intermediate sampled fields (fluxes, source terms, etc.).
    log_intermediate: bool = True
    # Print resolved run summary at startup.
    print_run_summary: bool = True

# ---------------------------------------------------------------------------
# Diagnostics configuration
# ---------------------------------------------------------------------------


# %% [markdown]
# ## Diagnostics Configuration
#
# These classes define optional post-run diagnostic plotting requests. They do
# not change the simulation physics.
#
# The dedicated notebooks
# `diagnostics_temporal_profiles.ipynb`,
# `diagnostics_spatial_snapshots.ipynb`, and
# `diagnostics_spatial_averages.ipynb` provide more flexible interactive
# plotting from saved run folders. The config-side diagnostics are kept here so
# scripted workflows can still request a standard set of outputs.


# %% [markdown]
# ### TemporalDiagnosticsConfig
#
# Time-history diagnostics saved or plotted after the run.
#
# Common quantities include voltages, discharge-current decompositions, CFL
# history, Picard iterations, adaptive-substep history, and particle inventory.
# Use `plot_groups` when same-nature quantities should be overlaid, such as
# `("V_app", "V_gap")` or current-decomposition terms.

# %%
@dataclass
class TemporalDiagnosticsConfig:
    """
    Post-run time-series diagnostics.

    `quantities` selects what to plot. Optional `plot_groups` lets users
    overlay multiple quantities on the same figure, for example:
      (("V_app", "V_gap"), ("I_discharge",),)
    """

    enabled: bool = True
    # Temporal quantity options: "V_app", "V_node", "V_source", "V_gap",
    # "I_discharge", "I_transport_plasma", "I_transport_circuit",
    # "I_emission_circuit", "I_emission_area", "I_displacement_gap", "cfl",
    # "picard_iterations", "adaptive_substeps", "adaptive_dt_sub",
    # "adaptive_cfl_est", and "particle_inventory".
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


# %% [markdown]
# ### SpatialDiagnosticsConfig
#
# Instantaneous spatial-profile diagnostics at selected saved times.
#
# - `quantities` selects fields such as density, potential, electric field,
#   fluxes, ionization coefficients, source terms, and transport coefficients.
# - `t_samples = None` means final saved snapshot only.
# - For one explicit sample time, remember Python tuple syntax:
#   `t_samples = (1.0e-6,)`.

# %%
@dataclass
class SpatialDiagnosticsConfig:
    """
    Post-run spatial diagnostics at selected times.

    `t_samples=None` means final-time only.
    """

    enabled: bool = True
    # Spatial quantity options: "ne", "ni", "phi", "E", "Gamma_i",
    # "Gamma_e", "townsend_alpha", "nu_i", "S_ion", "S", "mu_e", "D_e",
    # "mu_i", and "D_i".
    quantities: tuple[SpatialDiagnosticQuantity, ...] = ("ne", "E")
    plot_groups: tuple[tuple[SpatialDiagnosticQuantity, ...], ...] | None = None
    # Tuple of sample times [s]. For a single item, use trailing comma: (0.5e-6,).
    t_samples: tuple[float, ...] | None = None
    # Unit used for the plot x-axis. Options: "m", "cm", and "mm".
    x_unit: Literal["m", "cm", "mm"] = "cm"
    # Optional prefix for saving figures to files.
    savepath_prefix: str | None = None


# %% [markdown]
# ### AveragedSpatialDiagnosticsConfig
#
# Time-averaged spatial profiles. This is useful for RF/cyclic cases or for
# smoothing rapidly oscillating transient fields over a chosen time window.
#
# `mode = "time_window"` averages over `[t_avg_start, t_avg_end]`.
# `mode = "last_n_cycles"` uses `waveform.f_rf` and averages over the final
# `N_cycle_avg` RF cycles.

# %%
@dataclass
class AveragedSpatialDiagnosticsConfig:
    """
    Post-run time-averaged spatial diagnostics.

    Two averaging modes are supported:
    - "time_window": average over [t_avg_start, t_avg_end]
    - "last_n_cycles": average over the last N_cycle_avg RF cycles
    """

    enabled: bool = False
    # Spatial quantity options: "ne", "ni", "phi", "E", "Gamma_i",
    # "Gamma_e", "townsend_alpha", "nu_i", "S_ion", "S", "mu_e", "D_e",
    # "mu_i", and "D_i".
    quantities: tuple[SpatialDiagnosticQuantity, ...] = ("ne", "ni", "phi", "E")
    plot_groups: tuple[tuple[SpatialDiagnosticQuantity, ...], ...] | None = None
    # Averaging mode options:
    # - "time_window": average over [t_avg_start, t_avg_end]
    # - "last_n_cycles": average over the last N_cycle_avg RF cycles
    mode: AveragedSpatialMode = "time_window"
    # Used by mode="time_window". None -> full saved range.
    t_avg_start: float | None = None
    # Used by mode="time_window". None -> full saved range.
    t_avg_end: float | None = None
    # Used by mode="last_n_cycles". Must be > 0.
    N_cycle_avg: int = 1
    # Unit used for the plot x-axis. Options: "m", "cm", and "mm".
    x_unit: Literal["m", "cm", "mm"] = "cm"
    savepath_prefix: str | None = None


# %% [markdown]
# ### DiagnosticsConfig
#
# Container for the three diagnostics menus. Edit the nested blocks above for
# detailed selections. Most users will do deeper plotting in the diagnostics
# notebooks after the run has finished.

# %%
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


# %% [markdown]
# ## SimulationConfig
#
# This is the object constructed by `run_paschen_1d.ipynb`.
#
# Example override pattern:
#
# ```python
# cfg = SimulationConfig()
# cfg.run.run_name = "my_case_001"
# cfg.waveform.V_peak = 250.0
# cfg.circuit.C_ext = 18e-12
# ```
#
# The grouped structure below is intentionally explicit. It makes it clear
# which subsystem each knob belongs to and reduces accidental name collisions
# between unrelated controls.

# %%
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
    electron_swarm_data: ElectronSwarmDataConfig = field(
        default_factory=ElectronSwarmDataConfig
    )
    ion_transport: IonTransportConfig = field(default_factory=IonTransportConfig)
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


# %% [markdown]
# ## Runtime Containers
#
# The classes below are not user-editable input knobs. They are structured
# containers populated by the solver and helper functions.
#
# - `SimulationState` stores final fields and sampled scalar histories returned
#   by `run_simulation`.
# - `TransportCoeffs` stores baseline transport/reaction coefficients used by
#   initialization and diagnostic reporting.

# %%
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
    I_transport_plasma: np.ndarray | None = None
    I_transport_circuit: np.ndarray | None = None
    I_emission_circuit: np.ndarray | None = None
    I_emission_area: np.ndarray | None = None
    I_displacement_gap: np.ndarray | None = None
    V_node: np.ndarray | None = None
    V_source: np.ndarray | None = None
    mu_e_final: np.ndarray | None = None
    D_e_final: np.ndarray | None = None
    mu_i_final: np.ndarray | None = None
    D_i_final: np.ndarray | None = None
    picard_iterations: np.ndarray | None = None
    adaptive_substeps: np.ndarray | None = None
    adaptive_dt_sub: np.ndarray | None = None
    adaptive_cfl_est: np.ndarray | None = None

# %%
@dataclass
class TransportCoeffs:
    """Container for shared gas-state and reaction coefficients."""

    beta: float
    neutral_density: float
    pr: float
    T_e_eV: float
    T_i_eV: float
