"""
physics.py

Physics-facing helper routines for PASCHEN-1D.

This module contains five kinds of logic:
1) Applied-voltage waveform builders.
2) User-edit hooks for transport and ionization coefficients.
3) Swarm-data parsing and interpolation helpers for table-driven models.
4) Reference gas-state construction (neutral density, baseline scalars, beta).
5) Initial-state construction for phi, E, n_e, and n_i.

Conventions:
- SI units are used unless noted otherwise.
- Pressure inputs for the default empirical ionization/transport closures
  remain in Torr.
- The active runtime coefficient profiles are built by the
  ``compute_user_defined_*`` and ``build_*_profile`` functions below.
"""

from pathlib import Path
from typing import Callable

import numpy as np

from physical_constants import kB, e, m_e
from config import SimulationConfig, TransportCoeffs


_MU_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_D_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_ALPHA_OVER_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_ELECTRON_TRANSPORT_FALLBACK_WARNED: set[str] = set()
_ION_TRANSPORT_FALLBACK_WARNED: set[str] = set()
_ELECTRON_DIFFUSION_FALLBACK_WARNED: set[str] = set()
_ION_DIFFUSION_FALLBACK_WARNED: set[str] = set()
_TOWNSEND_ALPHA_FALLBACK_WARNED: set[str] = set()
_SWARM_DATA_SECTION_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


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
# Shared path, warning, and gas-state utilities
# ============================================================

def compute_background_neutral_density(cfg: SimulationConfig) -> np.float32:
    """
    Estimate the uniform background neutral gas number density.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation configuration containing:
        - p_Torr : gas pressure [Torr]
        - T_i    : ion temperature [K]

    Returns
    -------
    np.float32
        Background neutral number density [m^-3].

    Notes
    -----
    The current PASCHEN-1D model does not evolve a neutral continuity
    equation. Neutral depletion and gas heating are neglected, so the
    neutral background is treated as a fixed uniform reservoir.

    For the present transport-lookup workflow, the gas temperature is
    closed by the heavy-particle proxy T_gas = T_i, and the neutral
    density is computed from the ideal-gas relation N_g = p / (k_B T_gas),
    with pressure converted from Torr to Pa.
    """
    p_Pa = float(cfg.p_Torr) * 133.32236842105263
    T_gas = max(float(cfg.T_i), 1.0)
    return np.float32(p_Pa / (kB * T_gas))


def _resolve_project_path(path_str: str) -> Path:
    """
    Resolve a project-relative or absolute path used by PASCHEN-1D.
    """
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def _warn_fallback_once(
    warned_cache: set[str],
    prefix: str,
    source_selector: str,
    reason: str,
    *,
    fallback_target: str = "user_defined_equations",
    detail: str = "fell back to",
) -> None:
    """
    Print a one-time fallback warning keyed by the exact reason string.
    """
    if reason in warned_cache:
        return
    warned_cache.add(reason)
    print(
        f"{prefix} warning: "
        f"{source_selector}='swarm_data_table_interpolation' "
        f"{detail} {fallback_target} because {reason}."
    )


def load_swarm_data_section(
    path_str: str,
    section_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one named E/N section from a supported raw swarm-data output file.

    Parameters
    ----------
    path_str : str
        Path to the raw swarm-data output file.
    section_label : str
        Section label appearing after ``E/N (Td)`` in the header line,
        for example ``"Mobility *N (1/m/V/s)"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(en_td, values)`` in float64, sorted in ascending ``E/N``.
    """
    resolved = str(_resolve_project_path(path_str))
    cache_key = (resolved, section_label)
    cached = _SWARM_DATA_SECTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lines = Path(resolved).read_text(errors="replace").splitlines()
    target_header = f"E/N (Td)\t{section_label}"
    in_block = False
    pairs: list[tuple[float, float]] = []

    for line in lines:
        if not in_block:
            if line.strip() == target_header:
                in_block = True
            continue

        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("E/N (Td)"):
            break

        parts = stripped.split()
        if len(parts) < 2:
            continue

        try:
            en_val = float(parts[0])
            y_val = float(parts[1])
        except ValueError:
            continue
        pairs.append((en_val, y_val))

    if len(pairs) < 2:
        raise ValueError(
            f"Could not find a usable swarm-data section '{section_label}' in "
            f"'{resolved}'."
        )

    raw = np.asarray(pairs, dtype=np.float64)
    en_td = raw[:, 0]
    values = raw[:, 1]
    order = np.argsort(en_td)
    en_td = en_td[order]
    values = values[order]
    _SWARM_DATA_SECTION_CACHE[cache_key] = (en_td, values)
    return en_td, values


def _load_swarm_quantity_data(
    path_str: str,
    *,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
    section_label: str,
    source_label: str,
    value_label: str,
    allow_zero_values: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one E/N-dependent quantity from either a raw swarm-data file or a
    legacy two-column table.
    """
    resolved = str(_resolve_project_path(path_str))
    cached = cache.get(resolved)
    if cached is not None:
        return cached

    try:
        en_td, values = load_swarm_data_section(resolved, section_label)
    except ValueError:
        raw = np.loadtxt(resolved, comments="#", dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError(
                f"{source_label} '{resolved}' must contain either "
                f"a raw swarm-data output block or a two-column table with "
                f"E/N [Td] and {value_label}."
            )
        en_td = raw[:, 0].astype(np.float64, copy=False)
        values = raw[:, 1].astype(np.float64, copy=False)

    if en_td.size < 2:
        raise ValueError(
            f"{source_label} '{resolved}' must contain at least two data rows."
        )

    if allow_zero_values:
        invalid_values = np.any(values < 0.0)
        value_phrase = f"non-negative {value_label} values"
    else:
        invalid_values = np.any(values <= 0.0)
        value_phrase = f"strictly positive {value_label} values"

    if np.any(en_td <= 0.0) or invalid_values:
        raise ValueError(
            f"{source_label} '{resolved}' must contain strictly positive E/N "
            f"values and {value_phrase}."
        )

    order = np.argsort(en_td)
    en_td = en_td[order]
    values = values[order]
    cache[resolved] = (en_td, values)
    return en_td, values


def _interpolate_swarm_quantity_from_table(
    en_td_table: np.ndarray,
    values_table: np.ndarray,
    E_column: np.ndarray,
    neutral_density: float,
    *,
    density_operation: str,
    allow_zero_values: bool = False,
) -> np.ndarray:
    """
    Interpolate a generic swarm-data quantity defined on E/N.
    """
    N_g = max(float(neutral_density), 1.0)
    en_td_local = np.abs(E_column).astype(np.float64, copy=False) * 1.0e21 / N_g
    en_td_local = np.clip(en_td_local, en_td_table[0], en_td_table[-1])

    log_en_td = np.log10(en_td_table)
    safe_values = (
        np.maximum(values_table, 1.0e-300) if allow_zero_values else values_table
    )
    log_values = np.log10(safe_values)
    log_en_td_local = np.log10(en_td_local)
    log_values_local = np.interp(log_en_td_local, log_en_td, log_values)
    values_local = np.power(10.0, log_values_local)

    if density_operation == "divide_by_density":
        runtime_values = values_local / N_g
    elif density_operation == "multiply_by_density":
        runtime_values = values_local * N_g
    else:
        raise ValueError(f"Unknown density_operation: {density_operation}")

    return runtime_values.astype(np.float32, copy=False)


def load_electron_mobility_muN_data(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load electron-mobility data for mu_e(E/N).

    Parameters
    ----------
    path_str : str
        Path to either:
        - a supported raw swarm-data output file containing the section
          ``Mobility *N (1/m/V/s)``, or
        - a legacy two-column extracted table with:
          column 1 = E/N [Td], column 2 = mu_e * N [1/(m V s)].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(en_td, muN)`` in float64, sorted in ascending ``E/N``.

    Notes
    -----
    The preferred source is a supported raw swarm-data output file. Legacy extracted
    two-column tables are still accepted for backward compatibility.
    """
    return _load_swarm_quantity_data(
        path_str,
        cache=_MU_N_TABLE_CACHE,
        section_label="Mobility *N (1/m/V/s)",
        source_label="Electron-mobility source",
        value_label="mu_e*N",
    )


def load_electron_diffusion_DN_data(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load electron-diffusion data for D_e(E/N).

    Parameters
    ----------
    path_str : str
        Path to either:
        - a supported raw swarm-data output file containing the section
          ``Diffusion coefficient *N (1/m/s)``, or
        - a legacy two-column extracted table with:
          column 1 = E/N [Td], column 2 = D_e * N [1/(m s)].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(en_td, DN)`` in float64, sorted in ascending ``E/N``.
    """
    return _load_swarm_quantity_data(
        path_str,
        cache=_D_N_TABLE_CACHE,
        section_label="Diffusion coefficient *N (1/m/s)",
        source_label="Electron-diffusion source",
        value_label="D_e*N",
    )


def load_townsend_alpha_over_N_data(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Townsend alpha/N data for alpha(E/N).

    Parameters
    ----------
    path_str : str
        Path to either:
        - a supported raw swarm-data output file containing the section
          ``Townsend ioniz. coef. alpha/N (m2)``, or
        - a legacy two-column extracted table with:
          column 1 = E/N [Td], column 2 = alpha/N [m^2].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(en_td, alpha_over_N)`` in float64, sorted in ascending E/N.
    """
    return _load_swarm_quantity_data(
        path_str,
        cache=_ALPHA_OVER_N_TABLE_CACHE,
        section_label="Townsend ioniz. coef. alpha/N (m2)",
        source_label="Townsend-alpha source",
        value_label="alpha/N",
        allow_zero_values=True,
    )


def interpolate_electron_mobility_from_muN_table(
    cfg: SimulationConfig,
    E_column: np.ndarray,
    neutral_density: float,
) -> np.ndarray:
    """
    Interpolate mu_e(x) from a swarm-data mu_e(E/N) table.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation configuration containing the swarm-data source path.
    E_column : np.ndarray
        Local electric field [V/m].
    neutral_density : float
        Background neutral density [m^-3].

    Returns
    -------
    np.ndarray
        Electron mobility profile [m^2/(V s)], shape matching E_column.

    Notes
    -----
    The table stores mu_e * N versus E/N in Townsend. The runtime
    conversion is:

        (E/N)[Td] = |E| / N_g * 1e21
        mu_e      = (mu_e * N_g) / N_g

    A log-log interpolation is used because both axes span multiple
    decades. Values outside the tabulated E/N range are clamped to the
    nearest table endpoint.
    """
    source_path = cfg.electron_swarm_data_path
    en_td_table, muN_table = load_electron_mobility_muN_data(source_path)

    return _interpolate_swarm_quantity_from_table(
        en_td_table,
        muN_table,
        E_column,
        neutral_density,
        density_operation="divide_by_density",
    )


def interpolate_electron_diffusion_from_DN_table(
    cfg: SimulationConfig,
    E_column: np.ndarray,
    neutral_density: float,
) -> np.ndarray:
    """
    Interpolate D_e(x) from a swarm-data D_e(E/N) table.

    The table stores D_e * N versus E/N in Townsend. The runtime conversion is:

        (E/N)[Td] = |E| / N_g * 1e21
        D_e       = (D_e * N_g) / N_g

    Log-log interpolation is used because both axes span multiple decades.
    Values outside the tabulated E/N range are clamped to the nearest table
    endpoint.
    """
    source_path = cfg.electron_swarm_data_path
    en_td_table, DN_table = load_electron_diffusion_DN_data(source_path)

    return _interpolate_swarm_quantity_from_table(
        en_td_table,
        DN_table,
        E_column,
        neutral_density,
        density_operation="divide_by_density",
    )


def interpolate_townsend_alpha_from_alpha_over_N_table(
    cfg: SimulationConfig,
    E_column: np.ndarray,
    neutral_density: float,
) -> np.ndarray:
    """
    Interpolate Townsend alpha(x) from a swarm-data alpha/N table.

    The table stores alpha/N versus E/N in Townsend. The runtime conversion is:

        (E/N)[Td] = |E| / N_g * 1e21
        alpha     = (alpha/N) * N_g
    """
    source_path = (
        cfg.townsend_alpha_swarm_data_path
        if cfg.townsend_alpha_swarm_data_path is not None
        else cfg.electron_swarm_data_path
    )
    en_td_table, alpha_over_N_table = load_townsend_alpha_over_N_data(source_path)

    return _interpolate_swarm_quantity_from_table(
        en_td_table,
        alpha_over_N_table,
        E_column,
        neutral_density,
        density_operation="multiply_by_density",
        allow_zero_values=True,
    )


# ============================================================
# User-defined transport model hooks
# ============================================================

def compute_user_defined_electron_mobility_scalar(cfg: SimulationConfig) -> np.float32:
    """
    Return the default scalar electron mobility used by the user-defined
    electron-mobility profile.
    """
    p_Torr = cfg.p_Torr
    gas = cfg.gas.lower()
    if gas == "argon":
        mu_e_val = 29.3 / p_Torr
    elif gas == "nitrogen":
        mu_e_val = 30.4 / p_Torr
    else:
        raise NotImplementedError(f"Electron mobility not implemented for gas '{cfg.gas}'")
    return np.float32(mu_e_val)


def compute_user_defined_ion_mobility_scalar(cfg: SimulationConfig) -> np.float32:
    """
    Return the default scalar ion mobility used by the user-defined ion-mobility
    profile.
    """
    p_Torr = cfg.p_Torr
    gas = cfg.gas.lower()
    if gas == "argon":
        mu_i_val = 1.5e-1 / p_Torr
    elif gas == "nitrogen":
        mu_i_val = 2.09e-1 / p_Torr
    else:
        raise NotImplementedError(f"Ion mobility not implemented for gas '{cfg.gas}'")
    return np.float32(mu_i_val)


def compute_user_defined_electron_diffusion_scalar(cfg: SimulationConfig) -> np.float32:
    """
    Return the default scalar electron diffusion coefficient used by the
    user-defined electron-diffusion profile.
    """
    p_Torr = cfg.p_Torr
    gas = cfg.gas.lower()
    if gas == "argon":
        D_e_val = 29.3 / p_Torr
    elif gas == "nitrogen":
        mu_e_val = float(compute_user_defined_electron_mobility_scalar(cfg))
        D_e_val = mu_e_val * kB * cfg.T_e / e
    else:
        raise NotImplementedError(f"Electron diffusion not implemented for gas '{cfg.gas}'")
    return np.float32(D_e_val)


def compute_user_defined_ion_diffusion_scalar(cfg: SimulationConfig) -> np.float32:
    """
    Return the default scalar ion diffusion coefficient used by the
    user-defined ion-diffusion profile.
    """
    p_Torr = cfg.p_Torr
    gas = cfg.gas.lower()
    if gas == "argon":
        D_i_val = 0.006 / p_Torr
    elif gas == "nitrogen":
        mu_i_val = float(compute_user_defined_ion_mobility_scalar(cfg))
        D_i_val = mu_i_val * kB * cfg.T_i / e
    else:
        raise NotImplementedError(f"Ion diffusion not implemented for gas '{cfg.gas}'")
    return np.float32(D_i_val)


def compute_user_defined_electron_mobility(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
) -> np.ndarray:
    """
    Return the user-defined electron-mobility profile mu_e(x).

    This function is the intended edit point for custom empirical transport
    closures in PASCHEN-1D. Users may replace the current constant profile with
    any x-dependent or E-dependent expression they want, as long as the return
    value has shape ``(Nx,)`` and units of [m^2/(V s)].
    """
    del E_column
    return np.full_like(
        x_array,
        compute_user_defined_electron_mobility_scalar(cfg),
        dtype=np.float32,
    )


def compute_user_defined_ion_mobility(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
) -> np.ndarray:
    """
    Return the user-defined ion-mobility profile mu_i(x).

    This function is the intended edit point for custom ion-transport
    closures. The current default keeps the legacy empirical constant mobility.
    """
    del E_column
    return np.full_like(
        x_array,
        compute_user_defined_ion_mobility_scalar(cfg),
        dtype=np.float32,
    )


def compute_user_defined_electron_diffusion(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
) -> np.ndarray:
    """
    Return the user-defined electron-diffusion profile D_e(x).

    This is the intended edit point for custom empirical electron-diffusion
    closures. The current default keeps the legacy empirical constant
    diffusion coefficient.
    """
    del E_column
    return np.full_like(
        x_array,
        compute_user_defined_electron_diffusion_scalar(cfg),
        dtype=np.float32,
    )


def compute_user_defined_ion_diffusion(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
) -> np.ndarray:
    """
    Return the user-defined ion-diffusion profile D_i(x).

    This is the intended edit point for custom empirical ion-diffusion
    closures. The current default keeps the legacy empirical constant
    diffusion coefficient.
    """
    del E_column
    return np.full_like(
        x_array,
        compute_user_defined_ion_diffusion_scalar(cfg),
        dtype=np.float32,
    )


# ============================================================
# User-defined ionization/reaction-coefficient hooks
# ============================================================

def compute_user_defined_townsend_alpha(
    E_column: np.ndarray,
    p_Torr: float,
    pr: float,
    gas: str = "argon",
) -> np.ndarray:
    """
    Return the user-defined Townsend ionization coefficient alpha [1/m].

    Parameters
    ----------
    E_column : np.ndarray
        Local electric field array [V/m], shape (Nx,).
    p_Torr : float
        Gas pressure [Torr].
    pr : float
        Reduced pressure (dimensionless), typically p_Torr * T_ref / T_i.
        Currently unused by the default empirical closure. It remains in the
        signature so users can adopt pr-based alpha laws without changing the
        builder call path.
    gas : str, optional
        Gas species ("argon" or "nitrogen" at present). Controls A, B
        parameters in the empirical alpha(E/p) fit.

    Returns
    -------
    alpha_column : np.ndarray
        Townsend ionization coefficient alpha(x) [1/m], shape (Nx,).

    Default model
    -------------
    Uses the current empirical exponential fit:

        alpha/p = A * exp(-B * p / E)

    with pressure p in Torr and E in V/m. Rearranged:

        alpha = p * A * exp(-B * p / E)

    where A and B are gas-dependent empirical constants. In typical LTP
    tabulations, A has units [1/(m*Torr)] and B has units [V/(m*Torr)].

    A small floor on |E| is introduced to avoid numerical issues for
    very weak fields (E → 0).
    """
    del pr
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

def compute_user_defined_recombination_coefficient(cfg: SimulationConfig) -> np.float32:
    """
    Return the user-defined volumetric recombination coefficient beta.

    This is the intended edit point for the default volume recombination /
    loss coefficient used in the continuity source term. The current model
    keeps a single gas-independent constant value.
    """
    del cfg
    return np.float32(2.0e-13)


# ============================================================
# Reference gas-state builder
# ============================================================

def build_transport_reference_state(cfg: SimulationConfig) -> TransportCoeffs:
    """
    Compute baseline scalar coefficient references and shared gas parameters.

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
        - mu_e, mu_i : baseline scalar mobilities [m²/(V·s)]
        - D_e, D_i   : baseline scalar diffusion coefficients [m²/s]
        - beta       : baseline volume recombination coefficient [m³/s]
        - neutral_density : uniform background neutral density [m⁻³]
        - pr         : reduced pressure p * (T_ref / T_i) (dimensionless)
        - T_e_eV     : electron temperature in eV (for diagnostics)
        - T_i_eV     : ion temperature in eV (for diagnostics)

    Notes
    -----
    * The active runtime transport profiles are built by the
      ``compute_user_defined_*`` / ``build_*_profile`` functions in this
      module. The scalar mu/D values returned here are baseline references for
      diagnostics, legacy compatibility, and default constant-profile user
      equations.

    * The scalar beta value returned here is produced by the dedicated
      user-defined reaction-coefficient hook
      ``compute_user_defined_recombination_coefficient``.

    * The neutral background is treated as a fixed uniform reservoir, evaluated
      from the ideal-gas closure T_gas = T_i.
    """
    p_Torr = cfg.p_Torr
    T_i    = cfg.T_i
    neutral_density = compute_background_neutral_density(cfg)

    # Reduced pressure (Surzhikov-style scaling), typical form:
    #   pr = p * (T_ref / T_i)
    # with T_ref ≈ 300 K
    pr      = p_Torr * 300.0 / T_i
    T_e_eV  = 1.0
    T_i_eV  = 0.0258

    gas = cfg.gas.lower()

    if gas in ("argon", "nitrogen"):
        mu_e_val = compute_user_defined_electron_mobility_scalar(cfg)
        mu_i_val = compute_user_defined_ion_mobility_scalar(cfg)
        D_e_val = compute_user_defined_electron_diffusion_scalar(cfg)
        D_i_val = compute_user_defined_ion_diffusion_scalar(cfg)
        beta_val = compute_user_defined_recombination_coefficient(cfg)
    else:
        raise NotImplementedError(f"Gas '{cfg.gas}' not implemented yet.")

    return TransportCoeffs(
        mu_e=np.float32(mu_e_val),
        mu_i=np.float32(mu_i_val),
        D_e=np.float32(D_e_val),
        D_i=np.float32(D_i_val),
        beta=np.float32(beta_val),
        neutral_density=neutral_density,
        pr=pr,
        T_e_eV=T_e_eV,
        T_i_eV=T_i_eV,
    )

def build_electron_mobility_profile(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
    neutral_density: float,
) -> np.ndarray:
    """
    Build the electron-mobility profile mu_e(x) for the current step.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation configuration. The active selector is
        ``cfg.electron_transport_source``.
    x_array : np.ndarray
        Spatial grid [m], shape (Nx,).
    E_column : np.ndarray
        Local electric field [V/m], shape (Nx,).
    neutral_density : float
        Background neutral gas density [m^-3].

    Returns
    -------
    np.ndarray
        Electron mobility profile, shape (Nx,), dtype float32.

    Notes
    -----
    The ``"user_defined_equations"`` source uses the transport formulas
    implemented in this module (the current default returns the legacy
    empirical constant profile). The
    ``"swarm_data_table_interpolation"`` source attempts to use a swarm-data
    table for mu_e(E/N). If the source is incompatible with the selected gas,
    unavailable, or cannot be loaded, the code falls back to
    ``"user_defined_equations"`` and prints a one-time warning.
    """
    source = cfg.electron_transport_source

    if source == "user_defined_equations":
        return compute_user_defined_electron_mobility(
            cfg=cfg,
            x_array=x_array,
            E_column=E_column,
        )

    if source == "swarm_data_table_interpolation":
        table_gas = str(
            cfg.electron_swarm_data_gas
        ).strip().lower()
        if cfg.gas.strip().lower() != table_gas:
            _warn_fallback_once(
                _ELECTRON_TRANSPORT_FALLBACK_WARNED,
                "Electron-transport",
                "electron_transport_source",
                "the configured electron swarm-data gas does not match cfg.gas",
            )
            return compute_user_defined_electron_mobility(
                cfg=cfg,
                x_array=x_array,
                E_column=E_column,
            )

        try:
            return interpolate_electron_mobility_from_muN_table(
                cfg=cfg,
                E_column=E_column,
                neutral_density=neutral_density,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            _warn_fallback_once(
                _ELECTRON_TRANSPORT_FALLBACK_WARNED,
                "Electron-transport",
                "electron_transport_source",
                str(exc),
            )
            return compute_user_defined_electron_mobility(
                cfg=cfg,
                x_array=x_array,
                E_column=E_column,
            )

    raise ValueError(f"Unknown electron_transport_source: {source}")


def build_ion_mobility_profile(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
) -> np.ndarray:
    """
    Build the ion-mobility profile mu_i(x) for the current step.

    The ``"user_defined_equations"`` source uses the transport formulas
    implemented in this module. The
    ``"swarm_data_table_interpolation"`` source is reserved for a future
    ion swarm-data backend; until then it falls back to the user-defined
    equations and prints a one-time warning.
    """
    source = cfg.ion_transport_source

    if source == "user_defined_equations":
        return compute_user_defined_ion_mobility(
            cfg=cfg,
            x_array=x_array,
            E_column=E_column,
        )

    if source == "swarm_data_table_interpolation":
        _warn_fallback_once(
            _ION_TRANSPORT_FALLBACK_WARNED,
            "Ion-transport",
            "ion_transport_source",
            "no ion swarm-data interpolation backend is implemented yet",
        )
        return compute_user_defined_ion_mobility(
            cfg=cfg,
            x_array=x_array,
            E_column=E_column,
        )

    raise ValueError(f"Unknown ion_transport_source: {source}")


def build_electron_diffusion_profile(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
    neutral_density: float,
) -> np.ndarray:
    """
    Build the electron-diffusion profile D_e(x) for the current step.

    The ``"user_defined_equations"`` source uses the diffusion formulas
    implemented in this module. The
    ``"swarm_data_table_interpolation"`` source uses the same electron
    swarm-data file as the mobility interpolation path to recover D_e(E/N).
    If the source is incompatible with the selected gas, unavailable, or
    cannot be loaded, the code falls back to ``"user_defined_equations"``
    and prints a one-time warning.
    """
    source = cfg.electron_transport_source

    if source == "user_defined_equations":
        return compute_user_defined_electron_diffusion(
            cfg=cfg,
            x_array=x_array,
            E_column=E_column,
        )

    if source == "swarm_data_table_interpolation":
        table_gas = str(cfg.electron_swarm_data_gas).strip().lower()
        if cfg.gas.strip().lower() != table_gas:
            _warn_fallback_once(
                _ELECTRON_DIFFUSION_FALLBACK_WARNED,
                "Electron-diffusion",
                "electron_transport_source",
                "the configured electron swarm-data gas does not match cfg.gas",
                detail="is still using",
            )
            return compute_user_defined_electron_diffusion(
                cfg=cfg,
                x_array=x_array,
                E_column=E_column,
            )

        try:
            return interpolate_electron_diffusion_from_DN_table(
                cfg=cfg,
                E_column=E_column,
                neutral_density=neutral_density,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            _warn_fallback_once(
                _ELECTRON_DIFFUSION_FALLBACK_WARNED,
                "Electron-diffusion",
                "electron_transport_source",
                str(exc),
                detail="is still using",
            )
            return compute_user_defined_electron_diffusion(
                cfg=cfg,
                x_array=x_array,
                E_column=E_column,
            )

    raise ValueError(f"Unknown electron_transport_source: {source}")


def build_ion_diffusion_profile(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
) -> np.ndarray:
    """
    Build the ion-diffusion profile D_i(x) for the current step.

    The ``"user_defined_equations"`` source uses the diffusion formulas
    implemented in this module. The
    ``"swarm_data_table_interpolation"`` source is reserved for a future ion
    swarm-data diffusion backend; until then it falls back to the
    user-defined equations and prints a one-time warning.
    """
    source = cfg.ion_transport_source

    if source == "user_defined_equations":
        return compute_user_defined_ion_diffusion(
            cfg=cfg,
            x_array=x_array,
            E_column=E_column,
        )

    if source == "swarm_data_table_interpolation":
        _warn_fallback_once(
            _ION_DIFFUSION_FALLBACK_WARNED,
            "Ion-diffusion",
            "ion_transport_source",
            "no ion swarm-data diffusion backend is implemented yet",
            detail="is still using",
        )
        return compute_user_defined_ion_diffusion(
            cfg=cfg,
            x_array=x_array,
            E_column=E_column,
        )

    raise ValueError(f"Unknown ion_transport_source: {source}")
    

# ============================================================
# Townsend ionization coefficient alpha(E)
# ============================================================

def build_townsend_alpha_profile(
    cfg: SimulationConfig,
    E_column: np.ndarray,
    p_Torr: float,
    pr: float,
    gas: str,
    neutral_density: float,
) -> np.ndarray:
    """
    Build the Townsend ionization-coefficient profile alpha(x) [1/m].

    The ``"user_defined_equations"`` source uses the alpha(E, p) equations
    implemented in this module. The
    ``"swarm_data_table_interpolation"`` source interpolates alpha/N from the
    electron swarm-data file and converts it to alpha(x). If the source is
    incompatible with the selected gas, unavailable, or cannot be loaded, the
    code falls back to ``"user_defined_equations"`` and prints a one-time
    warning.
    """
    source = cfg.townsend_alpha_source

    if source == "user_defined_equations":
        return compute_user_defined_townsend_alpha(E_column, p_Torr, pr, gas).astype(
            np.float32, copy=False
        )

    if source == "swarm_data_table_interpolation":
        table_gas = str(
            cfg.townsend_alpha_swarm_data_gas
            if cfg.townsend_alpha_swarm_data_gas is not None
            else cfg.electron_swarm_data_gas
        ).strip().lower()
        if gas.strip().lower() != table_gas:
            _warn_fallback_once(
                _TOWNSEND_ALPHA_FALLBACK_WARNED,
                "Townsend-alpha",
                "townsend_alpha_source",
                "the configured Townsend-alpha swarm-data gas does not match cfg.gas",
            )
            return compute_user_defined_townsend_alpha(E_column, p_Torr, pr, gas).astype(
                np.float32, copy=False
            )

        try:
            return interpolate_townsend_alpha_from_alpha_over_N_table(
                cfg=cfg,
                E_column=E_column,
                neutral_density=neutral_density,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            _warn_fallback_once(
                _TOWNSEND_ALPHA_FALLBACK_WARNED,
                "Townsend-alpha",
                "townsend_alpha_source",
                str(exc),
            )
            return compute_user_defined_townsend_alpha(E_column, p_Torr, pr, gas).astype(
                np.float32, copy=False
            )

    raise ValueError(f"Unknown townsend_alpha_source: {source}")


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
    anode_electron_induced_yield: float,
    ni_boundary: float,
    mu_i: float,
    mu_e: float,
    ne_inner: float,
    T_e_eV: float,
    phi_boundary: float,
    phi_inner: float,
    dx: float,
    Gamma_ext: float = 0.0,
    use_vaughan_sey: bool = False,
    vaughan_Emax0_eV: float = 400.0,
    vaughan_dmax0: float = 3.2,
    vaughan_ks: float = 1.0,
    vaughan_z: float = 0.0,
    vaughan_E0: float = 0.0,
) -> float:
    """
    Electron-emission boundary closure in flux form, converted to density.

    Flux target follows side-specific closure rules:

    Cathode:
        Gamma_e = -[Gamma_ext + gamma * Gamma_i,incident]

    Anode:
        Gamma_e = +Gamma_ext - (1 - delta_ae) * Gamma_e,incident

    where:
      - gamma is the cathode ion-induced secondary-emission yield
        (constant in this model),
      - delta_ae is the anode electron-induced secondary-emission yield
        (constant or Vaughan-model value).

    The anode incident electron flux is estimated from the first interior
    electron density and local boundary field.

    For anode energy-dependent yield models, this routine computes a proxy
    electron impact energy [eV]:

        eps_proxy = (m_e / (2 e)) u_in^2 + C_th * T_e_eV,

    with u_in = Gamma_e,incident / n_e,inner and fixed C_th = 2.0.

    The target electron flux is converted to boundary density through the
    local boundary drift closure:

        Gamma_e = -mu_e * n_e * E.

    Parameters
    ----------
    boundary_side : str
        "anode" or "cathode".
    gamma : float
        Cathode ion-induced secondary electron emission coefficient
        (used only for boundary_side="cathode").
    anode_electron_induced_yield : float
        Anode electron-induced secondary electron emission yield (delta_ae).
    ni_boundary : float
        Ion density at the boundary cell.
    mu_i : float
        Ion mobility.
    mu_e : float
        Local electron mobility at the boundary where the closure is applied.
    ne_inner : float
        Electron density at the first interior cell adjacent to the boundary.
    T_e_eV : float
        Electron temperature proxy in eV.
    use_vaughan_sey : bool, optional
        If True, compute anode electron-induced yield from the Vaughan model
        using proxy impact energy; otherwise use constant
        anode_electron_induced_yield.
        This flag is used only for boundary_side="anode".
    vaughan_Emax0_eV, vaughan_dmax0, vaughan_ks, vaughan_z, vaughan_E0 : float
        Vaughan-model parameters. E0 is the threshold-offset energy in eV.
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
        # Cathode SEE uses constant gamma and incident-ion flux.
        Gamma_i_incident = max(Gamma_i_drift, 0.0)
        Gamma_e_target = -(Gamma_ext + gamma * Gamma_i_incident)
    else:
        # Anode electron-induced emission model:
        # Gamma_e = +Gamma_ext - (1 - delta_ae) * Gamma_e_incident
        Gamma_e_incident = max(mu_e * ne_inner * E_boundary, 0.0)
        n_inner_safe = max(ne_inner, 1e-30)
        u_in = Gamma_e_incident / n_inner_safe
        impact_energy_proxy_eV = (m_e / (2.0 * e)) * u_in * u_in + 2.0 * max(T_e_eV, 0.0)
        if use_vaughan_sey:
            E0 = max(vaughan_E0, 0.0)
            Emax = max(vaughan_Emax0_eV, 1e-12) * (
                1.0 + (max(vaughan_ks, 0.0) * vaughan_z * vaughan_z / (2.0 * np.pi))
            )
            dmax = max(vaughan_dmax0, 0.0) * (
                1.0 + (max(vaughan_ks, 0.0) * vaughan_z * vaughan_z / (2.0 * np.pi))
            )
            den_w = max(Emax - E0, 1e-12)
            w = max((impact_energy_proxy_eV - E0) / den_w, 0.0)
            if w <= 1.0:
                delta_ae = dmax * (w * np.exp(1.0 - w)) ** 0.56
            elif w < 3.6:
                delta_ae = dmax * (w * np.exp(1.0 - w)) ** 0.25
            else:
                delta_ae = dmax * 1.125 / (w ** 0.35)
        else:
            delta_ae = max(anode_electron_induced_yield, 0.0)
        Gamma_e_target = Gamma_ext - (1.0 - delta_ae) * Gamma_e_incident

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

    The mobility `mu_e` is the local anode-boundary electron mobility.
    """
    Ce = (mu_e * dt) / dx
    dphi_01 = (phi_inner - phi_left) / dx
    dphi_12 = (phi_inner2 - phi_inner) / dx

    den = 1.0 - Ce * dphi_01
    if den < 1e-12:
        den = 1e-12

    rhs_e = ne_curr_left - Ce * ne_next_inner * dphi_12
    return max(rhs_e / den, 0.0)
