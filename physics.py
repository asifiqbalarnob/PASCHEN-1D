"""
physics.py

Physics-facing helper routines for PASCHEN-1D.

This module contains six kinds of logic:
1) Applied-voltage waveform builders.
2) User-edit hooks for transport and ionization coefficients.
3) Swarm-data parsing and interpolation helpers for table-driven models.
4) Run-scoped swarm-table cache construction for high-performance interpolation.
5) Reference gas-state construction (neutral density, baseline scalars, beta).
6) Initial-state construction for phi, E, n_e, and n_i.

Conventions:
- SI units are used unless noted otherwise.
- Pressure inputs for the default empirical ionization/transport closures
  remain in Torr.
- The active runtime coefficient profiles are built by the
  ``compute_user_defined_*`` and ``build_*_profile`` functions below.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from physical_constants import kB, e, m_e
from config import (
    SimulationConfig,
    TransportCoeffs,
    TransportSourceMode,
)


_MU_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_D_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_ALPHA_OVER_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_NU_OVER_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_ELECTRON_TRANSPORT_FALLBACK_WARNED: set[str] = set()
_ELECTRON_DIFFUSION_FALLBACK_WARNED: set[str] = set()
_TOWNSEND_ALPHA_FALLBACK_WARNED: set[str] = set()
_SWARM_DATA_SECTION_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
_SWARM_DATA_ENERGY_SECTION_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
_SWARM_DATA_AXIS_CACHE: dict[str, tuple[bool, bool]] = {}


@dataclass
class _EoverNInterpolator:
    """
    Precomputed log-log interpolator for E/N-axis swarm quantities.

    This object holds interpolation-ready arrays so runtime calls avoid repeated
    path resolution, file parsing, and log-transform setup.
    """
    en_td_min: float
    en_td_max: float
    log_en_td: np.ndarray
    log_values: np.ndarray
    density_operation: str

    def evaluate(self, E_column: np.ndarray, neutral_density: float) -> np.ndarray:
        """
        Evaluate the interpolated runtime quantity on the local electric field.
        """
        N_g = max(float(neutral_density), 1.0)
        en_td_local = np.abs(E_column).astype(np.float64, copy=False) * 1.0e21 / N_g
        en_td_local = np.clip(en_td_local, self.en_td_min, self.en_td_max)
        log_en_td_local = np.log10(en_td_local)
        log_values_local = np.interp(log_en_td_local, self.log_en_td, self.log_values)
        values_local = np.power(10.0, log_values_local)

        if self.density_operation == "divide_by_density":
            runtime_values = values_local / N_g
        elif self.density_operation == "multiply_by_density":
            runtime_values = values_local * N_g
        else:
            raise ValueError(f"Unknown density_operation: {self.density_operation}")

        return runtime_values.astype(np.float32, copy=False)


@dataclass
class _EnergyInterpolator:
    """
    Preloaded linear interpolator for energy-axis swarm quantities.
    """
    eps_grid_eV: np.ndarray
    values_grid: np.ndarray
    quantity_name: str

    def evaluate(self, eps_query_eV: np.ndarray, policy: str) -> np.ndarray:
        """
        Evaluate with configured out-of-range policy ("clip" or "error").
        """
        eps_min = float(self.eps_grid_eV[0])
        eps_max = float(self.eps_grid_eV[-1])
        eps_safe = _apply_energy_range_policy(
            eps_query_eV.astype(np.float64, copy=False),
            eps_min,
            eps_max,
            policy,
            quantity_name=self.quantity_name,
        )
        return np.interp(eps_safe, self.eps_grid_eV, self.values_grid)


@dataclass
class SwarmRuntimeInterpolationCache:
    """
    Run-scoped cache of swarm-data interpolation objects.

    The cache preloads all needed swarm sections once at simulation startup and
    exposes lightweight evaluators used inside the time-stepping loop.
    """
    electron_transport_source: TransportSourceMode = "user_defined_equation"
    townsend_alpha_source_mode: str = "user_defined_equation"
    ionization_frequency_source_mode: str = "interpolate_from_e_over_n_table"
    impact_ionization_model: str = "from_townsend_alpha"
    electron_mu_eovern_interp: Optional[_EoverNInterpolator] = None
    electron_D_eovern_interp: Optional[_EoverNInterpolator] = None
    alpha_eovern_interp: Optional[_EoverNInterpolator] = None
    nu_over_N_eovern_interp: Optional[_EoverNInterpolator] = None
    alpha_energy_interp: Optional[_EnergyInterpolator] = None
    electron_mu_energy_interp: Optional[_EnergyInterpolator] = None
    electron_D_energy_interp: Optional[_EnergyInterpolator] = None
    nu_over_N_energy_interp: Optional[_EnergyInterpolator] = None
    nu_m_over_N_energy_interp: Optional[_EnergyInterpolator] = None
    loss_over_N_energy_interp: Optional[_EnergyInterpolator] = None

    def electron_mobility_from_field(
        self,
        cfg: SimulationConfig,
        x_array: np.ndarray,
        E_column: np.ndarray,
        neutral_density: float,
    ) -> np.ndarray:
        """
        Evaluate mu_e(x) from either cached swarm interpolation or user hook.
        """
        if (
            self.electron_transport_source == "swarm_data_table_interpolation"
            and self.electron_mu_eovern_interp is not None
        ):
            return self.electron_mu_eovern_interp.evaluate(E_column, neutral_density)
        return compute_user_defined_electron_mobility(cfg, x_array, E_column)

    def electron_diffusion_from_field(
        self,
        cfg: SimulationConfig,
        x_array: np.ndarray,
        E_column: np.ndarray,
        neutral_density: float,
    ) -> np.ndarray:
        """
        Evaluate D_e(x) from either cached swarm interpolation or user hook.
        """
        if (
            self.electron_transport_source == "swarm_data_table_interpolation"
            and self.electron_D_eovern_interp is not None
        ):
            return self.electron_D_eovern_interp.evaluate(E_column, neutral_density)
        return compute_user_defined_electron_diffusion(cfg, x_array, E_column)

    def townsend_alpha_from_field(
        self,
        E_column: np.ndarray,
        p_Torr: float,
        pr: float,
        gas: str,
        neutral_density: float,
    ) -> np.ndarray:
        """
        Evaluate alpha(x) from cached swarm interpolation or user hook.
        """
        if (
            self.townsend_alpha_source_mode == "interpolate_from_e_over_n_table"
            and self.alpha_eovern_interp is not None
        ):
            return self.alpha_eovern_interp.evaluate(E_column, neutral_density)
        return compute_user_defined_townsend_alpha(E_column, p_Torr, pr, gas).astype(
            np.float32, copy=False
        )

    def townsend_alpha_from_energy(
        self,
        mean_energy_eV: np.ndarray,
        neutral_density: float,
        out_of_range_policy: str,
    ) -> np.ndarray:
        """
        Evaluate alpha(x) [1/m] from cached energy-axis alpha/N interpolation.
        """
        if self.alpha_energy_interp is None:
            raise RuntimeError("Energy-axis Townsend-alpha interpolator is not initialized.")
        alpha_over_N_local = self.alpha_energy_interp.evaluate(
            mean_energy_eV, out_of_range_policy
        )
        N_g = max(float(neutral_density), 1.0)
        return (alpha_over_N_local * N_g).astype(np.float32, copy=False)

    def ionization_frequency_from_field(
        self,
        E_column: np.ndarray,
        neutral_density: float,
    ) -> np.ndarray:
        """
        Evaluate nu_i(x) [s^-1] from cached E/N-axis nu_i/N interpolation.
        """
        if self.nu_over_N_eovern_interp is None:
            raise RuntimeError("E/N-axis ionization-frequency interpolator is not initialized.")
        return self.nu_over_N_eovern_interp.evaluate(E_column, neutral_density)

    def electron_mobility_from_energy(
        self,
        mean_energy_eV: np.ndarray,
        neutral_density: float,
        out_of_range_policy: str,
    ) -> np.ndarray:
        """
        Evaluate mu_e(x) from cached energy-axis interpolation.
        """
        if self.electron_mu_energy_interp is None:
            raise RuntimeError("LEA electron mobility interpolator is not initialized.")
        muN_local = self.electron_mu_energy_interp.evaluate(mean_energy_eV, out_of_range_policy)
        N_g = max(float(neutral_density), 1.0)
        return (muN_local / N_g).astype(np.float32, copy=False)

    def electron_diffusion_from_energy(
        self,
        mean_energy_eV: np.ndarray,
        neutral_density: float,
        out_of_range_policy: str,
    ) -> np.ndarray:
        """
        Evaluate D_e(x) from cached energy-axis interpolation.
        """
        if self.electron_D_energy_interp is None:
            raise RuntimeError("LEA electron diffusion interpolator is not initialized.")
        DN_local = self.electron_D_energy_interp.evaluate(mean_energy_eV, out_of_range_policy)
        N_g = max(float(neutral_density), 1.0)
        return (DN_local / N_g).astype(np.float32, copy=False)

    def ionization_frequency_from_energy(
        self,
        mean_energy_eV: np.ndarray,
        neutral_density: float,
        out_of_range_policy: str,
    ) -> np.ndarray:
        """
        Evaluate nu_i(x) [s^-1] from cached energy-axis interpolation.
        """
        if self.nu_over_N_energy_interp is None:
            raise RuntimeError("LEA ionization-frequency interpolator is not initialized.")
        nu_over_N_local = self.nu_over_N_energy_interp.evaluate(
            mean_energy_eV, out_of_range_policy
        )
        N_g = max(float(neutral_density), 1.0)
        return (nu_over_N_local * N_g).astype(np.float32, copy=False)

    def momentum_frequency_from_energy(
        self,
        mean_energy_eV: np.ndarray,
        neutral_density: float,
        out_of_range_policy: str,
    ) -> np.ndarray:
        """
        Evaluate nu_m(x) [s^-1] from cached energy-axis interpolation.
        """
        if self.nu_m_over_N_energy_interp is None:
            raise RuntimeError("LEA momentum-frequency interpolator is not initialized.")
        nu_m_over_N_local = self.nu_m_over_N_energy_interp.evaluate(
            mean_energy_eV, out_of_range_policy
        )
        N_g = max(float(neutral_density), 1.0)
        return (nu_m_over_N_local * N_g).astype(np.float32, copy=False)

    def energy_loss_rate_over_N_from_energy(
        self,
        mean_energy_eV: np.ndarray,
        out_of_range_policy: str,
    ) -> np.ndarray:
        """
        Evaluate (P_loss/N)(x) [eV m^3/s] from cached energy-axis interpolation.
        """
        if self.loss_over_N_energy_interp is None:
            raise RuntimeError("LEA energy-loss interpolator is not initialized.")
        return self.loss_over_N_energy_interp.evaluate(
            mean_energy_eV, out_of_range_policy
        ).astype(np.float32, copy=False)


# ============================================================
# Applied-voltage waveforms
# ============================================================

def make_voltage_waveform(cfg: SimulationConfig) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build the applied-voltage function V_app(t) from cfg.waveform.waveform_type.

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
    Supported waveform types (cfg.waveform.waveform_type):

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
    if cfg.waveform.waveform_type == "gaussian":
        def V_app_func(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t)
            return cfg.waveform.V_peak * np.exp(-((t - cfg.waveform.t_peak) / cfg.waveform.tau) ** 2)

    elif cfg.waveform.waveform_type == "dc":
        def V_app_func(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t)
            return cfg.waveform.V_peak * np.ones_like(t)

    elif cfg.waveform.waveform_type == "step":
        # Small nonzero floor to avoid exactly zero applied voltage
        # (can help prevent degeneracies in some models).
        min_V = 1e-15

        def V_app_func(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t)
            return (
                cfg.waveform.V_peak * ((t >= cfg.waveform.tV_start) & (t <= cfg.waveform.tV_end)) +
                min_V      * ((t < cfg.waveform.tV_start) | (t > cfg.waveform.tV_end))
            )

    # --- pure RF (optionally with DC bias) ---
    elif cfg.waveform.waveform_type == "rf":
        omega_rf = 2.0 * np.pi * cfg.waveform.f_rf
        V0       = cfg.waveform.V_dc
        Vrf      = cfg.waveform.V_peak
        phi_rf   = cfg.waveform.phi_rf

        def V_app_func(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t)
            return V0 + Vrf * np.sin(omega_rf * t + phi_rf)

    else:
        raise ValueError(f"Unknown waveform_type: {cfg.waveform.waveform_type}")

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
    p_Pa = float(cfg.plasma_state.p_Torr) * 133.32236842105263
    T_gas = max(float(cfg.plasma_state.T_i), 1.0)
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
    fallback_target: str = "user_defined_equation",
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


def _normalize_electron_kinetics_mode(cfg: SimulationConfig) -> str:
    """
    Return the normalized electron-kinetics mode string.
    """
    mode = str(cfg.plasma.electron_kinetics_model).strip().lower()
    if mode not in {
        "user_defined_electron_kinetics",
        "local_field_approximation",
    }:
        raise ValueError(f"Unknown electron_kinetics_model: {cfg.plasma.electron_kinetics_model}")
    return mode


def _resolve_electron_transport_source(cfg: SimulationConfig) -> TransportSourceMode:
    """
    Resolve electron transport-source selection for the active kinetics mode.
    """
    mode = _normalize_electron_kinetics_mode(cfg)
    if mode == "user_defined_electron_kinetics":
        return "user_defined_equation"
    if mode == "local_field_approximation":
        return cfg.local_field_approximation.electron_transport_source
    raise ValueError(f"Unsupported electron kinetics mode: {mode}")


def _resolve_ion_transport_source(cfg: SimulationConfig) -> TransportSourceMode:
    """
    Resolve ion transport-source selection for the active kinetics mode.
    """
    ion_mode = str(cfg.plasma.ion_kinetics_model).strip().lower()
    if ion_mode == "user_defined_ion_kinetics":
        return "user_defined_equation"
    raise ValueError(f"Unsupported ion kinetics model: {cfg.plasma.ion_kinetics_model}")


def _resolve_electron_swarm_path(cfg: SimulationConfig) -> str:
    """
    Resolve the active electron swarm-data file path for the current mode.
    """
    return cfg.local_field_approximation.electron_swarm_data_path


def _resolve_townsend_alpha_source_mode(cfg: SimulationConfig) -> str:
    """
    Resolve Townsend-alpha source mode for the alpha-based ionization branch.
    """
    mode = str(cfg.townsend_coefficient.townsend_alpha_source_mode).strip().lower()
    if mode not in {
        "user_defined_equation",
        "interpolate_from_e_over_n_table",
    }:
        raise ValueError(
            "Unknown townsend_alpha_source_mode: "
            f"{cfg.townsend_coefficient.townsend_alpha_source_mode}"
        )
    return mode


def _resolve_townsend_alpha_path(cfg: SimulationConfig) -> str:
    """
    Resolve Townsend-alpha swarm-data path for interpolation modes.
    """
    if cfg.townsend_coefficient.townsend_alpha_swarm_data_path is not None:
        return cfg.townsend_coefficient.townsend_alpha_swarm_data_path

    return cfg.local_field_approximation.electron_swarm_data_path


def _resolve_impact_ionization_model(cfg: SimulationConfig) -> str:
    """
    Normalize top-level impact-ionization branch selector.
    """
    model = str(cfg.plasma.impact_ionization_model).strip().lower()
    if model not in {
        "from_townsend_alpha",
        "from_ionization_frequency",
    }:
        raise ValueError(f"Unknown impact_ionization_model: {cfg.plasma.impact_ionization_model}")
    return model


def _resolve_ionization_frequency_source_mode(cfg: SimulationConfig) -> str:
    """
    Resolve direct ionization-frequency source mode.
    """
    mode = str(cfg.ionization_frequency_source.ionization_frequency_source_mode).strip().lower()
    if mode not in {
        "user_defined_equation",
        "interpolate_from_e_over_n_table",
    }:
        raise ValueError(
            "Unknown ionization_frequency_source_mode: "
            f"{cfg.ionization_frequency_source.ionization_frequency_source_mode}"
        )
    return mode


def _resolve_ionization_frequency_path(cfg: SimulationConfig) -> str:
    """
    Resolve swarm-data path for direct nu_i interpolation mode.
    """
    if cfg.ionization_frequency_source.ionization_frequency_swarm_data_path is not None:
        return cfg.ionization_frequency_source.ionization_frequency_swarm_data_path

    return cfg.local_field_approximation.electron_swarm_data_path


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
    in_block = False
    pairs: list[tuple[float, float]] = []

    for line in lines:
        if not in_block:
            stripped_header = line.strip()
            if stripped_header.startswith("E/N (Td)") and (section_label in stripped_header):
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


def load_swarm_energy_section(
    path_str: str,
    section_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one named energy-axis section from a supported raw swarm-data output file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(energy_eV, values)`` in float64, sorted in ascending energy.
    """
    resolved = str(_resolve_project_path(path_str))
    cache_key = (resolved, section_label)
    cached = _SWARM_DATA_ENERGY_SECTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lines = Path(resolved).read_text(errors="replace").splitlines()
    in_block = False
    pairs: list[tuple[float, float]] = []

    for line in lines:
        if not in_block:
            stripped_header = line.strip()
            if stripped_header.startswith("Energy (eV)") and (section_label in stripped_header):
                in_block = True
            continue

        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Energy (eV)"):
            break

        parts = stripped.split()
        if len(parts) < 2:
            continue

        try:
            eps_val = float(parts[0])
            y_val = float(parts[1])
        except ValueError:
            continue
        pairs.append((eps_val, y_val))

    if len(pairs) < 2:
        raise ValueError(
            f"Could not find a usable swarm-data energy section '{section_label}' in "
            f"'{resolved}'."
        )

    raw = np.asarray(pairs, dtype=np.float64)
    energy_eV = raw[:, 0]
    values = raw[:, 1]
    order = np.argsort(energy_eV)
    energy_eV = energy_eV[order]
    values = values[order]
    _SWARM_DATA_ENERGY_SECTION_CACHE[cache_key] = (energy_eV, values)
    return energy_eV, values


def _detect_swarm_data_axes(path_str: str) -> tuple[bool, bool]:
    """
    Detect whether a swarm-data file contains E/N-axis and/or Energy-axis headers.

    Returns
    -------
    tuple[bool, bool]
        (has_eoverN_axis, has_energy_axis)
    """
    resolved = str(_resolve_project_path(path_str))
    cached = _SWARM_DATA_AXIS_CACHE.get(resolved)
    if cached is not None:
        return cached

    text = Path(resolved).read_text(errors="replace")
    has_eoverN = "E/N (Td)\t" in text
    has_energy = "Energy (eV)\t" in text
    out = (has_eoverN, has_energy)
    _SWARM_DATA_AXIS_CACHE[resolved] = out
    return out


def _ensure_swarm_axis_compatibility(
    path_str: str,
    *,
    expected_axis: str,
    source_label: str,
) -> None:
    """
    Validate that a swarm-data file contains the expected axis family.
    """
    has_eoverN, has_energy = _detect_swarm_data_axes(path_str)
    resolved = str(_resolve_project_path(path_str))

    if expected_axis == "eovern":
        if has_energy and (not has_eoverN):
            raise ValueError(
                f"{source_label} '{resolved}' appears to be energy-axis only "
                "(contains 'Energy (eV)' headers but no 'E/N (Td)' headers). "
                "Use an E/N-axis file for user_defined_electron_kinetics/local_field_approximation mode."
            )
        return

    if expected_axis == "energy":
        if has_eoverN and (not has_energy):
            raise ValueError(
                f"{source_label} '{resolved}' appears to be E/N-axis only "
                "(contains 'E/N (Td)' headers but no 'Energy (eV)' headers). "
                "Use an energy-axis file for mean-energy interpolation utilities."
            )
        return

    raise ValueError(f"Unknown expected_axis: {expected_axis}")


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

    _ensure_swarm_axis_compatibility(
        resolved,
        expected_axis="eovern",
        source_label=source_label,
    )

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


def _build_eovern_interpolator(
    en_td_table: np.ndarray,
    values_table: np.ndarray,
    *,
    density_operation: str,
    allow_zero_values: bool = False,
) -> _EoverNInterpolator:
    """
    Build a reusable log-log E/N interpolator object from tabulated arrays.
    """
    safe_values = (
        np.maximum(values_table, 1.0e-300) if allow_zero_values else values_table
    )
    return _EoverNInterpolator(
        en_td_min=float(en_td_table[0]),
        en_td_max=float(en_td_table[-1]),
        log_en_td=np.log10(en_td_table),
        log_values=np.log10(safe_values),
        density_operation=density_operation,
    )


def _build_total_loss_over_N_energy_interpolator(
    eps_el: np.ndarray,
    pel_over_N: np.ndarray,
    eps_inel: np.ndarray,
    pinel_over_N: np.ndarray,
) -> _EnergyInterpolator:
    """
    Build a reusable energy-loss interpolator for (P_el + P_inel)/N.

    The combined curve is built on the union of available mean-energy grids.
    Inelastic losses are treated as zero below their tabulated onset energy so
    low-energy LEA states are not artificially excluded.
    """
    eps_grid = np.unique(np.concatenate((eps_el, eps_inel))).astype(np.float64, copy=False)

    # Elastic term: use tabulated trend across the full covered range.
    pel_grid = np.interp(eps_grid, eps_el, pel_over_N)

    # Inelastic term: physically zero below first inelastic threshold/onset.
    pinel_grid = np.interp(
        eps_grid,
        eps_inel,
        pinel_over_N,
        left=0.0,
        right=float(pinel_over_N[-1]),
    )
    ptotal_grid = pel_grid + pinel_grid

    return _EnergyInterpolator(
        eps_grid_eV=eps_grid,
        values_grid=ptotal_grid,
        quantity_name="P_loss/N(energy)",
    )


def build_swarm_interpolation_cache(cfg: SimulationConfig) -> SwarmRuntimeInterpolationCache:
    """
    Preload and validate swarm-data tables once for the active run configuration.

    Returns
    -------
    SwarmRuntimeInterpolationCache
        Cache object with ready-to-use interpolation evaluators for:
        - E/N-axis mobility, diffusion, Townsend alpha (as configured),
        - optional direct E/N-axis ionization-frequency interpolation.

    Notes
    -----
    This routine centralizes all expensive swarm-data warmup work outside the
    runtime loop so transport/source evaluation inside `run_simulation` only
    performs light interpolation calls.
    """
    cache = SwarmRuntimeInterpolationCache()

    kinetics_mode = _normalize_electron_kinetics_mode(cfg)
    cache.electron_transport_source = _resolve_electron_transport_source(cfg)
    cache.townsend_alpha_source_mode = _resolve_townsend_alpha_source_mode(cfg)
    cache.ionization_frequency_source_mode = _resolve_ionization_frequency_source_mode(cfg)
    cache.impact_ionization_model = _resolve_impact_ionization_model(cfg)

    # Field-based electron transport (LFA mode).
    if cache.electron_transport_source == "swarm_data_table_interpolation":
        source_path = _resolve_electron_swarm_path(cfg)
        try:
            en_mu, muN = load_electron_mobility_muN_data(source_path)
            en_D, DN = load_electron_diffusion_DN_data(source_path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            _warn_fallback_once(
                _ELECTRON_TRANSPORT_FALLBACK_WARNED,
                "Electron-transport",
                "electron_transport_source",
                str(exc),
            )
            _warn_fallback_once(
                _ELECTRON_DIFFUSION_FALLBACK_WARNED,
                "Electron-diffusion",
                "electron_transport_source",
                str(exc),
                detail="is still using",
            )
            cache.electron_transport_source = "user_defined_equation"
        else:
            cache.electron_mu_eovern_interp = _build_eovern_interpolator(
                en_mu,
                muN,
                density_operation="divide_by_density",
            )
            cache.electron_D_eovern_interp = _build_eovern_interpolator(
                en_D,
                DN,
                density_operation="divide_by_density",
            )

    # Townsend alpha interpolation (for alpha-based ionization mode).
    if cache.townsend_alpha_source_mode != "user_defined_equation":
        alpha_path = _resolve_townsend_alpha_path(cfg)
        try:
            if cache.townsend_alpha_source_mode == "interpolate_from_e_over_n_table":
                en_alpha, alpha_over_N = load_townsend_alpha_over_N_data(alpha_path)
                cache.alpha_eovern_interp = _build_eovern_interpolator(
                    en_alpha,
                    alpha_over_N,
                    density_operation="multiply_by_density",
                    allow_zero_values=True,
                )
        except (FileNotFoundError, OSError, ValueError) as exc:
            _warn_fallback_once(
                _TOWNSEND_ALPHA_FALLBACK_WARNED,
                "Townsend-alpha",
                "townsend_alpha_source_mode",
                str(exc),
            )
            cache.townsend_alpha_source_mode = "user_defined_equation"

    # Direct ionization-frequency interpolation (nu_i from table).
    if (
        cache.impact_ionization_model == "from_ionization_frequency"
        and cache.ionization_frequency_source_mode != "user_defined_equation"
    ):
        ion_path = _resolve_ionization_frequency_path(cfg)
        if cache.ionization_frequency_source_mode == "interpolate_from_e_over_n_table":
            en_nu, nu_over_N = load_total_ionization_frequency_over_N_data(ion_path)
            cache.nu_over_N_eovern_interp = _build_eovern_interpolator(
                en_nu,
                nu_over_N,
                density_operation="multiply_by_density",
                allow_zero_values=True,
            )
    return cache


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


def load_townsend_alpha_over_N_vs_energy_data(
    path_str: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Townsend alpha/N data versus mean energy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(energy_eV, alpha_over_N)`` in float64.
    """
    _ensure_swarm_axis_compatibility(
        path_str,
        expected_axis="energy",
        source_label="Townsend-alpha source",
    )
    return load_swarm_energy_section(path_str, "Townsend ioniz. coef. alpha/N (m2)")


def load_total_ionization_frequency_over_N_data(
    path_str: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load total ionization-frequency data (nu_i/N) versus E/N.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(en_td, nu_over_N)`` in float64, sorted in ascending E/N.
    """
    return _load_swarm_quantity_data(
        path_str,
        cache=_NU_OVER_N_TABLE_CACHE,
        section_label="Total ionization freq. /N (m3/s)",
        source_label="Ionization-frequency source",
        value_label="nu_i/N",
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
    source_path = _resolve_electron_swarm_path(cfg)
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
    source_path = _resolve_electron_swarm_path(cfg)
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
        _resolve_townsend_alpha_path(cfg)
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


def interpolate_townsend_alpha_from_alpha_over_N_energy_table(
    cfg: SimulationConfig,
    mean_energy_eV: np.ndarray,
    neutral_density: float,
    out_of_range_policy: str = "clip",
) -> np.ndarray:
    """
    Interpolate Townsend alpha(x) from an energy-axis alpha/N table.
    """
    source_path = _resolve_townsend_alpha_path(cfg)
    eps_grid, alpha_over_N = load_townsend_alpha_over_N_vs_energy_data(source_path)
    alpha_over_N_local = _interp_energy_axis_quantity(
        mean_energy_eV.astype(np.float64, copy=False),
        eps_grid,
        alpha_over_N,
        out_of_range_policy,
        quantity_name="alpha/N(energy)",
    )
    N_g = max(float(neutral_density), 1.0)
    return (alpha_over_N_local * N_g).astype(np.float32, copy=False)


def load_total_ionization_frequency_over_N_vs_energy_data(
    path_str: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load total ionization-frequency data (nu_i/N) versus mean energy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(energy_eV, nu_over_N)`` in float64.
    """
    _ensure_swarm_axis_compatibility(
        path_str,
        expected_axis="energy",
        source_label="LEA ionization source",
    )
    return load_swarm_energy_section(path_str, "Total ionization freq. /N (m3/s)")


def load_momentum_frequency_over_N_vs_energy_data(
    path_str: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load electron momentum-transfer frequency data (nu_m/N) versus mean energy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(energy_eV, nu_m_over_N)`` in float64.
    """
    _ensure_swarm_axis_compatibility(
        path_str,
        expected_axis="energy",
        source_label="LEA momentum-frequency source",
    )
    return load_swarm_energy_section(path_str, "Momentum frequency /N (m3/s)")


def load_mobility_times_N_vs_energy_data(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load electron mobility*neutral-density data versus mean energy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(energy_eV, muN)`` in float64.
    """
    _ensure_swarm_axis_compatibility(
        path_str,
        expected_axis="energy",
        source_label="LEA electron transport source",
    )
    return load_swarm_energy_section(path_str, "Mobility *N (1/m/V/s)")


def load_diffusion_times_N_vs_energy_data(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load electron diffusion*neutral-density data versus mean energy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(energy_eV, DN)`` in float64.
    """
    _ensure_swarm_axis_compatibility(
        path_str,
        expected_axis="energy",
        source_label="LEA electron transport source",
    )
    return load_swarm_energy_section(path_str, "Diffusion coefficient *N (1/m/s)")


def load_elastic_power_loss_over_N_vs_energy_data(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load elastic electron-energy loss power over N versus mean energy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(energy_eV, P_elastic_over_N)`` in float64, where values are in
        [eV m^3 / s].
    """
    _ensure_swarm_axis_compatibility(
        path_str,
        expected_axis="energy",
        source_label="LEA energy-loss source",
    )
    return load_swarm_energy_section(path_str, "Elastic power loss /N (eV m3/s)")


def load_inelastic_power_loss_over_N_vs_energy_data(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load inelastic electron-energy loss power over N versus mean energy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(energy_eV, P_inelastic_over_N)`` in float64, where values are in
        [eV m^3 / s].
    """
    _ensure_swarm_axis_compatibility(
        path_str,
        expected_axis="energy",
        source_label="LEA energy-loss source",
    )
    return load_swarm_energy_section(path_str, "Inelastic power loss /N (eV m3/s)")


def _apply_energy_range_policy(
    eps_local_eV: np.ndarray,
    eps_min: float,
    eps_max: float,
    policy: str,
    *,
    quantity_name: str,
) -> np.ndarray:
    """
    Apply out-of-range policy for local mean-energy interpolation requests.
    """
    if policy == "clip":
        return np.clip(eps_local_eV, eps_min, eps_max)

    if policy == "error":
        below = np.any(eps_local_eV < eps_min)
        above = np.any(eps_local_eV > eps_max)
        if below or above:
            e_min_local = float(np.min(eps_local_eV))
            e_max_local = float(np.max(eps_local_eV))
            raise ValueError(
                f"Local mean energy for {quantity_name} is out of swarm-table range: "
                f"local=[{e_min_local:.6g}, {e_max_local:.6g}] eV, "
                f"table=[{eps_min:.6g}, {eps_max:.6g}] eV."
            )
        return eps_local_eV

    raise ValueError(
        f"Unknown local_energy_out_of_swarm_table_range_policy: {policy}"
    )


def _interp_energy_axis_quantity(
    eps_query_eV: np.ndarray,
    eps_grid_eV: np.ndarray,
    values_grid: np.ndarray,
    policy: str,
    *,
    quantity_name: str,
) -> np.ndarray:
    """
    Interpolate an energy-axis swarm-data quantity with configurable
    out-of-range behavior.
    """
    eps_min = float(eps_grid_eV[0])
    eps_max = float(eps_grid_eV[-1])
    eps_safe = _apply_energy_range_policy(
        eps_query_eV.astype(np.float64, copy=False),
        eps_min,
        eps_max,
        policy,
        quantity_name=quantity_name,
    )
    return np.interp(eps_safe, eps_grid_eV, values_grid)


def interpolate_electron_mobility_from_energy_swarm(
    cfg: SimulationConfig,
    mean_energy_eV: np.ndarray,
    neutral_density: float,
    out_of_range_policy: str,
) -> np.ndarray:
    """
    Interpolate electron mobility from an energy-axis swarm-data file.
    """
    source_path = _resolve_electron_swarm_path(cfg)
    eps_grid, muN_grid = load_mobility_times_N_vs_energy_data(source_path)
    muN_local = _interp_energy_axis_quantity(
        mean_energy_eV.astype(np.float64, copy=False),
        eps_grid,
        muN_grid,
        out_of_range_policy,
        quantity_name="mu_e(energy)",
    )
    N_g = max(float(neutral_density), 1.0)
    mu_local = muN_local / N_g
    return mu_local.astype(np.float32, copy=False)


def interpolate_electron_diffusion_from_energy_swarm(
    cfg: SimulationConfig,
    mean_energy_eV: np.ndarray,
    neutral_density: float,
    out_of_range_policy: str,
) -> np.ndarray:
    """
    Interpolate electron diffusion coefficient from an energy-axis swarm-data file.
    """
    source_path = _resolve_electron_swarm_path(cfg)
    eps_grid, DN_grid = load_diffusion_times_N_vs_energy_data(source_path)
    DN_local = _interp_energy_axis_quantity(
        mean_energy_eV.astype(np.float64, copy=False),
        eps_grid,
        DN_grid,
        out_of_range_policy,
        quantity_name="D_e(energy)",
    )
    N_g = max(float(neutral_density), 1.0)
    D_local = DN_local / N_g
    return D_local.astype(np.float32, copy=False)


def build_energy_loss_rate_over_N_profile_from_energy_swarm(
    cfg: SimulationConfig,
    mean_energy_eV: np.ndarray,
    out_of_range_policy: str,
) -> np.ndarray:
    """
    Build total electron energy-loss-rate coefficient profile from an
    energy-axis swarm-data file.

    Returns
    -------
    np.ndarray
        ``(P_loss / N)(x)`` in [eV m^3 / s], with
        ``P_loss = ne * N * (P_loss / N)``.
    """
    source_path = _resolve_electron_swarm_path(cfg)
    eps_el, pel_over_N = load_elastic_power_loss_over_N_vs_energy_data(source_path)
    eps_inel, pinel_over_N = load_inelastic_power_loss_over_N_vs_energy_data(source_path)

    total_loss_interp = _build_total_loss_over_N_energy_interpolator(
        eps_el,
        pel_over_N,
        eps_inel,
        pinel_over_N,
    )
    ptotal_local = total_loss_interp.evaluate(
        mean_energy_eV.astype(np.float64, copy=False),
        out_of_range_policy,
    )
    return ptotal_local.astype(np.float32, copy=False)


def build_ionization_frequency_profile_from_energy_swarm(
    cfg: SimulationConfig,
    mean_energy_eV: np.ndarray,
    neutral_density: float,
    out_of_range_policy: str = "clip",
) -> np.ndarray:
    """
    Build nu_i(x) [s^-1] from an energy-axis swarm-data file.

    The interpolation uses local mean electron energy [eV] directly:
      1) interpolate ``nu_i/N`` on the energy axis,
      2) multiply by neutral density ``N`` to get ``nu_i`` [s^-1].
    """
    source_path = _resolve_ionization_frequency_path(cfg)
    eps_grid, nu_over_N = load_total_ionization_frequency_over_N_vs_energy_data(source_path)
    nu_over_N_local = _interp_energy_axis_quantity(
        mean_energy_eV.astype(np.float64, copy=False),
        eps_grid,
        nu_over_N,
        out_of_range_policy,
        quantity_name="nu_i/N(energy)",
    )
    N_g = max(float(neutral_density), 1.0)
    nu_local = nu_over_N_local * N_g
    return nu_local.astype(np.float32, copy=False)


# ============================================================
# User-defined transport model hooks
# ============================================================

def compute_user_defined_electron_mobility_scalar(cfg: SimulationConfig) -> np.float32:
    """
    Return the default scalar electron mobility used by the user-defined
    electron-mobility profile.
    """
    p_Torr = cfg.plasma_state.p_Torr
    gas = cfg.plasma_state.gas.lower()
    if gas == "argon":
        mu_e_val = 29.3 / p_Torr
    elif gas == "nitrogen":
        mu_e_val = 30.4 / p_Torr
    else:
        raise NotImplementedError(f"Electron mobility not implemented for gas '{cfg.plasma_state.gas}'")
    return np.float32(mu_e_val)


def compute_user_defined_ion_mobility_scalar(cfg: SimulationConfig) -> np.float32:
    """
    Return the default scalar ion mobility used by the user-defined ion-mobility
    profile.
    """
    p_Torr = cfg.plasma_state.p_Torr
    gas = cfg.plasma_state.gas.lower()
    if gas == "argon":
        mu_i_val = 1.5e-1 / p_Torr
    elif gas == "nitrogen":
        mu_i_val = 2.09e-1 / p_Torr
    else:
        raise NotImplementedError(f"Ion mobility not implemented for gas '{cfg.plasma_state.gas}'")
    return np.float32(mu_i_val)


def compute_user_defined_electron_diffusion_scalar(cfg: SimulationConfig) -> np.float32:
    """
    Return the default scalar electron diffusion coefficient used by the
    user-defined electron-diffusion profile.
    """
    p_Torr = cfg.plasma_state.p_Torr
    gas = cfg.plasma_state.gas.lower()
    if gas == "argon":
        D_e_val = 29.3 / p_Torr
    elif gas == "nitrogen":
        mu_e_val = float(compute_user_defined_electron_mobility_scalar(cfg))
        D_e_val = mu_e_val * kB * cfg.plasma_state.T_e / e
    else:
        raise NotImplementedError(f"Electron diffusion not implemented for gas '{cfg.plasma_state.gas}'")
    return np.float32(D_e_val)


def compute_user_defined_ion_diffusion_scalar(cfg: SimulationConfig) -> np.float32:
    """
    Return the default scalar ion diffusion coefficient used by the
    user-defined ion-diffusion profile.
    """
    p_Torr = cfg.plasma_state.p_Torr
    gas = cfg.plasma_state.gas.lower()
    if gas == "argon":
        D_i_val = 0.006 / p_Torr
    elif gas == "nitrogen":
        mu_i_val = float(compute_user_defined_ion_mobility_scalar(cfg))
        D_i_val = mu_i_val * kB * cfg.plasma_state.T_i / e
    else:
        raise NotImplementedError(f"Ion diffusion not implemented for gas '{cfg.plasma_state.gas}'")
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


def compute_user_defined_ionization_frequency(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
) -> np.ndarray:
    """
    Return user-defined impact-ionization frequency nu_i(x) [s^-1].

    Default closure is intentionally conservative and backward compatible:
    1) compute user-defined Townsend alpha(E) [1/m],
    2) compute user-defined electron drift speed magnitude |u_e| = |mu_e E| [m/s],
    3) set nu_i = alpha * |u_e| [s^-1].

    Users can replace this function with any direct nu_i model without
    changing runtime branching in ``paschen_1d.py``.
    """
    p_Torr = float(cfg.plasma_state.p_Torr)
    # Retain legacy reduced-pressure argument path for alpha hook.
    pr = p_Torr * 300.0 / max(float(cfg.plasma_state.T_i), 1.0)
    gas = cfg.plasma_state.gas
    alpha_local = compute_user_defined_townsend_alpha(
        E_column=E_column,
        p_Torr=p_Torr,
        pr=pr,
        gas=gas,
    ).astype(np.float32, copy=False)
    mu_e_local = compute_user_defined_electron_mobility(
        cfg=cfg,
        x_array=x_array,
        E_column=E_column,
    ).astype(np.float32, copy=False)
    u_e_local = mu_e_local * E_column
    return (alpha_local * np.abs(u_e_local)).astype(np.float32, copy=False)

def compute_user_defined_recombination_coefficient(cfg: SimulationConfig) -> np.float32:
    """
    Return the user-defined volumetric recombination coefficient beta.

    This is the intended edit point for the default volume recombination /
    loss coefficient used in the continuity source term.
    """
    return np.float32(cfg.recombination.recombination_coefficient)


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
    p_Torr = cfg.plasma_state.p_Torr
    T_i    = cfg.plasma_state.T_i
    neutral_density = compute_background_neutral_density(cfg)

    # Reduced pressure (Surzhikov-style scaling), typical form:
    #   pr = p * (T_ref / T_i)
    # with T_ref ≈ 300 K
    pr      = p_Torr * 300.0 / T_i
    T_e_eV  = float(kB * float(cfg.plasma_state.T_e) / e)
    T_i_eV  = float(kB * float(cfg.plasma_state.T_i) / e)

    gas = cfg.plasma_state.gas.lower()

    if gas in ("argon", "nitrogen"):
        mu_e_val = compute_user_defined_electron_mobility_scalar(cfg)
        mu_i_val = compute_user_defined_ion_mobility_scalar(cfg)
        D_e_val = compute_user_defined_electron_diffusion_scalar(cfg)
        D_i_val = compute_user_defined_ion_diffusion_scalar(cfg)
        beta_val = compute_user_defined_recombination_coefficient(cfg)
    else:
        raise NotImplementedError(f"Gas '{cfg.plasma_state.gas}' not implemented yet.")

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
        Simulation configuration. The active selector is resolved from
        electron kinetics mode and the corresponding kinetics dataclass.
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
    The ``"user_defined_equation"`` source uses the transport formulas
    implemented in this module (the current default returns the legacy
    empirical constant profile). The
    ``"swarm_data_table_interpolation"`` source attempts to use a swarm-data
    table for mu_e(E/N). If the source is unavailable or cannot be loaded,
    the code falls back to
    ``"user_defined_equation"`` and prints a one-time warning.
    """
    source = _resolve_electron_transport_source(cfg)

    if source == "user_defined_equation":
        return compute_user_defined_electron_mobility(
            cfg=cfg,
            x_array=x_array,
            E_column=E_column,
        )

    if source == "swarm_data_table_interpolation":
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

    Ion kinetics is currently user-defined only. This routine applies
    ``compute_user_defined_ion_mobility``.
    """
    _resolve_ion_transport_source(cfg)
    return compute_user_defined_ion_mobility(
        cfg=cfg,
        x_array=x_array,
        E_column=E_column,
    )


def build_electron_diffusion_profile(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
    neutral_density: float,
) -> np.ndarray:
    """
    Build the electron-diffusion profile D_e(x) for the current step.

    The ``"user_defined_equation"`` source uses the diffusion formulas
    implemented in this module. The
    ``"swarm_data_table_interpolation"`` source uses the same electron
    swarm-data file as the mobility interpolation path to recover D_e(E/N).
    If the source is unavailable or cannot be loaded, the code falls back to
    ``"user_defined_equation"``
    and prints a one-time warning.
    """
    source = _resolve_electron_transport_source(cfg)

    if source == "user_defined_equation":
        return compute_user_defined_electron_diffusion(
            cfg=cfg,
            x_array=x_array,
            E_column=E_column,
        )

    if source == "swarm_data_table_interpolation":
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

    Ion kinetics is currently user-defined only. This routine applies
    ``compute_user_defined_ion_diffusion``.
    """
    _resolve_ion_transport_source(cfg)
    return compute_user_defined_ion_diffusion(
        cfg=cfg,
        x_array=x_array,
        E_column=E_column,
    )
    

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
    mean_energy_eV: np.ndarray | None = None,
    out_of_range_policy: str = "clip",
) -> np.ndarray:
    """
    Build the Townsend ionization-coefficient profile alpha(x) [1/m].

    Source modes:
    - user_defined_equation
    - interpolate_from_e_over_n_table
    """
    source_mode = _resolve_townsend_alpha_source_mode(cfg)

    if source_mode == "user_defined_equation":
        return compute_user_defined_townsend_alpha(E_column, p_Torr, pr, gas).astype(
            np.float32, copy=False
        )

    if source_mode == "interpolate_from_e_over_n_table":
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
                "townsend_alpha_source_mode",
                str(exc),
            )
            return compute_user_defined_townsend_alpha(E_column, p_Torr, pr, gas).astype(
                np.float32, copy=False
            )

    raise ValueError(f"Unknown townsend_alpha_source_mode: {source_mode}")


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
    n0    = cfg.plasma_state.n0

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
