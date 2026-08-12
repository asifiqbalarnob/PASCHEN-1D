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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from physical_constants import kB, e, m_e
from electron_transport import (
    ElectronSwarmTable,
    SECTION_DIFFUSION,
    SECTION_IONIZATION_FREQUENCY,
    SECTION_MOBILITY,
    SECTION_TOWNSEND,
    load_electron_swarm_table,
)
from data_paths import (
    ELECTRON_SWARM_DATA_DIR,
    resolve_electron_swarm_data_file,
    resolve_ion_swarm_data_file,
)
from ion_transport import (
    IonTableInterpolator,
    load_ion_transport_table,
    validate_table_identity,
    validate_table_pair,
)
from config import (
    SimulationConfig,
    TransportCoeffs,
    TransportSourceMode,
)


_MU_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_D_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_ALPHA_OVER_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_NU_OVER_N_TABLE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_SWARM_DATA_SECTION_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


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
    out_of_range_policy: str
    quantity_name: str
    below_range_count: int = 0
    above_range_count: int = 0
    evaluated_value_count: int = 0
    requested_min_Td: float = np.inf
    requested_max_Td: float = -np.inf

    def evaluate(self, E_column: np.ndarray, neutral_density: float) -> np.ndarray:
        """
        Evaluate the interpolated runtime quantity on the local electric field.
        """
        N_g = float(neutral_density)
        if not np.isfinite(N_g) or N_g <= 0.0:
            raise ValueError(f"Neutral density must be finite and positive; got {N_g!r}")
        if not np.all(np.isfinite(E_column)):
            raise ValueError(f"Electric field for {self.quantity_name} contains NaN or infinity.")
        en_td_local = np.abs(E_column).astype(np.float64, copy=False) * 1.0e21 / N_g
        below = en_td_local < self.en_td_min
        above = en_td_local > self.en_td_max
        below_count = int(np.count_nonzero(below))
        above_count = int(np.count_nonzero(above))
        self.below_range_count += below_count
        self.above_range_count += above_count
        self.evaluated_value_count += int(en_td_local.size)
        if en_td_local.size:
            self.requested_min_Td = min(self.requested_min_Td, float(np.min(en_td_local)))
            self.requested_max_Td = max(self.requested_max_Td, float(np.max(en_td_local)))
        if self.out_of_range_policy == "error" and (below_count or above_count):
            raise ValueError(
                f"Local E/N for {self.quantity_name} is outside its electron table: "
                f"requested [{float(np.min(en_td_local)):.6g}, "
                f"{float(np.max(en_td_local)):.6g}] Td; table "
                f"[{self.en_td_min:.6g}, {self.en_td_max:.6g}] Td."
            )
        if self.out_of_range_policy != "clip":
            raise ValueError(
                f"Unknown electron table out-of-range policy: {self.out_of_range_policy!r}"
            )
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

    def coverage(self) -> dict[str, float | int | str | None]:
        """Return accumulated table-range diagnostics for run metadata."""
        return {
            "quantity": self.quantity_name,
            "policy": self.out_of_range_policy,
            "table_min_Td": self.en_td_min,
            "table_max_Td": self.en_td_max,
            "evaluated_value_count": self.evaluated_value_count,
            "below_range_count": self.below_range_count,
            "above_range_count": self.above_range_count,
            "requested_min_Td": (
                self.requested_min_Td if self.evaluated_value_count else None
            ),
            "requested_max_Td": (
                self.requested_max_Td if self.evaluated_value_count else None
            ),
        }


@dataclass
class SwarmRuntimeInterpolationCache:
    """
    Run-scoped cache of swarm-data interpolation objects.

    The cache preloads all needed swarm sections once at simulation startup and
    exposes lightweight evaluators used inside the time-stepping loop.
    """
    electron_transport_source: TransportSourceMode = "user_defined_equation"
    ion_mobility_source_mode: str = "user_defined_equation"
    ion_diffusion_source_mode: str = "user_defined_equation"
    townsend_alpha_source_mode: str = "user_defined_equation"
    ionization_frequency_source_mode: str = "interpolate_from_e_over_n_table"
    impact_ionization_model: str = "from_townsend_alpha"
    electron_mu_eovern_interp: Optional[_EoverNInterpolator] = None
    electron_D_eovern_interp: Optional[_EoverNInterpolator] = None
    ion_mobility_eovern_interp: Optional[IonTableInterpolator] = None
    ion_diffusion_eovern_interp: Optional[IonTableInterpolator] = None
    alpha_eovern_interp: Optional[_EoverNInterpolator] = None
    nu_over_N_eovern_interp: Optional[_EoverNInterpolator] = None
    electron_tables: dict[str, ElectronSwarmTable] = field(default_factory=dict)
    ion_pair_provenance: Optional[dict] = None

    def ion_mobility_from_field(
        self,
        cfg: SimulationConfig,
        x_array: np.ndarray,
        E_column: np.ndarray,
        neutral_density: float,
    ) -> np.ndarray:
        """Evaluate the active positive-ion mobility profile."""
        if self.ion_mobility_source_mode == "swarm_data_table_interpolation":
            if self.ion_mobility_eovern_interp is None:
                raise RuntimeError("Ion mobility interpolator was not initialized.")
            return self.ion_mobility_eovern_interp.evaluate(E_column, neutral_density)
        if self.ion_mobility_source_mode == "user_defined_equation":
            return compute_user_defined_ion_mobility(cfg, x_array, E_column)
        raise RuntimeError(
            f"Unsupported ion mobility source: {self.ion_mobility_source_mode!r}"
        )

    def ion_diffusion_from_field(
        self,
        cfg: SimulationConfig,
        x_array: np.ndarray,
        E_column: np.ndarray,
        neutral_density: float,
        ion_mobility: np.ndarray,
    ) -> np.ndarray:
        """Evaluate table, Einstein, or user-defined ion diffusion."""
        if self.ion_diffusion_source_mode == "swarm_data_table_interpolation":
            if self.ion_diffusion_eovern_interp is None:
                raise RuntimeError("Ion diffusion interpolator was not initialized.")
            return self.ion_diffusion_eovern_interp.evaluate(E_column, neutral_density)
        if self.ion_diffusion_source_mode == "einstein_relation":
            return (
                ion_mobility.astype(np.float64, copy=False)
                * kB
                * float(cfg.plasma_state.T_i)
                / e
            ).astype(np.float32, copy=False)
        if self.ion_diffusion_source_mode == "user_defined_equation":
            return compute_user_defined_ion_diffusion(cfg, x_array, E_column)
        raise RuntimeError(
            f"Unsupported ion diffusion source: {self.ion_diffusion_source_mode!r}"
        )

    def ion_transport_provenance(self) -> dict:
        """Return normalized source provenance for output metadata."""
        result = {
            "mobility_source_mode": self.ion_mobility_source_mode,
            "diffusion_source_mode": self.ion_diffusion_source_mode,
        }
        if self.ion_mobility_eovern_interp is not None:
            result["mobility_table"] = self.ion_mobility_eovern_interp.table.provenance()
            result["mobility_coverage"] = self.ion_mobility_eovern_interp.coverage()
        if self.ion_diffusion_eovern_interp is not None:
            result["diffusion_table"] = self.ion_diffusion_eovern_interp.table.provenance()
            result["diffusion_coverage"] = self.ion_diffusion_eovern_interp.coverage()
        if self.ion_pair_provenance is not None:
            result["compatible_pair"] = self.ion_pair_provenance
        return result

    def electron_transport_provenance(self) -> dict:
        """Return authenticated electron sources and E/N coverage diagnostics."""
        interpolators = {
            "mobility": self.electron_mu_eovern_interp,
            "diffusion": self.electron_D_eovern_interp,
            "townsend_alpha": self.alpha_eovern_interp,
            "ionization_frequency": self.nu_over_N_eovern_interp,
        }
        return {
            "tables": {
                role: table.provenance() for role, table in self.electron_tables.items()
            },
            "coverage": {
                role: interp.coverage()
                for role, interp in interpolators.items()
                if interp is not None
            },
        }

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

    - "table", "tabulated", or "measured_table":
        V(t) is linearly interpolated from a CSV table. Headered and
        headerless numeric CSV files are accepted; columns are selected by
        zero-based index. Optional config attributes on cfg.waveform:
            table_path, table_time_column, table_voltage_column,
            table_time_scale, table_time_offset, table_voltage_scale,
            table_voltage_offset.
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

    elif cfg.waveform.waveform_type in ("table", "tabulated", "measured_table"):
        table_path = Path(getattr(cfg.waveform, "table_path"))
        table_time_column = int(getattr(cfg.waveform, "table_time_column", 0))
        table_voltage_column = int(getattr(cfg.waveform, "table_voltage_column", 1))
        table_time_scale = float(getattr(cfg.waveform, "table_time_scale", 1.0))
        table_time_offset = float(getattr(cfg.waveform, "table_time_offset", 0.0))
        table_voltage_scale = float(getattr(cfg.waveform, "table_voltage_scale", 1.0))
        table_voltage_offset = float(getattr(cfg.waveform, "table_voltage_offset", 0.0))

        raw = np.genfromtxt(table_path, delimiter=",", dtype=np.float64)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2 or raw.shape[1] <= max(table_time_column, table_voltage_column):
            raise ValueError(
                "Voltage table must contain the requested time and voltage "
                f"columns: {table_path}"
            )
        time_raw = raw[:, table_time_column]
        voltage_raw = raw[:, table_voltage_column]

        table_t = table_time_scale * np.asarray(time_raw, dtype=np.float64) + table_time_offset
        table_v = table_voltage_scale * np.asarray(voltage_raw, dtype=np.float64) + table_voltage_offset
        finite = np.isfinite(table_t) & np.isfinite(table_v)
        if np.count_nonzero(finite) < 2:
            raise ValueError(f"Voltage table must contain at least two finite rows: {table_path}")
        table_t = table_t[finite]
        table_v = table_v[finite]
        order = np.argsort(table_t)
        table_t = table_t[order]
        table_v = table_v[order]
        unique_t, unique_idx = np.unique(table_t, return_index=True)
        table_t = unique_t
        table_v = table_v[unique_idx]

        def V_app_func(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t)
            return np.interp(t, table_t, table_v, left=table_v[0], right=table_v[-1])

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


def _resolve_electron_data_path(path_str: str) -> Path:
    """Resolve an electron table relative to ``electron_swarm_data``."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        root = ELECTRON_SWARM_DATA_DIR.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Resolved electron table is outside '{root}': {resolved}"
            ) from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"Electron swarm-data file not found: {resolved}")
        return resolved
    return resolve_electron_swarm_data_file(path_str)


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


def _normalize_ion_kinetics_mode(cfg: SimulationConfig) -> str:
    """Return the validated ion-kinetics selector."""
    ion_mode = str(cfg.plasma.ion_kinetics_model).strip().lower()
    if ion_mode not in {
        "user_defined_ion_kinetics",
        "local_field_ion_kinetics",
    }:
        raise ValueError(f"Unsupported ion kinetics model: {cfg.plasma.ion_kinetics_model}")
    return ion_mode


def _resolve_ion_mobility_source(cfg: SimulationConfig) -> str:
    if _normalize_ion_kinetics_mode(cfg) == "user_defined_ion_kinetics":
        return "user_defined_equation"
    source = str(cfg.ion_transport.mobility_source_mode).strip().lower()
    if source not in {"user_defined_equation", "swarm_data_table_interpolation"}:
        raise ValueError(f"Unsupported ion mobility source: {source!r}")
    return source


def _resolve_ion_diffusion_source(cfg: SimulationConfig) -> str:
    if _normalize_ion_kinetics_mode(cfg) == "user_defined_ion_kinetics":
        return "user_defined_equation"
    source = str(cfg.ion_transport.diffusion_source_mode).strip().lower()
    if source not in {
        "user_defined_equation",
        "swarm_data_table_interpolation",
        "einstein_relation",
    }:
        raise ValueError(f"Unsupported ion diffusion source: {source!r}")
    return source


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
    table = load_electron_swarm_table(path_str)
    resolved = str(table.path)
    cache_key = (resolved, section_label)
    cached = _SWARM_DATA_SECTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    section = table.section(section_label)
    en_td = section.reduced_field_Td
    values = section.reduced_values_SI
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
    Load one authenticated E/N-dependent quantity from a bundled BOLSIG+ table.
    """
    resolved = str(_resolve_electron_data_path(path_str))
    cached = cache.get(resolved)
    if cached is not None:
        return cached

    en_td, values = load_swarm_data_section(path_str, section_label)

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

    if (
        not np.all(np.isfinite(en_td))
        or not np.all(np.isfinite(values))
        or np.any(en_td <= 0.0)
        or np.any(np.diff(en_td) <= 0.0)
        or invalid_values
    ):
        raise ValueError(
            f"{source_label} '{resolved}' must contain a finite, positive, unique, "
            f"strictly increasing E/N axis and {value_phrase}."
        )
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
    out_of_range_policy: str,
    quantity_name: str,
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
        out_of_range_policy=out_of_range_policy,
        quantity_name=quantity_name,
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

    _normalize_electron_kinetics_mode(cfg)
    _normalize_ion_kinetics_mode(cfg)
    cache.electron_transport_source = _resolve_electron_transport_source(cfg)
    cache.ion_mobility_source_mode = _resolve_ion_mobility_source(cfg)
    cache.ion_diffusion_source_mode = _resolve_ion_diffusion_source(cfg)
    cache.townsend_alpha_source_mode = _resolve_townsend_alpha_source_mode(cfg)
    cache.ionization_frequency_source_mode = _resolve_ionization_frequency_source_mode(cfg)
    cache.impact_ionization_model = _resolve_impact_ionization_model(cfg)
    electron_policy = cfg.electron_swarm_data.out_of_range_policy

    def configured_electron_table(path_value: str, role: str) -> ElectronSwarmTable:
        table = load_electron_swarm_table(
            path_value,
            configured_gas=cfg.plasma_state.gas,
            gas_temperature_K=cfg.plasma_state.T_i,
            temperature_tolerance_K=(
                cfg.electron_swarm_data.gas_temperature_tolerance_K
            ),
        )
        cache.electron_tables[role] = table
        return table

    # Field-based electron transport (LFA mode).
    if cache.electron_transport_source == "swarm_data_table_interpolation":
        source_path = _resolve_electron_swarm_path(cfg)
        table = configured_electron_table(source_path, "electron_transport")
        mobility = table.section(SECTION_MOBILITY)
        diffusion = table.section(SECTION_DIFFUSION)
        cache.electron_mu_eovern_interp = _build_eovern_interpolator(
            mobility.reduced_field_Td,
            mobility.reduced_values_SI,
            density_operation="divide_by_density",
            out_of_range_policy=electron_policy,
            quantity_name="electron mobility",
        )
        cache.electron_D_eovern_interp = _build_eovern_interpolator(
            diffusion.reduced_field_Td,
            diffusion.reduced_values_SI,
            density_operation="divide_by_density",
            out_of_range_policy=electron_policy,
            quantity_name="electron diffusion",
        )

    # Positive-ion local-field transport. Table identity and temperature are
    # checked before the time loop; a requested table never silently falls back.
    ion_cfg = cfg.ion_transport
    if cache.ion_mobility_source_mode == "swarm_data_table_interpolation":
        if not ion_cfg.mobility_table_path:
            raise ValueError(
                "ion_transport.mobility_table_path is required for table mobility."
            )
        mobility_table = load_ion_transport_table(
            resolve_ion_swarm_data_file(ion_cfg.mobility_table_path)
        )
        validate_table_identity(
            mobility_table,
            expected_quantity="reduced_mobility",
            configured_ion=ion_cfg.positive_ion,
            configured_neutral=cfg.plasma_state.gas,
            gas_temperature_K=cfg.plasma_state.T_i,
            temperature_tolerance_K=ion_cfg.gas_temperature_tolerance_K,
        )
        cache.ion_mobility_eovern_interp = IonTableInterpolator.from_table(
            mobility_table, ion_cfg.out_of_range_policy
        )
    if cache.ion_diffusion_source_mode == "swarm_data_table_interpolation":
        if not ion_cfg.diffusion_table_path:
            raise ValueError(
                "ion_transport.diffusion_table_path is required for table diffusion."
            )
        diffusion_table = load_ion_transport_table(
            resolve_ion_swarm_data_file(ion_cfg.diffusion_table_path)
        )
        validate_table_identity(
            diffusion_table,
            expected_quantity="reduced_longitudinal_diffusion",
            configured_ion=ion_cfg.positive_ion,
            configured_neutral=cfg.plasma_state.gas,
            gas_temperature_K=cfg.plasma_state.T_i,
            temperature_tolerance_K=ion_cfg.gas_temperature_tolerance_K,
        )
        cache.ion_diffusion_eovern_interp = IonTableInterpolator.from_table(
            diffusion_table, ion_cfg.out_of_range_policy
        )
    if (
        cache.ion_mobility_eovern_interp is not None
        and cache.ion_diffusion_eovern_interp is not None
    ):
        cache.ion_pair_provenance = validate_table_pair(
            cache.ion_mobility_eovern_interp.table,
            cache.ion_diffusion_eovern_interp.table,
        )

    # Townsend alpha interpolation (for alpha-based ionization mode).
    if (
        cache.impact_ionization_model == "from_townsend_alpha"
        and cache.townsend_alpha_source_mode != "user_defined_equation"
    ):
        alpha_path = _resolve_townsend_alpha_path(cfg)
        if cache.townsend_alpha_source_mode == "interpolate_from_e_over_n_table":
            table = configured_electron_table(alpha_path, "townsend_alpha")
            townsend = table.section(SECTION_TOWNSEND)
            cache.alpha_eovern_interp = _build_eovern_interpolator(
                townsend.reduced_field_Td,
                townsend.reduced_values_SI,
                density_operation="multiply_by_density",
                out_of_range_policy=electron_policy,
                quantity_name="Townsend alpha",
                allow_zero_values=True,
            )

    # Direct ionization-frequency interpolation (nu_i from table).
    if (
        cache.impact_ionization_model == "from_ionization_frequency"
        and cache.ionization_frequency_source_mode != "user_defined_equation"
    ):
        ion_path = _resolve_ionization_frequency_path(cfg)
        if cache.ionization_frequency_source_mode == "interpolate_from_e_over_n_table":
            table = configured_electron_table(ion_path, "ionization_frequency")
            ionization_frequency = table.section(SECTION_IONIZATION_FREQUENCY)
            cache.nu_over_N_eovern_interp = _build_eovern_interpolator(
                ionization_frequency.reduced_field_Td,
                ionization_frequency.reduced_values_SI,
                density_operation="multiply_by_density",
                out_of_range_policy=electron_policy,
                quantity_name="ionization frequency",
                allow_zero_values=True,
            )
    return cache


def load_electron_mobility_muN_data(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load electron-mobility data for mu_e(E/N).

    Parameters
    ----------
    path_str : str
        Bundled BOLSIG+ filename containing ``Mobility *N (1/m/V/s)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays ``(en_td, muN)`` in float64, sorted in ascending ``E/N``.

    Notes
    -----
    The file must be authenticated by ``electron_swarm_data/manifest.json``.
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
        Bundled BOLSIG+ filename containing ``Diffusion coefficient *N (1/m/s)``.

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
        Bundled BOLSIG+ filename containing Townsend ``alpha/N``.

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
        - beta       : baseline volume recombination coefficient [m³/s]
        - neutral_density : uniform background neutral density [m⁻³]
        - pr         : reduced pressure p * (T_ref / T_i) (dimensionless)
        - T_e_eV     : electron temperature in eV (for diagnostics)
        - T_i_eV     : ion temperature in eV (for diagnostics)

    Notes
    -----
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

    beta_val = compute_user_defined_recombination_coefficient(cfg)

    return TransportCoeffs(
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
    ``"swarm_data_table_interpolation"`` uses the configured swarm table.
    Missing or invalid files are fatal configuration errors.
    """
    source = _resolve_electron_transport_source(cfg)

    if source == "user_defined_equation":
        return compute_user_defined_electron_mobility(
            cfg=cfg,
            x_array=x_array,
            E_column=E_column,
        )

    if source == "swarm_data_table_interpolation":
        return interpolate_electron_mobility_from_muN_table(
            cfg=cfg,
            E_column=E_column,
            neutral_density=neutral_density,
        )

    raise ValueError(f"Unknown electron_transport_source: {source}")


def build_ion_mobility_profile(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
    neutral_density: float,
    interpolation_cache: SwarmRuntimeInterpolationCache,
) -> np.ndarray:
    """
    Build the configured ion-mobility profile mu_i(x) for the current step.
    """
    return interpolation_cache.ion_mobility_from_field(
        cfg, x_array, E_column, neutral_density
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
    Missing or invalid table sources are fatal configuration errors.
    """
    source = _resolve_electron_transport_source(cfg)

    if source == "user_defined_equation":
        return compute_user_defined_electron_diffusion(
            cfg=cfg,
            x_array=x_array,
            E_column=E_column,
        )

    if source == "swarm_data_table_interpolation":
        return interpolate_electron_diffusion_from_DN_table(
            cfg=cfg,
            E_column=E_column,
            neutral_density=neutral_density,
        )

    raise ValueError(f"Unknown electron_transport_source: {source}")


def build_ion_diffusion_profile(
    cfg: SimulationConfig,
    x_array: np.ndarray,
    E_column: np.ndarray,
    neutral_density: float,
    ion_mobility: np.ndarray,
    interpolation_cache: SwarmRuntimeInterpolationCache,
) -> np.ndarray:
    """
    Build the configured longitudinal ion-diffusion profile D_i(x).
    """
    return interpolation_cache.ion_diffusion_from_field(
        cfg, x_array, E_column, neutral_density, ion_mobility
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
        return interpolate_townsend_alpha_from_alpha_over_N_table(
            cfg=cfg,
            E_column=E_column,
            neutral_density=neutral_density,
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
