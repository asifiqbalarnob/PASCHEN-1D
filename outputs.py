"""
outputs.py

I/O utilities for the PASCHEN-1D drift-diffusion-Poisson solver.

This module handles:

1. Creation of on-disk, memory-mapped arrays for:
   - Field snapshots (phi, E)
   - Species densities (n_e, n_i)
   - Optional diagnostics (Gamma_i, Gamma_e, townsend_alpha, nu_i, S_ion, S,
     mu_e, D_e)
   - Scalar time histories (V_gap, CFL, I_discharge, current decomposition)
   - Adaptive-substepping time histories (substep count, dt_sub, CFL estimate)

2. A small dataclass `OutputHandles` that collects references to all
   memmapped arrays so the main driver can pass them around easily.

3. A `write_snapshot` helper that writes one snapshot of the current
   plasma state (and optional diagnostics) into the preallocated files.

The design is deliberately simple: all arrays are row-major with
shape (Nsave, Nx) for spatial snapshots or (Nt,) for scalars. Spatial
snapshots use np.float32 to keep files compact. Circuit-sensitive scalar
voltage/current histories use np.float64 so slow circuit relaxation is not
rounded away when the macro time step is very small. Files are created under a
directory named after `cfg.run.run_name`.
"""

from dataclasses import dataclass
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Optional

import numpy as np

from config import SimulationConfig
from version import __release_date__, __version__

SNAPSHOT_DTYPE = np.float32
SCALAR_HISTORY_DTYPE = np.float64
CONTROL_HISTORY_DTYPE = np.float32


def _software_provenance() -> dict:
    """Return release, runtime, dependency, and source-revision provenance."""
    dependency_versions = {}
    for package in ("numpy", "scipy", "numba", "tqdm", "matplotlib"):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        git_commit = result.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        git_commit = None
    return {
        "name": "PASCHEN-1D",
        "version": __version__,
        "release_date": __release_date__,
        "git_commit": git_commit,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": dependency_versions,
    }


def _metadata_float(value):
    """Return a JSON-safe representation for numeric metadata values."""
    value = float(value)
    if np.isnan(value):
        return None
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return value


# ============================================================
# Dataclass container for all outputs
# ============================================================


@dataclass
class OutputHandles:
    """
    Container for memory-mapped output arrays.

    Shapes
    ------
    All spatial snapshots:
        (Nsave, Nx)
    All scalar time histories:
        (Nt,)

    Attributes
    ----------
    phi_sampled, E_sampled : np.ndarray
        Sampled potential and electric field.
    n_e_sampled, n_i_sampled : np.ndarray
        Sampled electron and ion densities.
    Gamma_i_sampled, Gamma_e_sampled : np.ndarray or None
        Sampled ion and electron fluxes (optional; None if disabled).
    townsend_alpha_sampled : np.ndarray or None
        Sampled Townsend ionization coefficients (optional).
    nu_i_sampled : np.ndarray or None
        Sampled ionization frequencies (optional).
    S_ion_sampled : np.ndarray or None
        Sampled ionization-only source term (+nu_i * n_e or +k_ion*N*n_e) (optional).
    S_sampled : np.ndarray or None
        Sampled source term (e.g., ionization − recombination) (optional).
    mu_e_sampled : np.ndarray or None
        Sampled local electron mobility [m^2/(V s)] (optional).
    D_e_sampled : np.ndarray or None
        Sampled local electron diffusion coefficient [m^2/s] (optional).
    V_gap : np.ndarray
        Time history of plasma gap voltage.
    c_cfl : np.ndarray
        Time history of CFL diagnostic values.
    I_discharge : np.ndarray
        Time history of discharge current.
    I_transport_plasma : np.ndarray
        Physical plasma transport current from Gamma_i and Gamma_e.
    I_transport_circuit : np.ndarray
        Transport component recovered from I_discharge - I_displacement_gap;
        should match I_transport_plasma after direct surface-emission circuit
        coupling is removed.
    I_emission_circuit : np.ndarray
        Direct-emission circuit residual diagnostic; expected to be near zero
        after direct surface-emission circuit coupling is removed.
    I_emission_area : np.ndarray
        Signed external-emission surface current represented over the modeled
        1D area. This is not directly added to I_discharge.
    I_displacement_gap : np.ndarray
        Capacitive load-side current component, I_discharge - I_transport_circuit.
        For ordinary gap-only topologies this is the geometric gas-gap
        displacement current; for C_ext topologies it also includes load-side
        C_ext charging.
    adaptive_substeps : np.ndarray
        Per-macro-step adaptive substep counts (1 when adaptive mode is off).
    adaptive_dt_sub : np.ndarray
        Effective per-macro-step substep size [s].
    adaptive_cfl_est : np.ndarray
        Pre-substep drift-CFL estimate at macro-step start.
    """
    phi_sampled: np.ndarray
    E_sampled: np.ndarray
    n_e_sampled: np.ndarray
    n_i_sampled: np.ndarray
    Gamma_i_sampled: Optional[np.ndarray]
    Gamma_e_sampled: Optional[np.ndarray]
    townsend_alpha_sampled: Optional[np.ndarray]
    nu_i_sampled: Optional[np.ndarray]
    S_ion_sampled: Optional[np.ndarray]
    S_sampled: Optional[np.ndarray]
    mu_e_sampled: Optional[np.ndarray]
    D_e_sampled: Optional[np.ndarray]
    mu_i_sampled: Optional[np.ndarray]
    D_i_sampled: Optional[np.ndarray]
    V_gap: np.ndarray
    V_node: Optional[np.ndarray]
    V_source: Optional[np.ndarray]
    c_cfl: np.ndarray
    I_discharge: np.ndarray
    I_transport_plasma: np.ndarray
    I_transport_circuit: np.ndarray
    I_emission_circuit: np.ndarray
    I_emission_area: np.ndarray
    I_displacement_gap: np.ndarray
    picard_iterations: np.ndarray
    adaptive_substeps: np.ndarray
    adaptive_dt_sub: np.ndarray
    adaptive_cfl_est: np.ndarray

# ============================================================
# Low-level file creation helper
# ============================================================


def create_file(
    path: str | Path,
    shape: tuple[int, ...],
    dtype: np.dtype = np.float32,
) -> None:
    """
    Create and zero-initialize a memory-mapped binary file.

    This helper:
    - Ensures the parent directory exists.
    - Allocates a NumPy memmap with the requested shape and dtype.
    - Fills it with zeros and flushes to disk.
    - Closes the memmap immediately (so it can be reopened later).

    Parameters
    ----------
    path : str or pathlib.Path
        File path where the memmap will be created.
    shape : tuple[int, ...]
        Shape of the array to be stored in the memmap,
        e.g. (Nsave, Nx) or (Nt,).
    dtype : np.dtype, optional
        Data type of the stored array. Default is np.float32.

    Notes
    -----
    The file can later be reopened with:

        np.memmap(path, mode="r+" or "readwrite", dtype=dtype, shape=shape)

    This routine does *not* return the memmap; it only creates and zeros it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr[:] = 0.0
    arr.flush()
    # Delete reference so the file handle is closed
    del arr


# ============================================================
# Output allocation / memmap setup
# ============================================================


def allocate_outputs(cfg: SimulationConfig, Nt: int, Nx: int) -> OutputHandles:
    """
    Allocate and open all memory-mapped output arrays for a given run.

    This function:
    1. Computes the number of saved snapshots (Nsave) based on Nt and
       cfg.output.save_every.
    2. Creates a subdirectory named `cfg.run.run_name`.
    3. Creates zero-initialized memmap files for:
         - phi, E, n_e, n_i       (always)
         - Gamma_i, Gamma_e, townsend_alpha, nu_i, S_ion, S, mu_e, D_e
           (if cfg.output.log_intermediate is True)
         - V_gap, c_cfl, I_discharge, current decomposition
           (scalar time histories)
    4. Reopens those files in "readwrite" mode and wraps them in an
       OutputHandles dataclass.

    Parameters
    ----------
    cfg : SimulationConfig
        Simulation configuration; only `run_name`, `save_every`,
        and `log_intermediate` are used here.
    Nt : int
        Total number of time steps in the simulation.
    Nx : int
        Number of spatial grid points.

    Returns
    -------
    handles : OutputHandles
        Dataclass with references to all memmapped arrays.

    Notes
    -----
    Number of snapshots:
        Nsave = floor((Nt - 1) / save_every) + 1

    so that the code can save at indices 0, save_every, 2*save_every, ...
    up to (Nsave - 1)*save_every <= Nt - 1.
    """
    SAVE_EVERY = cfg.output.save_every
    LOG_INTERMEDIATE = cfg.output.log_intermediate

    # Number of saved snapshots along time
    Nsave = int((Nt - 1) // SAVE_EVERY + 1)

    outdir = Path(cfg.run.run_name)
    outdir.mkdir(exist_ok=True)

    # --- Field snapshot paths ---
    phi_path = outdir / "phi_sampled_mm.dat"
    E_path = outdir / "E_sampled_mm.dat"
    ne_path = outdir / "ne_sampled_mm.dat"
    ni_path = outdir / "ni_sampled_mm.dat"

    # --- Diagnostic snapshot paths (optional) ---
    Gamma_i_path = outdir / "Gamma_i_sampled_mm.dat"
    Gamma_e_path = outdir / "Gamma_e_sampled_mm.dat"
    townsend_alpha_path = outdir / "townsend_alpha_sampled_mm.dat"
    nu_i_path = outdir / "nu_i_sampled_mm.dat"
    S_ion_path = outdir / "S_ion_sampled_mm.dat"
    S_path   = outdir / "S_sampled_mm.dat"
    mu_e_path = outdir / "mu_e_sampled_mm.dat"
    D_e_path = outdir / "D_e_sampled_mm.dat"
    mu_i_path = outdir / "mu_i_sampled_mm.dat"
    D_i_path = outdir / "D_i_sampled_mm.dat"


    # --- Scalar time histories ---
    Vgap_path = outdir / "Vgap_mm.dat"
    c_cfl_path = outdir / "c_cfl_mm.dat"
    Idis_path = outdir / "Idischarge_mm.dat"
    Vnode_path = outdir / "Vnode_mm.dat"
    Vsource_path = outdir / "Vsource_mm.dat"
    I_transport_plasma_path = outdir / "I_transport_plasma_mm.dat"
    I_transport_circuit_path = outdir / "I_transport_circuit_mm.dat"
    I_emission_circuit_path = outdir / "I_emission_circuit_mm.dat"
    I_emission_area_path = outdir / "I_emission_area_mm.dat"
    I_displacement_gap_path = outdir / "I_displacement_gap_mm.dat"
    picard_iterations_path = outdir / "picard_iterations_mm.dat"
    adaptive_substeps_path = outdir / "adaptive_substeps_mm.dat"
    adaptive_dt_sub_path = outdir / "adaptive_dt_sub_mm.dat"
    adaptive_cfl_est_path = outdir / "adaptive_cfl_est_mm.dat"
    circuit_type = str(getattr(cfg.circuit, "circuit_type", ""))
    has_node_voltage = circuit_type in {
        "R0_Cp",
        "R0_Cp_Rm",
        "R0_Rm_Cext",
        "R0_Cs_Cp",
        "R0_Cs_Ls_Cp",
        "R0_Cs_Cp_Rm",
        "R0_Cs_Ls_Cp_Rm",
        "R0_Cs_Ls_Cp_Lp",
        "R0_Cs_Ls_Cp_Lp_Rm_Cext",
    }
    has_source_voltage = False

    # --- Create files (zero-initialized) ---
    create_file(phi_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
    create_file(E_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
    create_file(ne_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
    create_file(ni_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)

    if LOG_INTERMEDIATE:
        create_file(Gamma_i_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
        create_file(Gamma_e_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
        create_file(townsend_alpha_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
        create_file(nu_i_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
        create_file(S_ion_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
        create_file(S_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
        create_file(mu_e_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
        create_file(D_e_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
        create_file(mu_i_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)
        create_file(D_i_path, (Nsave, Nx), dtype=SNAPSHOT_DTYPE)


    create_file(Vgap_path, (Nt,), dtype=SCALAR_HISTORY_DTYPE)
    if has_node_voltage:
        create_file(Vnode_path, (Nt,), dtype=SCALAR_HISTORY_DTYPE)
    if has_source_voltage:
        create_file(Vsource_path, (Nt,), dtype=SCALAR_HISTORY_DTYPE)
    create_file(c_cfl_path, (Nt,), dtype=CONTROL_HISTORY_DTYPE)
    create_file(Idis_path, (Nt,), dtype=SCALAR_HISTORY_DTYPE)
    create_file(I_transport_plasma_path, (Nt,), dtype=SCALAR_HISTORY_DTYPE)
    create_file(I_transport_circuit_path, (Nt,), dtype=SCALAR_HISTORY_DTYPE)
    create_file(I_emission_circuit_path, (Nt,), dtype=SCALAR_HISTORY_DTYPE)
    create_file(I_emission_area_path, (Nt,), dtype=SCALAR_HISTORY_DTYPE)
    create_file(I_displacement_gap_path, (Nt,), dtype=SCALAR_HISTORY_DTYPE)
    create_file(picard_iterations_path, (Nt,), dtype=CONTROL_HISTORY_DTYPE)
    create_file(adaptive_substeps_path, (Nt,), dtype=CONTROL_HISTORY_DTYPE)
    create_file(adaptive_dt_sub_path, (Nt,), dtype=CONTROL_HISTORY_DTYPE)
    create_file(adaptive_cfl_est_path, (Nt,), dtype=CONTROL_HISTORY_DTYPE)

    # --- Open memmaps ---
    phi_sampled = np.memmap(phi_path, mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
    E_sampled = np.memmap(E_path, mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
    n_e_sampled = np.memmap(ne_path, mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
    n_i_sampled = np.memmap(ni_path, mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))

    if LOG_INTERMEDIATE:
        Gamma_i_sampled = np.memmap(Gamma_i_path, mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
        Gamma_e_sampled = np.memmap(Gamma_e_path, mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
        townsend_alpha_sampled   = np.memmap(townsend_alpha_path,   mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
        nu_i_sampled    = np.memmap(nu_i_path,    mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
        S_ion_sampled   = np.memmap(S_ion_path,   mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
        S_sampled       = np.memmap(S_path,       mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
        mu_e_sampled = np.memmap(mu_e_path, mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
        D_e_sampled = np.memmap(D_e_path, mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
        mu_i_sampled = np.memmap(mu_i_path, mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
        D_i_sampled = np.memmap(D_i_path, mode="readwrite", dtype=SNAPSHOT_DTYPE, shape=(Nsave, Nx))
    else:
        Gamma_i_sampled = Gamma_e_sampled = townsend_alpha_sampled = nu_i_sampled = S_ion_sampled = S_sampled = None
        mu_e_sampled = D_e_sampled = mu_i_sampled = D_i_sampled = None

    V_gap = np.memmap(Vgap_path, mode="readwrite", dtype=SCALAR_HISTORY_DTYPE, shape=(Nt,))
    V_node = (
        np.memmap(Vnode_path, mode="readwrite", dtype=SCALAR_HISTORY_DTYPE, shape=(Nt,))
        if has_node_voltage
        else None
    )
    V_source = (
        np.memmap(Vsource_path, mode="readwrite", dtype=SCALAR_HISTORY_DTYPE, shape=(Nt,))
        if has_source_voltage
        else None
    )
    c_cfl = np.memmap(c_cfl_path, mode="readwrite", dtype=CONTROL_HISTORY_DTYPE, shape=(Nt,))
    I_discharge = np.memmap(Idis_path, mode="readwrite", dtype=SCALAR_HISTORY_DTYPE, shape=(Nt,))
    I_transport_plasma = np.memmap(
        I_transport_plasma_path, mode="readwrite", dtype=SCALAR_HISTORY_DTYPE, shape=(Nt,)
    )
    I_transport_circuit = np.memmap(
        I_transport_circuit_path, mode="readwrite", dtype=SCALAR_HISTORY_DTYPE, shape=(Nt,)
    )
    I_emission_circuit = np.memmap(
        I_emission_circuit_path, mode="readwrite", dtype=SCALAR_HISTORY_DTYPE, shape=(Nt,)
    )
    I_emission_area = np.memmap(
        I_emission_area_path, mode="readwrite", dtype=SCALAR_HISTORY_DTYPE, shape=(Nt,)
    )
    I_displacement_gap = np.memmap(
        I_displacement_gap_path, mode="readwrite", dtype=SCALAR_HISTORY_DTYPE, shape=(Nt,)
    )
    picard_iterations = np.memmap(
        picard_iterations_path, mode="readwrite", dtype=CONTROL_HISTORY_DTYPE, shape=(Nt,)
    )
    adaptive_substeps = np.memmap(
        adaptive_substeps_path, mode="readwrite", dtype=CONTROL_HISTORY_DTYPE, shape=(Nt,)
    )
    adaptive_dt_sub = np.memmap(
        adaptive_dt_sub_path, mode="readwrite", dtype=CONTROL_HISTORY_DTYPE, shape=(Nt,)
    )
    adaptive_cfl_est = np.memmap(
        adaptive_cfl_est_path, mode="readwrite", dtype=CONTROL_HISTORY_DTYPE, shape=(Nt,)
    )

    return OutputHandles(
        phi_sampled=phi_sampled,
        E_sampled=E_sampled,
        n_e_sampled=n_e_sampled,
        n_i_sampled=n_i_sampled,
        Gamma_i_sampled=Gamma_i_sampled,
        Gamma_e_sampled=Gamma_e_sampled,
        townsend_alpha_sampled=townsend_alpha_sampled,
        nu_i_sampled=nu_i_sampled,
        S_ion_sampled=S_ion_sampled,
        S_sampled=S_sampled,
        mu_e_sampled=mu_e_sampled,
        D_e_sampled=D_e_sampled,
        mu_i_sampled=mu_i_sampled,
        D_i_sampled=D_i_sampled,
        V_gap=V_gap,
        V_node=V_node,
        V_source=V_source,
        c_cfl=c_cfl,
        I_discharge=I_discharge,
        I_transport_plasma=I_transport_plasma,
        I_transport_circuit=I_transport_circuit,
        I_emission_circuit=I_emission_circuit,
        I_emission_area=I_emission_area,
        I_displacement_gap=I_displacement_gap,
        picard_iterations=picard_iterations,
        adaptive_substeps=adaptive_substeps,
        adaptive_dt_sub=adaptive_dt_sub,
        adaptive_cfl_est=adaptive_cfl_est,
    )


def write_run_metadata(
    cfg: SimulationConfig,
    *,
    Nt: int,
    Nx: int,
    dt: float,
    dx: float,
    adaptive_stats: dict | None = None,
    hotloop_stats: dict | None = None,
    bc_poisson_picard_stats: dict | None = None,
    electron_transport_provenance: dict | None = None,
    ion_transport_provenance: dict | None = None,
) -> None:
    """
    Write lightweight run metadata for post-processing/replotting.

    The metadata is stored as JSON in:
        <run_name>/run_metadata.json
    """
    outdir = Path(cfg.run.run_name)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "run_metadata.json"

    payload = {
        "software": _software_provenance(),
        "run_name": cfg.run.run_name,
        "Nt": int(Nt),
        "Nx": int(Nx),
        "T_total": float(cfg.run.T_total),
        "L": float(cfg.geometry.L),
        "A": float(cfg.geometry.A),
        "gas": str(cfg.plasma_state.gas),
        "p_Torr": float(cfg.plasma_state.p_Torr),
        "T_e": float(cfg.plasma_state.T_e),
        "T_i": float(cfg.plasma_state.T_i),
        "n0": float(cfg.plasma_state.n0),
        "save_every": int(cfg.output.save_every),
        "snapshot_dtype": np.dtype(SNAPSHOT_DTYPE).name,
        "scalar_history_dtype": np.dtype(SCALAR_HISTORY_DTYPE).name,
        "control_history_dtype": np.dtype(CONTROL_HISTORY_DTYPE).name,
        "dt": float(dt),
        "dx": float(dx),
        "hotloop_backend": str(cfg.numerics.hotloop_backend),
        "numba_parallel": bool(cfg.numerics.numba_parallel),
        "use_adaptive_substepping": bool(cfg.numerics.use_adaptive_substepping),
        "target_cfl_substep": float(cfg.numerics.target_cfl_substep),
        "max_substeps": int(cfg.numerics.max_substeps),
        "adaptive_substep_overflow_policy": str(cfg.numerics.adaptive_substep_overflow_policy),
        "adaptive_substep_warn_every": int(cfg.numerics.adaptive_substep_warn_every),
        "bc_poisson_picard_min_iter": int(cfg.numerics.bc_poisson_picard_min_iter),
        "bc_poisson_picard_max_iter": int(cfg.numerics.bc_poisson_picard_max_iter),
        "bc_poisson_picard_tol": float(cfg.numerics.bc_poisson_picard_tol),
        "waveform_type": str(cfg.waveform.waveform_type),
        "V_peak": float(cfg.waveform.V_peak),
        "tV_start": float(cfg.waveform.tV_start),
        "tV_end": float(cfg.waveform.tV_end),
        "tau": float(cfg.waveform.tau),
        "t_peak": float(cfg.waveform.t_peak),
        "f_rf": float(cfg.waveform.f_rf),
        "V_dc": float(cfg.waveform.V_dc),
        "phi_rf": float(cfg.waveform.phi_rf),
        "gamma": float(cfg.emission.gamma),
        "shared_laser_t0": float(cfg.emission.shared_laser_t0),
        "shared_laser_U_J": float(cfg.emission.shared_laser_U_J),
        "cathode_laser_t0": float(cfg.emission.cathode_laser_t0),
        "cathode_laser_U_J": float(cfg.emission.cathode_laser_U_J),
        "cathode_enable_quantum_pulse_emission": bool(
            getattr(cfg.emission, "cathode_enable_quantum_pulse_emission", False)
        ),
        "enable_cathode_external_emission": bool(cfg.emission.enable_cathode_external_emission),
        "circuit_type": str(cfg.circuit.circuit_type),
        "circuit_time_scheme": str(cfg.circuit.circuit_time_scheme),
        "R0": _metadata_float(cfg.circuit.R0),
        "C_s": _metadata_float(cfg.circuit.C_s),
        "L_s": _metadata_float(cfg.circuit.L_s),
        "C_p": _metadata_float(cfg.circuit.C_p),
        "L_p": _metadata_float(cfg.circuit.L_p),
        "R_m": _metadata_float(cfg.circuit.R_m),
        "C_ext": _metadata_float(getattr(cfg.circuit, "C_ext", 0.0)),
        "plasma_model": {
            "electron_kinetics_model": str(cfg.plasma.electron_kinetics_model),
            "ion_kinetics_model": str(cfg.plasma.ion_kinetics_model),
            "impact_ionization_model": str(cfg.plasma.impact_ionization_model),
            "recombination_model": str(cfg.plasma.recombination_model),
        },
        "transport_sources": {
            "electron_transport_source": str(
                cfg.local_field_approximation.electron_transport_source
            ),
            "electron_swarm_data_path": str(
                cfg.local_field_approximation.electron_swarm_data_path
            ),
            "electron_table_out_of_range_policy": str(
                cfg.electron_swarm_data.out_of_range_policy
            ),
            "electron_table_provenance": electron_transport_provenance or {},
            "townsend_alpha_source_mode": str(
                cfg.townsend_coefficient.townsend_alpha_source_mode
            ),
            "townsend_alpha_swarm_data_path": str(
                cfg.townsend_coefficient.townsend_alpha_swarm_data_path
            ),
            "ionization_frequency_source_mode": str(
                cfg.ionization_frequency_source.ionization_frequency_source_mode
            ),
            "ionization_frequency_swarm_data_path": str(
                cfg.ionization_frequency_source.ionization_frequency_swarm_data_path
            ),
            "positive_ion": str(cfg.ion_transport.positive_ion),
            "ion_mobility_source_mode": str(cfg.ion_transport.mobility_source_mode),
            "ion_diffusion_source_mode": str(cfg.ion_transport.diffusion_source_mode),
            "ion_mobility_table_path": cfg.ion_transport.mobility_table_path,
            "ion_diffusion_table_path": cfg.ion_transport.diffusion_table_path,
            "ion_table_out_of_range_policy": str(cfg.ion_transport.out_of_range_policy),
            "ion_table_provenance": ion_transport_provenance or {},
        },
        "volume_sources": {
            "enable_volume_sources": bool(cfg.boundary.enable_volume_sources),
            "enable_ionization_source": bool(cfg.boundary.enable_ionization_source),
            "enable_recombination_sink": bool(cfg.boundary.enable_recombination_sink),
            "recombination_coefficient": _metadata_float(
                cfg.recombination.recombination_coefficient
            ),
        },
        "boundary_modes": {
            "anode_ion_boundary": str(cfg.boundary.anode_ion_boundary),
            "anode_electron_boundary": str(cfg.boundary.anode_electron_boundary),
            "cathode_ion_boundary": str(cfg.boundary.cathode_ion_boundary),
            "cathode_electron_boundary": str(cfg.boundary.cathode_electron_boundary),
        },
        "circuit": {
            "circuit_type": str(cfg.circuit.circuit_type),
            "circuit_time_scheme": str(cfg.circuit.circuit_time_scheme),
            "R0": _metadata_float(cfg.circuit.R0),
            "C_s": _metadata_float(cfg.circuit.C_s),
            "L_s": _metadata_float(cfg.circuit.L_s),
            "C_p": _metadata_float(cfg.circuit.C_p),
            "L_p": _metadata_float(cfg.circuit.L_p),
            "R_m": _metadata_float(cfg.circuit.R_m),
            "C_ext": _metadata_float(getattr(cfg.circuit, "C_ext", 0.0)),
        },
        "emission": {
            "gamma": _metadata_float(cfg.emission.gamma),
            "anode_electron_induced_yield": _metadata_float(
                cfg.emission.anode_electron_induced_yield
            ),
            "enable_external_emission": bool(cfg.emission.enable_external_emission),
            "electrode_material_mode": str(cfg.emission.electrode_material_mode),
            "enable_anode_external_emission": bool(
                cfg.emission.enable_anode_external_emission
            ),
            "enable_cathode_external_emission": bool(
                cfg.emission.enable_cathode_external_emission
            ),
            "anode_mechanisms": {
                "constant_J": bool(cfg.emission.anode_enable_constant_J_emission),
                "fn": bool(cfg.emission.anode_enable_fn_emission),
                "mg": bool(cfg.emission.anode_enable_mg_emission),
                "rd": bool(cfg.emission.anode_enable_rd_emission),
                "quantum_pulse": bool(cfg.emission.anode_enable_quantum_pulse_emission),
            },
            "cathode_mechanisms": {
                "constant_J": bool(cfg.emission.cathode_enable_constant_J_emission),
                "fn": bool(cfg.emission.cathode_enable_fn_emission),
                "mg": bool(cfg.emission.cathode_enable_mg_emission),
                "rd": bool(cfg.emission.cathode_enable_rd_emission),
                "quantum_pulse": bool(
                    cfg.emission.cathode_enable_quantum_pulse_emission
                ),
            },
            "shared_quantum_pulse": {
                "laser_t0": _metadata_float(cfg.emission.shared_laser_t0),
                "laser_U_J": _metadata_float(cfg.emission.shared_laser_U_J),
                "laser_tau_p_s": _metadata_float(cfg.emission.shared_laser_tau_p_s),
                "laser_theta_deg": _metadata_float(cfg.emission.shared_laser_theta_deg),
                "emission_lambda_m": _metadata_float(
                    cfg.emission.shared_emission_lambda_m
                ),
                "laser_wx_m": _metadata_float(cfg.emission.shared_laser_wx_m),
                "laser_wy_m": _metadata_float(cfg.emission.shared_laser_wy_m),
            },
        },
        "current_decomposition_outputs": [
            "I_transport_plasma_mm.dat",
            "I_transport_circuit_mm.dat",
            "I_emission_circuit_mm.dat",
            "I_emission_area_mm.dat",
            "I_displacement_gap_mm.dat",
        ],
    }
    for key in (
        "table_path",
        "table_time_column",
        "table_voltage_column",
        "table_time_scale",
        "table_time_offset",
        "table_voltage_scale",
        "table_voltage_offset",
    ):
        if hasattr(cfg.waveform, key):
            payload[key] = getattr(cfg.waveform, key)
    if adaptive_stats:
        payload["adaptive_stats"] = adaptive_stats
    if hotloop_stats:
        payload["hotloop_stats"] = hotloop_stats
    if bc_poisson_picard_stats:
        payload["bc_poisson_picard_stats"] = bc_poisson_picard_stats
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



# ============================================================
# Snapshot writing helper
# ============================================================


def write_snapshot(
    k: int,
    *,
    # destinations (preallocated memmaps or arrays)
    n_i_sampled: np.ndarray,
    n_e_sampled: np.ndarray,
    phi_sampled: np.ndarray,
    E_sampled: np.ndarray,
    # current fields
    ni: np.ndarray,
    ne: np.ndarray,
    phi: np.ndarray,
    E: np.ndarray,
    # optional intermediates
    log_intermediate: bool = False,
    Gamma_i_sampled: np.ndarray | None = None,
    Gamma_e_sampled: np.ndarray | None = None,
    townsend_alpha_sampled: np.ndarray | None = None,
    nu_i_sampled: np.ndarray | None = None,
    S_ion_sampled: np.ndarray | None = None,
    S_sampled: np.ndarray | None = None,
    mu_e_sampled: np.ndarray | None = None,
    D_e_sampled: np.ndarray | None = None,
    mu_i_sampled: np.ndarray | None = None,
    D_i_sampled: np.ndarray | None = None,
    Gamma_i: np.ndarray | None = None,
    Gamma_e: np.ndarray | None = None,
    townsend_alpha: np.ndarray | None = None,
    nu: np.ndarray | None = None,
    S_ion: np.ndarray | None = None,
    S: np.ndarray | None = None,
    mu_e: np.ndarray | None = None,
    D_e: np.ndarray | None = None,
    mu_i: np.ndarray | None = None,
    D_i: np.ndarray | None = None,
) -> None:
    """
    Store the current plasma state (and optionally diagnostic fields)
    into the k-th snapshot slot.

    This is a thin, explicit writer: you pass in the "current" 1D
    profiles (ni, ne, phi, E, and optionally Gamma_i, Gamma_e,
    townsend_alpha, nu, S),
    along with the preallocated (Nsave, Nx) destination arrays, and
    it writes a single row (index k) into each output array.

    Parameters
    ----------
    k : int
        Snapshot index along the first axis of the sampled arrays
        (0 <= k < n_i_sampled.shape[0]).
    n_i_sampled, n_e_sampled, phi_sampled, E_sampled : np.ndarray
        Preallocated (Nsave, Nx) arrays or memmaps where ion density,
        electron density, potential, and electric field snapshots are stored.
    ni, ne, phi, E : np.ndarray
        Current 1D profiles (shape (Nx,)) of ion density, electron density,
        potential, and electric field to be written.
    log_intermediate : bool, optional
        If True, also write diagnostic quantities (fluxes, townsend_alpha,
        nu_i, S_ion, S) to the corresponding *_sampled arrays. Default is False.
    Gamma_i_sampled, Gamma_e_sampled, townsend_alpha_sampled, nu_i_sampled, S_ion_sampled, S_sampled : np.ndarray or None
        (Nsave, Nx) diagnostic snapshot arrays (required if
        log_intermediate=True).
    mu_e_sampled, D_e_sampled : np.ndarray or None
        Optional (Nsave, Nx) sampled transport diagnostic arrays.
    Gamma_i, Gamma_e, townsend_alpha, nu, S_ion, S : np.ndarray or None
        Current 1D diagnostic profiles (shape (Nx,)) to be stored when
        log_intermediate=True.
    mu_e, D_e : np.ndarray or None
        Optional current 1D transport profiles (shape (Nx,)).

    Notes
    -----
    This function does not call `.flush()` on the memmaps; the main
    driver can decide how often to flush (e.g., at the end of the run
    or every few thousand steps).
    """
    # Basic bounds check for safety
    if not 0 <= k < n_i_sampled.shape[0]:
        raise IndexError(f"snapshot index k={k} out of range")

    # --- Primary fields ---
    n_i_sampled[k, :] = ni
    n_e_sampled[k, :] = ne
    phi_sampled[k, :] = phi
    E_sampled[k, :] = E

    # --- Optional diagnostics ---
    if log_intermediate:
        # Base diagnostics must always exist if logging is on
        if not all(
            x is not None
            for x in (Gamma_i_sampled, Gamma_e_sampled, townsend_alpha_sampled, nu_i_sampled,
                      Gamma_i,   Gamma_e,   townsend_alpha,   nu)
        ):
            raise ValueError("log_intermediate=True but some core arrays/inputs are None.")

        Gamma_i_sampled[k, :] = Gamma_i
        Gamma_e_sampled[k, :] = Gamma_e
        townsend_alpha_sampled[k, :] = townsend_alpha
        nu_i_sampled[k, :]    = nu
        if S_ion_sampled is not None and S_ion is not None:
            S_ion_sampled[k, :] = S_ion

        # Common plasma source.
        if S_sampled is not None and S is not None:
            S_sampled[k, :] = S

        if mu_e_sampled is not None and mu_e is not None:
            mu_e_sampled[k, :] = mu_e
        if D_e_sampled is not None and D_e is not None:
            D_e_sampled[k, :] = D_e
        if mu_i_sampled is not None and mu_i is not None:
            mu_i_sampled[k, :] = mu_i
        if D_i_sampled is not None and D_i is not None:
            D_i_sampled[k, :] = D_i
