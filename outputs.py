"""
outputs.py

I/O utilities for the PASCHEN-1D drift-diffusion-Poisson solver.

This module handles:

1. Creation of on-disk, memory-mapped arrays for:
   - Field snapshots (phi, E)
   - Species densities (n_e, n_i)
   - Optional diagnostics (Gamma_i, Gamma_e, townsend_alpha, nu_i, S)
   - Scalar time histories (V_gap, CFL, I_discharge)

2. A small dataclass `OutputHandles` that collects references to all
   memmapped arrays so the main driver can pass them around easily.

3. A `write_snapshot` helper that writes one snapshot of the current
   plasma state (and optional diagnostics) into the preallocated files.

The design is deliberately simple: all arrays are row-major with
shape (Nsave, Nx) for spatial snapshots or (Nt,) for scalars, using
np.float32 to keep files compact. Files are created under a directory
named after `cfg.run_name`.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np

from config import SimulationConfig


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
    S_sampled : np.ndarray or None
        Sampled source term (e.g., ionization − recombination) (optional).
    V_gap : np.ndarray
        Time history of plasma gap voltage.
    c_cfl : np.ndarray
        Time history of CFL diagnostic values.
    I_discharge : np.ndarray
        Time history of discharge current.
    """
    phi_sampled: np.ndarray
    E_sampled: np.ndarray
    n_e_sampled: np.ndarray
    n_i_sampled: np.ndarray
    Gamma_i_sampled: Optional[np.ndarray]
    Gamma_e_sampled: Optional[np.ndarray]
    townsend_alpha_sampled: Optional[np.ndarray]
    nu_i_sampled: Optional[np.ndarray]
    S_sampled: Optional[np.ndarray]
    V_gap: np.ndarray
    c_cfl: np.ndarray
    I_discharge: np.ndarray

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
       cfg.save_every.
    2. Creates a subdirectory named `cfg.run_name`.
    3. Creates zero-initialized memmap files for:
         - phi, E, n_e, n_i       (always)
         - Gamma_i, Gamma_e, townsend_alpha, nu_i, S
           (if cfg.log_intermediate is True)
         - V_gap, c_cfl, I_discharge (scalar time histories)
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
    SAVE_EVERY = cfg.save_every
    LOG_INTERMEDIATE = cfg.log_intermediate

    # Number of saved snapshots along time
    Nsave = int((Nt - 1) // SAVE_EVERY + 1)

    outdir = Path(cfg.run_name)
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
    S_path   = outdir / "S_sampled_mm.dat"


    # --- Scalar time histories ---
    Vgap_path = outdir / "Vgap_mm.dat"
    c_cfl_path = outdir / "c_cfl_mm.dat"
    Idis_path = outdir / "Idischarge_mm.dat"

    # --- Create files (zero-initialized) ---
    create_file(phi_path, (Nsave, Nx))
    create_file(E_path, (Nsave, Nx))
    create_file(ne_path, (Nsave, Nx))
    create_file(ni_path, (Nsave, Nx))

    if LOG_INTERMEDIATE:
        create_file(Gamma_i_path, (Nsave, Nx))
        create_file(Gamma_e_path, (Nsave, Nx))
        create_file(townsend_alpha_path, (Nsave, Nx))
        create_file(nu_i_path, (Nsave, Nx))
        create_file(S_path, (Nsave, Nx))


    create_file(Vgap_path, (Nt,))
    create_file(c_cfl_path, (Nt,))
    create_file(Idis_path, (Nt,))

    # --- Open memmaps ---
    phi_sampled = np.memmap(phi_path, mode="readwrite", dtype=np.float32, shape=(Nsave, Nx))
    E_sampled = np.memmap(E_path, mode="readwrite", dtype=np.float32, shape=(Nsave, Nx))
    n_e_sampled = np.memmap(ne_path, mode="readwrite", dtype=np.float32, shape=(Nsave, Nx))
    n_i_sampled = np.memmap(ni_path, mode="readwrite", dtype=np.float32, shape=(Nsave, Nx))

    if LOG_INTERMEDIATE:
        Gamma_i_sampled = np.memmap(Gamma_i_path, mode="readwrite", dtype=np.float32, shape=(Nsave, Nx))
        Gamma_e_sampled = np.memmap(Gamma_e_path, mode="readwrite", dtype=np.float32, shape=(Nsave, Nx))
        townsend_alpha_sampled   = np.memmap(townsend_alpha_path,   mode="readwrite", dtype=np.float32, shape=(Nsave, Nx))
        nu_i_sampled    = np.memmap(nu_i_path,    mode="readwrite", dtype=np.float32, shape=(Nsave, Nx))
        S_sampled       = np.memmap(S_path,       mode="readwrite", dtype=np.float32, shape=(Nsave, Nx))
    else:
        Gamma_i_sampled = Gamma_e_sampled = townsend_alpha_sampled = nu_i_sampled = S_sampled = None

    V_gap = np.memmap(Vgap_path, mode="readwrite", dtype=np.float32, shape=(Nt,))
    c_cfl = np.memmap(c_cfl_path, mode="readwrite", dtype=np.float32, shape=(Nt,))
    I_discharge = np.memmap(Idis_path, mode="readwrite", dtype=np.float32, shape=(Nt,))

    return OutputHandles(
        phi_sampled=phi_sampled,
        E_sampled=E_sampled,
        n_e_sampled=n_e_sampled,
        n_i_sampled=n_i_sampled,
        Gamma_i_sampled=Gamma_i_sampled,
        Gamma_e_sampled=Gamma_e_sampled,
        townsend_alpha_sampled=townsend_alpha_sampled,
        nu_i_sampled=nu_i_sampled,
        S_sampled=S_sampled,
        V_gap=V_gap,
        c_cfl=c_cfl,
        I_discharge=I_discharge,
    )


def write_run_metadata(
    cfg: SimulationConfig,
    *,
    Nt: int,
    Nx: int,
    dt: float,
    dx: float,
) -> None:
    """
    Write lightweight run metadata for post-processing/replotting.

    The metadata is stored as JSON in:
        <run_name>/run_metadata.json
    """
    outdir = Path(cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "run_metadata.json"

    payload = {
        "run_name": cfg.run_name,
        "Nt": int(Nt),
        "Nx": int(Nx),
        "T_total": float(cfg.T_total),
        "L": float(cfg.L),
        "A": float(cfg.A),
        "save_every": int(cfg.save_every),
        "dt": float(dt),
        "dx": float(dx),
        "waveform_type": str(cfg.waveform_type),
        "V_peak": float(cfg.V_peak),
        "tV_start": float(cfg.tV_start),
        "tV_end": float(cfg.tV_end),
        "tau": float(cfg.tau),
        "t_peak": float(cfg.t_peak),
        "f_rf": float(cfg.f_rf),
        "V_dc": float(cfg.V_dc),
        "phi_rf": float(cfg.phi_rf),
    }
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
    S_sampled: np.ndarray | None = None,
    Gamma_i: np.ndarray | None = None,
    Gamma_e: np.ndarray | None = None,
    townsend_alpha: np.ndarray | None = None,
    nu: np.ndarray | None = None,
    S: np.ndarray | None = None,
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
        nu_i, S) to the corresponding *_sampled arrays. Default is False.
    Gamma_i_sampled, Gamma_e_sampled, townsend_alpha_sampled, nu_i_sampled, S_sampled : np.ndarray or None
        (Nsave, Nx) diagnostic snapshot arrays (required if
        log_intermediate=True).
    Gamma_i, Gamma_e, townsend_alpha, nu, S : np.ndarray or None
        Current 1D diagnostic profiles (shape (Nx,)) to be stored when
        log_intermediate=True.

    Raises
    ------
    AssertionError
        If k is out of range or if log_intermediate=True but any of the
        required diagnostic arrays/values is missing.

    Notes
    -----
    This function does not call `.flush()` on the memmaps; the main
    driver can decide how often to flush (e.g., at the end of the run
    or every few thousand steps).
    """
    # Basic bounds check for safety
    assert 0 <= k < n_i_sampled.shape[0], f"snapshot index k={k} out of range"

    # --- Primary fields ---
    n_i_sampled[k, :] = ni
    n_e_sampled[k, :] = ne
    phi_sampled[k, :] = phi
    E_sampled[k, :] = E

    # --- Optional diagnostics ---
    if log_intermediate:
        # Base diagnostics must always exist if logging is on
        assert all(
            x is not None
            for x in (Gamma_i_sampled, Gamma_e_sampled, townsend_alpha_sampled, nu_i_sampled,
                      Gamma_i,   Gamma_e,   townsend_alpha,   nu)
        ), "log_intermediate=True but some core arrays/inputs are None."

        Gamma_i_sampled[k, :] = Gamma_i
        Gamma_e_sampled[k, :] = Gamma_e
        townsend_alpha_sampled[k, :] = townsend_alpha
        nu_i_sampled[k, :]    = nu

        # Common plasma source.
        if S_sampled is not None and S is not None:
            S_sampled[k, :] = S
