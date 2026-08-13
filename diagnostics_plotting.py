"""
diagnostics_plotting.py

Plotting utilities for PASCHEN-1D saved-run diagnostics.

This module contains both the original quick-look plotting helpers and the
array-level plotting backend used by the public diagnostics notebooks. It can
plot scalar time histories, time-dependent spatial summaries, instantaneous
spatial profiles, time/cycle-averaged spatial profiles, grouped diagnostics,
and derived sheath diagnostics.

All plotting assumes SI units on input. User-facing helpers expose scale/unit
knobs for common display units such as ns/us, mm/cm, microampere currents, and
field magnitudes in kV/cm.

For publication-style figures, call `set_notebook_plot_style(...)` or
`set_publication_style(...)` once at the start of a notebook or script.
"""

from pathlib import Path
from typing import Callable, Sequence
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


CURRENT_TEMPORAL_QUANTITIES = {
    "I_discharge",
    "I_transport_plasma",
    "I_transport_circuit",
    "I_emission_circuit",
    "I_emission_area",
    "I_displacement_gap",
}


# ============================================================
# Global style helper for publication-quality figures
# ============================================================

def set_publication_style(
    fontsize: float = 10.0,
    usetex: bool = False,
) -> None:
    """
    Configure matplotlib rcParams for publication-quality figures.

    Call this once at the beginning of your analysis/plotting script, e.g.:

        from diagnostics_plotting import set_publication_style
        set_publication_style(fontsize=10, usetex=False)

    Parameters
    ----------
    fontsize : float, optional
        Base font size for labels, ticks, and legends. Typical journal
        figures look good with 8–12 pt depending on the column width.
    usetex : bool, optional
        If True, use LaTeX for text rendering (requires a LaTeX
        installation). If False (default), use matplotlib's internal
        mathtext engine.
    """
    mpl.rcParams.update({
        # Font / text
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize * 0.9,
        "ytick.labelsize": fontsize * 0.9,
        "legend.fontsize": fontsize * 0.9,
        "text.usetex": usetex,
        "font.family": "serif" if usetex else "sans-serif",

        # Lines and axes
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,

        # Figure / save
        "figure.dpi": 150,
        "savefig.dpi": 600,        # high-res export
        "savefig.bbox": "tight",
    })


# ============================================================
# Drift-CFL diagnostic figure
# ============================================================

def plot_cfl_time_history(
    time: np.ndarray,
    c_cfl: np.ndarray,
    savepath: str | None = None,
) -> None:
    """
    Plot the drift-CFL number as a function of time.

    Parameters
    ----------
    time : np.ndarray
        Time array [s], shape (Nt,).
    c_cfl : np.ndarray
        Drift-CFL number at each time step (dimensionless), shape (Nt,).
    savepath : str or None, optional
        If provided, save the figure to this path (e.g. 'CFL_Number.pdf'
        or 'CFL_Number.png'). If None, the figure is not saved.

    Notes
    -----
    The x-axis is shown in nanoseconds; internally we multiply by 1e9.
    """
    fig, ax = plt.subplots(figsize=(3.2, 2.6))  # ~ single-column figure
    ax.plot(time * 1e9, c_cfl)
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Drift CFL number")
    ax.set_title("(a) Drift CFL")
    ax.grid(True)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600)
    plt.show()


# ============================================================
# Voltages and discharge current vs time
# ============================================================

def plot_voltages_and_current(
    time: np.ndarray,
    V_gap: np.ndarray,
    I_discharge: np.ndarray,
    V_app_func: Callable[[np.ndarray], np.ndarray],
    T: float,
    savepath: str | None = "voltage_current_last_run.pdf",
) -> None:
    """
    Plot applied vs gap voltage, and discharge current vs time.

    Parameters
    ----------
    time : np.ndarray
        Time array [s], shape (Nt,).
    V_gap : np.ndarray
        Gap voltage time history [V], shape (Nt,).
    I_discharge : np.ndarray
        Total discharge current time history [A], shape (Nt,).
    V_app_func : callable
        Applied voltage function V_app(t) [V]. Should accept a NumPy
        array of times (in seconds) and return an array of voltages of
        the same shape.
    T : float
        Total simulation time [s], used to set x-axis limits.
    savepath : str or None, optional
        If provided, path to save the figure. Using a '.pdf' extension
        is recommended for publication-quality vector output.
        If None, the figure is not saved.

    Notes
    -----
    - Panel (left): applied and gap voltages in kV vs time in ns.
    - Panel (right): discharge current in mA vs time in ns.
    - The first current sample (index 0) is often zero or ill-defined
      during setup, so the current trace omits time[0] and I_discharge[0].
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8))
    axs = axs.flatten()

    # Panel 1: Applied vs gap voltage (kV)
    axs[0].plot(
        time * 1e9,
        V_app_func(time) * 1e-3,
        label="Applied Voltage",
    )
    axs[0].plot(
        time * 1e9,
        V_gap * 1e-3,
        label="Gap Voltage",
    )
    axs[0].set_xlabel("Time [ns]")
    axs[0].set_ylabel("Voltage [kV]")
    axs[0].set_xlim(0.0, T * 1e9)
    axs[0].legend(frameon=False)
    axs[0].grid(True)

    # Panel 2: Discharge current (mA)
    axs[1].plot(time[1:] * 1e9, I_discharge[1:] * 1e3)
    axs[1].set_xlabel("Time [ns]")
    axs[1].set_ylabel("Discharge Current [mA]")
    axs[1].set_xlim(0.0, T * 1e9)
    axs[1].grid(True)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600)
    plt.show()


# ============================================================
# Spatial profiles (densities, potential, field)
# ============================================================

def plot_spatial_profiles(
    x: np.ndarray,
    ne: np.ndarray,
    ni: np.ndarray,
    phi: np.ndarray,
    E: np.ndarray,
    x_unit: str = "mm",
    savepath: str | None = None,
) -> None:
    """
    Plot n_e, n_i, phi, and E as functions of position.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid [m], shape (Nx,).
    ne : np.ndarray
        Electron density [m⁻³], shape (Nx,).
    ni : np.ndarray
        Ion density [m⁻³], shape (Nx,).
    phi : np.ndarray
        Electric potential [V], shape (Nx,).
    E : np.ndarray
        Electric field [V/m], shape (Nx,).
    x_unit : str, optional
        Unit for the x-axis in the plot: "mm", "cm", or "m".
        Default is "mm".
    savepath : str or None, optional
        If provided, save the figure to this path. A '.pdf' or '.eps'
        extension is recommended for journal-ready output. If None, the
        figure is not saved.

    Notes
    -----
    The function produces a 2x2 panel figure:
        (1,1) densities      – n_e and n_i
        (1,2) potential phi    – in volts
        (2,1) electric field – in V/m
        (2,2) intentionally left blank for possible future use.
    """
    # Convert x to the requested plotting unit
    if x_unit == "mm":
        x_plot = x * 1e3
        xlabel = "x [mm]"
    elif x_unit == "cm":
        x_plot = x * 1e2
        xlabel = "x [cm]"
    else:
        x_plot = x
        xlabel = "x [m]"

    fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.8))
    axs = axs.ravel()

    # Panel 1: Densities
    axs[0].plot(x_plot, ne, label="n_e")
    axs[0].plot(x_plot, ni, label="n_i", ls="--")
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("Density [m$^{-3}$]")
    axs[0].set_title("Densities")
    axs[0].legend(frameon=False)
    axs[0].grid(True)

    # Panel 2: Potential
    axs[1].plot(x_plot, phi)
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("Potential $\\phi$ [V]")
    axs[1].set_title("Potential")
    axs[1].grid(True)

    # Panel 3: Electric field
    axs[2].plot(x_plot, E)
    axs[2].set_xlabel(xlabel)
    axs[2].set_ylabel("Electric Field [V/m]")
    axs[2].set_title("Electric Field")
    axs[2].grid(True)

    # Panel 4 intentionally left blank to keep a compact 2x2 layout.
    axs[3].axis("off")

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600)
    plt.show()


def plot_selected_temporal_quantity(
    time: np.ndarray,
    quantity: str,
    values: np.ndarray,
    t_start: float | None = None,
    t_end: float | None = None,
    savepath: str | None = None,
) -> None:
    """
    Plot one temporal diagnostic quantity over a selected time window.

    Supported `quantity` values are configured from `config.py`:
    V_app, V_gap, I_discharge, current decomposition, cfl, diffusion_cfl,
    picard_iterations, adaptive_substeps, adaptive_dt_sub, adaptive_cfl_est,
    adaptive_diffusion_cfl_est.
    """
    if t_start is None:
        t_start = float(time[0])
    if t_end is None:
        t_end = float(time[-1])
    if t_end < t_start:
        t_start, t_end = t_end, t_start

    mask = (time >= t_start) & (time <= t_end)
    if not np.any(mask):
        print(f"Temporal diagnostic '{quantity}' skipped: empty time window.")
        return

    y = values[mask]
    x_ns = time[mask] * 1e9

    ylabel = quantity
    title = quantity
    if quantity in ("V_app", "V_node", "V_source", "V_gap"):
        y = y * 1e-3
        ylabel = "Voltage [kV]"
        title = f"{quantity} vs time"
    elif quantity in CURRENT_TEMPORAL_QUANTITIES:
        y = y * 1e3
        ylabel = "Current [mA]"
        title = f"{quantity} vs time"
    elif quantity == "cfl":
        ylabel = "Drift CFL number"
        title = "Drift CFL vs time"
    elif quantity == "diffusion_cfl":
        ylabel = "Diffusion stability number"
        title = "Diffusion stability vs time"
    elif quantity == "picard_iterations":
        ylabel = "Picard iterations per macro step"
        title = "Picard iterations vs time"
    elif quantity == "adaptive_substeps":
        ylabel = "Substeps per macro step"
        title = "Adaptive substeps vs time"
    elif quantity == "adaptive_dt_sub":
        ylabel = "Substep dt [s]"
        title = "Adaptive substep dt vs time"
    elif quantity == "adaptive_cfl_est":
        ylabel = "Estimated macro drift CFL"
        title = "Estimated macro drift CFL vs time"
    elif quantity == "adaptive_diffusion_cfl_est":
        ylabel = "Estimated macro diffusion stability"
        title = "Estimated macro diffusion stability vs time"

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(x_ns, y)
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600)
    plt.show()


def plot_selected_temporal_group(
    time: np.ndarray,
    quantities: tuple[str, ...],
    values_map: dict[str, np.ndarray],
    t_start: float | None = None,
    t_end: float | None = None,
    savepath: str | None = None,
) -> None:
    """
    Plot multiple temporal diagnostics in one axes (overlay).

    This is intended for same-unit overlays such as (V_app, V_gap).
    """
    if len(quantities) == 0:
        return

    if t_start is None:
        t_start = float(time[0])
    if t_end is None:
        t_end = float(time[-1])
    if t_end < t_start:
        t_start, t_end = t_end, t_start

    mask = (time >= t_start) & (time <= t_end)
    if not np.any(mask):
        print(f"Temporal diagnostic group {quantities} skipped: empty time window.")
        return

    x_ns = time[mask] * 1e9
    fig, ax = plt.subplots(figsize=(3.8, 2.9))

    unit_label = None
    label_map = {
        "V_app": "V_app",
        "V_node": "V_node",
        "V_source": "V_source",
        "V_gap": "V_gap",
        "I_discharge": "I_discharge",
        "I_transport_plasma": "I_transport_plasma",
        "I_transport_circuit": "I_transport_circuit",
        "I_emission_circuit": "I_emission_circuit",
        "I_emission_area": "I_emission_area",
        "I_displacement_gap": "I_displacement_gap",
        "cfl": "drift_cfl",
        "diffusion_cfl": "diffusion_cfl",
        "picard_iterations": "picard_iterations",
        "adaptive_substeps": "adaptive_substeps",
        "adaptive_dt_sub": "adaptive_dt_sub",
        "adaptive_cfl_est": "adaptive_drift_cfl_est",
        "adaptive_diffusion_cfl_est": "adaptive_diffusion_cfl_est",
    }

    for q in quantities:
        if q not in values_map:
            continue
        y = values_map[q][mask]
        if q in ("V_app", "V_node", "V_source", "V_gap"):
            y = y * 1e-3
            this_unit = "Voltage [kV]"
        elif q in CURRENT_TEMPORAL_QUANTITIES:
            y = y * 1e3
            this_unit = "Current [mA]"
        elif q == "adaptive_substeps":
            this_unit = "Substeps per macro step"
        elif q == "adaptive_dt_sub":
            this_unit = "Substep dt [s]"
        elif q == "adaptive_cfl_est":
            this_unit = "Estimated macro drift CFL"
        elif q == "adaptive_diffusion_cfl_est":
            this_unit = "Estimated macro diffusion stability"
        elif q == "cfl":
            this_unit = "Drift CFL number"
        elif q == "diffusion_cfl":
            this_unit = "Diffusion stability number"
        elif q == "picard_iterations":
            this_unit = "Picard iterations per macro step"
        else:
            this_unit = "Value"

        if unit_label is None:
            unit_label = this_unit
        elif unit_label != this_unit:
            unit_label = "Mixed units"

        ax.plot(x_ns, y, label=label_map.get(q, q))

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(unit_label if unit_label is not None else "Value")
    ax.set_title(" + ".join(quantities))
    ax.grid(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600)
    plt.show()


def plot_particle_inventory(
    time: np.ndarray,
    N_e: np.ndarray,
    N_i: np.ndarray,
    t_start: float | None = None,
    t_end: float | None = None,
    savepath: str | None = None,
) -> None:
    """
    Plot total electron/ion particle inventory versus time.

    Curves are normalized by their initial values for easier correctness checks.
    """
    if t_start is None:
        t_start = float(time[0])
    if t_end is None:
        t_end = float(time[-1])
    if t_end < t_start:
        t_start, t_end = t_end, t_start

    mask = (time >= t_start) & (time <= t_end)
    if not np.any(mask):
        print("Particle inventory diagnostic skipped: empty time window.")
        return

    Ne0 = float(N_e[0]) if N_e.size else 1.0
    Ni0 = float(N_i[0]) if N_i.size else 1.0
    if abs(Ne0) < 1e-30:
        Ne0 = 1.0
    if abs(Ni0) < 1e-30:
        Ni0 = 1.0

    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    ax.plot(time[mask] * 1e9, N_e[mask] / Ne0, label="N_e / N_e0")
    ax.plot(time[mask] * 1e9, N_i[mask] / Ni0, label="N_i / N_i0")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Normalized inventory")
    ax.set_title("Particle Inventory (saved snapshots)")
    ax.text(
        0.02,
        0.02,
        "Computed at snapshot times (save_every cadence)",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
    )
    ax.grid(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600)
    plt.show()


def plot_selected_spatial_quantity(
    x: np.ndarray,
    quantity: str,
    profiles: np.ndarray,
    sample_times: np.ndarray,
    x_unit: str = "mm",
    savepath: str | None = None,
) -> None:
    """
    Plot one spatial diagnostic quantity at one or more sample times.

    Parameters
    ----------
    x : np.ndarray
        Spatial grid [m], shape (Nx,).
    quantity : str
        Quantity label, e.g. 'ne', 'phi', 'Gamma_i', 'townsend_alpha'.
    profiles : np.ndarray
        Array with shape (Ns, Nx), one profile per sample time.
    sample_times : np.ndarray
        Times [s] associated with each profile, shape (Ns,). These are
        the actual sampled times used by the caller.
    x_unit : str, optional
        Unit for x-axis ('mm', 'cm', 'm').
    savepath : str | None, optional
        If provided, save figure to this path.
    """
    if profiles.ndim != 2 or profiles.shape[0] != sample_times.size:
        raise ValueError("profiles must have shape (Ns, Nx) and match sample_times.")

    if x_unit == "mm":
        x_plot = x * 1e3
        xlabel = "x [mm]"
    elif x_unit == "cm":
        x_plot = x * 1e2
        xlabel = "x [cm]"
    else:
        x_plot = x
        xlabel = "x [m]"

    ylabel_map = {
        "ne": "n_e [m$^{-3}$]",
        "ni": "n_i [m$^{-3}$]",
        "phi": "Potential [V]",
        "E": "Electric Field [V/m]",
        "Gamma_i": "Gamma_i [m$^{-2}$ s$^{-1}$]",
        "Gamma_e": "Gamma_e [m$^{-2}$ s$^{-1}$]",
        "townsend_alpha": "Townsend alpha [m$^{-1}$]",
        "nu_i": "nu_i [s$^{-1}$]",
        "S_ion": "S_ion [m$^{-3}$ s$^{-1}$]",
        "S": "S [m$^{-3}$ s$^{-1}$]",
        "mu_e": "mu_e [m$^2$ V$^{-1}$ s$^{-1}$]",
        "D_e": "D_e [m$^2$ s$^{-1}$]",
        "mu_i": "mu_i [m$^2$ V$^{-1}$ s$^{-1}$]",
        "D_i": "D_i [m$^2$ s$^{-1}$]",
    }

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for idx in range(sample_times.size):
        ax.plot(x_plot, profiles[idx], label=f"t={sample_times[idx]*1e9:.1f} ns")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_map.get(quantity, quantity))
    ax.set_title(f"{quantity} profiles")
    if sample_times.size <= 6:
        ax.legend(frameon=False)
    ax.grid(True)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600)
    plt.show()


def plot_selected_spatial_group(
    x: np.ndarray,
    quantities: tuple[str, ...],
    profiles_map: dict[str, np.ndarray],
    sample_times: np.ndarray,
    x_unit: str = "mm",
    savepath: str | None = None,
) -> None:
    """
    Plot multiple spatial quantities in one axes.

    If multiple sample times are requested, each curve is labeled by
    quantity and time.
    """
    if len(quantities) == 0:
        return

    if x_unit == "mm":
        x_plot = x * 1e3
        xlabel = "x [mm]"
    elif x_unit == "cm":
        x_plot = x * 1e2
        xlabel = "x [cm]"
    else:
        x_plot = x
        xlabel = "x [m]"

    ylabel_map = {
        "ne": "Density [m$^{-3}$]",
        "ni": "Density [m$^{-3}$]",
        "phi": "Potential [V]",
        "E": "Electric Field [V/m]",
        "Gamma_i": "Gamma [m$^{-2}$ s$^{-1}$]",
        "Gamma_e": "Gamma [m$^{-2}$ s$^{-1}$]",
        "townsend_alpha": "Townsend alpha [m$^{-1}$]",
        "nu_i": "nu_i [s$^{-1}$]",
        "S_ion": "S_ion [m$^{-3}$ s$^{-1}$]",
        "S": "S [m$^{-3}$ s$^{-1}$]",
        "mu_e": "mu_e [m$^2$ V$^{-1}$ s$^{-1}$]",
        "D_e": "D_e [m$^2$ s$^{-1}$]",
        "mu_i": "mu_i [m$^2$ V$^{-1}$ s$^{-1}$]",
        "D_i": "D_i [m$^2$ s$^{-1}$]",
    }

    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    ylabel = None
    for q in quantities:
        arr = profiles_map.get(q)
        if arr is None:
            continue
        this_ylabel = ylabel_map.get(q, q)
        if ylabel is None:
            ylabel = this_ylabel
        elif ylabel != this_ylabel:
            ylabel = "Mixed units"

        if arr.ndim != 2 or arr.shape[0] != sample_times.size:
            raise ValueError(f"profiles for {q} must have shape (Ns, Nx).")

        for idx in range(sample_times.size):
            if sample_times.size == 1:
                lbl = q
            else:
                lbl = f"{q}, t={sample_times[idx]*1e9:.1f} ns"
            ax.plot(x_plot, arr[idx], label=lbl)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else "Value")
    ax.set_title(" + ".join(quantities))
    if len(ax.lines) <= 10:
        ax.legend(frameon=False)
    ax.grid(True)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600)
    plt.show()


def plot_averaged_spatial_quantity(
    x: np.ndarray,
    quantity: str,
    profile: np.ndarray,
    averaging_label: str,
    x_unit: str = "mm",
    savepath: str | None = None,
) -> None:
    """
    Plot one time-averaged spatial diagnostic profile.

    The input profile is already averaged over the selected time window or
    over the requested number of RF cycles.
    """
    if x_unit == "mm":
        x_plot = x * 1e3
        xlabel = "x [mm]"
    elif x_unit == "cm":
        x_plot = x * 1e2
        xlabel = "x [cm]"
    else:
        x_plot = x
        xlabel = "x [m]"

    ylabel_map = {
        "ne": "n_e [m$^{-3}$]",
        "ni": "n_i [m$^{-3}$]",
        "phi": "Potential [V]",
        "E": "Electric Field [V/m]",
        "Gamma_i": "Gamma_i [m$^{-2}$ s$^{-1}$]",
        "Gamma_e": "Gamma_e [m$^{-2}$ s$^{-1}$]",
        "townsend_alpha": "Townsend alpha [m$^{-1}$]",
        "nu_i": "nu_i [s$^{-1}$]",
        "S_ion": "S_ion [m$^{-3}$ s$^{-1}$]",
        "S": "S [m$^{-3}$ s$^{-1}$]",
        "mu_e": "mu_e [m$^2$ V$^{-1}$ s$^{-1}$]",
        "D_e": "D_e [m$^2$ s$^{-1}$]",
        "mu_i": "mu_i [m$^2$ V$^{-1}$ s$^{-1}$]",
        "D_i": "D_i [m$^2$ s$^{-1}$]",
    }

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(x_plot, profile, label=quantity)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_map.get(quantity, quantity))
    ax.set_title(f"{quantity} averaged profile")
    ax.text(
        0.02,
        0.02,
        averaging_label,
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
    )
    ax.grid(True)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600)
    plt.show()


def plot_averaged_spatial_group(
    x: np.ndarray,
    quantities: tuple[str, ...],
    profiles_map: dict[str, np.ndarray],
    averaging_label: str,
    x_unit: str = "mm",
    savepath: str | None = None,
) -> None:
    """
    Plot multiple time-averaged spatial profiles in one axes.
    """
    if len(quantities) == 0:
        return

    if x_unit == "mm":
        x_plot = x * 1e3
        xlabel = "x [mm]"
    elif x_unit == "cm":
        x_plot = x * 1e2
        xlabel = "x [cm]"
    else:
        x_plot = x
        xlabel = "x [m]"

    ylabel_map = {
        "ne": "Density [m$^{-3}$]",
        "ni": "Density [m$^{-3}$]",
        "phi": "Potential [V]",
        "E": "Electric Field [V/m]",
        "Gamma_i": "Gamma [m$^{-2}$ s$^{-1}$]",
        "Gamma_e": "Gamma [m$^{-2}$ s$^{-1}$]",
        "townsend_alpha": "Townsend alpha [m$^{-1}$]",
        "nu_i": "nu_i [s$^{-1}$]",
        "S_ion": "S_ion [m$^{-3}$ s$^{-1}$]",
        "S": "S [m$^{-3}$ s$^{-1}$]",
        "mu_e": "mu_e [m$^2$ V$^{-1}$ s$^{-1}$]",
        "D_e": "D_e [m$^2$ s$^{-1}$]",
        "mu_i": "mu_i [m$^2$ V$^{-1}$ s$^{-1}$]",
        "D_i": "D_i [m$^2$ s$^{-1}$]",
    }

    fig, ax = plt.subplots(figsize=(4.3, 3.1))
    ylabel = None
    for q in quantities:
        if q not in profiles_map:
            continue
        this_ylabel = ylabel_map.get(q, q)
        if ylabel is None:
            ylabel = this_ylabel
        elif ylabel != this_ylabel:
            ylabel = "Mixed units"
        ax.plot(x_plot, profiles_map[q], label=q)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else "Value")
    ax.set_title(" + ".join(quantities))
    ax.text(
        0.02,
        0.02,
        averaging_label,
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.8,
    )
    ax.grid(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600)
    plt.show()


# ============================================================
# Shared flexible plotting backend for quick-look and notebooks
# ============================================================

DEFAULT_YLABELS = {
    "V_app": r"$V_\mathrm{app}$ [V]",
    "V_gap": r"$V_\mathrm{gap}$ [V]",
    "V_node": r"$V_\mathrm{node}$ [V]",
    "V_source": r"$V_\mathrm{source}$ [V]",
    "I_discharge": r"$I_\mathrm{discharge}$ [A]",
    "I_transport_plasma": r"$I_\mathrm{transport,plasma}$ [A]",
    "I_transport_circuit": r"$I_\mathrm{transport,circuit}$ [A]",
    "I_emission_circuit": r"$I_\mathrm{emission,circuit}$ [A]",
    "I_emission_area": r"$I_\mathrm{emission,area}$ [A]",
    "I_displacement_gap": r"$I_\mathrm{displacement,gap}$ [A]",
    "cfl": "Drift CFL number",
    "diffusion_cfl": "Diffusion stability number",
    "picard_iterations": "Picard iterations",
    "adaptive_substeps": "Adaptive substeps",
    "adaptive_dt_sub": r"Adaptive substep $\Delta t$ [s]",
    "adaptive_cfl_est": "Estimated macro drift CFL",
    "adaptive_diffusion_cfl_est": "Estimated macro diffusion stability",
    "ne": r"$n_e$ [m$^{-3}$]",
    "ni": r"$n_i$ [m$^{-3}$]",
    "phi": r"$\phi$ [V]",
    "E": r"$E$ [V/m]",
    "Gamma_i": r"$\Gamma_i$ [m$^{-2}$ s$^{-1}$]",
    "Gamma_e": r"$\Gamma_e$ [m$^{-2}$ s$^{-1}$]",
    "townsend_alpha": r"$\alpha$ [m$^{-1}$]",
    "nu_i": r"$\nu_i$ [s$^{-1}$]",
    "S_ion": r"$S_\mathrm{ion}$ [m$^{-3}$ s$^{-1}$]",
    "S": r"$S$ [m$^{-3}$ s$^{-1}$]",
    "mu_e": r"$\mu_e$ [m$^2$ V$^{-1}$ s$^{-1}$]",
    "D_e": r"$D_e$ [m$^2$ s$^{-1}$]",
    "mu_i": r"$\mu_i$ [m$^2$ V$^{-1}$ s$^{-1}$]",
    "D_i": r"$D_i$ [m$^2$ s$^{-1}$]",
}

DISPLAY_LABELS = {
    "V_app": r"$V_\mathrm{app}$",
    "V_gap": r"$V_\mathrm{gap}$",
    "V_node": r"$V_\mathrm{node}$",
    "V_source": r"$V_\mathrm{source}$",
    "I_discharge": r"$I_\mathrm{discharge}$",
    "I_transport_plasma": r"$I_\mathrm{transport,plasma}$",
    "I_transport_circuit": r"$I_\mathrm{transport,circuit}$",
    "I_emission_circuit": r"$I_\mathrm{emission,circuit}$",
    "I_emission_area": r"$I_\mathrm{emission,area}$",
    "I_displacement_gap": r"$I_\mathrm{displacement,gap}$",
    "cfl": "Drift CFL",
    "diffusion_cfl": "Diffusion stability",
    "picard_iterations": "Picard iterations",
    "adaptive_substeps": "Adaptive substeps",
    "adaptive_dt_sub": r"Adaptive $\Delta t_\mathrm{sub}$",
    "adaptive_cfl_est": "Adaptive drift-CFL estimate",
    "adaptive_diffusion_cfl_est": "Adaptive diffusion-stability estimate",
    "ne": r"$n_e$",
    "ni": r"$n_i$",
    "phi": r"$\phi$",
    "E": r"$E$",
    "Gamma_i": r"$\Gamma_i$",
    "Gamma_e": r"$\Gamma_e$",
    "townsend_alpha": r"$\alpha$",
    "nu_i": r"$\nu_i$",
    "S_ion": r"$S_\mathrm{ion}$",
    "S": r"$S$",
    "mu_e": r"$\mu_e$",
    "D_e": r"$D_e$",
    "mu_i": r"$\mu_i$",
    "D_i": r"$D_i$",
    "eta": r"$\eta$",
    "quasineutrality_metric": r"$\eta$",
    "rho": r"$\rho$",
    "abs_E": r"$|E|$",
}

SHEATH_DISPLAY_LABELS = {
    "edge_x_m": r"$x_s$",
    "width_m": r"$s$",
    "voltage_drop_abs_V": r"$|\Delta\phi_s|$",
    "peak_abs_E_V_m": r"$E_{s,\max}$",
    "space_charge_sigma_C_m2": r"$\sigma_s$",
    "space_charge_Q_C": r"$Q_s$",
}

SHEATH_DEFAULT_YLABELS = {
    "edge_x_m": r"$x_s$ [m]",
    "width_m": r"$s$ [m]",
    "voltage_drop_abs_V": r"$|\Delta\phi_s|$ [V]",
    "peak_abs_E_V_m": r"$E_{s,\max}$ [V/m]",
    "space_charge_sigma_C_m2": r"$\sigma_s$ [C/m$^2$]",
    "space_charge_Q_C": r"$Q_s$ [C]",
}


def display_label(name: str) -> str:
    """Return a Matplotlib-mathtext display label for a diagnostic name."""
    return DISPLAY_LABELS.get(name, name)


def sheath_metric_label(metric: str) -> str:
    """Return a Matplotlib-mathtext display label for a sheath metric."""
    return SHEATH_DISPLAY_LABELS.get(metric, metric)


def title_from_quantity(quantity: str, *, suffix: str = "") -> str:
    """Build a title from a diagnostic name while preserving math formatting."""
    return f"{display_label(quantity)}{suffix}"


def title_from_sheath_metric(metric: str, *, prefix: str = "") -> str:
    """Build a title from a sheath metric name while preserving math formatting."""
    return f"{prefix}{sheath_metric_label(metric)}"


def resolved_quantity_title(title, quantity: str, *, default_suffix: str = "") -> str:
    """Use a math label when the caller passes an internal diagnostic name."""
    default = title_from_quantity(quantity, suffix=default_suffix)
    if title is None:
        return default
    plain_defaults = {
        quantity,
        f"{quantity}{default_suffix}",
        f"{quantity} profiles",
        f"{quantity} averaged profile",
    }
    if title in plain_defaults:
        return default
    return title


def resolved_sheath_title(title, metric: str, *, prefix: str = "") -> str:
    """Use a math label when the caller passes an internal sheath metric name."""
    default = title_from_sheath_metric(metric, prefix=prefix)
    if title is None:
        return default
    plain_defaults = {metric, f"{prefix}{metric}"}
    if title in plain_defaults:
        return default
    return title


def resolved_context_title(title, profile_quantity: str) -> str:
    """Resolve sheath-context titles that may contain raw diagnostic names."""
    default = f"Sheath context: {display_label(profile_quantity)}"
    if title is None:
        return default
    prefix = "Sheath context: "
    if title == f"{prefix}{profile_quantity}":
        return default
    if title.startswith(prefix):
        raw_name = title[len(prefix):]
        if raw_name in SHEATH_DISPLAY_LABELS:
            return f"{prefix}{sheath_metric_label(raw_name)}"
        if raw_name in DISPLAY_LABELS:
            return f"{prefix}{display_label(raw_name)}"
    return title


def set_notebook_plot_style(
    *,
    font_size: float = 11,
    figure_dpi: int = 150,
    savefig_dpi: int = 600,
) -> None:
    """Set conservative publication-oriented Matplotlib defaults."""
    plt.rcParams.update(
        {
            "savefig.dpi": savefig_dpi,
            "figure.dpi": figure_dpi,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "legend.fontsize": max(font_size - 1, 1),
            "xtick.labelsize": max(font_size - 1, 1),
            "ytick.labelsize": max(font_size - 1, 1),
            "axes.linewidth": 1.0,
        }
    )


def apply_value_mode(y, mode: str):
    """Apply common signed/absolute transformations to plotted values."""
    y = np.asarray(y, dtype=np.float64)
    if mode == "raw":
        return y
    if mode == "abs":
        return np.abs(y)
    if mode == "positive":
        return np.maximum(y, 0.0)
    if mode == "negative_abs":
        return np.maximum(-y, 0.0)
    raise ValueError(f"Unknown value_mode: {mode}")


def window_mask(t, start=None, end=None) -> np.ndarray:
    """Return a boolean mask for an optional time window."""
    t = np.asarray(t, dtype=np.float64)
    mask = np.ones_like(t, dtype=bool)
    if start is not None:
        mask &= t >= float(start)
    if end is not None:
        mask &= t <= float(end)
    return mask


def reduce_xy_for_plot(xv, yv, max_points=30000):
    """Downsample dense lines while preserving local minima and maxima."""
    xv = np.asarray(xv)
    yv = np.asarray(yv)
    n = len(xv)
    if max_points is None or n <= max_points:
        return xv, yv
    bins = max(2, int(max_points) // 2)
    edges = np.linspace(0, n, bins + 1, dtype=np.int64)
    keep: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        seg = yv[lo:hi]
        finite = np.isfinite(seg)
        if not np.any(finite):
            keep.append(lo)
            continue
        local = np.flatnonzero(finite)
        vals = seg[finite]
        keep.append(lo + int(local[np.argmin(vals)]))
        keep.append(lo + int(local[np.argmax(vals)]))
    keep_arr = np.array(sorted(set(keep)), dtype=np.int64)
    return xv[keep_arr], yv[keep_arr]


def scale_time(t, unit: str) -> tuple[np.ndarray, str]:
    """Scale time array and return axis label."""
    if unit == "s":
        return np.asarray(t), "Time [s]"
    if unit == "ms":
        return np.asarray(t) * 1.0e3, "Time [ms]"
    if unit == "us":
        return np.asarray(t) * 1.0e6, r"Time [$\mu$s]"
    if unit == "ns":
        return np.asarray(t) * 1.0e9, "Time [ns]"
    raise ValueError(f"Unknown time unit: {unit}")


def scale_x(x, unit: str) -> tuple[np.ndarray, str]:
    """Scale position array and return axis label."""
    if unit == "m":
        return np.asarray(x), "x [m]"
    if unit == "cm":
        return np.asarray(x) * 1.0e2, "x [cm]"
    if unit == "mm":
        return np.asarray(x) * 1.0e3, "x [mm]"
    raise ValueError(f"Unknown x unit: {unit}")


def show_figure() -> None:
    """Show figures in notebooks while staying quiet in headless script tests."""
    if str(plt.get_backend()).lower() != "agg":
        plt.show()


def _apply_log_filter(y, yscale: str):
    if yscale == "log":
        return np.where(y > 0.0, y, np.nan)
    return y


def _save_if_requested(fig, savepath=None) -> None:
    if savepath is None:
        return
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, bbox_inches="tight")


def plot_temporal_quantity_from_arrays(
    time: np.ndarray,
    values: np.ndarray,
    quantity: str,
    *,
    t_start=None,
    t_end=None,
    t_unit="us",
    value_mode="raw",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    color="C0",
    linestyle="-",
    linewidth=2.0,
    marker=None,
    label=None,
    grid=True,
    legend=True,
    legend_loc="best",
    max_points=30000,
    ylim=None,
    xlim=None,
    savepath=None,
    show=True,
):
    """Plot one temporal diagnostic from arrays."""
    mask = window_mask(time, t_start, t_end)
    tp, xlabel = scale_time(np.asarray(time)[mask], t_unit)
    yp = apply_value_mode(np.asarray(values)[mask], value_mode) * y_scale_factor
    yp = _apply_log_filter(yp, yscale)
    tp, yp = reduce_xy_for_plot(tp, yp, max_points=max_points)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        tp,
        yp,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        label=label or display_label(quantity),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or DEFAULT_YLABELS.get(quantity, "Value"))
    ax.set_title(resolved_quantity_title(title, quantity))
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, alpha=0.35)
    if legend:
        ax.legend(frameon=True, loc=legend_loc)
    fig.tight_layout()
    _save_if_requested(fig, savepath)
    if show:
        show_figure()
    return fig, ax


def plot_temporal_group_from_series(
    series: Sequence[dict],
    *,
    t_start=None,
    t_end=None,
    t_unit="us",
    ylabel="Value",
    title="Grouped temporal diagnostics",
    xscale="linear",
    yscale="linear",
    figsize=(5.4, 3.6),
    linewidth=2.0,
    grid=True,
    legend=True,
    legend_loc="best",
    max_points=30000,
    xlim=None,
    ylim=None,
    savepath=None,
    show=True,
):
    """Plot a group of temporal series from dictionaries with time/value arrays."""
    fig, ax = plt.subplots(figsize=figsize)
    xlabel = "Time [s]"
    for item in series:
        t = np.asarray(item["time"], dtype=np.float64)
        y = np.asarray(item["values"], dtype=np.float64)
        q = item.get("quantity", "value")
        value_mode = item.get("value_mode", "raw")
        scale = float(item.get("scale", 1.0))
        mask = window_mask(t, t_start, t_end)
        tp, xlabel = scale_time(t[mask], t_unit)
        yp = apply_value_mode(y[mask], value_mode) * scale
        yp = _apply_log_filter(yp, yscale)
        tp, yp = reduce_xy_for_plot(tp, yp, max_points=max_points)
        ax.plot(
            tp,
            yp,
            color=item.get("color", None),
            linestyle=item.get("linestyle", "-"),
            linewidth=linewidth,
            marker=item.get("marker", None),
            label=item.get("label", display_label(q)),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, alpha=0.35)
    if legend:
        ax.legend(frameon=True, loc=legend_loc)
    fig.tight_layout()
    _save_if_requested(fig, savepath)
    if show:
        show_figure()
    return fig, ax


def plot_spatial_temporal_summary_from_arrays(
    time: np.ndarray,
    summary_values: np.ndarray,
    quantity: str,
    *,
    summary="max",
    t_start=None,
    t_end=None,
    t_unit="us",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    color="C0",
    linestyle="-",
    linewidth=2.0,
    marker=None,
    grid=True,
    legend=True,
    legend_loc="best",
    max_points=30000,
    ylim=None,
    xlim=None,
    savepath=None,
    show=True,
):
    """Plot a temporal summary derived from sampled spatial snapshots."""
    mask = window_mask(time, t_start, t_end)
    tp, xlabel = scale_time(np.asarray(time)[mask], t_unit)
    yp = np.asarray(summary_values, dtype=np.float64)[mask] * y_scale_factor
    yp = _apply_log_filter(yp, yscale)
    tp, yp = reduce_xy_for_plot(tp, yp, max_points=max_points)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        tp,
        yp,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        label=f"{display_label(quantity)} {summary}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or DEFAULT_YLABELS.get(quantity, "Value"))
    summary_title = f"{title_from_quantity(quantity)}: {summary} vs time"
    ax.set_title(summary_title if title is None or title == f"{quantity}: {summary} vs time" else title)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, alpha=0.35)
    if legend:
        ax.legend(frameon=True, loc=legend_loc)
    fig.tight_layout()
    _save_if_requested(fig, savepath)
    if show:
        show_figure()
    return fig, ax


def plot_spatial_profiles_from_arrays(
    x: np.ndarray,
    profiles: np.ndarray,
    sample_times: np.ndarray,
    quantity: str,
    *,
    x_unit="cm",
    time_unit_for_labels="us",
    value_mode="raw",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    cmap="viridis",
    linewidth=2.0,
    linestyle="-",
    grid=True,
    legend=True,
    legend_loc="best",
    colorbar=True,
    ylim=None,
    xlim=None,
    savepath=None,
    show=True,
):
    """Plot one spatial diagnostic at selected snapshot times."""
    profiles = np.asarray(profiles)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    if profiles.ndim == 1:
        profiles = profiles[np.newaxis, :]
    xp, xlabel = scale_x(x, x_unit)
    st_plot, st_label = scale_time(sample_times, time_unit_for_labels)

    fig, ax = plt.subplots(figsize=figsize)
    if len(sample_times) > 1 and colorbar:
        norm = mpl.colors.Normalize(vmin=float(np.min(st_plot)), vmax=float(np.max(st_plot)))
        cmap_obj = plt.get_cmap(cmap)
        for row, t_scaled in zip(profiles, st_plot):
            yp = apply_value_mode(row, value_mode) * y_scale_factor
            yp = _apply_log_filter(yp, yscale)
            ax.plot(
                xp,
                yp,
                color=cmap_obj(norm(float(t_scaled))),
                linewidth=linewidth,
                linestyle=linestyle,
            )
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(st_label)
    else:
        for row, t_scaled in zip(profiles, st_plot):
            yp = apply_value_mode(row, value_mode) * y_scale_factor
            yp = _apply_log_filter(yp, yscale)
            ax.plot(
                xp,
                yp,
                linewidth=linewidth,
                linestyle=linestyle,
                label=f"{t_scaled:.4g} {time_unit_for_labels}",
            )
        if legend:
            ax.legend(frameon=True, loc=legend_loc)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or DEFAULT_YLABELS.get(quantity, "Value"))
    ax.set_title(resolved_quantity_title(title, quantity, default_suffix=" profiles"))
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, alpha=0.35)
    fig.tight_layout()
    _save_if_requested(fig, savepath)
    if show:
        show_figure()
    return fig, ax


def plot_spatial_group_from_profiles(
    x: np.ndarray,
    profiles: Sequence[dict],
    *,
    t_label=None,
    x_unit="cm",
    ylabel="Value",
    title="Grouped spatial snapshot",
    xscale="linear",
    yscale="linear",
    figsize=(5.4, 3.6),
    linewidth=2.0,
    cmap="viridis",
    colorbar=False,
    colorbar_label=None,
    grid=True,
    legend=True,
    legend_loc="best",
    show_time_annotation=True,
    xlim=None,
    ylim=None,
    savepath=None,
    show=True,
):
    """Plot a group of spatial profiles on one axes."""
    xp, xlabel = scale_x(x, x_unit)
    fig, ax = plt.subplots(figsize=figsize)

    use_colorbar = bool(colorbar) and any("time_scaled" in item for item in profiles)
    if use_colorbar:
        time_values = np.asarray(
            [item.get("time_scaled", np.nan) for item in profiles],
            dtype=np.float64,
        )
        finite_time = np.isfinite(time_values)
        if np.any(finite_time):
            vmin = float(np.nanmin(time_values[finite_time]))
            vmax = float(np.nanmax(time_values[finite_time]))
            if vmin == vmax:
                pad = 0.5 if vmin == 0.0 else 0.05 * abs(vmin)
                vmin -= pad
                vmax += pad
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = plt.get_cmap(cmap)
        else:
            use_colorbar = False

    seen_labels = set()
    for item in profiles:
        yp = apply_value_mode(item["values"], item.get("value_mode", "raw"))
        yp = yp * float(item.get("scale", 1.0))
        yp = _apply_log_filter(yp, yscale)
        q = item.get("quantity", "value")
        line_kwargs = {}
        if use_colorbar and np.isfinite(item.get("time_scaled", np.nan)):
            line_kwargs["color"] = cmap_obj(norm(float(item["time_scaled"])))
        label = item.get("label", display_label(q))
        plot_label = label if label not in seen_labels else "_nolegend_"
        seen_labels.add(label)
        ax.plot(
            xp,
            yp,
            linewidth=linewidth,
            linestyle=item.get("linestyle", "-"),
            label=plot_label,
            **line_kwargs,
        )

    if use_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(colorbar_label or "Time")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else (t_label or "Grouped spatial snapshot"))
    if show_time_annotation and t_label is not None:
        ax.text(
            0.02,
            0.98,
            t_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, alpha=0.35)
    if legend:
        ax.legend(frameon=True, loc=legend_loc)
    fig.tight_layout()
    _save_if_requested(fig, savepath)
    if show:
        show_figure()
    return fig, ax


def plot_averaged_spatial_profile_from_array(
    x: np.ndarray,
    profile: np.ndarray,
    quantity: str,
    *,
    averaging_label=None,
    x_unit="cm",
    value_mode="raw",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    color="C0",
    linewidth=2.0,
    linestyle="-",
    grid=True,
    legend=True,
    legend_loc="best",
    ylim=None,
    xlim=None,
    savepath=None,
    show=True,
):
    """Plot one averaged spatial diagnostic profile."""
    xp, xlabel = scale_x(x, x_unit)
    yp = apply_value_mode(profile, value_mode) * y_scale_factor
    yp = _apply_log_filter(yp, yscale)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xp, yp, color=color, linewidth=linewidth, linestyle=linestyle, label=display_label(quantity))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or DEFAULT_YLABELS.get(quantity, "Value"))
    ax.set_title(resolved_quantity_title(title, quantity, default_suffix=" averaged profile"))
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if averaging_label:
        ax.text(
            0.02,
            0.98,
            averaging_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    if grid:
        ax.grid(True, alpha=0.35)
    if legend:
        ax.legend(frameon=True, loc=legend_loc)
    fig.tight_layout()
    _save_if_requested(fig, savepath)
    if show:
        show_figure()
    return fig, ax


def plot_averaged_spatial_group_from_profiles(
    x: np.ndarray,
    profiles: Sequence[dict],
    *,
    averaging_label=None,
    x_unit="cm",
    ylabel="Value",
    title="Grouped averaged spatial profiles",
    xscale="linear",
    yscale="linear",
    figsize=(5.4, 3.6),
    linewidth=2.0,
    grid=True,
    legend=True,
    legend_loc="best",
    xlim=None,
    ylim=None,
    savepath=None,
    show=True,
):
    """Plot a group of averaged spatial profiles."""
    xp, xlabel = scale_x(x, x_unit)
    fig, ax = plt.subplots(figsize=figsize)
    for item in profiles:
        yp = apply_value_mode(item["values"], item.get("value_mode", "raw"))
        yp = yp * float(item.get("scale", 1.0))
        yp = _apply_log_filter(yp, yscale)
        q = item.get("quantity", "value")
        ax.plot(xp, yp, linewidth=linewidth, label=item.get("label", display_label(q)))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if averaging_label:
        ax.text(
            0.02,
            0.98,
            averaging_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    if grid:
        ax.grid(True, alpha=0.35)
    if legend:
        ax.legend(frameon=True, loc=legend_loc)
    fig.tight_layout()
    _save_if_requested(fig, savepath)
    if show:
        show_figure()
    return fig, ax


def plot_sheath_temporal_metric_from_arrays(
    time: np.ndarray,
    anode_values: np.ndarray,
    cathode_values: np.ndarray,
    metric: str,
    *,
    t_start=None,
    t_end=None,
    t_unit="us",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    linewidth=2.0,
    grid=True,
    legend=True,
    legend_loc="best",
    ylim=None,
    xlim=None,
    savepath=None,
    show=True,
):
    """Plot anode/cathode sheath metric histories."""
    time = np.asarray(time, dtype=np.float64)
    mask = window_mask(time, t_start, t_end)
    tp, xlabel = scale_time(time[mask], t_unit)

    fig, ax = plt.subplots(figsize=figsize)
    for side, values in (("anode", anode_values), ("cathode", cathode_values)):
        yp = np.asarray(values, dtype=np.float64)[mask] * y_scale_factor
        yp = _apply_log_filter(yp, yscale)
        ax.plot(tp, yp, linewidth=linewidth, label=side)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or SHEATH_DEFAULT_YLABELS.get(metric, sheath_metric_label(metric)))
    ax.set_title(resolved_sheath_title(title, metric, prefix="Sheath "))
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, alpha=0.35)
    if legend:
        ax.legend(frameon=True, loc=legend_loc)
    fig.tight_layout()
    _save_if_requested(fig, savepath)
    if show:
        show_figure()
    return fig, ax


def plot_sheath_spatial_context_from_arrays(
    x: np.ndarray,
    profiles: np.ndarray,
    sample_times: np.ndarray,
    profile_quantity: str,
    *,
    anode_edges=None,
    cathode_edges=None,
    x_unit="cm",
    time_unit_for_labels="us",
    value_mode="raw",
    y_scale_factor=1.0,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    figsize=(5.0, 3.5),
    cmap="viridis",
    linewidth=2.0,
    linestyle="-",
    grid=True,
    legend=True,
    legend_loc="best",
    colorbar=True,
    ylim=None,
    xlim=None,
    savepath=None,
    show=True,
):
    """Plot spatial profiles with optional inferred sheath-edge markers."""
    profiles = np.asarray(profiles)
    if profiles.ndim == 1:
        profiles = profiles[np.newaxis, :]
    sample_times = np.asarray(sample_times, dtype=np.float64)
    xp, xlabel = scale_x(x, x_unit)
    st_plot, st_label = scale_time(sample_times, time_unit_for_labels)
    x_scale = 1.0 if x_unit == "m" else (1.0e2 if x_unit == "cm" else 1.0e3)

    fig, ax = plt.subplots(figsize=figsize)
    if len(sample_times) > 1 and colorbar:
        norm = mpl.colors.Normalize(vmin=float(np.min(st_plot)), vmax=float(np.max(st_plot)))
        cmap_obj = plt.get_cmap(cmap)
        for i, (row, t_scaled) in enumerate(zip(profiles, st_plot)):
            color = cmap_obj(norm(float(t_scaled)))
            yp = apply_value_mode(row, value_mode) * y_scale_factor
            yp = _apply_log_filter(yp, yscale)
            ax.plot(xp, yp, color=color, linewidth=linewidth, linestyle=linestyle)
            if anode_edges is not None and np.isfinite(anode_edges[i]):
                ax.axvline(float(anode_edges[i]) * x_scale, color=color, linestyle=":", alpha=0.65)
            if cathode_edges is not None and np.isfinite(cathode_edges[i]):
                ax.axvline(float(cathode_edges[i]) * x_scale, color=color, linestyle="--", alpha=0.65)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(st_label)
    else:
        for i, (row, t_scaled) in enumerate(zip(profiles, st_plot)):
            yp = apply_value_mode(row, value_mode) * y_scale_factor
            yp = _apply_log_filter(yp, yscale)
            ax.plot(
                xp,
                yp,
                linewidth=linewidth,
                linestyle=linestyle,
                label=f"{t_scaled:.4g} {time_unit_for_labels}",
            )
            if anode_edges is not None and np.isfinite(anode_edges[i]):
                ax.axvline(float(anode_edges[i]) * x_scale, color="0.4", linestyle=":", alpha=0.65)
            if cathode_edges is not None and np.isfinite(cathode_edges[i]):
                ax.axvline(float(cathode_edges[i]) * x_scale, color="0.4", linestyle="--", alpha=0.65)
        if legend:
            ax.legend(frameon=True, loc=legend_loc)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or DEFAULT_YLABELS.get(profile_quantity, "Value"))
    ax.set_title(resolved_context_title(title, profile_quantity))
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, alpha=0.35)
    fig.tight_layout()
    _save_if_requested(fig, savepath)
    if show:
        show_figure()
    return fig, ax
