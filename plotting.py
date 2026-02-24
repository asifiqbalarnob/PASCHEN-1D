"""
plotting.py

Lightweight plotting utilities for the PASCHEN-1D drift-diffusion-Poisson code.

This module provides convenience functions to visualize:

1. CFL number vs time               → plot_cfl_time_history
2. Applied/gap voltages and current → plot_voltages_and_current
3. Spatial profiles                 → plot_spatial_profiles

All plotting is done with matplotlib and assumes SI units on input
(time in seconds, voltages in volts, currents in amperes, lengths in meters);
the routines convert to ns / kV / mA / mm or cm where appropriate.

For publication-quality figures:
    • Call `set_publication_style()` once at the start of your script.
    • Prefer vector formats (e.g. '.pdf', '.eps') for journal submission.
"""

from typing import Callable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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

        from plotting import set_publication_style
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
# CFL diagnostic figure
# ============================================================

def plot_cfl_time_history(
    time: np.ndarray,
    c_cfl: np.ndarray,
    savepath: str | None = None,
) -> None:
    """
    Plot the CFL number as a function of time.

    Parameters
    ----------
    time : np.ndarray
        Time array [s], shape (Nt,).
    c_cfl : np.ndarray
        CFL number at each time step (dimensionless), shape (Nt,).
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
    ax.set_ylabel("CFL number")
    ax.set_title("(a) CFL Number")
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
    V_app, V_gap, I_discharge, cfl.
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
    if quantity in ("V_app", "V_gap"):
        y = y * 1e-3
        ylabel = "Voltage [kV]"
        title = f"{quantity} vs time"
    elif quantity == "I_discharge":
        y = y * 1e3
        ylabel = "Current [mA]"
        title = "I_discharge vs time"
    elif quantity == "cfl":
        ylabel = "CFL number"
        title = "CFL vs time"

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
        "V_gap": "V_gap",
        "I_discharge": "I_discharge",
        "cfl": "cfl",
    }

    for q in quantities:
        if q not in values_map:
            continue
        y = values_map[q][mask]
        if q in ("V_app", "V_gap"):
            y = y * 1e-3
            this_unit = "Voltage [kV]"
        elif q == "I_discharge":
            y = y * 1e3
            this_unit = "Current [mA]"
        else:
            this_unit = "CFL number"

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
        "S": "S [m$^{-3}$ s$^{-1}$]",
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
        "S": "S [m$^{-3}$ s$^{-1}$]",
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
