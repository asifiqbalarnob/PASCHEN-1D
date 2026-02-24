"""
postprocess.py

Post-processing helpers to regenerate plots from saved run files without
rerunning the simulation.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


TemporalQuantity = Literal["V_app", "V_gap", "I_discharge", "cfl", "particle_inventory"]
SpatialQuantity = Literal[
    "ne",
    "ni",
    "phi",
    "E",
    "Gamma_i",
    "Gamma_e",
    "townsend_alpha",
    "nu_i",
    "S",
]


@dataclass
class TemporalReplotStyle:
    t_unit: Literal["s", "ms", "us", "ns"] = "ns"
    xscale: Literal["linear", "log"] = "linear"
    yscale: Literal["linear", "log"] = "linear"
    figsize: tuple[float, float] = (4.2, 3.0)


@dataclass
class SpatialReplotStyle:
    x_unit: Literal["m", "cm", "mm"] = "mm"
    xscale: Literal["linear", "log"] = "linear"
    yscale: Literal["linear", "log"] = "linear"
    figsize: tuple[float, float] = (4.4, 3.1)


def _time_scale(unit: str) -> tuple[float, str]:
    if unit == "s":
        return 1.0, "Time [s]"
    if unit == "ms":
        return 1e3, "Time [ms]"
    if unit == "us":
        return 1e6, "Time [us]"
    return 1e9, "Time [ns]"


def _x_scale(unit: str) -> tuple[float, str]:
    if unit == "m":
        return 1.0, "x [m]"
    if unit == "cm":
        return 1e2, "x [cm]"
    return 1e3, "x [mm]"


def load_run_metadata(run_name: str) -> dict:
    path = Path(run_name) / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _v_app_from_metadata(time: np.ndarray, meta: dict) -> np.ndarray:
    waveform_type = meta["waveform_type"]
    if waveform_type == "dc":
        return meta["V_peak"] * np.ones_like(time)
    if waveform_type == "step":
        return (
            meta["V_peak"] * ((time >= meta["tV_start"]) & (time <= meta["tV_end"]))
            + 1e-15 * ((time < meta["tV_start"]) | (time > meta["tV_end"]))
        ).astype(np.float64)
    if waveform_type == "gaussian":
        return meta["V_peak"] * np.exp(-((time - meta["t_peak"]) / meta["tau"]) ** 2)
    if waveform_type == "rf":
        omega = 2.0 * np.pi * meta["f_rf"]
        return meta["V_dc"] + meta["V_peak"] * np.sin(omega * time + meta["phi_rf"])
    raise ValueError(f"Unknown waveform_type in metadata: {waveform_type}")


def _read_time_series(run_dir: Path, name: str, Nt: int) -> np.ndarray:
    return np.memmap(run_dir / name, mode="r", dtype=np.float32, shape=(Nt,))


def _spatial_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "ne": run_dir / "ne_sampled_mm.dat",
        "ni": run_dir / "ni_sampled_mm.dat",
        "phi": run_dir / "phi_sampled_mm.dat",
        "E": run_dir / "E_sampled_mm.dat",
        "Gamma_i": run_dir / "Gamma_i_sampled_mm.dat",
        "Gamma_e": run_dir / "Gamma_e_sampled_mm.dat",
        "townsend_alpha": run_dir / "townsend_alpha_sampled_mm.dat",
        "nu_i": run_dir / "nu_i_sampled_mm.dat",
        "S": run_dir / "S_sampled_mm.dat",
    }


def replot_from_saved(
    run_name: str,
    *,
    temporal_groups: tuple[tuple[TemporalQuantity, ...], ...] | None = None,
    spatial_groups: tuple[tuple[SpatialQuantity, ...], ...] | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    t_samples: tuple[float, ...] | None = None,
    temporal_style: TemporalReplotStyle | None = None,
    spatial_style: SpatialReplotStyle | None = None,
) -> None:
    """
    Regenerate diagnostics from saved files in <run_name>/.

    This function can be called repeatedly with different style/units/scales
    without rerunning simulation.
    """
    run_dir = Path(run_name)
    meta = load_run_metadata(run_name)

    Nt = int(meta["Nt"])
    Nx = int(meta["Nx"])
    T_total = float(meta["T_total"])
    L = float(meta["L"])
    A = float(meta["A"])
    save_every = int(meta["save_every"])

    temporal_style = temporal_style or TemporalReplotStyle()
    spatial_style = spatial_style or SpatialReplotStyle()

    time = np.linspace(0.0, T_total, Nt, dtype=np.float64)
    x = np.linspace(0.0, L, Nx, dtype=np.float64)

    # ---------- Temporal ----------
    if temporal_groups is None:
        temporal_groups = (
            ("V_app", "V_gap"),
            ("I_discharge",),
            ("cfl",),
            ("particle_inventory",),
        )

    t_factor, t_label = _time_scale(temporal_style.t_unit)
    tw0 = float(time[0]) if t_start is None else float(t_start)
    tw1 = float(time[-1]) if t_end is None else float(t_end)
    if tw1 < tw0:
        tw0, tw1 = tw1, tw0
    mask_full = (time >= tw0) & (time <= tw1)

    temporal_values = {
        "V_app": _v_app_from_metadata(time, meta),
        "V_gap": np.asarray(_read_time_series(run_dir, "Vgap_mm.dat", Nt), dtype=np.float64),
        "I_discharge": np.asarray(
            _read_time_series(run_dir, "Idischarge_mm.dat", Nt), dtype=np.float64
        ),
        "cfl": np.asarray(_read_time_series(run_dir, "c_cfl_mm.dat", Nt), dtype=np.float64),
    }

    # Particle inventory uses saved density snapshots.
    nsave = int((Nt - 1) // save_every + 1)
    saved_indices = np.arange(nsave, dtype=np.int64) * save_every
    saved_indices = np.minimum(saved_indices, Nt - 1)
    saved_times = time[saved_indices]
    ne_sampled = np.memmap(run_dir / "ne_sampled_mm.dat", mode="r", dtype=np.float32, shape=(nsave, Nx))
    ni_sampled = np.memmap(run_dir / "ni_sampled_mm.dat", mode="r", dtype=np.float32, shape=(nsave, Nx))
    N_e = A * np.trapz(np.asarray(ne_sampled, dtype=np.float64), x=x, axis=1)
    N_i = A * np.trapz(np.asarray(ni_sampled, dtype=np.float64), x=x, axis=1)

    for group in temporal_groups:
        if len(group) == 0:
            continue

        if "particle_inventory" in group:
            if len(group) > 1:
                print(f"Temporal group {group}: plotting 'particle_inventory' separately.")
            mask_inv = (saved_times >= tw0) & (saved_times <= tw1)
            if not np.any(mask_inv):
                print("Particle inventory skipped: empty selected time window.")
            else:
                Ne0 = N_e[0] if abs(N_e[0]) > 1e-30 else 1.0
                Ni0 = N_i[0] if abs(N_i[0]) > 1e-30 else 1.0
                fig, ax = plt.subplots(figsize=temporal_style.figsize)
                ax.plot(saved_times[mask_inv] * t_factor, N_e[mask_inv] / Ne0, label="N_e / N_e0")
                ax.plot(saved_times[mask_inv] * t_factor, N_i[mask_inv] / Ni0, label="N_i / N_i0")
                ax.set_xlabel(t_label)
                ax.set_ylabel("Normalized inventory")
                ax.set_xscale(temporal_style.xscale)
                ax.set_yscale(temporal_style.yscale)
                ax.set_title("Particle Inventory (saved snapshots)")
                ax.grid(True)
                ax.legend(frameon=False)
                fig.tight_layout()
                plt.show()

            # Continue with any non-inventory quantities in the same group.
            group = tuple(q for q in group if q != "particle_inventory")
            if len(group) == 0:
                continue

        valid = [q for q in group if q in temporal_values]
        if len(valid) == 0:
            print(f"Temporal group {group} has no available quantities.")
            continue

        fig, ax = plt.subplots(figsize=temporal_style.figsize)
        ylabel = None
        for q in valid:
            y = temporal_values[q][mask_full]
            if q in ("V_app", "V_gap"):
                y = y * 1e-3
                ylab = "Voltage [kV]"
            elif q == "I_discharge":
                y = y * 1e3
                ylab = "Current [mA]"
            else:
                ylab = "CFL number"
            if ylabel is None:
                ylabel = ylab
            elif ylabel != ylab:
                ylabel = "Mixed units"
            ax.plot(time[mask_full] * t_factor, y, label=q)

        ax.set_xlabel(t_label)
        ax.set_ylabel(ylabel if ylabel is not None else "Value")
        ax.set_xscale(temporal_style.xscale)
        ax.set_yscale(temporal_style.yscale)
        ax.set_title(" + ".join(valid))
        ax.grid(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()

    # ---------- Spatial ----------
    if spatial_groups is None:
        spatial_groups = (("ne", "ni"), ("phi",), ("E",))

    x_factor, x_label = _x_scale(spatial_style.x_unit)
    x_plot = x * x_factor
    paths = _spatial_paths(run_dir)
    sampled_arrays: dict[str, np.ndarray] = {}
    for q, p in paths.items():
        if p.exists():
            sampled_arrays[q] = np.memmap(p, mode="r", dtype=np.float32, shape=(nsave, Nx))

    if t_samples is None:
        requested = np.array([saved_times[-1]], dtype=np.float64)
    else:
        requested = np.asarray(t_samples, dtype=np.float64)
        requested = np.clip(requested, saved_times[0], saved_times[-1])

    for group in spatial_groups:
        if len(group) == 0:
            continue
        fig, ax = plt.subplots(figsize=spatial_style.figsize)
        ylabel = None
        has_curve = False
        for q in group:
            arr = sampled_arrays.get(q)
            if arr is None:
                print(f"Spatial quantity '{q}' missing in saved files; skipping.")
                continue
            if q in ("ne", "ni"):
                ylab = "Density [m$^{-3}$]"
            elif q == "phi":
                ylab = "Potential [V]"
            elif q == "E":
                ylab = "Electric Field [V/m]"
            elif q in ("Gamma_i", "Gamma_e"):
                ylab = "Gamma [m$^{-2}$ s$^{-1}$]"
            elif q == "townsend_alpha":
                ylab = "Townsend alpha [m$^{-1}$]"
            elif q == "nu_i":
                ylab = "nu_i [s$^{-1}$]"
            else:
                ylab = "Source [m$^{-3}$ s$^{-1}$]"
            if ylabel is None:
                ylabel = ylab
            elif ylabel != ylab:
                ylabel = "Mixed units"

            for t_req in requested:
                k = int(np.argmin(np.abs(saved_times - t_req)))
                lbl = q if requested.size == 1 else f"{q}, t={saved_times[k]*1e9:.1f} ns"
                ax.plot(x_plot, np.asarray(arr[k], dtype=np.float64), label=lbl)
                has_curve = True

        if not has_curve:
            plt.close(fig)
            print(f"Spatial group {group} has no available curves.")
            continue

        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel if ylabel is not None else "Value")
        ax.set_xscale(spatial_style.xscale)
        ax.set_yscale(spatial_style.yscale)
        ax.set_title(" + ".join(group))
        ax.grid(True)
        if len(ax.lines) <= 12:
            ax.legend(frameon=False)
        fig.tight_layout()
        plt.show()
