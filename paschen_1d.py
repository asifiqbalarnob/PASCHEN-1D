"""
paschen_1d.py

High-level simulation driver for the 1D drift-diffusion-Poisson model.

This module orchestrates:
- grid/time construction,
- waveform and external-circuit coupling,
- transport setup and initial conditions,
- KT+RK4 continuity updates for n_e and n_i,
- Poisson solve with Dirichlet boundaries (phi_left=V_gap, phi_right=0),
- optional anode/cathode external emission models,
- CFL diagnostics and snapshot output.

Typical usage (from command line):

    $ python paschen_1d.py

or from another script:

    from paschen_1d import run_simulation, SimulationConfig
    cfg = SimulationConfig(...)
    state = run_simulation(cfg)

The __main__ block provides a simple default test configuration.

Run-flow map (quick onboarding):
1. Build grid/time arrays from cfg.Nx, cfg.Nt, cfg.L, cfg.T_total.
2. Build V_app(t), transport coefficients, emission model, BC handler.
3. Allocate memmapped outputs in cfg.run_name.
4. Build initial fields and initialize circuit states.
5. For each time step: compute drift-diffusion fluxes.
6. Advance external circuit (step_circuit) to obtain new V_gap.
7. Build volumetric sources (Townsend ionization/recombination, optional toggles).
8. Advance n_i and n_e with KT + RK4, then enforce BC + Poisson fixed-point.
9. Compute CFL and write snapshots every cfg.save_every.
10. Return SimulationState; post-process via run_configured_diagnostics.
"""

import time as pytime
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import SimulationConfig, SimulationState
from outputs import allocate_outputs, write_snapshot, write_run_metadata
from physics import (
    make_voltage_waveform,
    set_transport_coefficients,
    build_initial_conditions,
    compute_townsend_alpha,
)
from numerics import (
    build_poisson_tridiag_interior,
    poisson_1d_dirichlet_interior,
    rk4_step,
    set_boundary_condition_implicit,
    CFL_test,
)
from plotting import (
    plot_selected_temporal_quantity,
    plot_selected_spatial_quantity,
    plot_selected_temporal_group,
    plot_selected_spatial_group,
    plot_particle_inventory,
)
from circuit import step_circuit
from circuit_implicit_euler import step_circuit_implicit_euler
from physical_constants import e
from emission import build_emission_model


def _format_waveform_summary(cfg: SimulationConfig) -> str:
    if cfg.waveform_type == "dc":
        return f"type=dc, V_peak={cfg.V_peak:.6g} V"
    if cfg.waveform_type == "step":
        return (
            f"type=step, V_peak={cfg.V_peak:.6g} V, "
            f"t_on={cfg.tV_start:.6g} s, t_off={cfg.tV_end:.6g} s"
        )
    if cfg.waveform_type == "gaussian":
        return (
            f"type=gaussian, V_peak={cfg.V_peak:.6g} V, "
            f"t_peak={cfg.t_peak:.6g} s, tau={cfg.tau:.6g} s"
        )
    if cfg.waveform_type == "rf":
        return (
            f"type=rf, V_peak={cfg.V_peak:.6g} V, f_rf={cfg.f_rf:.6g} Hz, "
            f"V_dc={cfg.V_dc:.6g} V, phi_rf={cfg.phi_rf:.6g} rad"
        )
    return f"type={cfg.waveform_type}"


def _enabled_external_emission_components_for_electrode(
    cfg: SimulationConfig,
    electrode: str,
) -> list[str]:
    prefix = "anode" if electrode == "anode" else "cathode"
    enabled = []
    if getattr(cfg, f"{prefix}_enable_constant_J_emission", False):
        enabled.append("constant_J")
    if getattr(cfg, f"{prefix}_enable_fn_emission", False):
        enabled.append("fn")
    if getattr(cfg, f"{prefix}_enable_mg_emission", False):
        enabled.append("mg")
    if getattr(cfg, f"{prefix}_enable_rd_emission", False):
        enabled.append("rd")
    if getattr(cfg, f"{prefix}_enable_quantum_pulse_emission", False):
        enabled.append("quantum_pulse")
    return enabled


def _print_run_config_summary(cfg: SimulationConfig, dt: float, dx: float) -> None:
    temporal = cfg.diagnostics.temporal
    spatial = cfg.diagnostics.spatial

    temporal_plot_mode = (
        f"grouped ({len(temporal.plot_groups)} groups)"
        if temporal.plot_groups is not None
        else f"separate ({len(temporal.quantities)} quantities)"
    )
    spatial_plot_mode = (
        f"grouped ({len(spatial.plot_groups)} groups)"
        if spatial.plot_groups is not None
        else f"separate ({len(spatial.quantities)} quantities)"
    )
    t_samples_text = (
        "final_time_only"
        if spatial.t_samples is None
        else ", ".join(f"{t:.3e}" for t in spatial.t_samples)
    )

    print("\n=== PASCHEN-1D RUN SUMMARY ===")
    print(f"run_name: {cfg.run_name}")
    print(
        f"geometry: L={cfg.L:.6g} m, A={cfg.A:.6g} m^2, "
        f"l={cfg.l:.6g} m, eps_r={cfg.eps_r:.6g}"
    )
    print(f"plasma: gas={cfg.gas}, p_Torr={cfg.p_Torr:.6g}, n0={cfg.n0:.6g} m^-3")
    print(
        f"grid/time: Nx={cfg.Nx}, Nt={cfg.Nt}, T_total={cfg.T_total:.6g} s, "
        f"dx={dx:.6g} m, dt={dt:.6g} s"
    )
    print(f"waveform: {_format_waveform_summary(cfg)}")
    print(
        f"circuit: type={cfg.circuit_type}, scheme={cfg.circuit_time_scheme}, "
        f"R0={cfg.R0:.6g}, C_s={cfg.C_s:.6g}, "
        f"L_s={cfg.L_s:.6g}, C_p={cfg.C_p:.6g}, L_p={cfg.L_p:.6g}, R_m={cfg.R_m:.6g}"
    )
    print(f"numerics: kt_limiter_theta={cfg.kt_limiter_theta:.6g}")
    print(
        "boundary-modes: "
        f"anode(i={cfg.anode_ion_boundary}, e={cfg.anode_electron_boundary}), "
        f"cathode(i={cfg.cathode_ion_boundary}, e={cfg.cathode_electron_boundary})"
    )
    print(
        "sources: "
        f"volume={cfg.enable_volume_sources}, ionization={cfg.enable_ionization_source}, "
        f"recombination={cfg.enable_recombination_sink}"
    )
    anode_components = _enabled_external_emission_components_for_electrode(cfg, "anode")
    cathode_components = _enabled_external_emission_components_for_electrode(cfg, "cathode")
    print(
        "emission: "
        f"enabled={cfg.enable_external_emission}, "
        f"material_mode={getattr(cfg, 'electrode_material_mode', 'shared')}, "
        f"anode={anode_components}, cathode={cathode_components}, "
        f"electrodes=(anode={cfg.enable_anode_external_emission}, cathode={cfg.enable_cathode_external_emission}), "
        f"to_circuit={cfg.enable_emission_in_circuit_current}"
    )
    print(
        "diagnostics-temporal: "
        f"enabled={temporal.enabled}, mode={temporal_plot_mode}, "
        f"window=[{temporal.t_start},{temporal.t_end}]"
    )
    print(
        "diagnostics-spatial: "
        f"enabled={spatial.enabled}, mode={spatial_plot_mode}, "
        f"t_samples={t_samples_text}, x_unit={spatial.x_unit}"
    )
    print("==============================\n")


def _print_config_warnings(cfg: SimulationConfig) -> None:
    warnings = []

    if cfg.Nx < 3:
        warnings.append("Nx < 3 can break boundary stencils.")
    if cfg.Nt < 2:
        warnings.append("Nt < 2 gives invalid time grid.")
    if cfg.T_total <= 0.0:
        warnings.append("T_total <= 0 is invalid.")
    if cfg.waveform_type == "step" and cfg.tV_end < cfg.tV_start:
        warnings.append("step waveform has tV_end < tV_start.")
    if getattr(cfg, "electrode_material_mode", "shared") not in ("shared", "separate"):
        warnings.append("electrode_material_mode must be 'shared' or 'separate'.")
    anode_components = _enabled_external_emission_components_for_electrode(cfg, "anode")
    cathode_components = _enabled_external_emission_components_for_electrode(cfg, "cathode")
    if cfg.enable_external_emission and (not anode_components) and (not cathode_components):
        warnings.append(
            "enable_external_emission=True but no anode/cathode external emission components are enabled."
        )
    if (not cfg.enable_external_emission) and cfg.enable_emission_in_circuit_current:
        warnings.append("enable_emission_in_circuit_current=True while enable_external_emission=False.")
    if (not cfg.enable_external_emission) and (anode_components or cathode_components):
        warnings.append(
            "External emission components are enabled but enable_external_emission=False."
        )
    if cfg.enable_external_emission and (
        (not cfg.enable_anode_external_emission)
        and (not cfg.enable_cathode_external_emission)
    ):
        warnings.append(
            "enable_external_emission=True but both anode/cathode external-emission electrode toggles are OFF."
        )
    if anode_components and (not cfg.enable_anode_external_emission):
        warnings.append(
            "Anode external-emission components are enabled but enable_anode_external_emission=False."
        )
    if cathode_components and (not cfg.enable_cathode_external_emission):
        warnings.append(
            "Cathode external-emission components are enabled but enable_cathode_external_emission=False."
        )
    if cfg.enable_anode_external_emission and cfg.anode_electron_boundary != "electron_emission":
        warnings.append(
            "enable_anode_external_emission=True but anode_electron_boundary!='electron_emission'; "
            "anode external emission will not be applied."
        )
    if cfg.enable_cathode_external_emission and cfg.cathode_electron_boundary != "electron_emission":
        warnings.append(
            "enable_cathode_external_emission=True but cathode_electron_boundary!='electron_emission'; "
            "cathode external emission will not be applied."
        )
    if (not cfg.enable_volume_sources) and (
        cfg.enable_ionization_source or cfg.enable_recombination_sink
    ):
        warnings.append("volume source master switch is OFF; ionization/recombination toggles are ignored.")
    if cfg.anode_ion_boundary == "implicit_drift_closure":
        warnings.append(
            "anode_ion_boundary='implicit_drift_closure' is not implemented in current BC kernel."
        )
    if cfg.cathode_ion_boundary == "electron_emission":
        warnings.append(
            "cathode_ion_boundary='electron_emission' is invalid for ions."
        )
    if cfg.anode_ion_boundary == "electron_emission":
        warnings.append(
            "anode_ion_boundary='electron_emission' is invalid for ions."
        )
    if cfg.cathode_electron_boundary == "implicit_drift_closure":
        warnings.append(
            "cathode_electron_boundary='implicit_drift_closure' is not implemented in current BC kernel."
        )

    if warnings:
        print("Config warnings:")
        for w in warnings:
            print(f"  - {w}")
        print("")


def run_configured_diagnostics(
    cfg: SimulationConfig,
    state: SimulationState,
    V_app_func,
) -> None:
    """
    Run user-selected temporal and spatial diagnostics after simulation.

    Behavior summary:
    - Temporal quantities are evaluated on `state.time`, then optionally
      windowed to [t_start, t_end] before plotting.
    - Spatial quantities are read from sampled memmaps written during runtime.
      Requested times are matched to the nearest saved snapshot time.
    - For final-time requests of ne/ni/phi/E, exact final in-memory profiles
      are used instead of nearest-snapshot lookup.

    This routine is post-processing only and does not affect solver logic.
    """
    # --------------------------
    # Temporal diagnostics
    # --------------------------
    temporal_cfg = cfg.diagnostics.temporal
    if temporal_cfg.enabled:
        temporal_values = {
            "V_app": np.asarray(V_app_func(state.time), dtype=np.float64),
            "V_gap": np.asarray(state.V_gap, dtype=np.float64),
            "I_discharge": np.asarray(state.I_discharge, dtype=np.float64),
            "cfl": np.asarray(state.c_cfl, dtype=np.float64),
        }

        particle_inventory_required = (
            ("particle_inventory" in temporal_cfg.quantities)
            or (
                temporal_cfg.plot_groups is not None
                and any("particle_inventory" in grp for grp in temporal_cfg.plot_groups)
            )
        )
        particle_inventory = None
        if particle_inventory_required:
            nsave = int((cfg.Nt - 1) // cfg.save_every + 1)
            saved_indices = np.arange(nsave, dtype=np.int64) * cfg.save_every
            saved_indices = np.minimum(saved_indices, cfg.Nt - 1)
            saved_times_inventory = state.time[saved_indices]

            outdir_inventory = Path(cfg.run_name)
            ne_path = outdir_inventory / "ne_sampled_mm.dat"
            ni_path = outdir_inventory / "ni_sampled_mm.dat"
            if ne_path.exists() and ni_path.exists():
                ne_sampled = np.memmap(ne_path, mode="r", dtype=np.float32, shape=(nsave, cfg.Nx))
                ni_sampled = np.memmap(ni_path, mode="r", dtype=np.float32, shape=(nsave, cfg.Nx))
                N_e = cfg.A * np.trapz(np.asarray(ne_sampled, dtype=np.float64), x=state.x, axis=1)
                N_i = cfg.A * np.trapz(np.asarray(ni_sampled, dtype=np.float64), x=state.x, axis=1)
                particle_inventory = (saved_times_inventory, N_e, N_i)
            else:
                print("Temporal diagnostic 'particle_inventory' unavailable; sampled density files missing.")

        temporal_groups = (
            temporal_cfg.plot_groups
            if temporal_cfg.plot_groups is not None
            else tuple((q,) for q in temporal_cfg.quantities)
        )

        for group in temporal_groups:
            group_no_inventory = tuple(q for q in group if q != "particle_inventory")
            if "particle_inventory" in group:
                if len(group_no_inventory) > 0:
                    print(
                        f"Temporal group {group} mixes 'particle_inventory' with scalar diagnostics; "
                        "plotting 'particle_inventory' separately."
                    )
                if particle_inventory is None:
                    if len(group_no_inventory) == 0:
                        continue
                else:
                    savepath = None
                    if temporal_cfg.savepath_prefix:
                        savepath = f"{temporal_cfg.savepath_prefix}_particle_inventory.pdf"
                    t_inv, N_e, N_i = particle_inventory
                    plot_particle_inventory(
                        time=t_inv,
                        N_e=N_e,
                        N_i=N_i,
                        t_start=temporal_cfg.t_start,
                        t_end=temporal_cfg.t_end,
                        savepath=savepath,
                    )

            valid_group = tuple(q for q in group_no_inventory if q in temporal_values)
            if len(valid_group) == 0:
                if "particle_inventory" not in group:
                    print(f"Temporal diagnostic group {group} is unknown; skipping.")
                continue

            savepath = None
            if temporal_cfg.savepath_prefix:
                savepath = f"{temporal_cfg.savepath_prefix}_{'_'.join(valid_group)}.pdf"

            if len(valid_group) == 1:
                quantity = valid_group[0]
                plot_selected_temporal_quantity(
                    time=state.time,
                    quantity=quantity,
                    values=temporal_values[quantity],
                    t_start=temporal_cfg.t_start,
                    t_end=temporal_cfg.t_end,
                    savepath=savepath,
                )
            else:
                plot_selected_temporal_group(
                    time=state.time,
                    quantities=valid_group,
                    values_map=temporal_values,
                    t_start=temporal_cfg.t_start,
                    t_end=temporal_cfg.t_end,
                    savepath=savepath,
                )

    # --------------------------
    # Spatial diagnostics
    # --------------------------
    spatial_cfg = cfg.diagnostics.spatial
    if not spatial_cfg.enabled:
        return

    requested_times = (
        np.asarray(spatial_cfg.t_samples, dtype=np.float64)
        if spatial_cfg.t_samples is not None
        else np.array([state.time[-1]], dtype=np.float64)
    )
    requested_times = np.clip(requested_times, state.time[0], state.time[-1])

    # Saved snapshot times corresponding to *_sampled memmaps.
    nsave = int((cfg.Nt - 1) // cfg.save_every + 1)
    saved_indices = np.arange(nsave, dtype=np.int64) * cfg.save_every
    saved_indices = np.minimum(saved_indices, cfg.Nt - 1)
    saved_times = state.time[saved_indices]

    outdir = Path(cfg.run_name)
    sampled_paths = {
        "ne": outdir / "ne_sampled_mm.dat",
        "ni": outdir / "ni_sampled_mm.dat",
        "phi": outdir / "phi_sampled_mm.dat",
        "E": outdir / "E_sampled_mm.dat",
        "Gamma_i": outdir / "Gamma_i_sampled_mm.dat",
        "Gamma_e": outdir / "Gamma_e_sampled_mm.dat",
        "townsend_alpha": outdir / "townsend_alpha_sampled_mm.dat",
        "nu_i": outdir / "nu_i_sampled_mm.dat",
        "S": outdir / "S_sampled_mm.dat",
    }

    # Exact final profiles available in memory.
    final_profiles = {
        "ne": state.ne_final,
        "ni": state.ni_final,
        "phi": state.phi_final,
        "E": state.E_final,
    }

    sampled_arrays: dict[str, np.ndarray] = {}
    for quantity in spatial_cfg.quantities:
        if quantity in sampled_arrays:
            continue
        path = sampled_paths.get(quantity)
        if path is not None and path.exists():
            sampled_arrays[quantity] = np.memmap(
                path, mode="r", dtype=np.float32, shape=(nsave, cfg.Nx)
            )

    dt = state.time[1] - state.time[0] if state.time.size > 1 else 0.0

    def collect_profiles_for_quantity(quantity: str) -> tuple[np.ndarray, np.ndarray] | None:
        profiles = []
        actual_times = []
        for t_req in requested_times:
            # Use exact final fields when final time is requested.
            if (
                quantity in final_profiles
                and abs(float(t_req) - float(state.time[-1])) <= 0.5 * dt
            ):
                profiles.append(np.asarray(final_profiles[quantity], dtype=np.float64))
                actual_times.append(float(state.time[-1]))
                continue

            if quantity not in sampled_arrays:
                return None

            k = int(np.argmin(np.abs(saved_times - t_req)))
            profiles.append(np.asarray(sampled_arrays[quantity][k], dtype=np.float64))
            actual_times.append(float(saved_times[k]))

        if len(profiles) == 0:
            return None
        return np.vstack(profiles), np.asarray(actual_times, dtype=np.float64)

    spatial_groups = (
        spatial_cfg.plot_groups
        if spatial_cfg.plot_groups is not None
        else tuple((q,) for q in spatial_cfg.quantities)
    )

    for group in spatial_groups:
        profiles_map: dict[str, np.ndarray] = {}
        actual_times_ref = None

        for quantity in group:
            collected = collect_profiles_for_quantity(quantity)
            if collected is None:
                print(f"Spatial diagnostic '{quantity}' unavailable; skipping in group {group}.")
                continue
            prof, times_q = collected
            profiles_map[quantity] = prof
            if actual_times_ref is None:
                actual_times_ref = times_q

        if len(profiles_map) == 0 or actual_times_ref is None:
            print(f"Spatial diagnostic group {group} unavailable; skipping.")
            continue

        valid_group = tuple(q for q in group if q in profiles_map)
        savepath = None
        if spatial_cfg.savepath_prefix:
            savepath = f"{spatial_cfg.savepath_prefix}_{'_'.join(valid_group)}.pdf"

        if len(valid_group) == 1:
            q = valid_group[0]
            plot_selected_spatial_quantity(
                x=state.x,
                quantity=q,
                profiles=profiles_map[q],
                sample_times=actual_times_ref,
                x_unit=spatial_cfg.x_unit,
                savepath=savepath,
            )
        else:
            plot_selected_spatial_group(
                x=state.x,
                quantities=valid_group,
                profiles_map=profiles_map,
                sample_times=actual_times_ref,
                x_unit=spatial_cfg.x_unit,
                savepath=savepath,
            )


def run_simulation(cfg: SimulationConfig) -> SimulationState:
    """
    Run one PASCHEN-1D simulation for the provided configuration.

    Parameters
    ----------
    cfg : SimulationConfig
        Physical, numerical, and circuit parameters for the run.

    Returns
    -------
    SimulationState
        Container holding:
            - cfg           : the input SimulationConfig (for provenance)
            - time          : 1D time array [s], length Nt
            - x             : 1D spatial grid [m], length Nx
            - V_gap         : gap voltage history [V], shape (Nt,)
            - I_discharge   : discharge current history [A], shape (Nt,)
            - c_cfl         : CFL number history, shape (Nt,)
            - ne_final      : final electron density profile [m^-3], shape (Nx,)
            - ni_final      : final ion    density profile [m^-3], shape (Nx,)
            - phi_final     : final potential profile [V], shape (Nx,)
            - E_final       : final electric field profile [V/m], shape (Nx,)

    Notes
    -----
    - Drift-diffusion update uses KT flux + explicit diffusion + RK4 in time.
    - Poisson equation is solved at each time step on interior nodes with a
      banded tridiagonal solver and Dirichlet BCs.
    - Boundary conditions for n_e and n_i are selected per electrode/species
      using config boundary modes:
      anode_ion_boundary, anode_electron_boundary,
      cathode_ion_boundary, cathode_electron_boundary.
    - External circuit dynamics are handled by `step_circuit` and include
      R0, Cp, Cs, Ls, Lp, Rm, and optional dielectric layers using the
      Adamovic convention.
    - Optional external emission is provided through `build_emission_model`
      and can be configured independently for anode/cathode.
      In the current formulation, emission contributes through:
        (i) boundary electron-emission closures at anode/cathode, and
        (ii) optionally, boundary electron-emission flux terms used in
             circuit current coupling.

    Onboarding map
    --------------
    The implementation body below follows numbered section headers:
    (1) discretization, (2) physics helpers, (3) circuit setup,
    (4) output allocation, (5) initialization, (6) initial snapshot,
    (7) Poisson matrix, (8) local flux functions, (9) time loop,
    (10) pack and return state.
    """
    # ------------------------------------------------------------
    # 1) Basic discretization setup
    # ------------------------------------------------------------
    Nt = cfg.Nt
    Nx = cfg.Nx
    T  = cfg.T_total

    dt = T / (Nt - 1)            # uniform time step [s]
    dx = cfg.L / (Nx - 1)        # uniform spatial step [m]

    if getattr(cfg, "print_run_summary", True):
        _print_run_config_summary(cfg, dt=dt, dx=dx)
    if getattr(cfg, "warn_on_config_mismatch", True):
        _print_config_warnings(cfg)

    time    = np.linspace(0.0, T, Nt, dtype=np.float64)       # 0 .. T
    x_array = np.linspace(0.0, cfg.L, Nx, dtype=np.float64)   # 0 .. L

    # ------------------------------------------------------------
    # 2) Physics helpers: waveform, transport, emission, BCs
    # ------------------------------------------------------------
    # Applied voltage waveform Vs(t)
    V_app_func = make_voltage_waveform(cfg)

    # Mobilities, diffusion, recombination, reduced pressure, etc.
    transport = set_transport_coefficients(cfg)

    # Emission model (may be None if emission is disabled in cfg)
    emission_model = build_emission_model(cfg)

    # Boundary conditions are applied via per-electrode/per-species mode knobs.
    bc_func = set_boundary_condition_implicit

    # Frequently used scalars.
    L      = cfg.L
    A      = cfg.A
    l      = cfg.l
    eps_r_ = cfg.eps_r
    gamma_ = cfg.gamma
    anode_electron_induced_yield_ = getattr(cfg, "anode_electron_induced_yield", 0.0)
    use_vaughan_sey_ = getattr(cfg, "use_vaughan_sey", False)
    vaughan_Emax0_eV_ = getattr(cfg, "vaughan_Emax0_eV", 400.0)
    vaughan_dmax0_ = getattr(cfg, "vaughan_dmax0", 3.2)
    vaughan_ks_ = getattr(cfg, "vaughan_ks", 1.0)
    vaughan_z_ = getattr(cfg, "vaughan_z", 0.0)
    vaughan_E0_ = getattr(cfg, "vaughan_E0", 0.0)
    p_Torr = cfg.p_Torr
    pr_    = transport.pr

    # ------------------------------------------------------------
    # 3) External-circuit setup (topology + state arrays)
    # ------------------------------------------------------------
    R0_val       = cfg.R0
    C_p          = cfg.C_p
    C_s          = getattr(cfg, "C_s", 0.0)
    L_s          = getattr(cfg, "L_s", 0.0)
    L_p          = getattr(cfg, "L_p", 0.0)
    R_m          = getattr(cfg, "R_m", 0.0)
    circuit_type = getattr(cfg, "circuit_type", "R0_Cp")
    circuit_time_scheme = getattr(cfg, "circuit_time_scheme", "explicit_euler")

    if circuit_time_scheme == "implicit_euler":
        circuit_stepper = step_circuit_implicit_euler
    elif circuit_time_scheme == "explicit_euler":
        circuit_stepper = step_circuit
    else:
        raise ValueError(f"Unknown circuit_time_scheme: {circuit_time_scheme}")

    # Circuit state arrays (allocated only when needed by topology).
    V_d = np.zeros(Nt, dtype=np.float32)  # dielectric mapping voltage state
    V_n = None      # node voltage
    V_Cs = None     # series capacitor voltage
    I_s = None      # series-branch current (R0-Cs-Ls)
    I_Lp = None     # shunt inductor current at node

    # Circuits that maintain a node voltage
    if circuit_type in (
        "R0_Cp",
        "R0_Cp_Rm",
        "R0_Cs_Cp",
        "R0_Cs_Ls_Cp",
        "R0_Cs_Cp_Rm",
        "R0_Cs_Ls_Cp_Rm",
        "R0_Cs_Ls_Cp_Lp",
        "R0_Cs_Ls_Cp_Lp_Rm",
    ):
        V_n = np.zeros(Nt, dtype=np.float32)

    # Circuits that include a series capacitor Cs
    if circuit_type in (
        "R0_Cs_Cp",
        "R0_Cs_Ls_Cp",
        "R0_Cs_Cp_Rm",
        "R0_Cs_Ls_Cp_Rm",
        "R0_Cs_Ls_Cp_Lp",
        "R0_Cs_Ls_Cp_Lp_Rm",
    ) and C_s > 0.0:
        V_Cs = np.zeros(Nt, dtype=np.float32)

    # Circuits with a series inductor L_s (requires a current state I_s)
    if circuit_type in (
        "R0_Cs_Ls_Cp",
        "R0_Cs_Ls_Cp_Rm",
        "R0_Cs_Ls_Cp_Lp",
        "R0_Cs_Ls_Cp_Lp_Rm",
    ) and L_s > 0.0:
        I_s = np.zeros(Nt, dtype=np.float32)

    # Circuits with a parallel inductor L_p at the node
    if circuit_type in ("R0_Cs_Ls_Cp_Lp", "R0_Cs_Ls_Cp_Lp_Rm") and L_p > 0.0:
        I_Lp = np.zeros(Nt, dtype=np.float32)

    # Convenience aliases.
    SAVE_EVERY       = cfg.save_every
    LOG_INTERMEDIATE = cfg.log_intermediate

    # ------------------------------------------------------------
    # 4) Allocate memory-mapped outputs
    # ------------------------------------------------------------
    outputs = allocate_outputs(cfg, Nt, Nx)
    phi_sampled = outputs.phi_sampled
    E_sampled = outputs.E_sampled
    n_e_sampled = outputs.n_e_sampled
    n_i_sampled = outputs.n_i_sampled
    Gamma_i_sampled = outputs.Gamma_i_sampled
    Gamma_e_sampled = outputs.Gamma_e_sampled
    townsend_alpha_sampled = outputs.townsend_alpha_sampled
    nu_i_sampled = outputs.nu_i_sampled
    S_sampled = outputs.S_sampled
    V_gap = outputs.V_gap
    c_cfl = outputs.c_cfl
    I_discharge = outputs.I_discharge

    # Write run metadata so plots can be regenerated without rerunning simulation.
    write_run_metadata(cfg, Nt=Nt, Nx=Nx, dt=dt, dx=dx)

    # ------------------------------------------------------------
    # 5) Initial conditions (phi, E, n_e, n_i, V_gap)
    # ------------------------------------------------------------
    phi0, E0, ne0, ni0, V0 = build_initial_conditions(cfg, x_array, V_app_func)
    V_gap[0] = V0  # initial gap voltage (usually Vs(t=0))

    # Initialize circuit-state arrays where applicable.
    if V_n is not None:
        # Start with node voltage equal to source at t=0.
        V_n[0] = V_app_func(0.0)

    if V_Cs is not None:
        V_Cs[0] = 0.0

    if I_s is not None:
        I_s[0] = 0.0

    if I_Lp is not None:
        I_Lp[0] = 0.0

    # Initialize dielectric mapping voltage from algebraic mapping.
    alpha_d = 1.0 + 2.0 * l / (eps_r_ * L)
    if V_n is not None:
        V_d[0] = float(V_n[0] - alpha_d * V_gap[0])
    else:
        V_d[0] = float(V_app_func(0.0) - alpha_d * V_gap[0])

    # Current/next-step fields (reused in-place).
    phi_curr = phi0.copy()
    E_curr   = E0.copy()
    ne_curr  = ne0.copy()
    ni_curr  = ni0.copy()

    phi_next = np.empty_like(phi_curr)
    E_next   = np.empty_like(E_curr)
    ne_next  = np.empty_like(ne_curr)
    ni_next  = np.empty_like(ni_curr)

    # Temporary arrays for gradients, fluxes, coefficients, and sources.
    grad_i      = np.empty(Nx, dtype=np.float32)
    grad_e      = np.empty(Nx, dtype=np.float32)
    Gamma_i_row = np.empty(Nx, dtype=np.float32)
    Gamma_e_row = np.empty(Nx, dtype=np.float32)
    townsend_alpha_row   = np.empty(Nx, dtype=np.float32)
    nu_row      = np.empty(Nx, dtype=np.float32)

    # ------------------------------------------------------------
    # 6) Store initial snapshot (k = 0)
    # ------------------------------------------------------------
    k0 = 0
    phi_sampled[k0, :] = phi_curr
    E_sampled[k0,   :] = E_curr
    n_e_sampled[k0, :] = ne_curr
    n_i_sampled[k0, :] = ni_curr

    if LOG_INTERMEDIATE:
        Gamma_i_sampled[k0, :] = 0.0
        Gamma_e_sampled[k0, :] = 0.0
        townsend_alpha_sampled[k0, :]   = 0.0
        nu_i_sampled[k0, :]    = 0.0
        S_sampled[k0,   :]     = 0.0

    # ------------------------------------------------------------
    # 7) Pre-build Poisson band matrix (interior Laplacian)
    # ------------------------------------------------------------
    ab_int = build_poisson_tridiag_interior(Nx, dtype=np.float64)

    # ------------------------------------------------------------
    # 8) Local drift flux functions for ions and electrons
    # ------------------------------------------------------------
    def ion_flux_local(n: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        Gamma_i = mu_i * n_i * E (drift only; diffusion handled separately).
        """
        return transport.mu_i * n * E

    def d_ion_flux_dn_local(n: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        d(Gamma_i)/dn = mu_i * E, used for KT local speed estimates.
        """
        return transport.mu_i * E

    def electron_flux_local(n: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        Gamma_e = -mu_e * n_e * E (drift only).
        """
        return -transport.mu_e * n * E

    def d_electron_flux_dn_local(n: np.ndarray, E: np.ndarray) -> np.ndarray:
        """
        d(Gamma_e)/dn = -mu_e * E, used for KT local speed estimates.
        """
        return -transport.mu_e * E

    # ------------------------------------------------------------
    # 9) Main time-integration loop
    # ------------------------------------------------------------
    start = pytime.perf_counter()

    for n_idx in tqdm(range(Nt - 1), mininterval=2, desc="Time stepping"):
        # (a) Density gradients and drift-diffusion fluxes.
        # np.gradient is used here for diffusion/diagnostic gradients.
        grad_i[:] = np.gradient(ni_curr, dx, edge_order=1).astype(
            grad_i.dtype, copy=False
        )
        grad_e[:] = np.gradient(ne_curr, dx, edge_order=1).astype(
            grad_e.dtype, copy=False
        )

        # Gamma = -D dn/dx ± mu n E
        Gamma_i_row[:] = -transport.D_i * grad_i + transport.mu_i * ni_curr * E_curr
        Gamma_e_row[:] = -transport.D_e * grad_e - transport.mu_e * ne_curr * E_curr

        t_next = float(time[n_idx + 1])

        # (b) Emission knob: optional circuit-coupling path.
        emission_active = emission_model is not None
        emission_to_circuit = emission_active and cfg.enable_emission_in_circuit_current

        if emission_active:
            J_emit_anode = (
                emission_model.current_density(
                    t=t_next,
                    V_gap=float(V_gap[n_idx]),
                    dt=dt,
                    E_surface=float(E_curr[0]),
                    electrode="anode",
                )
                if cfg.enable_anode_external_emission
                else 0.0
            )
            J_emit_cathode = (
                emission_model.current_density(
                    t=t_next,
                    V_gap=float(V_gap[n_idx]),
                    dt=dt,
                    E_surface=float(E_curr[-1]),
                    electrode="cathode",
                )
                if cfg.enable_cathode_external_emission
                else 0.0
            )
            Gamma_ext_anode = J_emit_anode / e
            Gamma_ext_cathode = J_emit_cathode / e
        else:
            Gamma_ext_anode = 0.0
            Gamma_ext_cathode = 0.0

        # Electron flux used in circuit coupling (optionally includes emission).
        if emission_to_circuit:
            Gamma_e_for_circuit = Gamma_e_row.copy()
            # Positive flux is along +x.
            # - Anode-emitted electrons move toward +x (add at left boundary).
            # - Cathode-emitted electrons move toward -x (subtract at right boundary).
            Gamma_e_for_circuit[0] += Gamma_ext_anode
            Gamma_e_for_circuit[-1] -= Gamma_ext_cathode
        else:
            Gamma_e_for_circuit = Gamma_e_row

        # Previous circuit state (scalars) for this step.
        V_n_prev = float(V_n[n_idx]) if V_n is not None else None
        V_d_prev = float(V_d[n_idx])
        V_Cs_prev   = float(V_Cs[n_idx])   if V_Cs   is not None else None
        I_s_prev    = float(I_s[n_idx])    if I_s    is not None else None
        I_Lp_prev   = float(I_Lp[n_idx])   if I_Lp   is not None else None

        # (c) Advance external circuit by one time step.
        # Coupling uses boundary-modified electron flux (includes optional
        # anode and/or cathode external-emission contributions).
        V_gap_new, I_new, V_d_new, V_n_new, V_Cs_new, I_s_new, I_Lp_new = circuit_stepper(
            circuit_type=circuit_type,
            V_app_func=V_app_func,
            t=t_next,
            dt=dt,
            V_gap_prev=float(V_gap[n_idx]),
            Gamma_i=Gamma_i_row,
            Gamma_e=Gamma_e_for_circuit,
            dx=dx,
            A=A,
            L=L,
            l=l,
            eps_r=eps_r_,
            R0=R0_val,
            C_s=C_s,
            C_p=C_p,
            R_m=R_m,
            L_s=L_s,
            L_p=L_p,
            V_d_prev=V_d_prev,
            V_n_prev=V_n_prev,
            V_Cs_prev=V_Cs_prev,
            I_s_prev=I_s_prev,
            I_Lp_prev=I_Lp_prev,
        )

        # Store updated circuit quantities.
        V_gap[n_idx + 1]   = V_gap_new
        I_discharge[n_idx] = I_new  # last entry remains default
        if V_d_new is None:
            raise RuntimeError("Circuit step did not return V_d_new.")
        V_d[n_idx + 1] = V_d_new

        if (V_n is not None) and (V_n_new is not None):
            V_n[n_idx + 1] = V_n_new

        if (V_Cs is not None) and (V_Cs_new is not None):
            V_Cs[n_idx + 1] = V_Cs_new

        if (I_s is not None) and (I_s_new is not None):
            I_s[n_idx + 1] = I_s_new

        if (I_Lp is not None) and (I_Lp_new is not None):
            I_Lp[n_idx + 1] = I_Lp_new

        # (d) Ionization/recombination source terms.
        # Townsend alpha(E) [1/m], ionization frequency nu_i = alpha*mu_e*|E|.
        townsend_alpha_row[:] = compute_townsend_alpha(
            E_curr, p_Torr, pr_, gas=cfg.gas
        ).astype(np.float32)
        nu_row[:] = townsend_alpha_row * transport.mu_e * np.abs(E_curr)

        if cfg.enable_volume_sources:
            ionization_source = (
                nu_row * ne_curr if cfg.enable_ionization_source else 0.0
            )
            recombination_sink = (
                transport.beta * ni_curr * ne_curr
                if cfg.enable_recombination_sink
                else 0.0
            )
            S_row = ionization_source - recombination_sink
        else:
            S_row = np.zeros_like(ne_curr)

        # (e) Drift-diffusion update for ions/electrons (KT + RK4).
        ni_next[:] = rk4_step(
            ni_curr,
            ion_flux_local,
            d_ion_flux_dn_local,
            E_curr,
            transport.D_i,
            S_row,
            dx,
            dt,
            kt_limiter_theta=cfg.kt_limiter_theta,
        )
        ne_next[:] = rk4_step(
            ne_curr,
            electron_flux_local,
            d_electron_flux_dn_local,
            E_curr,
            transport.D_e,
            S_row,
            dx,
            dt,
            kt_limiter_theta=cfg.kt_limiter_theta,
        )

        # (f) BC + Poisson fixed-point (Picard) iteration.
        # Iterate BC enforcement and Poisson solve using previous phi iterate.
        phi_iter = phi_curr.copy()
        for _ in range(10):
            # Enforce BCs on next-step densities with current phi iterate.
            ne_next, ni_next = bc_func(
                ne_next,
                ni_next,
                ne_curr,
                ni_curr,
                phi_iter,
                gamma_,
                anode_electron_induced_yield_,
                transport.T_e_eV,
                transport.mu_i,
                transport.mu_e,
                dx,
                dt,
                Gamma_ext_anode=Gamma_ext_anode,
                Gamma_ext_cathode=Gamma_ext_cathode,
                use_vaughan_sey=use_vaughan_sey_,
                vaughan_Emax0_eV=vaughan_Emax0_eV_,
                vaughan_dmax0=vaughan_dmax0_,
                vaughan_ks=vaughan_ks_,
                vaughan_z=vaughan_z_,
                vaughan_E0=vaughan_E0_,
                anode_ion_boundary=cfg.anode_ion_boundary,
                anode_electron_boundary=cfg.anode_electron_boundary,
                cathode_ion_boundary=cfg.cathode_ion_boundary,
                cathode_electron_boundary=cfg.cathode_electron_boundary,
            )

            # Solve Poisson with Dirichlet BCs:
            #   phi(0)      = V_gap[n_idx + 1]
            #   phi(Nx - 1) = 0
            poisson_1d_dirichlet_interior(
                n_i=ni_next,
                n_e=ne_next,
                dx=dx,
                phi_left=float(V_gap[n_idx + 1]),
                phi_right=0.0,
                ab_int=ab_int,
                phi_out=phi_next,
                E_out=E_next,
            )

            # Convergence check on phi iterate.
            if np.max(np.abs(phi_next - phi_iter)) < 1e-6:
                break
            phi_iter[:] = phi_next

        # (g) CFL diagnostic for drift term.
        c_cfl[n_idx + 1] = CFL_test(
            transport.mu_e,
            transport.mu_i,
            E_next,
            dt,
            dx,
            time,
            n_idx,
        )

        # (h) Save snapshots at configured interval.
        if (n_idx + 1) % SAVE_EVERY == 0:
            k = (n_idx + 1) // SAVE_EVERY

            # Save the same electron flux definition used for circuit coupling.
            Gamma_e_to_save = Gamma_e_for_circuit if emission_to_circuit else Gamma_e_row

            snapshot_kwargs = dict(
                k=k,
                n_i_sampled=n_i_sampled,
                n_e_sampled=n_e_sampled,
                phi_sampled=phi_sampled,
                E_sampled=E_sampled,
                ni=ni_next,
                ne=ne_next,
                phi=phi_next,
                E=E_next,
                log_intermediate=LOG_INTERMEDIATE,
                Gamma_i_sampled=Gamma_i_sampled,
                Gamma_e_sampled=Gamma_e_sampled,
                townsend_alpha_sampled=townsend_alpha_sampled,
                nu_i_sampled=nu_i_sampled,
                S_sampled=S_sampled,
                Gamma_i=Gamma_i_row,
                Gamma_e=Gamma_e_to_save,
                townsend_alpha=townsend_alpha_row,
                nu=nu_row,
                S=S_row,
            )
            write_snapshot(**snapshot_kwargs)

        # (i) Roll arrays: next -> current.
        ni_curr, ni_next   = ni_next, ni_curr
        ne_curr, ne_next   = ne_next, ne_curr
        phi_curr, phi_next = phi_next, phi_curr
        E_curr,   E_next   = E_next,   E_curr

    elapsed = pytime.perf_counter() - start
    print(f"Elapsed time: {elapsed:.6f} s")

    # ------------------------------------------------------------
    # 10) Pack final state and return.
    # ------------------------------------------------------------
    return SimulationState(
        cfg=cfg,
        time=time,
        x=x_array,
        V_gap=np.array(V_gap),
        I_discharge=np.array(I_discharge),
        c_cfl=np.array(c_cfl),
        ne_final=ne_curr.copy(),
        ni_final=ni_curr.copy(),
        phi_final=phi_curr.copy(),
        E_final=E_curr.copy(),
    )


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Simple smoke-test configuration when running this file directly.
    # -----------------------------------------------------------------
    cfg = SimulationConfig(
        waveform_type="step",
        V_peak=200.0,
        p_Torr=1.0,
        gas="argon",
        run_name="argon_pd1torrcm_vapp200V_step",
    )

    state = run_simulation(cfg)
    V_app_func = make_voltage_waveform(cfg)
    run_configured_diagnostics(cfg, state, V_app_func)
