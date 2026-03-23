# PASCHEN-1D Tutorial (Current Schema)

This tutorial matches the current `SimulationConfig` schema in `config.py`.

## 1. Run Flow

1. Choose a config file (`config.py` or one of the examples).
2. Run `paschen_1d_driver.ipynb`.
3. Inspect generated diagnostics.
4. Use `paschen_1d_postprocess_driver.ipynb` to replot saved runs without rerunning physics.

## 2. Minimal Driver Usage

```python
from config import SimulationConfig
from paschen_1d import run_simulation, run_configured_diagnostics
from physics import make_voltage_waveform

cfg = SimulationConfig()
state = run_simulation(cfg)
run_configured_diagnostics(cfg, state, make_voltage_waveform(cfg))
```

## 3. Configuration Layout

`SimulationConfig` is grouped as:

1. `run`
2. `numerics`
3. `geometry`
4. `plasma_state`
5. `plasma`
6. `user_defined_electron_kinetics`
7. `local_field_approximation`
8. `townsend_coefficient`
9. `ionization_frequency_source`
10. `recombination`
11. `waveform`
12. `boundary`
13. `circuit`
14. `emission`
15. `output`
16. `diagnostics`

Use grouped access everywhere (example: `cfg.geometry.L`, not `cfg.L`).

## 4. Physics-Model Selection

### 4.1 Electron kinetics

`cfg.plasma.electron_kinetics_model`:
- `"user_defined_electron_kinetics"`
- `"local_field_approximation"`

For LFA transport source:
- `cfg.local_field_approximation.electron_transport_source`
  - `"user_defined_equation"`
  - `"swarm_data_table_interpolation"`
- `cfg.local_field_approximation.electron_swarm_data_path`

### 4.2 Impact ionization

`cfg.plasma.impact_ionization_model`:
- `"from_townsend_alpha"`
- `"from_ionization_frequency"`

Townsend source (`from_townsend_alpha` branch):
- `cfg.townsend_coefficient.townsend_alpha_source_mode`
  - `"user_defined_equation"`
  - `"interpolate_from_e_over_n_table"`
- optional override path:
  - `cfg.townsend_coefficient.townsend_alpha_swarm_data_path`

Direct ionization-frequency source (`from_ionization_frequency` branch):
- `cfg.ionization_frequency_source.ionization_frequency_source_mode`
  - `"user_defined_equation"`
  - `"interpolate_from_e_over_n_table"`
- optional override path:
  - `cfg.ionization_frequency_source.ionization_frequency_swarm_data_path`

### 4.3 Recombination

- `cfg.recombination.recombination_coefficient`

## 5. Boundary Conditions

Per electrode/per species:
- `cfg.boundary.anode_ion_boundary`
- `cfg.boundary.anode_electron_boundary`
- `cfg.boundary.cathode_ion_boundary`
- `cfg.boundary.cathode_electron_boundary`

Allowed mode labels:
- `zero_density`
- `implicit_drift_closure`
- `electron_emission` (electrons only)

Common setup:

```python
cfg.boundary.anode_ion_boundary = "zero_density"
cfg.boundary.anode_electron_boundary = "implicit_drift_closure"
cfg.boundary.cathode_ion_boundary = "implicit_drift_closure"
cfg.boundary.cathode_electron_boundary = "electron_emission"
```

## 6. External Emission

Master switches:
- `cfg.emission.enable_external_emission`
- `cfg.emission.enable_anode_external_emission`
- `cfg.emission.enable_cathode_external_emission`
- `cfg.emission.enable_emission_in_circuit_current`

Per-electrode emission mechanisms (any combination):
- anode toggles:
  - `anode_enable_constant_J_emission`
  - `anode_enable_fn_emission`
  - `anode_enable_mg_emission`
  - `anode_enable_rd_emission`
  - `anode_enable_quantum_pulse_emission`
- cathode toggles:
  - `cathode_enable_constant_J_emission`
  - `cathode_enable_fn_emission`
  - `cathode_enable_mg_emission`
  - `cathode_enable_rd_emission`
  - `cathode_enable_quantum_pulse_emission`

Material parameter mode:
- `cfg.emission.electrode_material_mode = "shared"` or `"separate"`

SEE controls:
- cathode ion-induced SEE: `cfg.emission.gamma`
- anode electron-induced SEE:
  - constant: `cfg.emission.anode_electron_induced_yield`
  - Vaughan model: set `cfg.emission.use_vaughan_sey = True`

## 7. Numerics and Performance

Core controls:
- `cfg.numerics.Nx`, `cfg.numerics.Nt`
- `cfg.numerics.kt_limiter_theta`

Hot-loop backend:
- `cfg.numerics.hotloop_backend = "numpy"` or `"numba"`
- `cfg.numerics.numba_parallel` (usually slower for small 1D grids)

Adaptive substepping:
- `cfg.numerics.use_adaptive_substepping`
- `cfg.numerics.target_cfl_substep`
- `cfg.numerics.max_substeps`
- `cfg.numerics.adaptive_substep_overflow_policy`
- `cfg.numerics.adaptive_substep_warn_every`

BC+Poisson Picard controls:
- `cfg.numerics.bc_poisson_picard_min_iter`
- `cfg.numerics.bc_poisson_picard_max_iter`
- `cfg.numerics.bc_poisson_picard_tol`

## 8. Swarm-Data Files

Accepted file patterns:
1. Raw swarm text with `E/N (Td)` named sections.
2. Two-column tables (`E/N`, quantity).

Used section labels:
- `Mobility *N (1/m/V/s)`
- `Diffusion coefficient *N (1/m/s)`
- `Townsend ioniz. coef. alpha/N (m2)`
- `Total ionization frequency /N (m3/s)`

Bundled examples:
- `ar_swarm_output.dat`
- `n2_swarm_output.dat`
- `ar_swarm_output_full_EoverN.dat`
- `n2_swarm_output_full_EoverN.dat`

## 9. Diagnostics

### 9.1 Temporal (`cfg.diagnostics.temporal`)

Quantities:
- `V_app`, `V_gap`, `I_discharge`, `cfl`
- `picard_iterations`
- `adaptive_substeps`, `adaptive_dt_sub`, `adaptive_cfl_est`
- `particle_inventory`

Options:
- `quantities`
- `plot_groups`
- `t_start`, `t_end`
- `savepath_prefix`

### 9.2 Spatial (`cfg.diagnostics.spatial`)

Quantities:
- `ne`, `ni`, `phi`, `E`
- `Gamma_i`, `Gamma_e`
- `townsend_alpha`, `nu_i`, `S_ion`, `S`
- `mu_e`, `D_e`

Options:
- `quantities`
- `plot_groups`
- `t_samples` (`None` means final time)
- `x_unit`
- `savepath_prefix`

### 9.3 Averaged spatial (`cfg.diagnostics.averaged_spatial`)

Options:
- `enabled`
- `quantities`
- `plot_groups`
- `mode = "time_window" | "last_n_cycles"`
- `t_avg_start`, `t_avg_end`
- `N_cycle_avg`
- `x_unit`
- `savepath_prefix`

## 10. Output Files

Per run directory `<run_name>/`:

Scalar histories:
- `Vgap_mm.dat`
- `Idischarge_mm.dat`
- `c_cfl_mm.dat`
- `picard_iterations_mm.dat`
- `adaptive_substeps_mm.dat`
- `adaptive_dt_sub_mm.dat`
- `adaptive_cfl_est_mm.dat`

Sampled fields:
- `ne_sampled_mm.dat`, `ni_sampled_mm.dat`
- `phi_sampled_mm.dat`, `E_sampled_mm.dat`

Optional sampled intermediates (`cfg.output.log_intermediate=True`):
- `Gamma_i_sampled_mm.dat`, `Gamma_e_sampled_mm.dat`
- `townsend_alpha_sampled_mm.dat`, `nu_i_sampled_mm.dat`
- `S_ion_sampled_mm.dat`, `S_sampled_mm.dat`
- `mu_e_sampled_mm.dat`, `D_e_sampled_mm.dat`

Metadata:
- `run_metadata.json`

## 11. Practical Setup Checklist

Before long runs:
1. Verify startup run summary (geometry, waveform, circuit, BC, emission, diagnostics).
2. Confirm boundary/emission consistency:
   - if `enable_anode_external_emission=True`, set `anode_electron_boundary="electron_emission"`
   - if `enable_cathode_external_emission=True`, set `cathode_electron_boundary="electron_emission"`
3. Use `circuit_time_scheme="implicit_euler"` for stiff circuit parameter sets.
4. Start with reduced `Nt/Nx`, then scale up.
