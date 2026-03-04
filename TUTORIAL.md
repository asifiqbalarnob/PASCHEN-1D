# PASCHEN-1D Tutorial (Current Code Schema)

This tutorial matches the current code structure and `SimulationConfig` schema.

## 1. Typical Workflow

1. Edit `config.py` (or instantiate `SimulationConfig(...)` in notebook).
2. Run `paschen_1d_driver.ipynb`.
3. Inspect auto-generated diagnostics.
4. Replot from saved data in `paschen_1d_postprocess_driver.ipynb` for publication-style customization.

## 2. Running a Simulation

Use the notebook driver (`paschen_1d_driver.ipynb`) with:

```python
from config import SimulationConfig
from paschen_1d import run_simulation, run_configured_diagnostics
from physics import make_voltage_waveform

cfg = SimulationConfig()
state = run_simulation(cfg)
V_app_func = make_voltage_waveform(cfg)
run_configured_diagnostics(cfg, state, V_app_func)
```

The solver creates `<run_name>/` and writes sampled data + metadata.

## 3. Core Config Sections

## 3.1 Geometry and Plasma

- `L`, `A`, `l`, `eps_r`
- `gas`, `p_Torr`, `T_e`, `T_i`, `gamma`, `n0`

`gamma` is the cathode ion-induced secondary electron yield used when
`cathode_electron_boundary="electron_emission"`.

## 3.2 Coefficient Source Selection

The active code now separates the governing fluid equations from the source
used to supply the coefficients that appear in them.

Transport sources:

- `electron_transport_source`
- `ion_transport_source`

Ionization-coefficient source:

- `townsend_alpha_source`

Allowed source labels:

- `user_defined_equations`
- `swarm_data_table_interpolation`

`user_defined_equations` uses the editable closures in `physics.py`. Those
functions may return constant profiles or field-dependent profiles.

`swarm_data_table_interpolation` currently supports the electron-side path:

- `mu_e(E/N)`
- `D_e(E/N)`
- Townsend `alpha(E/N)` through tabulated `alpha/N`

The bundled BOLSIG+-generated swarm-data files are used for the electron-side
path. They do not provide ion transport coefficients, so the shipped example
cases keep ion transport in the `user_defined_equations` path. The config
schema still includes:

- `ion_swarm_data_path`
- `ion_swarm_data_gas`

so externally supplied ion swarm-data sources can be referenced through the
same interface.

Swarm-data file controls:

- `electron_swarm_data_path`
- `electron_swarm_data_gas`
- `ion_swarm_data_path`
- `ion_swarm_data_gas`

Optional dedicated Townsend-alpha overrides:

- `townsend_alpha_swarm_data_path`
- `townsend_alpha_swarm_data_gas`

If the Townsend-alpha override fields are left as `None`, the code reuses the
electron swarm-data source.

### Adding A New Gas

The `gas` field in the configuration is an internal PASCHEN-1D label. Changing `gas = "..."` by itself does not create support for a new gas. The same label must be used consistently anywhere the code selects gas-specific behavior.

There are two supported extension paths:

1. `user_defined_equations`
- Update the user-defined coefficient closures in `PASCHEN-1D/physics.py`.
- The main edit points are:
  - `compute_user_defined_electron_mobility(...)`
  - `compute_user_defined_ion_mobility(...)`
  - `compute_user_defined_electron_diffusion(...)`
  - `compute_user_defined_ion_diffusion(...)`
  - `compute_user_defined_townsend_alpha(...)`
  - `compute_user_defined_recombination_coefficient(...)`
- If the default reference-state logic needs gas-specific baseline values, also update `build_transport_reference_state(...)`.
- These user-defined functions may return either constant profiles or field-dependent spatial profiles.

2. `swarm_data_table_interpolation`
- Provide a compatible swarm-data file and point the configuration to it through:
  - `electron_swarm_data_path`
  - `electron_swarm_data_gas`
- If Townsend alpha also uses swarm-data interpolation, either:
  - leave `townsend_alpha_swarm_data_path` / `townsend_alpha_swarm_data_gas` as `None` to inherit from the electron swarm-data source, or
  - set explicit Townsend-alpha overrides.
- The `gas` label, `electron_swarm_data_gas`, and any Townsend-alpha swarm-data gas override must match exactly.

Current implementation scope:
- Electron mobility, electron diffusion, and Townsend alpha can use swarm-data interpolation.
- The shipped examples keep ion transport in `user_defined_equations`, because the bundled swarm-data files only provide the electron-side quantities used by the current workflow. The config schema still includes `ion_swarm_data_path` and `ion_swarm_data_gas` so externally supplied ion swarm-data sources can be referenced when available.


## 3.3 Circuit Setup

Choose topology with `circuit_type`:

- `dielectric_plasma`
- `R0_Cp`
- `R0_Cp_Rm`
- `R0_Cs_Cp`
- `R0_Cs_Cp_Rm`
- `R0_Cs_Ls_Cp`
- `R0_Cs_Ls_Cp_Rm`
- `R0_Cs_Ls_Cp_Lp`
- `R0_Cs_Ls_Cp_Lp_Rm`

Set solver with:

- `circuit_time_scheme = "explicit_euler"` or `"implicit_euler"`

Set element values:

- `R0`, `C_s`, `L_s`, `C_p`, `L_p`, `R_m`

For stiff parameter sets, use `"implicit_euler"`.

## 3.4 Applied Voltage

- `waveform_type` in `{"step", "gaussian", "dc", "rf"}`
- waveform-specific fields:
  - step/dc: `V_peak`, `tV_start`, `tV_end`
  - gaussian: `V_peak`, `tau`, `t_peak`
  - rf: `V_peak`, `f_rf`, `V_dc`, `phi_rf`

## 3.5 Grid and Numerics

- `Nt`, `Nx`, `T_total`
- `kt_limiter_theta`

## 3.6 Boundary Modes

Per electrode and species:

- `anode_ion_boundary`
- `anode_electron_boundary`
- `cathode_ion_boundary`
- `cathode_electron_boundary`

Allowed mode labels:

- `zero_density`
- `implicit_drift_closure`
- `electron_emission` (electrons only)

Example physically common setup:

```python
anode_ion_boundary = "zero_density"
anode_electron_boundary = "implicit_drift_closure"
cathode_ion_boundary = "implicit_drift_closure"
cathode_electron_boundary = "electron_emission"
```

Electron-emission mode is side-specific in the current implementation:

- Cathode `electron_emission`:
  - uses constant ion-induced SEE coefficient `gamma`
  - may include external emission flux at cathode if enabled
- Anode `electron_emission`:
  - uses electron-induced SEE
  - SEE can be constant (`anode_electron_induced_yield`) or Vaughan-based (`use_vaughan_sey=True`)
  - may include external emission flux at anode if enabled

## 3.7 Volume Source Controls

- `enable_volume_sources`
- `enable_ionization_source`
- `enable_recombination_sink`

These only affect volumetric continuity source terms (ionization/recombination).

## 3.8 External Emission Controls

Master switches:

- `enable_external_emission`
- `enable_anode_external_emission`
- `enable_cathode_external_emission`
- `enable_emission_in_circuit_current`

Per-electrode model toggles (any combination allowed):

- anode:
  - `anode_enable_constant_J_emission`
  - `anode_enable_fn_emission`
  - `anode_enable_mg_emission`
  - `anode_enable_rd_emission`
  - `anode_enable_quantum_pulse_emission`
- cathode:
  - `cathode_enable_constant_J_emission`
  - `cathode_enable_fn_emission`
  - `cathode_enable_mg_emission`
  - `cathode_enable_rd_emission`
  - `cathode_enable_quantum_pulse_emission`

Additional anode SEE controls:

- `anode_electron_induced_yield`
- `use_vaughan_sey`
- `vaughan_Emax0_eV`
- `vaughan_dmax0`
- `vaughan_ks`
- `vaughan_z`
- `vaughan_E0`

Vaughan normalization uses:
\[
w = \frac{E_{\text{impact}} - E_0}{E_{\max} - E_0},
\]
with `E0 = vaughan_E0`. In this code path, the anode impact energy is a
proxy computed from incoming normal electron drift and a fixed thermal term.

### Shared vs Separate Electrode Parameters

- `electrode_material_mode = "shared"`:
  - use `shared_*` parameter fields for both electrodes
- `electrode_material_mode = "separate"`:
  - use `anode_*` and `cathode_*` parameter fields independently

This supports different anode/cathode materials and laser/emission settings.

## 4. Swarm-Data Files

Bundled example swarm-data source files:

- `ar_swarm_output.dat`
- `n2_swarm_output.dat`

These are raw swarm-data output files already in the format expected by the
active parser. Additional compatible swarm-data files can be generated
externally and referenced through the config.

Accepted swarm-data file patterns:

1. Raw swarm-data output containing the required named section(s):
   - `Mobility *N (1/m/V/s)`
   - `Diffusion coefficient *N (1/m/s)`
   - `Townsend ioniz. coef. alpha/N (m2)`
   In each section, the parser reads:
   - column 1: `E/N` in `Td`
   - column 2: the corresponding tabulated quantity

2. A simple two-column table:
   - column 1: `E/N [Td]`
   - column 2: the requested quantity

Expected second-column quantity for a two-column table:
- electron transport source:
  - `mu_e * N` for mobility
  - `D_e * N` for diffusion
- Townsend-alpha source:
  - `alpha / N`

Numerical requirements:
- at least two data rows
- strictly positive `E/N`
- strictly positive `mu_e * N` and `D_e * N`
- non-negative `alpha / N`

## 5. Emission Models in This Version

- `constant_J`
  - time-windowed constant current density
- `fn`
  - Fowler-Nordheim field emission
- `mg`
  - cold Murphy-Good with Schottky-Nordheim image-force barrier correction
- `rd`
  - Richardson-Dushman thermionic emission
- `quantum_pulse`
  - pulsed quantum photoemission model

External emission enters through boundary electron-emission closure.

Important model split:
- `use_vaughan_sey` applies only to anode electron-induced SEE.
- Cathode SEE remains the constant-`gamma` model.

## 6. Diagnostics Configuration

Diagnostics are organized under:

- `cfg.diagnostics.temporal`
- `cfg.diagnostics.spatial`
- `cfg.diagnostics.averaged_spatial`

## 6.1 Temporal Diagnostics

Select quantities:

- `V_app`, `V_gap`, `I_discharge`, `cfl`, `particle_inventory`

Configure:

- `quantities=(...)`
- optional grouped overlays: `plot_groups=((...),(...))`
- optional time window: `t_start`, `t_end`
- optional file prefix: `savepath_prefix`

## 6.2 Spatial Diagnostics

Select quantities:

- `ne`, `ni`, `phi`, `E`, `Gamma_i`, `Gamma_e`, `townsend_alpha`, `nu_i`, `S`

Configure:

- `quantities=(...)`
- optional grouped overlays: `plot_groups=((...),(...))`
- sample times: `t_samples=(...)` or `None` for final time
- x-axis unit: `x_unit in {"m","cm","mm"}`
- optional file prefix: `savepath_prefix`

Example:

```python
cfg.diagnostics.temporal.plot_groups = (("V_app", "V_gap"), ("I_discharge",))
cfg.diagnostics.temporal.t_start = 0.5e-6
cfg.diagnostics.temporal.t_end = 1.0e-6

cfg.diagnostics.spatial.plot_groups = (("ne", "ni"), ("phi",), ("E",))
cfg.diagnostics.spatial.t_samples = (0.8e-6, 1.0e-6)
cfg.diagnostics.spatial.x_unit = "cm"
```

## 6.3 Averaged Spatial Diagnostics

Use this family for CCP-style benchmarking when cycle-averaged or time-window
averaged profiles are more meaningful than instantaneous snapshots.

Supported quantity labels are the same spatial profile labels used by
`cfg.diagnostics.spatial`, for example:

- `ne`, `ni`, `phi`, `E`
- `Gamma_i`, `Gamma_e`
- `townsend_alpha`, `nu_i`, `S`

Main controls:

- `enabled`
- `quantities`
- optional grouped overlays: `plot_groups=((...),(...))`
- `mode = "time_window"` or `mode = "last_n_cycles"`
- `t_avg_start`, `t_avg_end` for `time_window`
- `N_cycle_avg` for `last_n_cycles`
- `x_unit`

Example:

```python
cfg.diagnostics.averaged_spatial.enabled = True
cfg.diagnostics.averaged_spatial.plot_groups = (("ne", "ni"), ("phi",), ("E",))
cfg.diagnostics.averaged_spatial.mode = "last_n_cycles"
cfg.diagnostics.averaged_spatial.N_cycle_avg = 10
cfg.diagnostics.averaged_spatial.x_unit = "cm"
```

Important:
- averaged spatial diagnostics are computed from the saved snapshot cadence
  set by `save_every`
- if the run is RF-driven, `save_every` should be fine enough to resolve the
  RF waveform over the averaging window

## 7. Postprocessing Without Rerun

Use `paschen_1d_postprocess_driver.ipynb`:

- picks any run folder containing `run_metadata.json`
- replots directly from saved memmaps
- allows changing units/scales/grouping/labels/titles/limits
- supports averaged-spatial profile replots from saved runs
- saves publication-quality figures to:
  - `<run_name>/postprocess_figures/`

## 8. Case Templates

## 8.1 Zero-bias, no-activity baseline

- `waveform_type="dc"`, `V_peak=0.0`
- `enable_external_emission=False`
- `enable_volume_sources=False`

## 8.2 Cathode constant-J delayed turn-on

- `enable_external_emission=True`
- `enable_cathode_external_emission=True`
- `cathode_enable_constant_J_emission=True`
- in shared mode:
  - `shared_emission_J_const = ...`
  - `shared_emission_t_start = 0.5e-6`

## 8.3 Mixed electrode emission

Example:

- cathode: quantum + MG
- anode: FN only

Set:

- cathode toggles: `cathode_enable_quantum_pulse_emission=True`, `cathode_enable_mg_emission=True`
- anode toggles: `anode_enable_fn_emission=True`
- per-electrode enable switches ON
- use `electrode_material_mode="separate"` if material/laser parameters differ.

## 9. Outputs Written per Run

In `<run_name>/`:

- scalar time histories:
  - `Vgap_mm.dat`, `Idischarge_mm.dat`, `c_cfl_mm.dat`
- sampled fields:
  - `ne_sampled_mm.dat`, `ni_sampled_mm.dat`, `phi_sampled_mm.dat`, `E_sampled_mm.dat`
- optional sampled intermediates (if `log_intermediate=True`):
  - `Gamma_i_sampled_mm.dat`, `Gamma_e_sampled_mm.dat`
  - `townsend_alpha_sampled_mm.dat`, `nu_i_sampled_mm.dat`, `S_sampled_mm.dat`
- metadata:
  - `run_metadata.json`

## 10. Common Setup Checks

Before launching long runs:

1. Confirm run summary printout at startup (geometry, waveform, circuit, BCs, emission).
2. Keep boundary/electrode switches consistent:
   - if anode external emission is enabled, anode electron BC should be `electron_emission`
   - same logic for cathode.
3. Use implicit circuit scheme for stiff element choices.
4. Start with modest `Nt/Nx` for smoke tests before production runs.

## 11. Where to Extend Next

- Add/adjust new emission mechanisms in `emission.py`.
- Extend BC physics closures in `physics.py` and boundary orchestration in `numerics.py`.
- Add new diagnostics in `plotting.py` / `postprocess.py`.
