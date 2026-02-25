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

## 3.2 Circuit Setup

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

## 3.3 Applied Voltage

- `waveform_type` in `{"step", "gaussian", "dc", "rf"}`
- waveform-specific fields:
  - step/dc: `V_peak`, `tV_start`, `tV_end`
  - gaussian: `V_peak`, `tau`, `t_peak`
  - rf: `V_peak`, `f_rf`, `V_dc`, `phi_rf`

## 3.4 Grid and Numerics

- `Nt`, `Nx`, `T_total`
- `kt_limiter_theta`

## 3.5 Boundary Modes

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

## 3.6 Volume Source Controls

- `enable_volume_sources`
- `enable_ionization_source`
- `enable_recombination_sink`

These only affect volumetric continuity source terms (ionization/recombination).

## 3.7 External Emission Controls

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

## 4. Emission Models in This Version

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

## 5. Diagnostics Configuration

Diagnostics are organized under:

- `cfg.diagnostics.temporal`
- `cfg.diagnostics.spatial`

## 5.1 Temporal Diagnostics

Select quantities:

- `V_app`, `V_gap`, `I_discharge`, `cfl`, `particle_inventory`

Configure:

- `quantities=(...)`
- optional grouped overlays: `plot_groups=((...),(...))`
- optional time window: `t_start`, `t_end`
- optional file prefix: `savepath_prefix`

## 5.2 Spatial Diagnostics

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

## 6. Postprocessing Without Rerun

Use `paschen_1d_postprocess_driver.ipynb`:

- picks any run folder containing `run_metadata.json`
- replots directly from saved memmaps
- allows changing units/scales/grouping/labels/titles/limits
- saves publication-quality figures to:
  - `<run_name>/postprocess_figures/`

## 7. Case Templates

## 7.1 Zero-bias, no-activity baseline

- `waveform_type="dc"`, `V_peak=0.0`
- `enable_external_emission=False`
- `enable_volume_sources=False`

## 7.2 Cathode constant-J delayed turn-on

- `enable_external_emission=True`
- `enable_cathode_external_emission=True`
- `cathode_enable_constant_J_emission=True`
- in shared mode:
  - `shared_emission_J_const = ...`
  - `shared_emission_t_start = 0.5e-6`

## 7.3 Mixed electrode emission

Example:

- cathode: quantum + MG
- anode: FN only

Set:

- cathode toggles: `cathode_enable_quantum_pulse_emission=True`, `cathode_enable_mg_emission=True`
- anode toggles: `anode_enable_fn_emission=True`
- per-electrode enable switches ON
- use `electrode_material_mode="separate"` if material/laser parameters differ.

## 8. Outputs Written per Run

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

## 9. Common Setup Checks

Before launching long runs:

1. Confirm run summary printout at startup (geometry, waveform, circuit, BCs, emission).
2. Keep boundary/electrode switches consistent:
   - if anode external emission is enabled, anode electron BC should be `electron_emission`
   - same logic for cathode.
3. Use implicit circuit scheme for stiff element choices.
4. Start with modest `Nt/Nx` for smoke tests before production runs.

## 10. Where to Extend Next

- Add/adjust new emission mechanisms in `emission.py`.
- Extend BC physics closures in `physics.py` and boundary orchestration in `numerics.py`.
- Add new diagnostics in `plotting.py` / `postprocess.py`.
