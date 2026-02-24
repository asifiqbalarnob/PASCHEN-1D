# PASCHEN-1D

1D drift-diffusion-Poisson plasma solver with external-circuit coupling, configurable boundary conditions, and modular external electron-emission models.

## What This Package Contains

- Core solver:
  - `paschen_1d.py`
  - `numerics.py`
  - `physics.py`
  - `circuit.py`
  - `circuit_implicit_euler.py`
  - `emission.py`
- Configuration:
  - `config.py`
  - example presets:
    - `config_nitrogen_pulsed_discharge.py`
    - `config_argon_photoemission.py`
    - `config_argon_dc_glow_breakdown.py`
- Output and plotting:
  - `outputs.py`
  - `plotting.py`
  - `postprocess.py`
- Notebook drivers:
  - `paschen_1d_driver.ipynb` (run simulation + configured diagnostics)
  - `paschen_1d_postprocess_driver.ipynb` (replot from saved runs without rerunning)

## Main Features

- 1D electron/ion continuity with KT flux + RK4 update.
- 1D Poisson solve with Dirichlet electrode potentials.
- External circuit topologies (`R0/Cs/Ls/Cp/Lp/Rm` variants + dielectric-plasma mode).
- Explicit Euler and implicit Euler circuit stepping.
- Per-electrode boundary mode control (anode/cathode, ion/electron separately):
  - `zero_density`
  - `implicit_drift_closure`
  - `electron_emission` (electrons only)
- External emission through boundary-flux coupling (Eq. 11a-style form), including:
  - `constant_J`
  - Fowler-Nordheim (`fn`)
  - cold Murphy-Good with Schottky-Nordheim correction (`mg`)
  - Richardson-Dushman (`rd`)
  - `quantum_pulse`
- Per-electrode emission model toggles (different model combinations on anode/cathode).
- Shared vs separate electrode material/emission parameter sets.
- Config-driven diagnostics:
  - temporal (with optional grouped overlays and time-window selection)
  - spatial (with optional grouped overlays and chosen snapshot times)
- Postprocessing notebook for unit/scale/title/legend/axis-limit edits and figure export.

## Quick Start

1. Open `config.py` and set your case parameters.
2. Run `paschen_1d_driver.ipynb`:
   - creates `<run_name>/`
   - writes sampled fields and histories
   - writes `run_metadata.json`
   - generates configured diagnostics
3. Optional: open `paschen_1d_postprocess_driver.ipynb` to regenerate plots from saved data only.

## Key Configuration Entry Points

- Simulation identity and grid:
  - `run_name`, `L`, `A`, `Nt`, `Nx`, `T_total`
- Gas and plasma:
  - `gas`, `p_Torr`, `T_e`, `T_i`, `gamma`, `n0`
- Circuit:
  - `circuit_type`, `circuit_time_scheme`, `R0`, `C_s`, `L_s`, `C_p`, `L_p`, `R_m`
- Voltage waveform:
  - `waveform_type` + corresponding waveform parameters
- Boundary modes:
  - `anode_ion_boundary`, `anode_electron_boundary`
  - `cathode_ion_boundary`, `cathode_electron_boundary`
- Volume source toggles:
  - `enable_volume_sources`, `enable_ionization_source`, `enable_recombination_sink`
- External emission:
  - `enable_external_emission`
  - `enable_anode_external_emission`, `enable_cathode_external_emission`
  - per-model toggles per electrode (for simultaneous multi-model emission)
  - `electrode_material_mode = "shared" | "separate"`

## Outputs

For each run, outputs are written to `<run_name>/`:

- Time histories:
  - `Vgap_mm.dat`
  - `Idischarge_mm.dat`
  - `c_cfl_mm.dat`
- Sampled fields:
  - `ne_sampled_mm.dat`, `ni_sampled_mm.dat`
  - `phi_sampled_mm.dat`, `E_sampled_mm.dat`
  - optional sampled intermediates (if enabled)
- Metadata:
  - `run_metadata.json`

## Notes

- The startup run summary prints the resolved setup (geometry, circuit, boundary modes, emission toggles, diagnostics window).
- Non-fatal configuration mismatches are reported when `warn_on_config_mismatch=True`.
- For stiff circuit parameter sets, prefer `circuit_time_scheme="implicit_euler"`.

## Next Recommended Files

- Full user guide: `TUTORIAL.md`
- Main config schema: `config.py`

