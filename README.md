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
- Boundary electron-emission closure supports:
  - cathode ion-induced SEE via constant `gamma`
  - anode electron-induced SEE via constant `anode_electron_induced_yield` or Vaughan model
  - externally driven emission flux contributions at anode and/or cathode
- External emission models available to drive boundary emission flux:
  - `constant_J`
  - Fowler-Nordheim (`fn`)
  - cold Murphy-Good with Schottky-Nordheim correction (`mg`)
  - Richardson-Dushman (`rd`)
  - `quantum_pulse`
- Per-electrode emission model toggles (different model combinations on anode/cathode).
- Shared vs separate electrode material/emission parameter sets.
- Config-selectable coefficient sources:
  - `user_defined_equations`
  - `swarm_data_table_interpolation`
- Electron-side swarm-data interpolation currently supports:
  - `mu_e(E/N)`
  - `D_e(E/N)`
  - Townsend `alpha(E/N)` via tabulated `alpha/N`
- The shipped examples keep ion transport in the `user_defined_equations` path
  because the bundled BOLSIG+-generated swarm-data files do not provide ion
  transport coefficients. The config schema still includes `ion_swarm_data_*`
  fields so externally supplied ion swarm-data sources can be referenced.
- Config-driven diagnostics:
  - temporal (with optional grouped overlays and time-window selection)
  - spatial (with optional grouped overlays and chosen snapshot times)
  - averaged spatial (time-window averages or averages over the last `N` RF cycles)
- Postprocessing notebook for unit/scale/title/legend/axis-limit edits and figure export.

## Quick Start

1. Open `config.py` and set your case parameters.
2. Run `paschen_1d_driver.ipynb`:
   - creates `<run_name>/`
   - writes sampled fields and histories
   - writes `run_metadata.json`
   - generates configured diagnostics
3. Optional: open `paschen_1d_postprocess_driver.ipynb` to regenerate plots from saved data only.

Bundled example swarm-data source files:
- `ar_swarm_output.dat`
- `n2_swarm_output.dat`

Additional compatible swarm-data files can be generated externally and
referenced through the config.


## Key Configuration Entry Points

- Simulation identity and grid:
  - `run_name`, `L`, `A`, `Nt`, `Nx`, `T_total`
- Gas and plasma:
  - `gas`, `p_Torr`, `T_e`, `T_i`, `gamma`, `anode_electron_induced_yield`, `n0`
- Coefficient sources:
  - `electron_transport_source`, `ion_transport_source`, `townsend_alpha_source`
  - `electron_swarm_data_path`, `electron_swarm_data_gas`
  - `ion_swarm_data_path`, `ion_swarm_data_gas`
  - optional Townsend-alpha overrides:
    `townsend_alpha_swarm_data_path`, `townsend_alpha_swarm_data_gas`
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
- Anode Vaughan SEE controls:
  - `use_vaughan_sey`
  - `vaughan_Emax0_eV`, `vaughan_dmax0`, `vaughan_ks`, `vaughan_z`, `vaughan_E0`

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

## Swarm-Data File Format

Accepted swarm-data file patterns:

1. Raw swarm-data output containing the required named section(s):
   - `Mobility *N (1/m/V/s)`
   - `Diffusion coefficient *N (1/m/s)`
   - `Townsend ioniz. coef. alpha/N (m2)`
   In each section, PASCHEN-1D reads the first two numeric columns as:
   - column 1: `E/N` in `Td`
   - column 2: the corresponding tabulated quantity

2. A two-column table with:
   - column 1: `E/N [Td]`
   - column 2: the requested quantity

For two-column tables, the expected second-column quantity is:
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

## Notes

- The startup run summary prints the resolved setup (geometry, circuit, boundary modes, emission toggles, diagnostics window).
- Non-fatal configuration mismatches are reported when `warn_on_config_mismatch=True`.
- For stiff circuit parameter sets, prefer `circuit_time_scheme="implicit_euler"`.
- `use_vaughan_sey` affects only the anode `electron_emission` boundary path.
- Cathode SEE remains the constant-`gamma` model.
- `swarm_data_table_interpolation` is used in the shipped package for electron
  transport (`mu_e`, `D_e`) and the Townsend-ionization path. The bundled
  BOLSIG+-generated files do not include ion transport coefficients, so the
  provided example cases keep ion transport in the user-defined path. The
  config schema already exposes `ion_swarm_data_path` / `ion_swarm_data_gas`
  for externally supplied ion swarm-data sources.

## License and Citation

- This project is distributed under the `PASCHEN-1D Non-Commercial Citation License (PNCCL) v1.0`.
- Commercial use is not permitted without separate written permission.
- Any use, modification, redistribution, or publication of results obtained
  using this software must acknowledge/cite:

  Asif Iqbal, Yves Heri, Bingqing Wang, Lan Jin, Md Arifuzzaman Faisal, and
  Peng Zhang, "PASCHEN-1D: A one-dimensional fluid plasma solver with
  multi-mechanism surface emission and flexible external circuit coupling".

## Next Recommended Files

- Full user guide: `TUTORIAL.md`
- Main config schema: `config.py`
