# PASCHEN-1D

PASCHEN-1D is a 1D drift-diffusion-Poisson plasma solver with configurable
external-circuit coupling, boundary-condition closures, and modular
multi-mechanism external electron emission.

## Scope of This Release

This package supports two electron-kinetics branches:
- `user_defined_electron_kinetics`
- `local_field_approximation`

Electron-energy PDE modes are not part of this release.

## Package Contents

Core modules:
- `paschen_1d.py`
- `numerics.py`
- `numerics_jit.py`
- `physics.py`
- `circuit.py`
- `circuit_implicit_euler.py`
- `emission.py`
- `outputs.py`
- `plotting.py`
- `postprocess.py`
- `physical_constants.py`

Configuration:
- `config.py`
- `config_nitrogen_pulsed_discharge.py`
- `config_argon_photoemission.py`
- `config_argon_dc_glow_breakdown.py`

Notebook drivers:
- `paschen_1d_driver.ipynb`
- `paschen_1d_postprocess_driver.ipynb`

Bundled swarm-data examples:
- `ar_swarm_output_full_EoverN.dat`
- `n2_swarm_output_full_EoverN.dat`

## Core Capabilities

- 1D electron/ion continuity with KT flux update and RK4 stepping.
- 1D Poisson solve (Dirichlet electrodes).
- Configurable circuit topologies (`dielectric_plasma`, `R0/Cs/Ls/Cp/Lp/Rm` variants).
- Explicit and implicit Euler circuit solvers.
- Per-electrode/per-species boundary mode selection:
  - `zero_density`
  - `implicit_drift_closure`
  - `electron_emission` (electrons only)
- Electron-emission boundary closure with:
  - cathode ion-induced SEE (`gamma`)
  - anode electron-induced SEE (constant yield or Vaughan model)
  - optional externally driven emission fluxes at anode/cathode
- External emission mechanisms:
  - `constant_J`, `fn`, `mg`, `rd`, `quantum_pulse`
  - per-electrode independent toggles
  - shared or separate electrode material parameter sets
- Adaptive substepping (optional) for drift-CFL control.
- NumPy or Numba hot-loop backend (`hotloop_backend`).
- Three diagnostics families:
  - temporal
  - spatial
  - averaged spatial

## Configuration Structure

`SimulationConfig` is organized into grouped dataclasses in this order:

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

All runtime modules use grouped access (`cfg.geometry.L`, `cfg.run.T_total`, etc.).

## Coefficient and Ionization Source Selection

### Electron transport

- `plasma.electron_kinetics_model = "user_defined_electron_kinetics"`
  - uses user equations in `physics.py`.
- `plasma.electron_kinetics_model = "local_field_approximation"`
  - source selected by `local_field_approximation.electron_transport_source`:
    - `"user_defined_equation"`
    - `"swarm_data_table_interpolation"`

### Ionization frequency

Top-level selector:
- `plasma.impact_ionization_model`
  - `"from_townsend_alpha"`
  - `"from_ionization_frequency"`

Townsend path source:
- `townsend_coefficient.townsend_alpha_source_mode`
  - `"user_defined_equation"`
  - `"interpolate_from_e_over_n_table"`

Direct frequency path source:
- `ionization_frequency_source.ionization_frequency_source_mode`
  - `"user_defined_equation"`
  - `"interpolate_from_e_over_n_table"`

Recombination:
- `recombination.recombination_coefficient`

## Swarm-Data Format

The parser accepts either:

1. Raw swarm-output text containing named `E/N (Td)` sections, or
2. Two-column tables (`E/N [Td]`, quantity).

Required section labels by quantity:
- mobility: `Mobility *N (1/m/V/s)`
- diffusion: `Diffusion coefficient *N (1/m/s)`
- Townsend alpha: `Townsend ioniz. coef. alpha/N (m2)`
- ionization frequency: `Total ionization frequency /N (m3/s)`

Numerical requirements:
- at least 2 rows
- strictly positive `E/N`
- non-negative values for alpha/N and ionization frequency/N
- positive values for `mu*N` and `D*N`

## Quick Start

1. Edit `config.py` (or one of the example config files).
2. Run `paschen_1d_driver.ipynb`.
3. Results are written to `<run_name>/`.
4. Replot without rerun using `paschen_1d_postprocess_driver.ipynb`.

## Output Files

In `<run_name>/`:
- scalar histories: `Vgap_mm.dat`, `Idischarge_mm.dat`, `c_cfl_mm.dat`
- optional scalar histories: `picard_iterations_mm.dat`, `adaptive_substeps_mm.dat`, `adaptive_dt_sub_mm.dat`, `adaptive_cfl_est_mm.dat`
- sampled fields: `ne_sampled_mm.dat`, `ni_sampled_mm.dat`, `phi_sampled_mm.dat`, `E_sampled_mm.dat`
- optional sampled intermediates: `Gamma_i_sampled_mm.dat`, `Gamma_e_sampled_mm.dat`, `townsend_alpha_sampled_mm.dat`, `nu_i_sampled_mm.dat`, `S_ion_sampled_mm.dat`, `S_sampled_mm.dat`, `mu_e_sampled_mm.dat`, `D_e_sampled_mm.dat`
- run metadata: `run_metadata.json`

## License and Citation

- License file: `LICENSE`
- Commercial use is not permitted without separate written permission.
- Any use/modification/redistribution/publication must acknowledge/cite:

Asif Iqbal, Yves Heri, Bingqing Wang, Lan Jin, Md Arifuzzaman Faisal, and Peng Zhang,
"PASCHEN-1D: A one-dimensional fluid plasma solver with multi-mechanism surface
emission and flexible external circuit coupling".
