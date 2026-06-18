# PASCHEN-1D

PASCHEN-1D is a one-dimensional drift-diffusion-Poisson plasma solver with
surface-emission physics and external-circuit coupling. It is intended for
gas-breakdown, pulsed-discharge, plasma-surface interaction, and
plasma-circuit studies where electron/ion transport, sheath evolution,
emission, ionization, and the external circuit are strongly coupled.

## Scope of This Release

This release supports two electron-kinetics branches:

- `user_defined_electron_kinetics`
- `local_field_approximation`

The local-field branch can use user-defined transport/ionization formulas or
table interpolation from bundled/BOLSIG+-style swarm-output files. Electron
energy equation / energy PDE models are not included in this release.

## Key Capabilities

- 1D electron/ion continuity with Kurganov-Tadmor fluxes and slope limiting.
- Self-consistent Poisson solve with emission/sheath boundary iteration.
- Surface-emission coupling through boundary-face electron fluxes.
- Secondary, thermionic, field, Richardson-Dushman, constant-current-density,
  and quantum-pulse photoemission models.
- Configurable gas/transport/ionization models, including local-field tables.
- Explicit, implicit, and MNA-based circuit coupling.
- Unified maximum circuit topology
  `R0_Cs_Ls_Cp_Lp_Rm_Cext` for custom lumped circuits.
- Load-side stray capacitance support through `C_ext`.
- Optional dielectric-layer mapping at the electrodes.
- NumPy or Numba hot-loop backend.
- Saved-run diagnostics notebooks for temporal profiles, spatial snapshots,
  and time/cycle-averaged spatial profiles.

## Package Contents

Core solver modules:

- `paschen_1d.py`
- `numerics.py`
- `numerics_jit.py`
- `physics.py`
- `emission.py`
- `circuit.py`
- `circuit_implicit_euler.py`
- `circuit_mna.py`
- `outputs.py`
- `physical_constants.py`

Configuration and case loading:

- `config.py`
- `config_loader.py`
- `config_case_argon_photoemission_discharge.py`
- `config_case_argon_dc_discharge.py`
- `config_case_nitrogen_pulsed_discharge.py`

User notebooks:

- `run_paschen_1d.ipynb`
- `diagnostics_temporal_profiles.ipynb`
- `diagnostics_spatial_snapshots.ipynb`
- `diagnostics_spatial_averages.ipynb`

Diagnostics helpers:

- `diagnostics_io.py`
- `diagnostics_plotting.py`
- `derived_diagnostics.py`

Bundled swarm-data examples:

- `ar_swarm_output_full_EoverN.dat`
- `n2_swarm_output_full_EoverN.dat`

Release notes:

- `next_release_change_notes.html`

## Quick Start

1. Open `run_paschen_1d.ipynb`.
2. Choose a case by setting `CONFIG_MODULE`, for example:

   ```python
   CONFIG_MODULE = "config_case_argon_photoemission_discharge"
   ```

3. Run the notebook. Results are written to the folder named by
   `cfg.run.run_name`.
4. Use one of the diagnostics notebooks to inspect the saved run:

   - `diagnostics_temporal_profiles.ipynb`
   - `diagnostics_spatial_snapshots.ipynb`
   - `diagnostics_spatial_averages.ipynb`

`config.py` is the guided default configuration. It has the same layout as the
case files and includes notebook-style section markers/comments for all exposed
user knobs.

## Configuration Structure

`SimulationConfig` is organized into grouped dataclasses:

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

Runtime modules use grouped access such as `cfg.geometry.L`,
`cfg.run.T_total`, and `cfg.circuit.R_m`.

## Circuit Usage

For standard reduced circuits, choose a named topology with physical finite
values for the elements in that topology. Examples include:

- `dielectric_plasma`
- `R0_Cp`
- `R0_Cp_Rm`
- `R0_Rm_Cext`
- `R0_Cs_Cp`
- `R0_Cs_Cp_Rm`
- `R0_Cs_Ls_Cp`
- `R0_Cs_Ls_Cp_Rm`
- `R0_Cs_Ls_Cp_Lp`
- `R0_Cs_Ls_Cp_Lp_Rm`
- `R0_Cs_Ls_Cp_Lp_Rm_Cext`

For a custom circuit that is not available as a reduced topology, use:

```python
cfg.circuit.circuit_type = "R0_Cs_Ls_Cp_Lp_Rm_Cext"
cfg.circuit.circuit_time_scheme = "mna"
```

In MNA mode, elements may be removed with neutral values:

- `R0 = 0.0`
- `C_s = np.inf`
- `L_s = 0.0`
- `C_p = 0.0`
- `L_p = np.inf`
- `R_m = 0.0`
- `C_ext = 0.0`

## Diagnostics

The main run notebook includes optional quick-look plots immediately after a
simulation. For saved-run analysis, use the dedicated diagnostics notebooks.
Each notebook exposes user-editable plot controls, figure-saving knobs, and
optional grouped-plot controls.

Available diagnostics include:

- temporal circuit/plasma quantities such as `V_app`, `V_node`, `V_gap`,
  `I_discharge`, current components, CFL, Picard iterations, and particle
  inventory;
- spatial snapshots of `ne`, `ni`, `phi`, `E`, fluxes, ionization quantities,
  mobilities/diffusivities, and source terms;
- time-window or cycle-averaged spatial profiles;
- derived sheath metrics computed from saved density/potential/field profiles.

## Output Files

Each run writes a folder named by `cfg.run.run_name`. Common files include:

- scalar histories: `Vgap_mm.dat`, `Idischarge_mm.dat`, `c_cfl_mm.dat`
- optional scalar histories: `Vnode_mm.dat`, `Vsource_mm.dat`,
  `picard_iterations_mm.dat`, `adaptive_substeps_mm.dat`,
  `adaptive_dt_sub_mm.dat`, `adaptive_cfl_est_mm.dat`
- sampled fields: `ne_sampled_mm.dat`, `ni_sampled_mm.dat`,
  `phi_sampled_mm.dat`, `E_sampled_mm.dat`
- optional sampled intermediates: `Gamma_i_sampled_mm.dat`,
  `Gamma_e_sampled_mm.dat`, `townsend_alpha_sampled_mm.dat`,
  `nu_i_sampled_mm.dat`, `S_ion_sampled_mm.dat`, `S_sampled_mm.dat`,
  `mu_e_sampled_mm.dat`, `D_e_sampled_mm.dat`
- metadata: `run_metadata.json`

## Swarm-Data Format

The parser accepts either raw swarm-output text containing named `E/N (Td)`
sections or two-column tables. Recognized section labels include:

- `Mobility *N (1/m/V/s)`
- `Diffusion coefficient *N (1/m/s)`
- `Townsend ioniz. coef. alpha/N (m2)`
- `Total ionization frequency /N (m3/s)`

## License and Citation

- License file: `LICENSE`
- Commercial use is not permitted without separate written permission.
- Any use, modification, redistribution, or publication must acknowledge/cite:

Asif Iqbal, Yves Heri, Bingqing Wang, Lan Jin, Md Arifuzzaman Faisal, and
Peng Zhang, "PASCHEN-1D: A one-dimensional fluid plasma solver with
multi-mechanism surface emission and flexible external circuit coupling".
