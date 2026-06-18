# PASCHEN-1D Tutorial

This tutorial describes the public workflow for the current PASCHEN-1D release.
The recommended entry point is `run_paschen_1d.ipynb`; detailed saved-run
analysis is handled by the three diagnostics notebooks.

## 1. Run Workflow

1. Choose a configuration module in `run_paschen_1d.ipynb`.
2. Run the simulation.
3. Review optional quick-look plots in the run notebook.
4. Use diagnostics notebooks for saved-run plotting and derived quantities.

The default case modules are:

- `config.py`
- `config_case_argon_photoemission_discharge.py`
- `config_case_argon_dc_discharge.py`
- `config_case_nitrogen_pulsed_discharge.py`

## 2. Selecting a Case in the Run Notebook

In `run_paschen_1d.ipynb`, set:

```python
CONFIG_MODULE = "config_case_argon_photoemission_discharge"
```

Then run:

```python
from config_loader import load_simulation_case

SimulationConfig, run_simulation = load_simulation_case(CONFIG_MODULE)
cfg = SimulationConfig()
state = run_simulation(cfg)
```

The loader installs the selected case module as the active `config` module for
solver internals, so users do not need to rename files to `config.py`.

## 3. Configuration Layout

Every public config file follows the same grouped dataclass layout:

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

Use grouped access everywhere, for example `cfg.geometry.L`,
`cfg.circuit.R_m`, or `cfg.output.log_intermediate`.

## 4. Physics-Model Selection

### Electron Kinetics

`cfg.plasma.electron_kinetics_model` can be:

- `"user_defined_electron_kinetics"`
- `"local_field_approximation"`

For the local-field branch, set:

```python
cfg.local_field_approximation.electron_transport_source = "user_defined_equation"
```

or:

```python
cfg.local_field_approximation.electron_transport_source = "swarm_data_table_interpolation"
cfg.local_field_approximation.electron_swarm_data_path = "ar_swarm_output_full_EoverN.dat"
```

### Impact Ionization

The high-level selector is:

```python
cfg.plasma.impact_ionization_model = "from_townsend_alpha"
```

or:

```python
cfg.plasma.impact_ionization_model = "from_ionization_frequency"
```

Townsend-alpha source:

```python
cfg.townsend_coefficient.townsend_alpha_source_mode = "user_defined_equation"
```

or:

```python
cfg.townsend_coefficient.townsend_alpha_source_mode = "interpolate_from_e_over_n_table"
```

Direct ionization-frequency source:

```python
cfg.ionization_frequency_source.ionization_frequency_source_mode = "user_defined_equation"
```

or:

```python
cfg.ionization_frequency_source.ionization_frequency_source_mode = "interpolate_from_e_over_n_table"
```

### Recombination

For the current constant-coefficient model:

```python
cfg.recombination.enable_recombination_sink = True
cfg.recombination.recombination_coefficient = 2.0e-13
```

## 5. Boundary Conditions and Emission

Boundary modes are set separately for each species/electrode:

```python
cfg.boundary.anode_ion_boundary = "zero_density"
cfg.boundary.anode_electron_boundary = "implicit_drift_closure"
cfg.boundary.cathode_ion_boundary = "implicit_drift_closure"
cfg.boundary.cathode_electron_boundary = "electron_emission"
```

Allowed labels:

- `zero_density`
- `implicit_drift_closure`
- `electron_emission` for electron boundaries

Surface-emission switches live under `cfg.emission`. Common controls include:

- `enable_external_emission`
- `enable_anode_external_emission`
- `enable_cathode_external_emission`
- `gamma`
- per-electrode toggles for `constant_J`, `fn`, `mg`, `rd`, and
  `quantum_pulse` emission

If an external emission source is enabled on an electrode, the corresponding
electron boundary should use `electron_emission`.

## 6. Circuit Selection

Reduced topologies can be run with:

```python
cfg.circuit.circuit_type = "R0_Rm_Cext"
cfg.circuit.circuit_time_scheme = "explicit_euler"
```

or:

```python
cfg.circuit.circuit_time_scheme = "implicit_euler"
```

For the maximum configurable topology, use the MNA backend:

```python
cfg.circuit.circuit_type = "R0_Cs_Ls_Cp_Lp_Rm_Cext"
cfg.circuit.circuit_time_scheme = "mna"
```

In MNA mode, remove elements with neutral values:

- `R0 = 0.0`
- `C_s = np.inf`
- `L_s = 0.0`
- `C_p = 0.0`
- `L_p = np.inf`
- `R_m = 0.0`
- `C_ext = 0.0`

This is the preferred route for user-defined lumped circuits that are not one
of the reduced named topologies.

## 7. Numerics and Performance

Core controls:

- `cfg.numerics.Nx`
- `cfg.numerics.Nt`
- `cfg.numerics.kt_limiter_theta`

Hot-loop backend:

```python
cfg.numerics.hotloop_backend = "numpy"
```

or:

```python
cfg.numerics.hotloop_backend = "numba"
```

Adaptive substepping and BC+Poisson Picard settings are also exposed in
`cfg.numerics`. For new cases, start with reduced `Nx`/`Nt`, then scale upward
after the setup is stable.

## 8. Output and Diagnostics

Simulation output is written to `cfg.run.run_name`.

For saved-run analysis:

- `diagnostics_temporal_profiles.ipynb` plots scalar histories and temporal
  sheath metrics.
- `diagnostics_spatial_snapshots.ipynb` plots instantaneous spatial profiles
  and sheath context profiles.
- `diagnostics_spatial_averages.ipynb` plots time-window or cycle-averaged
  spatial profiles.

Each diagnostics notebook contains:

- a run-selection cell;
- a summary of available quantities;
- definition/description cells for each quantity;
- user-facing plot controls;
- save/figure-name knobs close to the quantity selection;
- optional grouped plotting cells.

## 9. Swarm-Data Files

Bundled examples:

- `ar_swarm_output_full_EoverN.dat`
- `n2_swarm_output_full_EoverN.dat`

The parser accepts raw swarm-output sections or two-column tables. Required
labels depend on the requested quantity:

- `Mobility *N (1/m/V/s)`
- `Diffusion coefficient *N (1/m/s)`
- `Townsend ioniz. coef. alpha/N (m2)`
- `Total ionization frequency /N (m3/s)`

## 10. Practical Checklist

Before running a long case:

1. Confirm `cfg.run.run_name` is unique.
2. Check geometry, pressure, gas temperature, and initial density.
3. Check boundary modes against enabled emission mechanisms.
4. Check the selected circuit topology and time scheme.
5. Enable `cfg.output.log_intermediate = True` if spatial diagnostics of
   fluxes, ionization, mobility, or diffusion are needed.
6. Run a short smoke case with reduced `Nt` before committing to a long run.
