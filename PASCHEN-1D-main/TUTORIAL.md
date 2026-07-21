# PASCHEN-1D Tutorial

This tutorial describes the current PASCHEN-1D workflow, including the
table-based electron and positive-ion transport framework. The recommended
entry point is `run_paschen_1d.ipynb`; the three diagnostics notebooks provide
saved-run analysis.

## 1. What the Model Solves

PASCHEN-1D is a one-dimensional drift-diffusion-Poisson plasma solver coupled
to surface-emission and external-circuit models. The present species model has
two charged fluids:

- electrons;
- one user-selected positive ion.

The neutral background is fixed by `plasma_state.gas`, pressure, and gas/ion
temperature. The code computes its number density from the ideal-gas relation

```text
N = p / (kB * T_i).
```

An ion transport dataset therefore always describes a pair: the transported
ion moving through a particular neutral gas. For example, the deuterium case
uses D3+ moving through D2. Selecting only `D3+` without also matching the D2
neutral would not define the transport problem.

This release does not yet evolve multiple positive ions, negative ions,
attachment chemistry, or a separate electron-energy equation. A table may
exist in the library without implying that all plasma chemistry for that gas
is modeled.

## 2. Install and Run

From the `PASCHEN-1D-main` directory, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Open `run_paschen_1d.ipynb`, set the case module, and run all cells:

```python
CONFIG_MODULE = "config_case_argon_photoemission_discharge"
```

The notebook uses the following public interface:

```python
from config_loader import load_simulation_case

SimulationConfig, run_simulation = load_simulation_case(CONFIG_MODULE)
cfg = SimulationConfig()
state = run_simulation(cfg)
```

`config_loader.py` imports the selected case normally. It does not rename a
file or replace another module. Every supplied case file independently defines
the same complete dataclass structure, with defaults tuned to that case.

Results are written to the directory named by `cfg.run.run_name`.

## 3. Supplied Cases

The current case modules are:

- `config` — generic guided configuration with argon photoemission defaults;
- `config_case_argon_photoemission_discharge` — standalone argon
  photoemission configuration;
- `config_case_argon_dc_discharge` — argon DC glow-discharge case;
- `config_case_nitrogen_pulsed_discharge` — nitrogen pulsed-discharge case;
- `config_case_deuterium_pulsed_discharge` — D2 pulsed discharge using BOLSIG+
  electron data and a matched D3+/D2 LXCat ion pair;
- `config_case_helium_photoemission_discharge` — helium table-transport smoke
  case using He+ mobility and Einstein-relation diffusion.

Use a unique `run_name` whenever parameters change. Reusing a populated output
directory can mix or overwrite results from different cases.

## 4. Standalone Configuration Modules

`config.py` and every `config_case_*.py` file define the full public dataclass
structure. Each standalone `SimulationConfig` contains these 18 blocks:

1. `run`
2. `numerics`
3. `geometry`
4. `plasma_state`
5. `plasma`
6. `user_defined_electron_kinetics`
7. `local_field_approximation`
8. `electron_swarm_data`
9. `ion_transport`
10. `townsend_coefficient`
11. `ionization_frequency_source`
12. `recombination`
13. `waveform`
14. `boundary`
15. `circuit`
16. `emission`
17. `output`
18. `diagnostics`

Use grouped paths such as `cfg.geometry.L`, `cfg.plasma_state.p_Torr`,
`cfg.ion_transport.positive_ion`, and `cfg.output.log_intermediate`.

### Create a New Case

Copy the complete generic configuration or the closest complete case, then
edit the dataclass defaults directly. For example:

```bash
cp config.py config_case_my_new_case.py
```

In the copied file, change `RunConfig.run_name`,
`PlasmaStateConfig.p_Torr`, `WaveformConfig.V_peak`, and any other desired
defaults. Keep the complete type aliases and dataclass field structure intact;
the configuration-schema regression test checks parity across supplied cases.

Save this as a Python module beside the other configuration files, then use
its filename without `.py` as `CONFIG_MODULE`.

## 5. Gas State and Physics Selectors

Set the neutral gas and thermodynamic state first:

```python
cfg.plasma_state.gas = "deuterium"
cfg.plasma_state.p_Torr = 3.0
cfg.plasma_state.T_i = 300.0
cfg.plasma_state.T_e = 11600.0
cfg.plasma_state.n0 = 1.0e10
```

`T_i` is also the neutral-gas temperature used to validate ion tables. The
solver does not evolve `T_i` or `T_e` as energy equations.

The high-level closures are selected under `cfg.plasma`:

```python
cfg.plasma.electron_kinetics_model = "local_field_approximation"
cfg.plasma.ion_kinetics_model = "local_field_ion_kinetics"
cfg.plasma.impact_ionization_model = "from_townsend_alpha"
cfg.plasma.recombination_model = "user_defined_constant_coefficient"
```

The built-in empirical transport and ionization equations cover argon and
nitrogen. A different gas should use compatible electron tables and a matched
positive-ion/neutral transport selection, unless the user deliberately adds
and verifies new equations in `physics.py`.

## 6. Electron Transport and Ionization

### Electron Kinetics

Two electron-kinetics branches are available:

```python
cfg.plasma.electron_kinetics_model = "user_defined_electron_kinetics"
```

or:

```python
cfg.plasma.electron_kinetics_model = "local_field_approximation"
```

For local-field approximation, choose user equations or an E/N table:

```python
cfg.local_field_approximation.electron_transport_source = (
    "swarm_data_table_interpolation"
)
cfg.local_field_approximation.electron_swarm_data_path = (
    "d2_swarm_output_full_EoverN.dat"
)
```

The local reduced field is computed from the instantaneous field and neutral
density. The table provides reduced coefficients, and PASCHEN-1D converts them
to local mobility and diffusion values.

Electron-table validation and range behavior are configured centrally:

```python
cfg.electron_swarm_data.out_of_range_policy = "error"  # or "clip"
cfg.electron_swarm_data.gas_temperature_tolerance_K = 5.0
```

### Impact Ionization

The Townsend-alpha route evaluates `nu_i = alpha * |u_e|`:

```python
cfg.plasma.impact_ionization_model = "from_townsend_alpha"
cfg.townsend_coefficient.townsend_alpha_source_mode = (
    "interpolate_from_e_over_n_table"
)
cfg.townsend_coefficient.townsend_alpha_swarm_data_path = (
    "d2_swarm_output_full_EoverN.dat"
)
```

The direct ionization-frequency route is:

```python
cfg.plasma.impact_ionization_model = "from_ionization_frequency"
cfg.ionization_frequency_source.ionization_frequency_source_mode = (
    "interpolate_from_e_over_n_table"
)
cfg.ionization_frequency_source.ionization_frequency_swarm_data_path = (
    "d2_swarm_output_full_EoverN.dat"
)
```

Set the table pathname in every active dataclass. Using a single local variable
in the case file prevents accidental mismatches:

```python
electron_table = "d2_swarm_output_full_EoverN.dat"
cfg.local_field_approximation.electron_swarm_data_path = electron_table
cfg.townsend_coefficient.townsend_alpha_swarm_data_path = electron_table
cfg.ionization_frequency_source.ionization_frequency_swarm_data_path = electron_table
```

### Electron Table Format

The bundled BOLSIG+ output parser recognizes sections including:

- `Mobility *N (1/m/V/s)`;
- `Diffusion coefficient *N (1/m/s)`;
- `Townsend ioniz. coef. alpha/N (m2)`;
- `Total ionization freq. /N (m3/s)`.

The file must contain every section required by the selected transport and
ionization modes. The validated electron table library is under
`electron_swarm_data/lxcat_2026-07-20/`. Configuration values are filenames or
paths relative to `electron_swarm_data`; never use a machine-specific absolute
path. Because bundled electron filenames are unique, the basename alone is
normally sufficient, for example `"d2_swarm_output_full_EoverN.dat"`.

## 7. Positive-Ion Transport

### Select the Ion/Neutral Pair

The neutral is set by:

```python
cfg.plasma_state.gas = "deuterium"
```

The transported positive ion is set independently:

```python
cfg.ion_transport.positive_ion = "D3+"
```

Together these mean D3+ drifting and diffusing through D2. PASCHEN-1D checks
the pair against the metadata embedded in each normalized table.

### Source Modes

The original user-equation branch is:

```python
cfg.plasma.ion_kinetics_model = "user_defined_ion_kinetics"
cfg.ion_transport.mobility_source_mode = "user_defined_equation"
cfg.ion_transport.diffusion_source_mode = "user_defined_equation"
```

For table-based local-field transport:

```python
cfg.plasma.ion_kinetics_model = "local_field_ion_kinetics"
cfg.ion_transport.mobility_source_mode = "swarm_data_table_interpolation"
cfg.ion_transport.diffusion_source_mode = "swarm_data_table_interpolation"
cfg.ion_transport.mobility_table_path = "<unique-mobility-filename>.csv"
cfg.ion_transport.diffusion_table_path = "<unique-diffusion-filename>.csv"
```

These values are resolved inside `ion_swarm_data`. Bundled normalized ion
filenames are unique, so the basename alone can also be used. A relative path
within that directory remains useful when selecting a table from the manifest.

Diffusion can instead be computed from the active ion mobility through the
Einstein relation:

```python
cfg.ion_transport.diffusion_source_mode = "einstein_relation"
cfg.ion_transport.diffusion_table_path = None
```

This gives `D_i = mu_i * kB * T_i / e`. It is an explicit modeling
assumption, not a replacement for measured diffusion data in every regime.

### Normalized LXCat Tables

The ion-data workflow has two layers:

- `ion_swarm_data/raw_lxcat/lxcat_ion_swarm_2026-07-21/` is a local,
  Git-ignored workspace created by the downloader for LXCat exports, catalog,
  URLs, and checksums; it is not distributed with PASCHEN-1D;
- `ion_swarm_data/normalized_lxcat_2026-07-21/` contains solver-ready CSV files
  and `manifest.json`.

The normalized release contains 2,268 accepted transport tables and 818
conservatively matched mobility/diffusion pairs across 22 neutral labels. It
also records 92 rejected candidate tables and five LXCat source-export errors;
these are retained as provenance and are not silently presented as usable
tables. Independent measurements are not averaged, spliced, or extrapolated
during normalization.

LXCat reduced mobility is normalized to `K0*N0`; PASCHEN-1D evaluates
`mu_i = (K0*N0)/N`. Reduced longitudinal diffusion is normalized to `N*Dz`,
and PASCHEN-1D evaluates `D_i = (N*Dz)/N`.

### Search the Manifest

The following cell lists compatible positive-ion pairs for a neutral:

```python
import json
from pathlib import Path

ion_root = Path("ion_swarm_data/normalized_lxcat_2026-07-21")
manifest = json.loads((ion_root / "manifest.json").read_text())

neutral = "D2"
pairs = [
    pair
    for pair in manifest["compatible_mobility_diffusion_pairs"]
    if pair["neutral"] == neutral
    and "+" in pair["ion"]
    and "-" not in pair["ion"]
]

for pair in pairs:
    print(
        pair["ion"],
        pair["gas_temperature_K"],
        pair["common_reduced_field_min_Td"],
        pair["common_reduced_field_max_Td"],
        pair["mobility_file"],
        pair["diffusion_file"],
    )
```

When selecting a pair, inspect all of the following:

- neutral and positive-ion identities;
- gas temperature;
- common E/N interval;
- database, reference, and source family;
- whether the mobility and diffusion grids are identical;
- any comments or limitations from the contributor.

The relative filenames in a pair record are relative to `ion_root`. In a case
file, prefix them with the normalized archive directory.

### Out-of-Range Policy

Ion tables have an explicit E/N policy:

```python
cfg.ion_transport.out_of_range_policy = "error"
```

stops when the field leaves the data interval, while:

```python
cfg.ion_transport.out_of_range_policy = "clip"
```

uses the nearest endpoint. Clipping prevents an interpolation failure but does
not extend the experimental validity of a dataset. A production analysis
should compare the simulated E/N distribution with the selected table range.

The gas-temperature check is controlled by:

```python
cfg.ion_transport.gas_temperature_tolerance_K = 1.0
```

## 8. Complete Deuterium Transport Example

The supplied deuterium case contains the following essential selections:

```python
cfg.plasma_state.gas = "deuterium"

electron_table = "d2_swarm_output_full_EoverN.dat"
cfg.plasma.electron_kinetics_model = "local_field_approximation"
cfg.local_field_approximation.electron_transport_source = (
    "swarm_data_table_interpolation"
)
cfg.local_field_approximation.electron_swarm_data_path = electron_table

cfg.townsend_coefficient.townsend_alpha_source_mode = (
    "interpolate_from_e_over_n_table"
)
cfg.townsend_coefficient.townsend_alpha_swarm_data_path = electron_table
cfg.ionization_frequency_source.ionization_frequency_source_mode = (
    "interpolate_from_e_over_n_table"
)
cfg.ionization_frequency_source.ionization_frequency_swarm_data_path = electron_table

cfg.plasma.ion_kinetics_model = "local_field_ion_kinetics"
cfg.ion_transport.positive_ion = "D3+"
cfg.ion_transport.mobility_source_mode = "swarm_data_table_interpolation"
cfg.ion_transport.diffusion_source_mode = "swarm_data_table_interpolation"
cfg.ion_transport.mobility_table_path = (
    "lxcat_606acd7d3f0d4373cf14_reduced_mobility.csv"
)
cfg.ion_transport.diffusion_table_path = (
    "lxcat_72c9a7cbf9c0f10b3463_reduced_longitudinal_diffusion.csv"
)
cfg.ion_transport.out_of_range_policy = "clip"
cfg.ion_transport.gas_temperature_tolerance_K = 1.0
```

This matched D3+/D2 pair is at 300 K and has a common measured range of
5.01–50.1 Td. The supplied case uses clipping because a complete pulsed run can
leave that interval. Interpret clipped portions as an uncertainty or data-gap
diagnostic, not as new LXCat measurements.

## 9. Recombination and Volume Sources

The volume-source switches live under `boundary`:

```python
cfg.boundary.enable_volume_sources = True
cfg.boundary.enable_ionization_source = True
cfg.boundary.enable_recombination_sink = True
```

The coefficient itself lives under `recombination`:

```python
cfg.recombination.recombination_coefficient = 2.0e-13  # m^3/s
```

The current sink is `S_rec = beta * n_e * n_i`. Setting
`enable_volume_sources = False` disables both ionization and recombination
regardless of their individual switches.

## 10. Boundary Conditions and Emission

Boundary modes are selected separately for each species and electrode:

```python
cfg.boundary.anode_ion_boundary = "zero_density"
cfg.boundary.anode_electron_boundary = "implicit_drift_closure"
cfg.boundary.cathode_ion_boundary = "implicit_drift_closure"
cfg.boundary.cathode_electron_boundary = "electron_emission"
```

Available modes are:

- `zero_density`;
- `implicit_drift_closure`;
- `electron_emission` for an emitting electron boundary.

Surface-emission controls live under `cfg.emission`. They include ion-induced
secondary emission through `gamma`, anode electron-induced emission, and
per-electrode constant-current, Fowler-Nordheim, Murphy-Good,
Richardson-Dushman, and quantum-pulse mechanisms. Multiple enabled external
mechanisms on one electrode are summed.

If an electrode emits electrons, its electron boundary must use
`electron_emission`. For photoemission, set physical laser widths and energy;
the backend maps the emission spot to the represented electrode area.

## 11. Waveforms and Circuits

Waveform types include `step`, `gaussian`, `dc`, `rf`, and tabulated/measured
tables. For example:

```python
cfg.waveform.waveform_type = "step"
cfg.waveform.V_peak = 1000.0
cfg.waveform.tV_start = 0.0
cfg.waveform.tV_end = 100.0e-9
```

The applied source waveform need not equal the plasma-gap voltage when an
external circuit is active.

A reduced circuit can use:

```python
cfg.circuit.circuit_type = "R0_Rm_Cext"
cfg.circuit.circuit_time_scheme = "explicit_euler"
```

Use `implicit_euler` for the supported reduced topologies when stiffness makes
an explicit update unsuitable. For a custom combination of lumped elements,
use the maximum MNA topology:

```python
cfg.circuit.circuit_type = "R0_Cs_Ls_Cp_Lp_Rm_Cext"
cfg.circuit.circuit_time_scheme = "mna"
```

In MNA mode, neutral values remove elements:

- `R0 = 0.0`;
- `C_s = np.inf`;
- `L_s = 0.0`;
- `C_p = 0.0`;
- `L_p = np.inf`;
- `R_m = 0.0`;
- `C_ext = 0.0`.

## 12. Numerics and Staged Testing

The macro step is `dt = run.T_total / (numerics.Nt - 1)`; the grid spacing follows
from `geometry.L` and `numerics.Nx`. Core controls include:

```python
cfg.numerics.Nx = 200
cfg.numerics.Nt = 1_000_000
cfg.numerics.hotloop_backend = "numba"  # or "numpy"
cfg.numerics.use_adaptive_substepping = True
cfg.numerics.target_cfl_substep = 0.5
cfg.numerics.max_substeps = 16
```

Adaptive substepping splits a macro step to control drift CFL. If it repeatedly
reaches `max_substeps`, increase `Nt` or revise the step controls rather than
treating a capped warning as proof of convergence.

### Stage 1: Configuration and Table Validation

Before a long run:

```python
from paschen_1d import validate_simulation_config
from physics import build_swarm_interpolation_cache

validate_simulation_config(cfg)
cache = build_swarm_interpolation_cache(cfg)
print(cache.ion_transport_provenance())
```

Startup validation fails before the time loop for unsupported selectors,
missing files, malformed tables, incorrect quantities or units, nonpositive or
nonfinite values, nonmonotonic grids, ion/neutral mismatches, and gas-temperature
mismatches. A requested table never silently falls back to an empirical model.

### Stage 2: Short Smoke Run

Preserve the original macro-step size when shortening a case:

```python
dt_original = cfg.run.T_total / (cfg.numerics.Nt - 1)
cfg.numerics.Nt = 1000
cfg.run.T_total = dt_original * (cfg.numerics.Nt - 1)
cfg.run.run_name = "my_new_case_smoke"
cfg.output.save_every = 100
state = run_simulation(cfg)
```

Check for finite densities, fields, currents, CFL history, and reasonable
table-range use. A smoke run verifies plumbing and early-time stability; it
does not validate a full pulse or steady discharge.

### Stage 3: Resolution and Full-Physics Runs

Restore the intended duration and resolution. Compare successively finer
`Nx`/`Nt` settings and verify that the quantities of interest converge. Then
run the full pulse or steady interval with a new output name.

### Stage 4: Regression Tests

Run the packaged tests after changing configuration, loaders, or physics:

```bash
python -m pytest -q
```

## 13. Outputs, Diagnostics, and Provenance

`cfg.output.save_every` controls saved sampling cadence.
`cfg.output.log_intermediate = True` adds the spatial transport, flux, and
source arrays needed by many diagnostics.

Common scalar histories include gap/source/node voltages, discharge-current
components, CFL, Picard iterations, and adaptive-step statistics. Always-saved
spatial fields include:

- `ne`, `ni`;
- `phi`, `E`.

Intermediate logging additionally includes:

- electron and ion fluxes;
- Townsend coefficient and ionization frequency;
- ionization and net source terms;
- `mu_e`, `D_e`, `mu_i`, and `D_i`.

Every completed run writes `run_metadata.json`. It records the PASCHEN-1D
version, Git commit when available, Python/platform and dependency versions,
resolved model selectors, gas state, numerical settings, circuit and source
switches. Electron and ion table entries include portable paths, checksums,
identities, temperatures, source citations, raw provenance, and accumulated
E/N range coverage.
Keep this file with any exported figures or reported results.

Use:

- `diagnostics_temporal_profiles.ipynb` for scalar histories and temporal
  sheath metrics;
- `diagnostics_spatial_snapshots.ipynb` for instantaneous profiles;
- `diagnostics_spatial_averages.ipynb` for time-window or final-cycle averages.

## 14. Adding Another Gas: End-to-End Checklist

1. Choose a validated BOLSIG+ electron table containing the sections required
   by the selected electron and impact-ionization modes.
2. Set `plasma_state.gas`, pressure, and `T_i`.
3. Use `local_field_approximation` with table interpolation for a gas that has
   no built-in empirical closure.
4. Set the same electron filename in every active electron transport,
   Townsend, and ionization-frequency dataclass.
5. Search the ion manifest for a physically appropriate positive ion moving
   through the chosen neutral.
6. Select a matched mobility/diffusion pair, or select a mobility table and
   deliberately choose Einstein diffusion.
7. Set the ion identity, both source modes, file paths, range policy, and gas
   temperature tolerance.
8. Verify waveform, boundaries, emission, source toggles, and circuit settings;
   these do not become correct automatically when the gas changes.
9. Run configuration/cache validation, a short smoke run, resolution checks,
   and only then the full case.
10. Inspect `run_metadata.json`, transport profiles, E/N coverage, conservation,
    and numerical convergence before interpreting the result physically.

## 15. Common Failures

- **Ion/neutral mismatch:** the configured gas does not match the neutral
  embedded in an ion table. Choose a table for the actual background gas.
- **Temperature mismatch:** `plasma_state.T_i` differs from the table
  temperature by more than `gas_temperature_tolerance_K`.
- **Missing electron section:** the BOLSIG+ file lacks a coefficient required
  by the active electron or ionization model.
- **Missing ion path:** a table source mode was selected without its CSV path.
- **Out-of-range exception:** local E/N left an ion table configured with
  `out_of_range_policy = "error"`. Check the physics before choosing clipping.
- **Unexpected empirical-gas error:** a non-argon/non-nitrogen case still uses
  a user-defined closure. Switch that closure to a compatible table.
- **Emission with the wrong boundary:** an external emission mechanism is
  enabled but the corresponding electron boundary is not `electron_emission`.
- **CFL or adaptive overflow warnings:** reduce the macro time step, revisit
  adaptive limits, and perform a convergence study.
- **Overwritten/mixed output:** assign a unique `run.run_name` for every case.

## 16. Rebuilding the Ion Tables

The checked-in normalized tables are ready to use. To reproduce them from
LXCat, first review and accept the current LXCat terms, then run:

```bash
python tools/download_lxcat_ion_data.py
python tools/normalize_lxcat_ion_data.py
```

The first command downloads source material and retrieval metadata into the
local, Git-ignored `ion_swarm_data/raw_lxcat/` directory. The second creates
self-describing solver tables and a manifest without averaging or splicing
independent datasets. Raw LXCat downloads must not be committed or included in
a release. Review source-export failures and rejected records in the regenerated
manifest before changing production selections.
