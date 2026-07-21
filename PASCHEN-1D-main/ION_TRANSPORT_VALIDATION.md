# Ion-transport rollout validation

Validated on 2026-07-21.

## LXCat acquisition and normalization

- Official ion-swarm catalog: 486 ion/neutral entries and 21,137 reported
  process selections across Phelps, UNAM, and Viehland.
- Deterministic download plan: 222 checksum-validated batches.
- Successful official exports: 21,132 process selections.
- Source-side export failures: five (`779`, `721`, `1197`, `723`, `725`).
  LXCat returned HTTP 500 for each record after bounded retries and isolated
  single-record export attempts. Exact labels and errors are retained in raw
  batch metadata and in the normalized manifest.
- Preserved raw layer: 758 verbatim ASCII files containing 12,772 tabulated
  process blocks.
- Candidate mobility/diffusion tables: 2,360.
- Strict PASCHEN-1D tables generated: 2,268.
- Rejected candidates: 92, each retained in the manifest with its data-quality
  reason (non-positive E/N, duplicate E/N coordinates, or too few points).
- Source-compatible mobility/diffusion pair choices: 818.
- Exhaustive audit: all 2,268 normalized files were checksum verified and
  loaded through the production strict loader; the manifest has no missing or
  orphan table files and all 818 pair references resolve.

## Code rollout

- `config.py` is the sole canonical configuration/dataclass definition.
- `IonTransportConfig` selects the positive ion, mobility source, diffusion
  source, table paths, range policy, and temperature tolerance.
- Startup validation rejects invalid selectors, absent paths, malformed tables,
  wrong quantities or units, ion/neutral mismatch, temperature mismatch,
  non-positive/non-monotonic tables, and invalid provenance checksums.
- The run-scoped interpolation cache now owns electron and ion table
  interpolators. Runtime reduced coefficients are converted with the actual
  neutral number density passed through the simulation.
- Table-based ion mobility, table-based longitudinal diffusion, and
  Einstein-relation diffusion are implemented.
- The outer argon/nitrogen dispatch gate was removed. Gas-specific empirical
  hooks remain intentionally limited to argon/nitrogen and are invoked only
  when the user explicitly selects those hooks.
- Saved diagnostics include `mu_i` and `D_i`. Run metadata includes ion source
  modes, dataset IDs, normalized and raw SHA-256 checksums, identities,
  temperature, database, source notes, and citation.

## Tests and legacy recovery

- Nine automated unit/integration tests pass, covering canonical case loading,
  required table paths, gas/ion aliases, strict identity rejection, reduced
  coefficient conversion, clip/error range behavior, table cache/profile
  evaluation, removal of the outer gas gate, and LXCat source-block parsing.
- A 2,000-step pre/post implementation regression was run for argon
  photoemission, nitrogen pulsed discharge, and argon DC glow discharge. Time,
  gap voltage, discharge current, electron density, ion density, potential,
  and electric field were bit-for-bit identical for all three cases.
- Full production legacy cases completed with finite saved histories and no
  nonconverged Poisson/boundary solves:

  - nitrogen pulsed discharge: 150,000 steps, peak CFL 0.848165;
  - argon DC glow discharge: 4,000,000 steps, peak CFL 0.253805;
  - argon photoemission discharge: 8,750,000 steps, peak realized CFL
    0.501188 with adaptive substepping.

## Deuterium production case

`config_case_deuterium_pulsed_discharge.py` inherits the nitrogen pulse case
and changes the gas/transport definitions without duplicating configuration
dataclasses.

- Neutral gas: D2 (`deuterium` in the configuration).
- Positive ion: D3+.
- Electron mobility, diffusion, and Townsend ionization: D2 BOLSIG+ E/N table.
- Ion mobility and longitudinal diffusion: matched 300 K raw Georgia Tech
  D3+/D2 records from the Viehland database, both on the same 14-point
  5.01-50.1 Td grid.
- Reduced-grid full-pulse stage: 15,000 steps, finite outputs, peak CFL
  0.219072, zero nonconverged solves.
- Full production stage: 150,000 steps and 1,000 cells in 53.7 s; all scalar
  and sampled spatial outputs finite; peak CFL 0.373252; zero nonconverged
  solves.

The selected ion-table range policy is explicitly `clip`. In the saved full
run, 68.97% of sampled cells were below 5.01 Td, 19.67% were within the measured
5.01-50.1 Td interval, and 11.36% were above 50.1 Td. The low-field clamp is the
measured limiting-mobility value; high-field results should be interpreted with
the documented 50.1 Td ceiling unless a broader mutually compatible D3+/D2
dataset becomes available.
