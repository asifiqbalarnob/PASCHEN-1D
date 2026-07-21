# Ion swarm data

This directory separates source preservation from solver-ready data.

- `raw_lxcat/lxcat_ion_swarm_2026-07-21/` contains the verbatim ASCII exports,
  LXCat output HTML, per-batch URLs/checksums, the contributor catalog, and the
  deterministic download plan.
- `normalized_lxcat_2026-07-21/manifest.json` indexes every raw process and
  identifies every normalized mobility/diffusion table. Candidate tables with
  non-positive values, duplicate E/N coordinates, or fewer than two points are
  retained in the manifest with explicit rejection reasons and are not exposed
  to the solver as usable tables.
- `normalized_lxcat_2026-07-21/tables/` contains self-describing PASCHEN-1D CSV
  tables. Independent datasets are never averaged, merged, or extrapolated.

Regenerate the archive and normalized layer with:

```bash
python tools/download_lxcat_ion_data.py
python tools/normalize_lxcat_ion_data.py
```

## Reduced-coefficient conversion

LXCat reduced mobility `K0` is converted from `cm2/(V s)` to `K0*N0` in
`1/(m V s)`, where `N0 = 100000/(kB*273.15)`. At runtime PASCHEN-1D computes
`mu_i = (K0*N0)/N`.

LXCat longitudinal diffusion `NDz` is converted from `1E20/(m s)` to `N*Dz`
in `1/(m s)`. At runtime PASCHEN-1D computes `D_i = (N*Dz)/N`.

Every normalized file embeds its ion, neutral, quantity, gas temperature,
database, citation, raw source path, and source checksum. The solver validates
these fields against the configuration before beginning a run.
