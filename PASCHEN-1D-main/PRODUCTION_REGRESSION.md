# Production Regression Record

The release audit retained completed production-resolution runs outside the distribution at `../PASCHEN-1D-production-results-2026-07-21/`. This keeps the downloadable source tree clean without discarding the recovery evidence.

On 2026-07-21, every saved scalar history and sampled spatial array in all four runs was reopened through `diagnostics_io.py` and checked for finite values. All runs reported zero adaptive-substep overflow events and zero nonconverged boundary/Poisson solves.

| Case | Grid (`Nt × Nx`) | Final gap voltage (V) | Peak absolute current (A) | Peak electron density (m⁻³) | Peak ion density (m⁻³) |
|---|---:|---:|---:|---:|---:|
| Argon photoemission | 8,750,000 × 200 | 223.803989963 | 7.2176256e-5 | 1.8449479e16 | 1.8716117e16 |
| Nitrogen pulsed discharge | 150,000 × 1,000 | -2784.68054127 | 62.8238694 | 1.0375466e20 | 7.9728324e19 |
| Argon 140 V DC glow | 4,000,000 × 200 | 133.129346336 | 3.5347768e-5 | 7.2888320e14 | 8.3553743e14 |
| Deuterium pulsed discharge | 150,000 × 1,000 | -480.006803732 | 62.0906938 | 4.9211462e19 | 5.7373225e19 |

These values are regression records, not universal physical benchmarks. The automated test suite separately uses small deterministic golden cases so CI can exercise the same pathways in seconds.
