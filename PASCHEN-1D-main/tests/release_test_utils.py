"""Shared construction helpers for fast release-regression simulations."""

from __future__ import annotations

from config import SimulationConfig


def make_miniature_config(
    cfg: SimulationConfig | None = None,
    *,
    run_name: str = "release_smoke",
    backend: str = "numpy",
) -> SimulationConfig:
    """Reduce a production case without changing its selected physics models."""
    cfg = SimulationConfig() if cfg is None else cfg
    cfg.run.run_name = run_name
    cfg.run.T_total = 1.0e-12
    cfg.numerics.Nt = 7
    cfg.numerics.Nx = 9
    cfg.numerics.hotloop_backend = backend
    cfg.numerics.numba_parallel = False
    cfg.numerics.use_adaptive_substepping = False
    cfg.numerics.bc_poisson_picard_min_iter = 1
    cfg.numerics.bc_poisson_picard_max_iter = 8
    cfg.output.save_every = 1
    cfg.output.log_intermediate = True
    cfg.output.print_run_summary = False

    cfg.emission.enable_external_emission = False
    cfg.emission.enable_anode_external_emission = False
    cfg.emission.enable_cathode_external_emission = False
    for electrode in ("anode", "cathode"):
        for mechanism in ("constant_J", "fn", "mg", "rd", "quantum_pulse"):
            setattr(
                cfg.emission,
                f"{electrode}_enable_{mechanism}_emission",
                False,
            )
    return cfg
