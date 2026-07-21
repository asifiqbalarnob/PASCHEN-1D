"""Import a PASCHEN-1D case without mutating Python's module registry."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Callable


def _module_name(config_module: str) -> str:
    """Normalize a config module name or filename."""
    name = str(config_module).strip()
    if not name:
        raise ValueError("config_module must be a non-empty module name or .py filename.")
    if name.endswith(".py"):
        name = name[:-3]
    return name


def load_config_module(config_module: str = "config") -> ModuleType:
    """
    Load a case module defining a ``SimulationConfig`` subclass.

    Parameters
    ----------
    config_module:
        Module name or filename for a config file defining ``SimulationConfig``.
    """
    name = _module_name(config_module)

    module = importlib.import_module(name)

    if not hasattr(module, "SimulationConfig"):
        raise AttributeError(f"{name!r} does not define SimulationConfig.")
    return module


def load_simulation_case(
    config_module: str = "config",
) -> tuple[type, Callable]:
    """
    Return ``(SimulationConfig, run_simulation)`` for the selected case config.

    This is the recommended helper for user-facing notebooks:

        from config_loader import load_simulation_case
        SimulationConfig, run_simulation = load_simulation_case(CONFIG_MODULE)
    """
    module = load_config_module(config_module)
    from paschen_1d import run_simulation

    return module.SimulationConfig, run_simulation
