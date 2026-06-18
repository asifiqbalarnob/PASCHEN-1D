"""
config_loader.py

Small backend helper for selecting PASCHEN-1D case configuration modules.

The solver modules import shared dataclass types from a module named
``config`` at import time. This helper lets user-facing notebooks select any
case config file without carrying the module-aliasing/reload details in the
notebook itself.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Callable


_CONFIG_DEPENDENT_MODULES = (
    "paschen_1d",
    "physics",
    "emission",
    "outputs",
)


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
    Load a case config and install it under the runtime module name ``config``.

    Parameters
    ----------
    config_module:
        Module name or filename for a config file defining ``SimulationConfig``
        and ``SimulationState``.
    """
    name = _module_name(config_module)

    for module_name in _CONFIG_DEPENDENT_MODULES:
        sys.modules.pop(module_name, None)

    sys.modules.pop("config", None)
    if name in sys.modules:
        module = importlib.reload(sys.modules[name])
    else:
        module = importlib.import_module(name)

    if not hasattr(module, "SimulationConfig"):
        raise AttributeError(f"{name!r} does not define SimulationConfig.")
    if not hasattr(module, "SimulationState"):
        raise AttributeError(f"{name!r} does not define SimulationState.")

    sys.modules["config"] = module
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
