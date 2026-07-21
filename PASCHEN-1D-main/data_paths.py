"""Portable resolution of bundled electron and ion swarm-data files."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ELECTRON_SWARM_DATA_DIR = PROJECT_ROOT / "electron_swarm_data"
ION_SWARM_DATA_DIR = PROJECT_ROOT / "ion_swarm_data"


def _resolve_bundled_data_file(
    file_value: str | Path,
    *,
    data_root: Path,
    data_label: str,
) -> Path:
    """Resolve a portable filename or relative path within one data library."""
    value = str(file_value).strip()
    if not value:
        raise ValueError(f"{data_label} filename must be non-empty.")

    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(
            f"{data_label} filename must be relative to '{data_root.name}', "
            f"not an absolute path: {value!r}"
        )
    if ".." in relative.parts:
        raise ValueError(
            f"{data_label} filename cannot leave '{data_root.name}': {value!r}"
        )

    root = data_root.resolve()
    direct = root / relative
    if direct.is_file():
        return direct.resolve()

    # The normalized libraries preserve source/species directory structure.
    # Their basenames are unique, so a user may select a table by filename only.
    if len(relative.parts) == 1 and root.is_dir():
        matches = sorted(path.resolve() for path in root.rglob(relative.name) if path.is_file())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            rendered = "\n- ".join(str(path.relative_to(root)) for path in matches)
            raise ValueError(
                f"Ambiguous {data_label} filename {value!r}; use one of these "
                f"paths relative to '{data_root.name}':\n- {rendered}"
            )

    raise FileNotFoundError(
        f"{data_label} file {value!r} was not found inside '{root}'. "
        f"Provide a filename or a path relative to '{data_root.name}'."
    )


def resolve_electron_swarm_data_file(file_value: str | Path) -> Path:
    """Resolve a file inside the bundled electron swarm-data library."""
    return _resolve_bundled_data_file(
        file_value,
        data_root=ELECTRON_SWARM_DATA_DIR,
        data_label="Electron swarm-data",
    )


def resolve_ion_swarm_data_file(file_value: str | Path) -> Path:
    """Resolve a file inside the bundled ion swarm-data library."""
    return _resolve_bundled_data_file(
        file_value,
        data_root=ION_SWARM_DATA_DIR,
        data_label="Ion swarm-data",
    )
