"""Strict manifest-backed loading of bundled BOLSIG+ electron swarm tables."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import data_paths


MANIFEST_FORMAT = "paschen-1d-electron-swarm-manifest-v1"
SECTION_MOBILITY = "Mobility *N (1/m/V/s)"
SECTION_DIFFUSION = "Diffusion coefficient *N (1/m/s)"
SECTION_TOWNSEND = "Townsend ioniz. coef. alpha/N (m2)"
SECTION_IONIZATION_FREQUENCY = "Total ionization freq. /N (m3/s)"
REQUIRED_SECTIONS = (
    SECTION_MOBILITY,
    SECTION_DIFFUSION,
    SECTION_TOWNSEND,
    SECTION_IONIZATION_FREQUENCY,
)


def canonical_electron_gas(value: str) -> str:
    """Normalize common gas names and formulas for strict identity checks."""
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {
        "argon": "ar",
        "nitrogen": "n2",
        "deuterium": "d2",
        "hydrogen": "h2",
        "helium": "he",
        "neon": "ne",
        "krypton": "kr",
        "xenon": "xe",
        "oxygen": "o2",
        "water": "h2o",
        "methane": "ch4",
        "carbondioxide": "co2",
        "carbonmonoxide": "co",
        "ammonia": "nh3",
    }
    return aliases.get(key, key)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ElectronSwarmSection:
    reduced_field_Td: np.ndarray
    reduced_values_SI: np.ndarray


@dataclass(frozen=True)
class ElectronSwarmTable:
    path: Path
    sha256: str
    manifest_record: dict[str, Any]
    sections: dict[str, ElectronSwarmSection]

    def section(self, label: str) -> ElectronSwarmSection:
        try:
            return self.sections[label]
        except KeyError as exc:
            raise ValueError(f"Electron table {self.path} has no section {label!r}") from exc

    def provenance(self) -> dict[str, Any]:
        record = self.manifest_record
        return {
            "path": self.path.resolve().relative_to(
                data_paths.ELECTRON_SWARM_DATA_DIR.resolve()
            ).as_posix(),
            "sha256": self.sha256,
            "gas": record["gas"],
            "canonical_gas": record["canonical_gas"],
            "header_gas": record["header_gas"],
            "gas_temperature_K": record["gas_temperature_K"],
            "lxcat_database": record["lxcat_database"],
            "bolsig_batch": record["batch"],
            "common_e_over_n_min_Td": record["common_e_over_n_min_Td"],
            "common_e_over_n_max_Td": record["common_e_over_n_max_Td"],
        }


_TABLE_CACHE: dict[tuple[str, str], ElectronSwarmTable] = {}


def _load_manifest() -> dict[str, Any]:
    path = data_paths.ELECTRON_SWARM_DATA_DIR / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Electron swarm-data manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed electron swarm-data manifest: {path}") from exc

    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"Unsupported electron manifest format in {path}: {manifest.get('format')!r}"
        )
    tables = manifest.get("tables")
    if not isinstance(tables, list) or len(tables) != manifest.get("table_count"):
        raise ValueError(f"Electron manifest table count is inconsistent: {path}")
    if tuple(manifest.get("required_sections", ())) != REQUIRED_SECTIONS:
        raise ValueError(f"Electron manifest required-section declaration is invalid: {path}")
    return manifest


def _manifest_record_for_path(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = data_paths.ELECTRON_SWARM_DATA_DIR.resolve()
    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Electron table is outside the bundled data root: {path}") from exc
    matches = [record for record in manifest["tables"] if record.get("filename") == relative]
    if len(matches) != 1:
        raise ValueError(
            f"Electron table {relative!r} does not have exactly one manifest record."
        )
    return matches[0]


def _parse_sections(path: Path, record: dict[str, Any]) -> dict[str, ElectronSwarmSection]:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Electron table is not valid UTF-8 text: {path}") from exc

    sections: dict[str, ElectronSwarmSection] = {}
    for label in REQUIRED_SECTIONS:
        pairs: list[tuple[float, float]] = []
        in_section = False
        header_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("E/N (Td)") and label in stripped:
                header_count += 1
                if header_count > 1:
                    raise ValueError(f"Duplicate electron section {label!r} in {path}")
                in_section = True
                continue
            if in_section and stripped.startswith("E/N (Td)"):
                break
            if not in_section or not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                pairs.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

        if header_count != 1 or len(pairs) < 2:
            raise ValueError(f"Missing or incomplete electron section {label!r} in {path}")
        raw = np.asarray(pairs, dtype=np.float64)
        axis = raw[:, 0]
        values = raw[:, 1]
        if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(values)):
            raise ValueError(f"Electron section {label!r} contains NaN or infinity: {path}")
        if np.any(axis <= 0.0) or np.any(np.diff(axis) <= 0.0):
            raise ValueError(
                f"Electron section {label!r} requires a positive, unique, strictly "
                f"increasing E/N axis: {path}"
            )
        if label in {SECTION_MOBILITY, SECTION_DIFFUSION}:
            invalid_values = np.any(values <= 0.0)
            requirement = "strictly positive"
        else:
            invalid_values = np.any(values < 0.0)
            requirement = "non-negative"
        if invalid_values:
            raise ValueError(
                f"Electron section {label!r} requires {requirement} values: {path}"
            )
        expected_rows = record.get("section_rows", {}).get(label)
        if expected_rows != len(axis):
            raise ValueError(
                f"Electron section {label!r} has {len(axis)} rows but the manifest "
                f"requires {expected_rows}: {path}"
            )
        axis.setflags(write=False)
        values.setflags(write=False)
        sections[label] = ElectronSwarmSection(axis, values)
    return sections


def load_electron_swarm_table(
    file_value: str | Path,
    *,
    configured_gas: str | None = None,
    gas_temperature_K: float | None = None,
    temperature_tolerance_K: float = 0.0,
) -> ElectronSwarmTable:
    """Load, authenticate, parse, and optionally identity-check an electron table."""
    path = data_paths.resolve_electron_swarm_data_file(file_value)
    manifest = _load_manifest()
    record = _manifest_record_for_path(path, manifest)
    actual_sha256 = sha256_file(path)
    expected_sha256 = str(record.get("sha256", ""))
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Electron table checksum mismatch for {path}; expected {expected_sha256}, "
            f"got {actual_sha256}."
        )

    if configured_gas is not None and (
        canonical_electron_gas(configured_gas)
        != canonical_electron_gas(str(record.get("canonical_gas", "")))
    ):
        raise ValueError(
            f"Electron table gas mismatch: configuration requests {configured_gas!r}, "
            f"but {path.name} is for {record.get('gas')!r}."
        )
    if gas_temperature_K is not None:
        requested = float(gas_temperature_K)
        tolerance = float(temperature_tolerance_K)
        table_temperature = float(record.get("gas_temperature_K"))
        if not np.isfinite(requested) or not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Electron table temperature and tolerance must be finite.")
        if abs(requested - table_temperature) > tolerance:
            raise ValueError(
                f"Electron table gas-temperature mismatch: configuration requests "
                f"{requested:g} K, table is {table_temperature:g} K, tolerance is "
                f"{tolerance:g} K."
            )

    cache_key = (str(path), actual_sha256)
    table = _TABLE_CACHE.get(cache_key)
    if table is None:
        table = ElectronSwarmTable(
            path=path,
            sha256=actual_sha256,
            manifest_record=record,
            sections=_parse_sections(path, record),
        )
        _TABLE_CACHE[cache_key] = table
    return table
