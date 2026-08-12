"""Strict loading, validation, and interpolation of normalized ion tables."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import data_paths


ION_TABLE_FORMAT = "paschen-1d-ion-transport-v1"
ION_MANIFEST_DIRECTORY = "normalized_lxcat_2026-07-21"
_MANIFEST_CACHE: dict[tuple[str, int, int], tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}


def canonical_neutral(value: str) -> str:
    """Return a comparison key for common gas names and LXCat formulas."""
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {
        "argon": "ar",
        "ar": "ar",
        "nitrogen": "n2",
        "n2": "n2",
        "deuterium": "d2",
        "d2": "d2",
        "hydrogen": "h2",
        "h2": "h2",
        "helium": "he",
        "he": "he",
        "neon": "ne",
        "ne": "ne",
        "krypton": "kr",
        "kr": "kr",
        "xenon": "xe",
        "xe": "xe",
        "oxygen": "o2",
        "o2": "o2",
    }
    return aliases.get(key, key)


def canonical_ion(value: str) -> str:
    """Normalize harmless typography while preserving isotope/state identity."""
    return re.sub(r"[\s^{}()]", "", str(value)).lower()


def resolve_data_path(path_value: str | Path) -> Path:
    """Resolve a config data path relative to the package or current directory."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        candidates = (path,)
    else:
        package_root = Path(__file__).resolve().parent
        candidates = (Path.cwd() / path, package_root / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Ion transport table not found; checked: {rendered}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ion_manifest() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and index the ion manifest once per on-disk revision."""
    normalized_root = (data_paths.ION_SWARM_DATA_DIR / ION_MANIFEST_DIRECTORY).resolve()
    manifest_path = normalized_root / "manifest.json"
    try:
        stat = manifest_path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Ion transport manifest not found: {manifest_path}") from exc
    cache_key = (str(manifest_path), stat.st_mtime_ns, stat.st_size)
    cached = _MANIFEST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed ion transport manifest: {manifest_path}") from exc
    if manifest.get("format") != ION_TABLE_FORMAT or manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported ion transport manifest schema: {manifest_path}")
    records = manifest.get("records")
    pairs = manifest.get("compatible_mobility_diffusion_pairs")
    if not isinstance(records, list) or not isinstance(pairs, list):
        raise ValueError(f"Ion transport manifest lists are malformed: {manifest_path}")
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        relative = record.get("normalized_file")
        if relative is None:
            continue
        if relative in index:
            raise ValueError(f"Duplicate normalized ion path in manifest: {relative!r}")
        index[str(relative)] = record
    value = (manifest, index)
    _MANIFEST_CACHE[cache_key] = value
    return value


def _bundled_manifest_record(path: Path) -> dict[str, Any] | None:
    """Return and authenticate the manifest record for a bundled normalized table."""
    normalized_root = (data_paths.ION_SWARM_DATA_DIR / ION_MANIFEST_DIRECTORY).resolve()
    try:
        relative = path.resolve().relative_to(normalized_root).as_posix()
    except ValueError:
        # Standalone tables remain loadable for development and unit testing.
        return None

    _, index = _load_ion_manifest()
    record = index.get(relative)
    if record is None:
        raise ValueError(
            f"Bundled ion table {relative!r} does not have a manifest record."
        )
    expected = str(record.get("normalized_file_sha256", ""))
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError(
            f"Ion table checksum mismatch for {path}; expected {expected}, got {actual}."
        )
    return record


@dataclass(frozen=True)
class IonTransportTable:
    path: Path
    metadata: dict[str, Any]
    reduced_field_Td: np.ndarray
    reduced_transport_SI: np.ndarray
    sha256: str

    @property
    def dataset_id(self) -> str:
        return str(self.metadata["dataset_id"])

    @property
    def quantity(self) -> str:
        return str(self.metadata["quantity"])

    def provenance(self) -> dict[str, Any]:
        try:
            portable_path = self.path.resolve().relative_to(
                data_paths.ION_SWARM_DATA_DIR.resolve()
            ).as_posix()
        except ValueError:
            portable_path = self.path.name
        return {
            "path": portable_path,
            "sha256": self.sha256,
            "dataset_id": self.dataset_id,
            "quantity": self.quantity,
            "ion": self.metadata["ion"],
            "neutral": self.metadata["neutral"],
            "gas_temperature_K": self.metadata.get("gas_temperature_K"),
            "database": self.metadata.get("database"),
            "process": self.metadata.get("process"),
            "parameters": self.metadata.get("parameters"),
            "group_comment": self.metadata.get("group_comment"),
            "source_family": self.metadata.get("source_family"),
            "comment": self.metadata.get("comment"),
            "updated": self.metadata.get("updated"),
            "permalink": self.metadata.get("permalink"),
            "reference": self.metadata.get("reference"),
            "source_file": self.metadata.get("source_file"),
            "source_file_sha256": self.metadata.get("source_file_sha256"),
        }


@dataclass
class IonTableInterpolator:
    table: IonTransportTable
    log_reduced_field_Td: np.ndarray
    log_reduced_transport_SI: np.ndarray
    out_of_range_policy: str
    below_range_count: int = 0
    above_range_count: int = 0
    evaluated_value_count: int = 0
    requested_min_Td: float = np.inf
    requested_max_Td: float = -np.inf

    @classmethod
    def from_table(
        cls, table: IonTransportTable, out_of_range_policy: str
    ) -> "IonTableInterpolator":
        if out_of_range_policy not in {"clip", "error"}:
            raise ValueError(
                "ion_transport.out_of_range_policy must be 'clip' or 'error'."
            )
        return cls(
            table=table,
            log_reduced_field_Td=np.log10(table.reduced_field_Td),
            log_reduced_transport_SI=np.log10(table.reduced_transport_SI),
            out_of_range_policy=out_of_range_policy,
        )

    def evaluate(self, electric_field_V_m: np.ndarray, neutral_density_m3: float) -> np.ndarray:
        density = float(neutral_density_m3)
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError(f"neutral density must be finite and positive; got {density!r}")
        reduced_field = np.abs(electric_field_V_m).astype(np.float64, copy=False) * 1.0e21 / density
        lower = float(self.table.reduced_field_Td[0])
        upper = float(self.table.reduced_field_Td[-1])
        below_count = int(np.count_nonzero(reduced_field < lower))
        above_count = int(np.count_nonzero(reduced_field > upper))
        self.below_range_count += below_count
        self.above_range_count += above_count
        self.evaluated_value_count += int(reduced_field.size)
        if reduced_field.size:
            self.requested_min_Td = min(
                self.requested_min_Td, float(np.min(reduced_field))
            )
            self.requested_max_Td = max(
                self.requested_max_Td, float(np.max(reduced_field))
            )
        if self.out_of_range_policy == "error" and (below_count or above_count):
            raise ValueError(
                f"Ion {self.table.quantity} E/N is outside dataset {self.table.dataset_id}: "
                f"requested [{float(np.min(reduced_field)):.6g}, "
                f"{float(np.max(reduced_field)):.6g}] Td; table [{lower:.6g}, {upper:.6g}] Td."
            )
        reduced_field = np.clip(reduced_field, lower, upper)
        reduced_value = np.power(
            10.0,
            np.interp(
                np.log10(reduced_field),
                self.log_reduced_field_Td,
                self.log_reduced_transport_SI,
            ),
        )
        return (reduced_value / density).astype(np.float32, copy=False)

    def coverage(self) -> dict[str, Any]:
        """Return accumulated table-range diagnostics for run metadata."""
        return {
            "policy": self.out_of_range_policy,
            "table_min_Td": float(self.table.reduced_field_Td[0]),
            "table_max_Td": float(self.table.reduced_field_Td[-1]),
            "evaluated_value_count": self.evaluated_value_count,
            "below_range_count": self.below_range_count,
            "above_range_count": self.above_range_count,
            "requested_min_Td": (
                self.requested_min_Td if self.evaluated_value_count else None
            ),
            "requested_max_Td": (
                self.requested_max_Td if self.evaluated_value_count else None
            ),
        }


def _parse_metadata_line(line: str, path: Path) -> tuple[str, Any]:
    content = line[1:].strip()
    if ":" not in content:
        raise ValueError(f"Malformed metadata line in {path}: {line.rstrip()}")
    key, encoded = content.split(":", 1)
    try:
        value = json.loads(encoded.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON metadata {key!r} in {path}") from exc
    return key.strip(), value


def load_ion_transport_table(path_value: str | Path) -> IonTransportTable:
    path = resolve_data_path(path_value)
    manifest_record = _bundled_manifest_record(path)
    metadata: dict[str, Any] = {}
    content_lines: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line in stream:
            if line.startswith("#"):
                key, value = _parse_metadata_line(line, path)
                if key in metadata:
                    raise ValueError(f"Duplicate metadata key {key!r} in {path}")
                metadata[key] = value
            elif line.strip():
                content_lines.append(line)

    required = {
        "format",
        "dataset_id",
        "quantity",
        "reduced_quantity",
        "units",
        "ion",
        "neutral",
        "gas_temperature_K",
        "database",
        "process",
        "parameters",
        "group_comment",
        "source_family",
        "permalink",
        "reference",
        "source_file",
        "source_file_sha256",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Ion table {path} is missing metadata: {', '.join(missing)}")
    if metadata["format"] != ION_TABLE_FORMAT:
        raise ValueError(
            f"Unsupported ion table format in {path}: {metadata['format']!r}"
        )
    if metadata["quantity"] not in {
        "reduced_mobility",
        "reduced_longitudinal_diffusion",
    }:
        raise ValueError(f"Unsupported ion quantity in {path}: {metadata['quantity']!r}")
    expected_descriptors = {
        "reduced_mobility": ("K0_times_N0", "1/(m V s)"),
        "reduced_longitudinal_diffusion": ("N_times_D_longitudinal", "1/(m s)"),
    }
    expected_reduced_quantity, expected_units = expected_descriptors[metadata["quantity"]]
    if metadata["reduced_quantity"] != expected_reduced_quantity:
        raise ValueError(
            f"Ion table {path} has reduced_quantity={metadata['reduced_quantity']!r}; "
            f"expected {expected_reduced_quantity!r}."
        )
    if metadata["units"] != expected_units:
        raise ValueError(
            f"Ion table {path} has units={metadata['units']!r}; expected {expected_units!r}."
        )
    try:
        gas_temperature = float(metadata["gas_temperature_K"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Ion table {path} has an invalid gas temperature.") from exc
    if not np.isfinite(gas_temperature) or gas_temperature <= 0.0:
        raise ValueError(f"Ion table {path} requires a finite positive gas temperature.")
    source_checksum = str(metadata["source_file_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", source_checksum) is None:
        raise ValueError(f"Ion table {path} has an invalid source-file SHA-256.")

    reader = csv.DictReader(content_lines)
    expected_columns = {"reduced_electric_field_Td", "reduced_transport_SI"}
    if reader.fieldnames is None or set(reader.fieldnames) != expected_columns:
        raise ValueError(
            f"Ion table {path} must contain exactly {sorted(expected_columns)}; "
            f"found {reader.fieldnames}."
        )
    rows = list(reader)
    try:
        field = np.asarray([float(row["reduced_electric_field_Td"]) for row in rows])
        values = np.asarray([float(row["reduced_transport_SI"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric ion transport row in {path}") from exc
    if len(field) < 2:
        raise ValueError(f"Ion table {path} must contain at least two points.")
    if not np.all(np.isfinite(field)) or not np.all(np.isfinite(values)):
        raise ValueError(f"Ion table {path} contains NaN or infinite values.")
    if np.any(field <= 0.0) or np.any(values <= 0.0):
        raise ValueError(f"Ion table {path} requires strictly positive E/N and values.")
    if np.any(np.diff(field) <= 0.0):
        raise ValueError(f"Ion table {path} E/N axis must be strictly increasing.")
    table_sha256 = sha256_file(path)
    if manifest_record is not None:
        manifest_checks = {
            "dataset_id": metadata["dataset_id"],
            "ion": metadata["ion"],
            "neutral": metadata["neutral"],
            "gas_temperature_K": metadata["gas_temperature_K"],
            "normalized_quantity": metadata["quantity"],
            "point_count": len(field),
        }
        mismatches = [
            key
            for key, value in manifest_checks.items()
            if manifest_record.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"Ion table metadata disagrees with its manifest for {path}: "
                + ", ".join(mismatches)
            )
    return IonTransportTable(path, metadata, field, values, table_sha256)


def validate_table_identity(
    table: IonTransportTable,
    *,
    expected_quantity: str,
    configured_ion: str,
    configured_neutral: str,
    gas_temperature_K: float,
    temperature_tolerance_K: float,
) -> None:
    """Reject table/config mismatches before the time loop begins."""
    problems: list[str] = []
    if table.quantity != expected_quantity:
        problems.append(
            f"quantity is {table.quantity!r}, expected {expected_quantity!r}"
        )
    if canonical_ion(table.metadata["ion"]) != canonical_ion(configured_ion):
        problems.append(
            f"ion is {table.metadata['ion']!r}, configured ion is {configured_ion!r}"
        )
    if canonical_neutral(table.metadata["neutral"]) != canonical_neutral(configured_neutral):
        problems.append(
            f"neutral is {table.metadata['neutral']!r}, configured gas is {configured_neutral!r}"
        )
    table_temperature = table.metadata.get("gas_temperature_K")
    if table_temperature is not None:
        difference = abs(float(table_temperature) - float(gas_temperature_K))
        if difference > float(temperature_tolerance_K):
            problems.append(
                f"gas temperature is {float(table_temperature):g} K, configured T_i is "
                f"{float(gas_temperature_K):g} K (tolerance {float(temperature_tolerance_K):g} K)"
            )
    if problems:
        raise ValueError(
            f"Ion transport table/config mismatch for {table.path}: " + "; ".join(problems)
        )


def validate_table_pair(
    mobility_table: IonTransportTable,
    diffusion_table: IonTransportTable,
) -> dict[str, Any]:
    """Require a mobility/diffusion selection listed as compatible in the manifest."""
    manifest_path = data_paths.ION_SWARM_DATA_DIR / ION_MANIFEST_DIRECTORY / "manifest.json"
    manifest, _ = _load_ion_manifest()

    matches = [
        pair
        for pair in manifest.get("compatible_mobility_diffusion_pairs", [])
        if pair.get("mobility_dataset_id") == mobility_table.dataset_id
        and pair.get("diffusion_dataset_id") == diffusion_table.dataset_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "Selected ion mobility and diffusion tables are not an approved compatible "
            f"pair in {manifest_path}: mobility={mobility_table.dataset_id}, "
            f"diffusion={diffusion_table.dataset_id}."
        )
    pair = matches[0]
    lower = max(
        float(mobility_table.reduced_field_Td[0]),
        float(diffusion_table.reduced_field_Td[0]),
    )
    upper = min(
        float(mobility_table.reduced_field_Td[-1]),
        float(diffusion_table.reduced_field_Td[-1]),
    )
    if lower >= upper:
        raise ValueError("Selected ion mobility and diffusion tables have no E/N overlap.")
    return dict(pair)
