#!/usr/bin/env python3
"""Index preserved LXCat ion-swarm exports and normalize PASCHEN-1D tables.

The raw files are never edited.  Every tabulated LXCat process becomes a
manifest record, while the two quantities used by PASCHEN-1D are additionally
written as small, self-describing CSV files:

* Mobility (Ko) -> reduced mobility K0*N0 [1/(m V s)]
* Diffusion x gas density (NDz) -> N*D_longitudinal [1/(m s)]

Independent publications, temperatures, and ion/neutral pairs remain separate.
The program deliberately does not average or splice datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


FORMAT_VERSION = "paschen-1d-ion-transport-v1"
STANDARD_NUMBER_DENSITY_M3 = 100_000.0 / (1.380649e-23 * 273.15)
MOBILITY_PROCESS = "Mobility (Ko)"
DIFFUSION_PROCESS = "Diffusion x gas density (NDz)"
SUPPORTED_PROCESS_MAP = {
    MOBILITY_PROCESS: (
        "reduced_mobility",
        "K0_times_N0",
        "1/(m V s)",
        1.0e-4 * STANDARD_NUMBER_DENSITY_M3,
    ),
    DIFFUSION_PROCESS: (
        "reduced_longitudinal_diffusion",
        "N_times_D_longitudinal",
        "1/(m s)",
        1.0e20,
    ),
}


@dataclass
class ParsedRecord:
    source_file: Path
    source_file_sha256: str
    source_block_index: int
    database: str
    permalink: str
    reference: str
    species: str
    ion: str
    neutral: str
    process: str
    parameters: str
    group_comment: str
    comment: str
    updated: str
    columns: str
    data: list[list[float]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(text: str) -> str:
    value = text.replace("+", "_plus_").replace("-", "_minus_")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "unnamed"


def source_family(comment: str) -> str:
    """Derive a conservative cross-quantity identity from an LXCat note."""

    value = re.sub(r"\b(?:mobility|diffusion)\b", "transport", comment, flags=re.I)
    return re.sub(r"\s+", " ", value).strip().casefold()


def parse_temperature_K(parameters: str) -> float | None:
    match = re.search(
        r"(?:Tgas|To)\s*=\s*([0-9.eE+\-]+)\s*K\b", parameters, re.IGNORECASE
    )
    return float(match.group(1)) if match else None


def _field(lines: list[str], start: int, label: str, stop_labels: tuple[str, ...]) -> str:
    prefix = f"{label}:"
    if not lines[start].startswith(prefix):
        return ""
    pieces = [lines[start][len(prefix) :].strip()]
    index = start + 1
    while index < len(lines):
        line = lines[index]
        if any(line.startswith(f"{item}:") for item in stop_labels):
            break
        if not line.strip() or re.match(r"^[*x\-]{8,}", line.strip()):
            break
        if line.startswith(" ") or line.startswith("\t"):
            pieces.append(line.strip())
            index += 1
            continue
        break
    return " ".join(piece for piece in pieces if piece)


def parse_lxcat_file(path: Path, raw_root: Path) -> list[ParsedRecord]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    lines = text.splitlines()
    file_hash = sha256_file(path)
    database = ""
    permalink = ""
    reference = ""
    group_comment = ""
    records: list[ParsedRecord] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("DATABASE:"):
            database = line.split(":", 1)[1].strip()
            group_comment = ""
        elif line.startswith("PERMLINK:"):
            permalink = line.split(":", 1)[1].strip()
        elif line.startswith("HOW TO REFERENCE:"):
            reference = _field(
                lines,
                index,
                "HOW TO REFERENCE",
                ("DATABASE", "SPECIES", "CONTACT", "DESCRIPTION"),
            )
        elif line.startswith("COMMENT:"):
            group_comment = _field(
                lines,
                index,
                "COMMENT",
                ("DATABASE", "SPECIES", "PROCESS", "PARAM.", "UPDATED", "COLUMNS"),
            )
        elif line.startswith("SPECIES:"):
            species_parts: list[str] = []
            species_cursor = index
            while species_cursor < len(lines) and lines[species_cursor].startswith(
                "SPECIES:"
            ):
                species_parts.append(
                    lines[species_cursor].split(":", 1)[1].strip()
                )
                species_cursor += 1
            # A few official LXCat exports wrap a long pair over two labeled
            # SPECIES lines (ion on the first, "/ neutral" on the second).
            species = " ".join(part for part in species_parts if part)
            if " / " not in species:
                raise ValueError(f"Malformed SPECIES line in {path}: {line}")
            ion, neutral = (part.strip() for part in species.rsplit(" / ", 1))
            columns_line = species_cursor
            while columns_line < len(lines) and not lines[columns_line].startswith("COLUMNS:"):
                if lines[columns_line].startswith(("SPECIES:", "DATABASE:")):
                    break
                columns_line += 1
            if columns_line >= len(lines) or not lines[columns_line].startswith("COLUMNS:"):
                raise ValueError(f"Incomplete LXCat process block in {path} at line {index + 1}")
            block = lines[index : columns_line + 1]
            labels: dict[str, int] = {}
            for offset, block_line in enumerate(block):
                for label in ("PROCESS", "PARAM.", "COMMENT", "UPDATED", "COLUMNS"):
                    if block_line.startswith(f"{label}:") and label not in labels:
                        labels[label] = offset
            if "PROCESS" not in labels or "COLUMNS" not in labels:
                raise ValueError(f"Incomplete LXCat process block in {path} at line {index + 1}")
            process = block[labels["PROCESS"]].split(":", 1)[1].strip()
            parameters = (
                _field(block, labels["PARAM."], "PARAM.", ("COMMENT", "UPDATED", "COLUMNS"))
                if "PARAM." in labels
                else ""
            )
            comment = (
                _field(block, labels["COMMENT"], "COMMENT", ("UPDATED", "COLUMNS"))
                if "COMMENT" in labels
                else ""
            )
            updated = (
                block[labels["UPDATED"]].split(":", 1)[1].strip()
                if "UPDATED" in labels
                else ""
            )
            columns = block[labels["COLUMNS"]].split(":", 1)[1].strip()
            data: list[list[float]] = []
            data_cursor = columns_line + 1
            while data_cursor < len(lines):
                stripped = lines[data_cursor].strip()
                if not stripped or re.match(r"^[*x\-]{3,}$", stripped):
                    if data:
                        break
                    data_cursor += 1
                    continue
                fields = re.split(r"[\t ,]+", stripped)
                try:
                    row = [float(item) for item in fields]
                except ValueError:
                    if data:
                        break
                    data_cursor += 1
                    continue
                if len(row) >= 2:
                    data.append(row)
                data_cursor += 1
            if not data:
                raise ValueError(f"No numeric data in {path} at line {index + 1}")
            records.append(
                ParsedRecord(
                    source_file=path.relative_to(raw_root),
                    source_file_sha256=file_hash,
                    source_block_index=len(records) + 1,
                    database=database,
                    permalink=permalink,
                    reference=reference,
                    species=species,
                    ion=ion,
                    neutral=neutral,
                    process=process,
                    parameters=parameters,
                    group_comment=group_comment,
                    comment=comment,
                    updated=updated,
                    columns=columns,
                    data=data,
                )
            )
            index = data_cursor
        index += 1
    return records


def record_id(record: ParsedRecord) -> str:
    identity = "\n".join(
        (
            str(record.source_file),
            str(record.source_block_index),
            record.database,
            record.species,
            record.process,
            record.parameters,
            record.group_comment,
            record.comment,
            record.updated,
        )
    )
    return "lxcat_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def write_normalized_csv(path: Path, metadata: dict, rows: list[tuple[float, float]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        for key, value in metadata.items():
            stream.write(f"# {key}: {json.dumps(value, ensure_ascii=False)}\n")
        writer = csv.writer(stream)
        writer.writerow(("reduced_electric_field_Td", "reduced_transport_SI"))
        writer.writerows((f"{x:.12e}", f"{y:.12e}") for x, y in rows)
    return sha256_file(path)


def normalize(raw_root: Path, output_root: Path) -> dict:
    if not (raw_root / "catalog.json").exists():
        raise FileNotFoundError(f"LXCat catalog not found: {raw_root / 'catalog.json'}")
    summary = json.loads((raw_root / "download_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") not in {"complete", "complete_with_source_errors"}:
        raise RuntimeError("Refusing to normalize an incomplete LXCat download archive.")

    source_export_failures: list[dict] = []
    for metadata_path in sorted((raw_root / "batches").glob("*/batch_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for failure in metadata.get("failed_processes", []):
            source_export_failures.append(
                {
                    "batch_id": metadata.get("batch_id"),
                    "ion_ids": metadata.get("ion_ids", []),
                    "neutral_ids": metadata.get("neutral_ids", []),
                    **failure,
                }
            )

    files = sorted((raw_root / "batches").glob("*/*.txt"))
    if not files:
        raise RuntimeError(f"No LXCat ASCII exports found below {raw_root}")
    records: list[ParsedRecord] = []
    for path in files:
        records.extend(parse_lxcat_file(path, raw_root))

    manifest_records: list[dict] = []
    normalized_count = 0
    rejected_normalization_count = 0
    pair_set: set[tuple[str, str]] = set()
    neutral_set: set[str] = set()
    ion_set: set[str] = set()
    for record in records:
        dataset_id = record_id(record)
        pair_set.add((record.ion, record.neutral))
        ion_set.add(record.ion)
        neutral_set.add(record.neutral)
        temperature_K = parse_temperature_K(record.parameters)
        axis_values = [row[0] for row in record.data]
        axis_values_sha256 = hashlib.sha256(
            json.dumps(axis_values, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        item = {
            "dataset_id": dataset_id,
            "database": record.database,
            "permalink": record.permalink,
            "reference": record.reference,
            "species": record.species,
            "ion": record.ion,
            "neutral": record.neutral,
            "charge_sign": "positive" if "+" in record.ion else "negative" if "-" in record.ion else "unknown",
            "process": record.process,
            "parameters": record.parameters,
            "gas_temperature_K": temperature_K,
            "group_comment": record.group_comment,
            "source_family": source_family(record.group_comment),
            "comment": record.comment,
            "updated": record.updated,
            "columns": record.columns,
            "point_count": len(record.data),
            "axis_min": min(row[0] for row in record.data),
            "axis_max": max(row[0] for row in record.data),
            "axis_values_sha256": axis_values_sha256,
            "source_file": str(Path("raw_lxcat") / raw_root.name / record.source_file),
            "source_file_sha256": record.source_file_sha256,
            "source_block_index": record.source_block_index,
            "normalization_status": "unsupported_process",
            "normalization_issues": [],
            "normalized_file": None,
        }
        if record.process in SUPPORTED_PROCESS_MAP:
            quantity, reduced_name, units, factor = SUPPORTED_PROCESS_MAP[record.process]
            rows = sorted((row[0], row[1] * factor) for row in record.data)
            issues: list[str] = []
            if len(rows) < 2:
                issues.append("fewer than two tabulated points")
            if any(not all(math.isfinite(value) for value in row) for row in rows):
                issues.append("NaN or infinite tabulated value")
            if any(field <= 0.0 for field, _ in rows):
                issues.append("non-positive reduced-electric-field value")
            if any(value <= 0.0 for _, value in rows):
                issues.append("non-positive reduced-transport value")
            if any(right[0] <= left[0] for left, right in zip(rows, rows[1:])):
                issues.append("duplicate reduced-electric-field value")
            if issues:
                item["normalization_status"] = "not_solver_usable"
                item["normalization_issues"] = issues
                rejected_normalization_count += 1
                manifest_records.append(item)
                continue
            relative = Path("tables") / slug(record.neutral) / slug(record.ion) / f"{dataset_id}_{quantity}.csv"
            metadata = {
                "format": FORMAT_VERSION,
                "dataset_id": dataset_id,
                "quantity": quantity,
                "reduced_quantity": reduced_name,
                "units": units,
                "ion": record.ion,
                "neutral": record.neutral,
                "gas_temperature_K": temperature_K,
                "database": record.database,
                "process": record.process,
                "parameters": record.parameters,
                "group_comment": record.group_comment,
                "source_family": source_family(record.group_comment),
                "comment": record.comment,
                "updated": record.updated,
                "permalink": record.permalink,
                "reference": record.reference,
                "source_file": item["source_file"],
                "source_file_sha256": record.source_file_sha256,
            }
            checksum = write_normalized_csv(output_root / relative, metadata, rows)
            item.update(
                {
                    "normalization_status": "normalized",
                    "normalized_file": str(relative),
                    "normalized_file_sha256": checksum,
                    "normalized_quantity": quantity,
                    "normalized_units": units,
                    "normalized_axis": "reduced_electric_field_Td",
                }
            )
            normalized_count += 1
        manifest_records.append(item)

    raw_catalog = json.loads((raw_root / "catalog.json").read_text(encoding="utf-8"))
    pair_groups: dict[tuple, dict[str, list[dict]]] = {}
    for item in manifest_records:
        quantity = item.get("normalized_quantity")
        if quantity not in {"reduced_mobility", "reduced_longitudinal_diffusion"}:
            continue
        # Without a source note LXCat provides no defensible basis for claiming
        # that independently exported mobility and diffusion tables belong
        # together. They remain available as individual manifest records.
        if not item["source_family"]:
            continue
        key = (
            item["database"],
            item["ion"],
            item["neutral"],
            item["gas_temperature_K"],
            item["parameters"],
            item["source_family"],
        )
        pair_groups.setdefault(key, {}).setdefault(quantity, []).append(item)
    compatible_pairs: list[dict] = []
    for group in pair_groups.values():
        for mobility in group.get("reduced_mobility", []):
            for diffusion in group.get("reduced_longitudinal_diffusion", []):
                overlap_min = max(mobility["axis_min"], diffusion["axis_min"])
                overlap_max = min(mobility["axis_max"], diffusion["axis_max"])
                if overlap_min > overlap_max:
                    continue
                identity = f"{mobility['dataset_id']}\n{diffusion['dataset_id']}"
                compatible_pairs.append(
                    {
                        "pair_id": "pair_" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                        "ion": mobility["ion"],
                        "neutral": mobility["neutral"],
                        "gas_temperature_K": mobility["gas_temperature_K"],
                        "database": mobility["database"],
                        "mobility_dataset_id": mobility["dataset_id"],
                        "mobility_file": mobility["normalized_file"],
                        "diffusion_dataset_id": diffusion["dataset_id"],
                        "diffusion_file": diffusion["normalized_file"],
                        "common_reduced_field_min_Td": overlap_min,
                        "common_reduced_field_max_Td": overlap_max,
                        "identical_reduced_field_grid": (
                            mobility["axis_values_sha256"]
                            == diffusion["axis_values_sha256"]
                        ),
                        "source_family": mobility["source_family"],
                        "mobility_group_comment": mobility["group_comment"],
                        "diffusion_group_comment": diffusion["group_comment"],
                        "reference": mobility["reference"],
                    }
                )
    manifest = {
        "schema_version": 1,
        "format": FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_archive": str(Path("raw_lxcat") / raw_root.name),
        "raw_catalog_retrieved_at_utc": raw_catalog.get("retrieved_at_utc"),
        "raw_download_completed_at_utc": summary.get("completed_at_utc"),
        "source": "LXCat ion swarm data center",
        "source_url": "https://us.lxcat.net/data/",
        "normalization": {
            "mobility": "K0[cm2/(V s)] * 1e-4 * N0, N0=100000/(kB*273.15)",
            "standard_number_density_m-3": STANDARD_NUMBER_DENSITY_M3,
            "diffusion": "NDz[1E20/(m s)] * 1e20",
            "policy": "Every independent LXCat dataset is preserved separately; no averaging or splicing.",
        },
        "statistics": {
            "raw_ascii_file_count": len(files),
            "process_record_count": len(records),
            "candidate_transport_table_count": (
                normalized_count + rejected_normalization_count
            ),
            "normalized_table_count": normalized_count,
            "rejected_transport_table_count": rejected_normalization_count,
            "ion_count": len(ion_set),
            "neutral_count": len(neutral_set),
            "ion_neutral_pair_count": len(pair_set),
            "compatible_mobility_diffusion_pair_count": len(compatible_pairs),
            "source_export_failure_count": len(source_export_failures),
        },
        "source_export_failures": source_export_failures,
        "compatible_mobility_diffusion_pairs": sorted(
            compatible_pairs, key=lambda item: item["pair_id"]
        ),
        "records": sorted(manifest_records, key=lambda item: item["dataset_id"]),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=project_root / "ion_swarm_data" / "raw_lxcat" / "lxcat_ion_swarm_2026-07-21",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "ion_swarm_data" / "normalized_lxcat_2026-07-21",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = normalize(args.raw_root.resolve(), args.output_root.resolve())
    print(json.dumps(manifest["statistics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
