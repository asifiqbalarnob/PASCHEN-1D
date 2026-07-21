"""Authenticate every bundled electron table and normalized ion dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from electron_transport import load_electron_swarm_table
from ion_transport import load_ion_transport_table, sha256_file


def main() -> None:
    project = PROJECT_ROOT
    electron_root = project / "electron_swarm_data"
    electron_manifest = json.loads((electron_root / "manifest.json").read_text())
    electron_records = electron_manifest["tables"]
    for record in electron_records:
        load_electron_swarm_table(record["filename"])

    ion_root = project / "ion_swarm_data"
    normalized_root = ion_root / "normalized_lxcat_2026-07-21"
    ion_manifest = json.loads((normalized_root / "manifest.json").read_text())
    normalized_records = [
        record for record in ion_manifest["records"] if record.get("normalized_file")
    ]
    dataset_ids = set()
    for record in normalized_records:
        if record["dataset_id"] in dataset_ids:
            raise ValueError(f"Duplicate normalized ion dataset ID: {record['dataset_id']}")
        dataset_ids.add(record["dataset_id"])
        table = load_ion_transport_table(normalized_root / record["normalized_file"])
        if table.dataset_id != record["dataset_id"]:
            raise ValueError(f"Ion manifest/table ID mismatch: {table.path}")

    expected_count = int(ion_manifest["statistics"]["normalized_table_count"])
    if len(normalized_records) != expected_count:
        raise ValueError(
            f"Ion manifest declares {expected_count} normalized tables but contains "
            f"{len(normalized_records)} records."
        )
    for pair in ion_manifest["compatible_mobility_diffusion_pairs"]:
        if pair["mobility_dataset_id"] not in dataset_ids:
            raise ValueError(f"Unknown mobility dataset in pair {pair['pair_id']}")
        if pair["diffusion_dataset_id"] not in dataset_ids:
            raise ValueError(f"Unknown diffusion dataset in pair {pair['pair_id']}")

    raw_sources = {
        (str(record["source_file"]), str(record["source_file_sha256"]))
        for record in ion_manifest["records"]
        if record.get("source_file") and record.get("source_file_sha256")
    }
    for relative, expected_sha256 in sorted(raw_sources):
        path = ion_root / relative
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"Raw LXCat source checksum mismatch: {path}")

    print(
        f"Authenticated {len(electron_records)} electron tables, "
        f"{len(normalized_records)} normalized ion tables, "
        f"{len(ion_manifest['compatible_mobility_diffusion_pairs'])} compatible pairs, "
        f"and {len(raw_sources)} preserved raw ion exports."
    )


if __name__ == "__main__":
    main()
