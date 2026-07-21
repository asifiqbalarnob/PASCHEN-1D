"""Build the release manifest for bundled BOLSIG+ electron swarm tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


FORMAT = "paschen-1d-electron-swarm-manifest-v1"
REQUIRED_SECTIONS = (
    "Mobility *N (1/m/V/s)",
    "Diffusion coefficient *N (1/m/s)",
    "Townsend ioniz. coef. alpha/N (m2)",
    "Total ionization freq. /N (m3/s)",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_header(path: Path) -> tuple[float, str]:
    text = path.read_text(encoding="utf-8", errors="strict")
    temperature_match = re.search(
        r"^Gas temperature \(K\)\s+([0-9.Ee+-]+)\s*$", text, re.MULTILINE
    )
    fraction_match = re.search(r"^Mole fraction\s+(.+?)\s+1\.000\s*$", text, re.MULTILINE)
    if temperature_match is None or fraction_match is None:
        raise ValueError(f"Could not identify gas and temperature in {path}")
    return float(temperature_match.group(1)), fraction_match.group(1).strip()


def section_axes(path: Path) -> dict[str, list[float]]:
    axes: dict[str, list[float]] = {}
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    for label in REQUIRED_SECTIONS:
        values: list[float] = []
        in_section = False
        for line in lines:
            stripped = line.strip()
            if not in_section:
                if stripped.startswith("E/N (Td)") and label in stripped:
                    in_section = True
                continue
            if stripped.startswith("E/N (Td)"):
                break
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                values.append(float(parts[0]))
                float(parts[1])
            except ValueError:
                continue
        if len(values) < 2:
            raise ValueError(f"Missing or incomplete section {label!r} in {path}")
        axes[label] = values
    return axes


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repository_root = project_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tables",
        type=Path,
        default=project_root / "electron_swarm_data" / "lxcat_2026-07-20",
    )
    parser.add_argument(
        "--generation-manifest",
        type=Path,
        default=(
            repository_root
            / "bolsigplus032016-mac"
            / "lxcat_complete_2026-07-20"
            / "generation_manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "electron_swarm_data" / "manifest.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed manifest differs instead of rewriting it.",
    )
    args = parser.parse_args()

    generation = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
    usable = {
        str(item["slug"]): item
        for item in generation["results"]
        if item.get("usable") is True
    }
    records = []
    for path in sorted(args.tables.glob("*_swarm_output_full_EoverN.dat")):
        slug = path.name.removesuffix("_swarm_output_full_EoverN.dat")
        source = usable.get(slug)
        if source is None:
            raise ValueError(f"No usable generation record for {path.name}")
        temperature, header_gas = parse_header(path)
        axes = section_axes(path)
        section_rows = {label: len(axis) for label, axis in axes.items()}
        common_min = max(axis[0] for axis in axes.values())
        common_max = min(axis[-1] for axis in axes.values())
        records.append(
            {
                "filename": f"lxcat_2026-07-20/{path.name}",
                "sha256": sha256_file(path),
                "gas": source["gas"],
                "canonical_gas": slug,
                "header_gas": header_gas,
                "gas_temperature_K": temperature,
                "lxcat_database": source["database"],
                "batch": source["batch"],
                "section_rows": section_rows,
                "common_e_over_n_min_Td": common_min,
                "common_e_over_n_max_Td": common_max,
            }
        )

    payload = {
        "format": FORMAT,
        "generated_on": generation["generated_on"],
        "solver": generation["solver"],
        "source_generation_manifest_sha256": sha256_file(args.generation_manifest),
        "required_sections": list(REQUIRED_SECTIONS),
        "table_count": len(records),
        "tables": records,
    }
    if len(records) != len(usable):
        raise ValueError(
            f"Manifest would contain {len(records)} tables but generation metadata has "
            f"{len(usable)} usable records"
        )
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.check:
        committed = args.output.read_text(encoding="utf-8")
        if committed != rendered:
            raise SystemExit(
                f"Electron manifest is stale: run {Path(__file__).name} without --check."
            )
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
