"""Build a reproducible wheel containing code and both bundled swarm libraries."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from version import __release_date__, __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    dist = PROJECT_ROOT / "dist"
    dist.mkdir(exist_ok=True)
    release_time = datetime.strptime(__release_date__, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = str(int(release_time.timestamp()))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(PROJECT_ROOT),
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist),
        ],
        check=True,
        env=environment,
    )
    wheel = dist / f"paschen_1d-{__version__}-py3-none-any.whl"
    checksum = sha256_file(wheel)
    checksum_path = wheel.with_suffix(wheel.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {wheel.name}\n", encoding="ascii")
    print(wheel)
    print(checksum_path)


if __name__ == "__main__":
    main()
