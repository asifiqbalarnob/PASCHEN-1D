"""Build a deterministic source archive containing code, notebooks, and swarm data."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import sys
import tarfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from version import __version__


EXCLUDED_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(root: Path) -> list[Path]:
    output_directories = {
        path.parent.resolve() for path in root.rglob("run_metadata.json")
    }
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in path.parts
        ):
            continue
        if any(parent in output_directories for parent in path.resolve().parents):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def main() -> None:
    root = PROJECT_ROOT
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    archive = dist / f"PASCHEN-1D-{__version__}.tar.gz"
    prefix = f"PASCHEN-1D-{__version__}"
    with archive.open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as compressed, tarfile.open(mode="w", fileobj=compressed) as tar:
        for path in included_files(root):
            info = tar.gettarinfo(
                str(path), arcname=f"{prefix}/{path.relative_to(root).as_posix()}"
            )
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                tar.addfile(info, stream)
    checksum = sha256_file(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    print(archive)
    print(checksum_path)


if __name__ == "__main__":
    main()
