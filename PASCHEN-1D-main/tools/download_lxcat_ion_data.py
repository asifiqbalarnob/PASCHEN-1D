#!/usr/bin/env python3
"""Download the complete LXCat ion-swarm catalog through the public export UI.

The LXCat data center limits a displayed selection to 1000 processes.  This
downloader builds deterministic batches from the contributor catalog, submits
each batch through the same forms as the web UI, preserves the generated ASCII
files verbatim, and records URLs and SHA-256 checksums for reproducibility.

The downloader is resumable: a completed batch is skipped when its metadata
and downloaded files are still present with matching checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://us.lxcat.net"
ION_DATABASE_IDS = ("19", "192", "26")  # Phelps, UNAM, Viehland
ION_DATABASE_NAMES = {
    "19": "Phelps",
    "192": "UNAM",
    "26": "Viehland",
}
# The UI can display 1000 processes, but some mixed selections near that limit
# fail during asynchronous ASCII generation.  A conservative 400-record cap
# has proven reliable and still keeps the full catalog to a manageable number
# of resumable batches.
MAX_PROCESSES_PER_BATCH = 400
# LXCat's export backend is less reliable when one request combines many ion
# species, even when the displayed record count is the same. Keep those bins
# smaller while retaining the proven 400-record cap for a single ion split
# across neutral backgrounds.
# Exporting many unrelated ions together can trigger deterministic HTTP 500
# responses in LXCat even at low record counts. One complete ion per batch is
# slower but reliable, resumable, and preserves a simple audit boundary.
MAX_MIXED_PROCESSES_PER_BATCH = 1
REQUEST_DELAY_SECONDS = 0.35
REQUEST_TIMEOUT_SECONDS = 180
USER_AGENT = "PASCHEN-1D-LXCat-ion-downloader/1.0 (research data preservation)"


@dataclass(frozen=True)
class PairRecord:
    database_id: str
    ion_id: str
    neutral_id: str
    neutral_label: str
    reported_process_count: int
    href: str


@dataclass(frozen=True)
class DownloadBatch:
    batch_id: str
    ion_ids: tuple[str, ...]
    neutral_ids: tuple[str, ...] | None
    expected_process_count: int


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(text: str) -> str:
    value = re.sub(r"\.txt$", "", text, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "data"


def _form_values(soup: BeautifulSoup, field: str) -> list[str]:
    return [element["value"] for element in soup.select(f'input[name="{field}"]')]


class LXCatIonDownloader:
    def __init__(self, output_dir: Path, delay_seconds: float = REQUEST_DELAY_SECONDS):
        self.output_dir = output_dir
        self.delay_seconds = max(float(delay_seconds), 0.0)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _sleep(self) -> None:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)

    def _get(self, url: str, **kwargs) -> requests.Response:
        return self._request("GET", url, **kwargs)

    def _post(self, url: str, data, **kwargs) -> requests.Response:
        return self._request("POST", url, data=data, **kwargs)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Issue a polite request with bounded retry for transient server failures."""
        retryable_statuses = {429, 500, 502, 503, 504}
        last_response: requests.Response | None = None
        retry_attempts = max(int(kwargs.pop("_retry_attempts", 6)), 1)
        for attempt in range(retry_attempts):
            self._sleep()
            response = self.session.request(
                method,
                urljoin(BASE_URL, url),
                timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
            last_response = response
            if response.status_code not in retryable_statuses:
                response.raise_for_status()
                return response
            if attempt < retry_attempts - 1:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(2.0**attempt, 20.0)
                )
                print(
                    f"  LXCat returned HTTP {response.status_code}; "
                    f"retrying in {wait_seconds:.1f} s",
                    flush=True,
                )
                time.sleep(wait_seconds)
        assert last_response is not None
        last_response.raise_for_status()
        return last_response

    def _accept_reference_terms(self, response: requests.Response) -> requests.Response:
        """Follow LXCat's mandatory citation acknowledgement when presented."""
        current = response
        for _ in range(8):
            if "/instructions/how_reference.php" not in current.url:
                return current
            soup = BeautifulSoup(current.text, "html.parser")
            accept = next(
                (
                    anchor["href"]
                    for anchor in soup.find_all("a", href=True)
                    if "ref1=" in anchor["href"]
                ),
                None,
            )
            if accept is None:
                raise RuntimeError(
                    f"Could not accept LXCat reference terms at {current.url}"
                )
            current = self._get(accept)
        raise RuntimeError("LXCat reference acceptance did not converge.")

    def fetch_catalog(self) -> tuple[list[PairRecord], dict[str, str]]:
        """Return current ion pair records and first-species labels."""
        response = self._get("/contributors/")
        soup = BeautifulSoup(response.text, "html.parser")

        # Obtain the authoritative set of charged first-species IDs exposed by
        # the ion-filtered data-center workflow.  This cleanly excludes the
        # electron records that coexist with ions in the UNAM database.
        self._post(
            "/data/set_type.php",
            data=[("type", "swrm"), ("spec[]", "ions")],
        )
        response_species = self._post(
            "/data/set_databases.php",
            data=[("db[]", database_id) for database_id in ION_DATABASE_IDS],
        )
        species_soup = BeautifulSoup(response_species.text, "html.parser")
        ion_ids = set(_form_values(species_soup, "spec[]"))
        ion_labels: dict[str, str] = {}
        for checkbox in species_soup.select('input[name="spec[]"]'):
            parent_text = " ".join(checkbox.parent.stripped_strings)
            ion_labels[checkbox["value"]] = re.sub(r"\s+", " ", parent_text).strip()

        pairs: list[PairRecord] = []
        seen: set[tuple[str, str, str]] = set()
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if "preselect.php" not in href or "t=swrm" not in href:
                continue
            query = parse_qs(urlparse(href).query)
            database_id = query.get("d", [""])[0]
            ion_id = query.get("a", [""])[0]
            neutral_id = query.get("b", [""])[0]
            if database_id not in ION_DATABASE_IDS or ion_id not in ion_ids or not neutral_id:
                continue
            key = (database_id, ion_id, neutral_id)
            if key in seen:
                continue
            label = " ".join(anchor.stripped_strings)
            count_match = re.search(r"\[(\d+)\]\s*$", label)
            if not count_match:
                continue
            seen.add(key)
            pairs.append(
                PairRecord(
                    database_id=database_id,
                    ion_id=ion_id,
                    neutral_id=neutral_id,
                    neutral_label=re.sub(r"\s*\[\d+\]\s*$", "", label).strip(),
                    reported_process_count=int(count_match.group(1)),
                    href=urljoin(BASE_URL, href),
                )
            )

        if not pairs:
            raise RuntimeError("LXCat contributor catalog yielded no ion-swarm pairs.")
        return pairs, ion_labels

    @staticmethod
    def build_download_plan(pairs: Iterable[PairRecord]) -> list[DownloadBatch]:
        """Build deterministic batches expected to stay below LXCat's cap."""
        by_ion: dict[str, list[PairRecord]] = {}
        for pair in pairs:
            by_ion.setdefault(pair.ion_id, []).append(pair)

        whole_ion_items: list[tuple[str, int, tuple[str, ...]]] = []
        neutrals_by_ion: dict[str, tuple[str, ...]] = {}
        batches: list[DownloadBatch] = []
        for ion_id, records in sorted(by_ion.items(), key=lambda item: int(item[0])):
            total = sum(record.reported_process_count for record in records)
            if total <= MAX_PROCESSES_PER_BATCH:
                neutral_ids = tuple(
                    sorted({record.neutral_id for record in records}, key=int)
                )
                whole_ion_items.append((ion_id, total, neutral_ids))
                neutrals_by_ion[ion_id] = neutral_ids
                continue

            # Very large theoretical ions are split by neutral background.
            current_neutrals: list[str] = []
            current_total = 0
            for record in sorted(records, key=lambda item: int(item.neutral_id)):
                if current_neutrals and current_total + record.reported_process_count > MAX_PROCESSES_PER_BATCH:
                    batch_id = f"ion_{ion_id}_neutral_{'_'.join(current_neutrals)}"
                    batches.append(
                        DownloadBatch(
                            batch_id=batch_id,
                            ion_ids=(ion_id,),
                            neutral_ids=tuple(current_neutrals),
                            expected_process_count=current_total,
                        )
                    )
                    current_neutrals = []
                    current_total = 0
                current_neutrals.append(record.neutral_id)
                current_total += record.reported_process_count
            if current_neutrals:
                batch_id = f"ion_{ion_id}_neutral_{'_'.join(current_neutrals)}"
                batches.append(
                    DownloadBatch(
                        batch_id=batch_id,
                        ion_ids=(ion_id,),
                        neutral_ids=tuple(current_neutrals),
                        expected_process_count=current_total,
                    )
                )

        # First-fit decreasing packing for the remaining complete ion species.
        bins: list[tuple[list[str], int]] = []
        for ion_id, count, _ in sorted(
            whole_ion_items, key=lambda item: (-item[1], int(item[0]))
        ):
            for index, (ion_ids, total) in enumerate(bins):
                if total + count <= MAX_MIXED_PROCESSES_PER_BATCH:
                    ion_ids.append(ion_id)
                    bins[index] = (ion_ids, total + count)
                    break
            else:
                bins.append(([ion_id], count))

        for index, (ion_ids, total) in enumerate(bins, start=1):
            neutral_ids = tuple(
                sorted(
                    {
                        neutral
                        for ion_id in ion_ids
                        for neutral in neutrals_by_ion[ion_id]
                    },
                    key=int,
                )
            )
            batches.append(
                DownloadBatch(
                    batch_id=f"ions_{index:03d}_{ion_ids[0]}_{ion_ids[-1]}",
                    ion_ids=tuple(sorted(ion_ids, key=int)),
                    neutral_ids=neutral_ids,
                    expected_process_count=total,
                )
            )
        return sorted(batches, key=lambda batch: batch.batch_id)

    def _start_selection(self, batch: DownloadBatch) -> tuple[list[str], requests.Response]:
        self._post(
            "/data/set_type.php",
            data=[("type", "swrm"), ("spec[]", "ions")],
        )
        response = self._post(
            "/data/set_databases.php",
            data=[("db[]", database_id) for database_id in ION_DATABASE_IDS],
        )
        available_ions = set(_form_values(BeautifulSoup(response.text, "html.parser"), "spec[]"))
        missing_ions = set(batch.ion_ids) - available_ions
        if missing_ions:
            raise RuntimeError(f"Ion IDs disappeared from LXCat: {sorted(missing_ions)}")

        response = self._post(
            "/data/set_specA.php",
            data=[("spec[]", ion_id) for ion_id in batch.ion_ids],
        )
        response = self._accept_reference_terms(response)
        soup = BeautifulSoup(response.text, "html.parser")
        selected_neutrals = list(batch.neutral_ids or ())

        # LXCat automatically advances through a stage whenever only one
        # choice exists. Drive the form by the fields actually returned rather
        # than assuming that every intermediate page is shown.
        if _form_values(soup, "proc[]"):
            return selected_neutrals, response

        available_neutrals = _form_values(soup, "spec[]")
        if available_neutrals:
            selected_neutrals = (
                available_neutrals
                if batch.neutral_ids is None
                else [neutral for neutral in batch.neutral_ids if neutral in available_neutrals]
            )
            if not selected_neutrals:
                raise RuntimeError(
                    f"Batch {batch.batch_id} has no available neutral backgrounds."
                )
            response = self._post(
                "/data/set_specB.php",
                data=[("spec[]", neutral_id) for neutral_id in selected_neutrals],
            )
            response = self._accept_reference_terms(response)
            soup = BeautifulSoup(response.text, "html.parser")
            if _form_values(soup, "proc[]"):
                return selected_neutrals, response

        groups = _form_values(soup, "gr[]")
        if groups:
            response = self._post(
                "/data/set_groups.php",
                data=[("gr[]", group_id) for group_id in groups],
            )
            response = self._accept_reference_terms(response)
            soup = BeautifulSoup(response.text, "html.parser")
            if _form_values(soup, "proc[]"):
                return selected_neutrals, response

        # A single remaining process is auto-selected by LXCat, which skips
        # the process form and opens the asynchronous output page directly.
        if "/data/output.php" in response.url or "/cache/" in response.url:
            return selected_neutrals, response

        raise RuntimeError(
            f"Batch {batch.batch_id} did not reach the LXCat process-selection page "
            f"(last URL: {response.url})."
        )

    def _submit_processes(
        self, process_ids: list[str], *, retry_attempts: int = 6
    ) -> requests.Response:
        data = [("proc[]", process_id) for process_id in process_ids]
        response = self._post(
            "/data/set_processes.php",
            data=data,
            _retry_attempts=retry_attempts,
        )
        return self._accept_reference_terms(response)

    def _wait_for_output(self, response: requests.Response) -> requests.Response:
        deadline = time.monotonic() + 300.0
        current = response
        while "please wait" in current.text.lower():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"LXCat output generation timed out: {current.url}")
            time.sleep(1.0)
            current = self._get(current.url)
        return current

    def _batch_is_complete(self, batch: DownloadBatch, batch_dir: Path) -> bool:
        metadata_path = batch_dir / "batch_metadata.json"
        if not metadata_path.exists():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if metadata.get("status") not in {"complete", "complete_with_source_errors"}:
            return False
        if metadata.get("ion_ids") != list(batch.ion_ids):
            return False
        if metadata.get("expected_process_count") != batch.expected_process_count:
            return False
        if (
            int(metadata.get("actual_process_count", -1))
            + int(metadata.get("source_error_process_count", -1))
            != batch.expected_process_count
        ):
            return False
        if batch.neutral_ids is not None and metadata.get("neutral_ids") != list(
            batch.neutral_ids
        ):
            return False
        for record in metadata.get("files", []):
            path = batch_dir / record["name"]
            if not path.exists() or _sha256_file(path) != record["sha256"]:
                return False
        return True

    def _save_output_response(
        self,
        response: requests.Response,
        batch_dir: Path,
        *,
        prefix: str,
        used_names: set[str],
        files: list[dict],
    ) -> None:
        """Preserve one generated LXCat output page and all ASCII links."""
        response = self._wait_for_output(response)
        output_soup = BeautifulSoup(response.text, "html.parser")
        links = [
            anchor["href"]
            for anchor in output_soup.find_all("a", href=True)
            if anchor["href"].lower().endswith(".txt")
        ]
        if not links:
            raise RuntimeError(f"LXCat generated no ASCII links: {response.url}")

        index_name = f"{prefix}lxcat_output_index.html"
        index_path = batch_dir / index_name
        index_path.write_bytes(response.content)
        files.append(
            {
                "name": index_path.name,
                "source_url": response.url,
                "sha256": _sha256_bytes(response.content),
                "bytes": len(response.content),
            }
        )
        for index, href in enumerate(links, start=1):
            source_url = urljoin(response.url, href)
            data_response = self._get(source_url)
            original_name = Path(urlparse(href).path).name
            base_name = f"{prefix}{index:03d}_{_slug(original_name)}.txt"
            name = base_name
            suffix = 2
            while name in used_names:
                name = f"{Path(base_name).stem}_{suffix}.txt"
                suffix += 1
            used_names.add(name)
            path = batch_dir / name
            path.write_bytes(data_response.content)
            files.append(
                {
                    "name": name,
                    "original_name": original_name,
                    "source_url": source_url,
                    "sha256": _sha256_bytes(data_response.content),
                    "bytes": len(data_response.content),
                }
            )

    def download_batch(self, batch: DownloadBatch) -> dict:
        batch_dir = self.output_dir / "batches" / batch.batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        if self._batch_is_complete(batch, batch_dir):
            return json.loads((batch_dir / "batch_metadata.json").read_text(encoding="utf-8"))

        try:
            selected_neutrals, selection_response = self._start_selection(batch)
        except requests.HTTPError as exc:
            if batch.expected_process_count != 1:
                raise
            # For an auto-selected single process the failing HTTP request is
            # itself the export operation, so there is no process form to save.
            metadata = {
                "status": "complete_with_source_errors",
                "batch_id": batch.batch_id,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "ion_ids": list(batch.ion_ids),
                "neutral_ids": list(batch.neutral_ids or ()),
                "expected_process_count": 1,
                "actual_process_count": 0,
                "source_error_process_count": 1,
                "failed_processes": [
                    {
                        "process_id": "auto_selected_single_process",
                        "label": "",
                        "error": str(exc),
                    }
                ],
                "database_ids": list(ION_DATABASE_IDS),
                "database_names": [ION_DATABASE_NAMES[item] for item in ION_DATABASE_IDS],
                "output_urls": [],
                "split_process_export": False,
                "auto_selected_process": True,
                "files": [],
            }
            (batch_dir / "batch_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
            )
            print("  source export unavailable for auto-selected process; recorded", flush=True)
            return metadata

        process_soup = BeautifulSoup(selection_response.text, "html.parser")
        warning_text = " ".join(process_soup.stripped_strings)
        warning_match = re.search(
            r"(\d+) records have been found.*only first 1000", warning_text, flags=re.IGNORECASE
        )
        if warning_match:
            raise RuntimeError(
                f"Batch {batch.batch_id} exceeded the LXCat display cap with "
                f"{warning_match.group(1)} processes; split the plan more finely."
            )
        process_ids = _form_values(process_soup, "proc[]")
        auto_selected_process = not process_ids and (
            "/data/output.php" in selection_response.url
            or "/cache/" in selection_response.url
        )
        if not process_ids and not auto_selected_process:
            raise RuntimeError(f"Batch {batch.batch_id} yielded no processes.")

        files: list[dict] = []
        used_names: set[str] = set()
        output_urls: list[str] = []
        failed_processes: list[dict] = []
        process_labels = {
            checkbox["value"]: " ".join(checkbox.parent.stripped_strings)
            for checkbox in process_soup.select('input[name="proc[]"]')
        }
        split_export = False
        if auto_selected_process:
            process_ids = ["auto_selected_single_process"]
            self._save_output_response(
                selection_response,
                batch_dir,
                prefix="",
                used_names=used_names,
                files=files,
            )
            output_urls.append(selection_response.url)
        else:
            try:
                response = self._submit_processes(process_ids)
            except requests.HTTPError:
                split_export = True
                print(
                    f"  combined export failed for {batch.batch_id}; "
                    f"retrying {len(process_ids)} processes individually",
                    flush=True,
                )
                for part_index, process_id in enumerate(process_ids, start=1):
                    _, fresh_response = self._start_selection(batch)
                    fresh_process_soup = BeautifulSoup(fresh_response.text, "html.parser")
                    fresh_ids = _form_values(fresh_process_soup, "proc[]")
                    if process_id not in fresh_ids:
                        raise RuntimeError(
                            f"LXCat process ID {process_id} disappeared while splitting "
                            f"batch {batch.batch_id}."
                        )
                    try:
                        response = self._submit_processes(
                            [process_id], retry_attempts=1
                        )
                    except requests.HTTPError as exc:
                        failed_processes.append(
                            {
                                "process_id": process_id,
                                "label": process_labels.get(process_id, ""),
                                "error": str(exc),
                            }
                        )
                        print(
                            f"  source export unavailable for process {process_id}; recorded",
                            flush=True,
                        )
                        continue
                    self._save_output_response(
                        response,
                        batch_dir,
                        prefix=f"part_{part_index:04d}_",
                        used_names=used_names,
                        files=files,
                    )
                    output_urls.append(response.url)
            else:
                self._save_output_response(
                    response,
                    batch_dir,
                    prefix="",
                    used_names=used_names,
                    files=files,
                )
                output_urls.append(response.url)

        if failed_processes:
            selection_bytes = str(process_soup).encode("utf-8")
            selection_path = batch_dir / "process_selection_index.html"
            selection_path.write_bytes(selection_bytes)
            files.append(
                {
                    "name": selection_path.name,
                    "source_url": "LXCat process-selection page",
                    "sha256": _sha256_bytes(selection_bytes),
                    "bytes": len(selection_bytes),
                }
            )

        metadata = {
            "status": (
                "complete_with_source_errors" if failed_processes else "complete"
            ),
            "batch_id": batch.batch_id,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "ion_ids": list(batch.ion_ids),
            "neutral_ids": selected_neutrals,
            "expected_process_count": batch.expected_process_count,
            "actual_process_count": len(process_ids) - len(failed_processes),
            "source_error_process_count": len(failed_processes),
            "failed_processes": failed_processes,
            "database_ids": list(ION_DATABASE_IDS),
            "database_names": [ION_DATABASE_NAMES[item] for item in ION_DATABASE_IDS],
            "output_urls": output_urls,
            "split_process_export": split_export,
            "auto_selected_process": auto_selected_process,
            "files": files,
        }
        (batch_dir / "batch_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return metadata

    def run(self, *, max_batches: int | None = None) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pairs, ion_labels = self.fetch_catalog()
        plan = self.build_download_plan(pairs)
        catalog = {
            "catalog_url": urljoin(BASE_URL, "/contributors/"),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "database_ids": list(ION_DATABASE_IDS),
            "database_names": ION_DATABASE_NAMES,
            "ion_labels": ion_labels,
            "pair_count": len(pairs),
            "reported_process_count": sum(pair.reported_process_count for pair in pairs),
            "pairs": [pair.__dict__ for pair in pairs],
        }
        (self.output_dir / "catalog.json").write_text(
            json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.output_dir / "download_plan.json").write_text(
            json.dumps([batch.__dict__ for batch in plan], indent=2, sort_keys=True),
            encoding="utf-8",
        )

        selected_plan = plan if max_batches is None else plan[:max_batches]
        completed: list[dict] = []
        for index, batch in enumerate(selected_plan, start=1):
            print(
                f"[{index}/{len(selected_plan)}] {batch.batch_id}: "
                f"expected {batch.expected_process_count} processes",
                flush=True,
            )
            metadata = self.download_batch(batch)
            completed.append(metadata)
            print(
                f"  complete: {metadata['actual_process_count']} processes, "
                f"{len(metadata['files']) - 1} ASCII files",
                flush=True,
            )

        actual_process_count = sum(
            int(metadata["actual_process_count"]) for metadata in completed
        )
        source_error_process_count = sum(
            int(metadata.get("source_error_process_count", 0)) for metadata in completed
        )
        catalog_process_count = sum(pair.reported_process_count for pair in pairs)
        if (
            len(selected_plan) == len(plan)
            and actual_process_count + source_error_process_count != catalog_process_count
        ):
            raise RuntimeError(
                "Completed batch count does not match the LXCat contributor catalog: "
                f"downloaded {actual_process_count}, source errors "
                f"{source_error_process_count}, catalog reports {catalog_process_count}."
            )

        summary = {
            "status": (
                "partial"
                if len(selected_plan) != len(plan)
                else "complete_with_source_errors"
                if source_error_process_count
                else "complete"
            ),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "planned_batches": len(plan),
            "completed_batches_in_this_run": len(completed),
            "catalog_pair_count": len(pairs),
            "catalog_reported_process_count": sum(
                pair.reported_process_count for pair in pairs
            ),
            "downloaded_process_count": actual_process_count,
            "source_error_process_count": source_error_process_count,
        }
        (self.output_dir / "download_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = (
        Path(__file__).resolve().parents[1]
        / "ion_swarm_data"
        / "raw_lxcat"
        / f"lxcat_ion_swarm_{date.today().isoformat()}"
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Download only the first N deterministic batches (for smoke testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LXCatIonDownloader(args.output_dir, delay_seconds=args.delay).run(
        max_batches=args.max_batches
    )


if __name__ == "__main__":
    main()
