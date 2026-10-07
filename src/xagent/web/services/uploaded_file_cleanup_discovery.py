"""Bounded directory discovery for one retained cleanup manifest.

Open streams are only a process-local optimization, never durable directory
cursors. A restart or another worker starts a new complete scan while retaining
all captured obligations. A stream page is replayed until its SQL revision is
acknowledged; an uncommitted page cannot silently advance discovery.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import os
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.file_storage.storage import (
    atomic_copy_temp_prefix,
    materialized_key_directory,
)
from .uploaded_file_cleanup_resources import (
    CleanupResourceUncertain,
    _identity,
    _parent_descriptor,
    capture_resource,
    dispose_resource,
)

DISCOVERY_ENTRIES_PER_RUN = 256
DISPOSALS_PER_RUN = 128
MAX_OPEN_DISCOVERY_STREAMS = 8
DISCOVERY_STREAM_IDLE_SECONDS = 3600.0
MAX_DISPOSAL_DEPTH = 64


class CleanupBudgetExhausted(RuntimeError):
    """The retained claim has more bounded work to do on a later activation."""


@dataclass
class CleanupWorkBudget:
    entries: int = DISCOVERY_ENTRIES_PER_RUN
    disposals: int = DISPOSALS_PER_RUN

    def entry(self) -> None:
        if self.entries <= 0:
            raise CleanupBudgetExhausted()
        self.entries -= 1

    def disposal(self) -> None:
        if self.disposals <= 0:
            raise CleanupBudgetExhausted()
        self.disposals -= 1


def _temp_job(resource: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    prefixes: tuple[str, ...]
    target = Path(resource["path"])
    if resource["root"] == manifest["roots"][1]:
        prefixes = (f".{target.name}.", atomic_copy_temp_prefix(target.name))
    else:
        legacy = f".{hashlib.sha256(manifest['file_id'].encode()).hexdigest()[:24]}.{target.name}."
        current = atomic_copy_temp_prefix(target.name, owner=manifest["file_id"])
        prefixes = (legacy, "." + legacy, current, "." + current)
    return {
        "kind": "temporary",
        "path": str(target.parent),
        "root": resource["root"],
        "prefixes": list(dict.fromkeys(prefixes)),
    }


def prepare_discovery(manifest: dict[str, Any], filename: str) -> None:
    """Add resumable obligations without changing previously captured evidence."""
    if "discovery" in manifest:
        return
    key_directory = materialized_key_directory(
        Path(manifest["roots"][1]), manifest["storage_key"]
    )
    manifest["discovery"] = {
        "local": {
            "position": 0,
            "revision": 0,
            "complete": "local" in manifest["done"],
            "jobs": [
                {
                    "kind": "materialized",
                    "path": str(key_directory),
                    "root": manifest["roots"][1],
                    "filename": Path(filename).name,
                }
            ]
            + [_temp_job(resource, manifest) for resource in manifest["local"]],
        },
        "previews": {
            "position": 0,
            "revision": 0,
            "complete": "previews" in manifest["done"],
            "jobs": [
                {
                    "kind": "preview",
                    "path": str(Path(manifest["roots"][2]) / name),
                    "root": manifest["roots"][2],
                }
                for name in ("pptx_pdf_cache", "svg_png_cache")
            ],
        },
    }
    manifest["discovery"]["preview_check"] = copy.deepcopy(
        manifest["discovery"]["previews"]
    )
    manifest["preview_check"] = None
    manifest["version"] = 2


class _Stream:
    def __init__(self, job: dict[str, Any], revision: int):
        self.stack = ExitStack()
        try:
            self.parent = self.stack.enter_context(
                _parent_descriptor(Path(job["path"]) / "entry", Path(job["root"]))
            )
            self.identity = _identity(os.fstat(self.parent))
            if job.get("identity") is not None and job["identity"] != self.identity[:3]:
                raise CleanupResourceUncertain("Discovery directory was replaced")
            job["identity"] = self.identity[:3]
            self.entries = os.scandir(self.parent)
            self.stack.callback(self.entries.close)
        except BaseException:
            self.stack.close()
            raise
        self.revision = revision
        self.pending: tuple[list[dict[str, Any]], list[dict[str, Any]], bool] | None = (
            None
        )
        self.job = copy.deepcopy(job)
        self.touched = time.monotonic()

    def validate(self, job: dict[str, Any]) -> None:
        with _parent_descriptor(
            Path(job["path"]) / "entry", Path(job["root"])
        ) as current:
            # Mutations can invalidate readdir ordering. Restart instead of
            # treating an EOF on a changed directory as a completion proof.
            identity = _identity(os.fstat(current))
            if identity[:3] != self.identity[:3]:
                raise CleanupResourceUncertain("Discovery directory was replaced")
            if identity != self.identity:
                raise CleanupBudgetExhausted()

    def close(self) -> None:
        self.stack.close()


_streams: dict[tuple[str, str], _Stream] = {}
_stream_lock = threading.Lock()


def close_discovery_streams(generation: str | None = None) -> None:
    """Release optimizations; committed manifests remain sufficient for restart."""
    with _stream_lock:
        for key in tuple(_streams):
            if generation is None or key[0] == generation:
                _streams.pop(key).close()


atexit.register(close_discovery_streams)


def _capture_entry(
    entry: os.DirEntry[str], job: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = Path(job["path"]) / entry.name
    kind = job["kind"]
    checksum = None
    allow_directory = False
    if kind == "materialized":
        if entry.is_symlink():
            raise CleanupResourceUncertain("materialization namespace has a symlink")
        if not entry.is_dir(follow_symlinks=False):
            return None, None
        path /= job["filename"]
        from .managed_file_ref import _checksum_to_sha256_hex

        checksum = _checksum_to_sha256_hex(manifest["checksum"] or "")
    elif kind == "temporary":
        if not entry.name.startswith(tuple(job["prefixes"])) or not entry.name.endswith(
            ".tmp"
        ):
            return None, None
    else:
        file_id = manifest["file_id"]
        if Path(file_id).name != file_id or file_id in {".", ".."}:
            raise CleanupResourceUncertain("Invalid preview owner identity")
        allow_directory = entry.name.startswith(f".{file_id}.preview-")
        owned = (
            entry.name == f"{file_id}.preview.pdf"
            or (
                entry.name.startswith(f"{file_id}.")
                and ".preview.png" in entry.name
                and (entry.name.endswith(".preview.png") or entry.name.endswith(".tmp"))
            )
            or (
                entry.name.startswith(f"{file_id}.preview.pdf.")
                and entry.name.endswith(".tmp")
            )
            or allow_directory
        )
        if not owned:
            return None, None
    resource = capture_resource(
        path,
        root=Path(job["root"]),
        generation=manifest["generation"],
        checksum=checksum,
        allow_directory=allow_directory,
    )
    return resource, _temp_job(resource, manifest) if kind == "materialized" else None


def advance_discovery(
    manifest: dict[str, Any], phase: str, budget: CleanupWorkBudget
) -> None:
    """Capture at most the remaining entry budget, retaining a page for SQL retry."""
    state = manifest["discovery"][phase]
    key = manifest["generation"], phase
    with _stream_lock:
        for old_key, old_stream in tuple(_streams.items()):
            if time.monotonic() - old_stream.touched >= DISCOVERY_STREAM_IDLE_SECONDS:
                _streams.pop(old_key).close()
        while not state["complete"]:
            budget.entry()
            position, revision = state["position"], state["revision"]
            job = state["jobs"][position]
            stream = _streams.get(key)
            if stream is not None and (
                stream.job != job
                or revision not in {stream.revision, stream.revision + 1}
            ):
                _streams.pop(key).close()
                stream = None
            if stream is None:
                if len(_streams) >= MAX_OPEN_DISCOVERY_STREAMS:
                    raise CleanupBudgetExhausted()
                try:
                    stream = _Stream(job, revision)
                except FileNotFoundError:
                    state["position"] += 1
                    state["revision"] += 1
                    state["complete"] = state["position"] == len(state["jobs"])
                    continue
                _streams[key] = stream
            stream.touched = time.monotonic()
            try:
                stream.validate(job)
                if revision == stream.revision + 1:
                    stream.revision = revision
                    stream.pending = None
                if stream.pending is None:
                    found: list[dict[str, Any]] = []
                    jobs: list[dict[str, Any]] = []
                    exhausted = False
                    while budget.entries > 0:
                        budget.entry()
                        try:
                            entry = next(stream.entries)
                        except StopIteration:
                            exhausted = True
                            break
                        resource, child_job = _capture_entry(entry, job, manifest)
                        if resource is not None:
                            found.append(resource)
                        if child_job is not None:
                            jobs.append(child_job)
                    stream.validate(job)
                    stream.pending = found, jobs, exhausted
                found, jobs, exhausted = stream.pending
                destination = manifest[phase]
                if destination is None:
                    destination = manifest[phase] = []
                existing = {resource["path"]: resource for resource in destination}
                for resource in found:
                    prior = existing.get(resource["path"])
                    if prior is None:
                        destination.append(copy.deepcopy(resource))
                        existing[resource["path"]] = resource
                    elif resource["identity"] is not None and (
                        prior["identity"] != resource["identity"]
                        or prior["parent"] != resource["parent"]
                    ):
                        raise CleanupResourceUncertain(
                            "Discovered resource was replaced"
                        )
                for child_job in jobs:
                    if child_job not in state["jobs"]:
                        state["jobs"].append(copy.deepcopy(child_job))
                state["revision"] += 1
                if exhausted:
                    state["position"] += 1
                    state["complete"] = state["position"] == len(state["jobs"])
                    # Keep EOF replayable until the next persisted position is seen.
                if budget.entries <= 0 and not state["complete"]:
                    return
            except BaseException:
                _streams.pop(key).close()
                raise
        if key in _streams:
            _streams.pop(key).close()


def dispose_directory(
    parent: int, name: str, budget: CleanupWorkBudget, depth: int = 0
) -> None:
    """Drain an owned quarantine without an unbounded recursive enumeration."""
    if depth >= MAX_DISPOSAL_DEPTH:
        raise CleanupResourceUncertain(
            "Converter directory exceeds cleanup depth limit"
        )
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    descriptor = os.open(
        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
    )
    try:
        if _identity(os.fstat(descriptor))[:3] != _identity(before)[:3]:
            raise CleanupResourceUncertain("Converter directory was replaced")
        while True:
            budget.entry()
            # Close the stream before changing its directory. Deleted entries
            # provide restart progress; directory offsets never authorize it.
            with os.scandir(descriptor) as entries:
                entry = next(entries, None)
            if entry is None:
                break
            if entry.is_dir(follow_symlinks=False):
                dispose_directory(descriptor, entry.name, budget, depth + 1)
            else:
                budget.disposal()
                os.unlink(entry.name, dir_fd=descriptor)
        if (
            _identity(os.stat(name, dir_fd=parent, follow_symlinks=False))[:3]
            != _identity(before)[:3]
        ):
            raise CleanupResourceUncertain("Converter directory was replaced")
        budget.disposal()
        os.rmdir(name, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(descriptor)


def dispose_resource_page(
    manifest: dict[str, Any], phase: str, budget: CleanupWorkBudget
) -> None:
    """Checkpoint a bounded prefix, without revisiting every disposed resource."""
    positions = manifest.setdefault("disposal_positions", {})
    position = positions.get(phase, 0)
    resources = manifest[phase] or []
    while position < len(resources):
        budget.disposal()
        resource = resources[position]
        if not resource.get("disposed"):
            dispose_resource(resource, budget=budget)
            resource["disposed"] = True
        position += 1
        positions[phase] = position


def validate_local_evidence(manifest: dict[str, Any]) -> None:
    """Known uncertainty must not reserve a discovery stream indefinitely."""
    if "local" in manifest["done"]:
        return
    uncertainty = manifest.get("uncertain_materialization") or next(
        (
            resource["uncertain"]
            for resource in manifest["local"]
            if resource.get("uncertain")
        ),
        None,
    )
    if uncertainty:
        raise CleanupResourceUncertain(uncertainty)
