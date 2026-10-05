"""Resource snapshots and safe disposal for the claimed-upload lifecycle."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote, urlsplit
from uuid import uuid4

from ...config import (
    get_external_upload_dirs,
    get_file_materialize_dir,
    get_file_storage_options,
    get_storage_root,
    get_uploads_dir,
)
from ...core.file_storage import get_user_file_storage
from ...core.file_storage.storage import normalize_storage_key
from ...core.workspace import scoped_user_root
from .managed_file_ref import _checksum_to_sha256_hex


class CleanupResourceUncertain(RuntimeError):
    """A retained manifest needs retry or operator reconciliation."""


def _canonical_root(path: Path) -> Path:
    return path.expanduser().resolve()


def _locator(uri: str) -> str:
    """Persist routing identity without embedding authentication material."""
    parts = urlsplit(uri)
    return (
        parts._replace(netloc=parts.netloc.rsplit("@", 1)[-1], query="", fragment="")
        .geturl()
        .rstrip("/")
    )


def _provider_routing() -> dict[str, str]:
    options = get_file_storage_options()
    client = options.get("client_kwargs") or {}
    return {
        "endpoint": _locator(
            str(options.get("endpoint_url") or client.get("endpoint_url") or "")
        ),
        "region": str(client.get("region_name") or ""),
    }


def _identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns]


def _matches_resource(info: os.stat_result, expected: list[int] | None) -> bool:
    if expected is None:
        return False
    actual = _identity(info)
    return (
        actual[:3] == expected[:3] if stat.S_ISDIR(info.st_mode) else actual == expected
    )


@contextmanager
def _parent_descriptor(path: Path, root: Path) -> Iterator[int]:
    """Pin directories without following symlinks below the configured root."""
    relative = path.relative_to(root)
    if not relative.parts:
        raise CleanupResourceUncertain("Cleanup cannot remove a managed root")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def capture_resource(
    path: Path,
    *,
    root: Path,
    generation: str,
    checksum: str | None = None,
    allow_directory: bool = False,
) -> dict[str, Any]:
    """Capture a bounded owned locator before any destructive operation."""
    resource: dict[str, Any] = {
        "path": str(path),
        "root": str(root),
        "quarantine": f".{path.name}.cleanup-{generation}",
        "checksum": checksum,
        "identity": None,
        "parent": None,
    }
    try:
        with _parent_descriptor(path, root) as parent:
            resource["parent"] = _identity(os.fstat(parent))[:3]
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            resource["identity"] = _identity(info)
            if not stat.S_ISREG(info.st_mode) and not (
                allow_directory and stat.S_ISDIR(info.st_mode)
            ):
                resource["uncertain"] = "unsupported resource or symlink"
            elif stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                resource["preserved"] = "shared hard-linked source"
    except FileNotFoundError:
        pass
    except OSError as exc:
        if exc.errno not in {errno.ELOOP, errno.ENOTDIR}:
            raise
        resource["uncertain"] = "unverifiable path boundary"
    except ValueError:
        resource["uncertain"] = "unverifiable path boundary"
    return resource


def build_cleanup_manifest(snapshot: Any) -> dict[str, Any]:
    """Snapshot metadata, configured locators and local replacement evidence.

    The caller has returned its SQL connection. This performs only filesystem
    metadata reads; content verification belongs to execution after claim commit.
    """
    generation = uuid4().hex
    uploads = _canonical_root(get_uploads_dir())
    materialize = _canonical_root(get_file_materialize_dir())
    previews = _canonical_root(get_storage_root())
    checksum = _checksum_to_sha256_hex(snapshot.checksum or "")
    key = normalize_storage_key(snapshot.storage_key, strict=False)
    storage = get_user_file_storage(snapshot.user_id)
    expected_uri = f"{_locator(storage.base_uri)}/{quote(key, safe='/')}"
    manifest: dict[str, Any] = {
        "version": 1,
        "generation": generation,
        "file_id": snapshot.file_id,
        "user_id": snapshot.user_id,
        "storage_key": key,
        "checksum": snapshot.checksum,
        "etag": snapshot.etag,
        "storage_backend": snapshot.storage_backend or storage.backend,
        "storage_uri": _locator(snapshot.storage_uri)
        if snapshot.storage_uri
        else expected_uri,
        "base_uri": _locator(storage.base_uri),
        "routing": _provider_routing(),
        "roots": [str(uploads), str(materialize), str(previews)],
        "local": [],
        "previews": None,
        "done": [],
    }
    # User containment is necessary even for task-less rows; an external
    # allowlist permits reading a source, never destroying it.
    path = Path(os.path.abspath(Path(snapshot.storage_path).expanduser()))
    owner_root = scoped_user_root(uploads, snapshot.user_id)
    external = tuple(_canonical_root(p) for p in get_external_upload_dirs())
    if path.is_relative_to(owner_root) and not any(
        path.is_relative_to(root) for root in external
    ):
        manifest["local"].append(
            capture_resource(
                path, root=uploads, generation=generation, checksum=checksum
            )
        )
        if not checksum:
            manifest["local"][-1]["uncertain"] = (
                "managed source has no SHA-256 evidence"
            )
    else:
        manifest["preserved_source"] = str(path)
    if checksum:
        cached = (
            materialize
            / hashlib.sha256(key.encode()).hexdigest()[:16]
            / checksum
            / Path(snapshot.filename).name
        )
        if not any(resource["path"] == str(cached) for resource in manifest["local"]):
            manifest["local"].append(
                capture_resource(
                    cached, root=materialize, generation=generation, checksum=checksum
                )
            )
    else:
        manifest["uncertain_materialization"] = "materialization checksum unavailable"
    for resource in tuple(manifest["local"]):
        target = Path(resource["path"])
        prefix = (
            f".{target.name}."
            if Path(resource["root"]) == materialize
            else f".{hashlib.sha256(str(snapshot.file_id).encode()).hexdigest()[:24]}.{target.name}."
        )
        try:
            with _parent_descriptor(target, Path(resource["root"])) as parent:
                names = os.listdir(parent)
        except FileNotFoundError:
            continue
        except OSError as exc:
            if exc.errno not in {errno.ELOOP, errno.ENOTDIR}:
                raise
            continue
        for name in names:
            prefixes = (
                (prefix,)
                if Path(resource["root"]) == materialize
                else (prefix, "." + prefix)
            )
            if name.startswith(prefixes) and name.endswith(".tmp"):
                manifest["local"].append(
                    capture_resource(
                        target.parent / name,
                        root=Path(resource["root"]),
                        generation=generation,
                    )
                )
    return manifest


def validate_configuration(manifest: dict[str, Any]) -> None:
    current = [
        str(_canonical_root(get_uploads_dir())),
        str(_canonical_root(get_file_materialize_dir())),
        str(_canonical_root(get_storage_root())),
    ]
    storage = get_user_file_storage(manifest["user_id"])
    uri = f"{_locator(storage.base_uri)}/{quote(manifest['storage_key'], safe='/')}"
    if (
        current != manifest["roots"]
        or storage.backend != manifest["storage_backend"]
        or _locator(storage.base_uri) != manifest["base_uri"]
        or _provider_routing() != manifest["routing"]
        or uri != manifest["storage_uri"]
    ):
        raise CleanupResourceUncertain("Cleanup storage configuration changed")
    # Validate ownership with the same scope boundary used by storage consumers.
    storage._scoped(manifest["storage_key"], strict=False)


def dispose_resource(resource: dict[str, Any]) -> None:
    """Quarantine before deleting; an unexpected replacement is never unlinked.

    The quarantine locator is already durable. Exit after rename therefore
    leaves a retryable resource, including when the original path is absent.
    """
    if resource.get("preserved"):
        return
    if resource.get("uncertain"):
        raise CleanupResourceUncertain(resource["uncertain"])
    path, root = Path(resource["path"]), Path(resource["root"])
    try:
        with _parent_descriptor(path, root) as parent:
            if resource["parent"] is not None and (
                _identity(os.fstat(parent))[:3] != resource["parent"]
            ):
                raise CleanupResourceUncertain("Cleanup parent was replaced")
            quarantine = resource["quarantine"]
            try:
                info = os.stat(quarantine, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    return
                if not _matches_resource(info, resource["identity"]):
                    raise CleanupResourceUncertain("Cleanup source was replaced")
                if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                    raise CleanupResourceUncertain("Cleanup source became shared")
                os.rename(path.name, quarantine, src_dir_fd=parent, dst_dir_fd=parent)
                os.fsync(parent)
                info = os.stat(quarantine, dir_fd=parent, follow_symlinks=False)
            if not _matches_resource(info, resource["identity"]):
                raise CleanupResourceUncertain(
                    "Retained quarantine contains a replacement"
                )
            if stat.S_ISDIR(info.st_mode):
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise CleanupResourceUncertain(
                        "Directory disposal requires safe rmtree"
                    )
                shutil.rmtree(quarantine, dir_fd=parent)
            else:
                if resource.get("checksum"):
                    descriptor = os.open(
                        quarantine, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent
                    )
                    with os.fdopen(descriptor, "rb") as source:
                        digest = hashlib.file_digest(source, "sha256").hexdigest()
                        if digest != resource["checksum"]:
                            raise CleanupResourceUncertain(
                                "Managed copy checksum changed"
                            )
                    if (
                        _identity(
                            os.stat(quarantine, dir_fd=parent, follow_symlinks=False)
                        )
                        != resource["identity"]
                    ):
                        raise CleanupResourceUncertain(
                            "Quarantine changed during verification"
                        )
                os.unlink(quarantine, dir_fd=parent)
            os.fsync(parent)
            # A replacement published while quarantining is not our resource.
            try:
                os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return
            raise CleanupResourceUncertain("Cleanup path has a replacement")
    except FileNotFoundError:
        # A disappearing child inside rmtree is not evidence that its owned
        # quarantine disappeared. Verify both top-level locators before success.
        try:
            with _parent_descriptor(path, root) as parent:
                if resource["parent"] is not None and (
                    _identity(os.fstat(parent))[:3] != resource["parent"]
                ):
                    raise CleanupResourceUncertain("Cleanup parent was replaced")
                for name in (path.name, resource["quarantine"]):
                    try:
                        os.stat(name, dir_fd=parent, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    raise CleanupResourceUncertain(
                        "Resource remains after interrupted disposal"
                    )
        except FileNotFoundError:
            return


def capture_previews(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(manifest["roots"][2])
    file_id = manifest["file_id"]
    if Path(file_id).name != file_id or file_id in {".", ".."}:
        raise CleanupResourceUncertain("Invalid preview owner identity")
    resources = []
    for name in ("pptx_pdf_cache", "svg_png_cache"):
        directory = root / name
        try:
            with _parent_descriptor(directory / "entry", root) as parent:
                names = os.listdir(parent)
        except FileNotFoundError:
            continue
        for entry in names:
            owned = (
                entry == f"{file_id}.preview.pdf"
                or (
                    entry.startswith(f"{file_id}.")
                    and ".preview.png" in entry
                    and (entry.endswith(".preview.png") or entry.endswith(".tmp"))
                )
                or (
                    entry.startswith(f"{file_id}.preview.pdf.")
                    and entry.endswith(".tmp")
                )
                or entry.startswith(f".{file_id}.preview-")
            )
            if owned:
                resources.append(
                    capture_resource(
                        directory / entry,
                        root=root,
                        generation=manifest["generation"],
                        allow_directory=entry.startswith(f".{file_id}.preview-"),
                    )
                )
    return resources
