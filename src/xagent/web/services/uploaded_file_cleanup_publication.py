"""Prevent late local/preview publication across claimed-upload cleanup."""

from __future__ import annotations

import asyncio
import copy
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace
from typing import Any, Callable, Iterator, TypeVar, cast

from filelock import Timeout
from sqlalchemy.engine import Connection
from sqlalchemy.orm import object_session, sessionmaker
from sqlalchemy.orm.exc import UnmappedInstanceError

from ...core.tools.core.RAG_tools.storage.file_reference import (
    _drain,
    file_cleanup_lock,
    has_cleanup_fence,
)
from ..models.database import get_optional_session_local, release_db_connection_if_clean
from ..models.uploaded_file import UploadedFile
from ..models.uploaded_file_cleanup_fence import UploadedFileCleanupFence

_Operation = TypeVar("_Operation", bound=Callable[..., Any])
_COPY_FIELDS = (
    "id",
    "user_id",
    "file_id",
    "filename",
    "storage_path",
    "storage_key",
    "storage_status",
    "checksum",
    "task_id",
    "etag",
    "storage_backend",
    "storage_uri",
)


def _validate_publication(file_id: str, sessions: Any, snapshot: Any = None) -> None:
    from .managed_file_ref import DurableObjectMissingError

    if sessions is None:
        if has_cleanup_fence(file_id):
            raise DurableObjectMissingError("File is unavailable")
        return
    with sessions() as db:
        current = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).first()
        if current is None:
            retired = db.get(UploadedFileCleanupFence, file_id) is not None
            if (
                retired
                or (snapshot is not None and snapshot.id is not None)
                or has_cleanup_fence(file_id)
            ):
                raise DurableObjectMissingError("File is unavailable")
            return
        if current.storage_status not in {"available", "legacy"}:
            raise DurableObjectMissingError("File is unavailable")
        if snapshot is not None and any(
            getattr(current, field) != getattr(snapshot, field)
            for field in (
                "id",
                "user_id",
                "storage_path",
                "storage_key",
                "checksum",
                "etag",
            )
            if getattr(snapshot, field) is not None
        ):
            raise DurableObjectMissingError("File generation is unavailable")


def guard_managed_copy_publication(operation: _Operation) -> _Operation:
    @wraps(operation)
    def publish(self: Any, *args: Any, **kwargs: Any) -> Any:
        snapshot = SimpleNamespace(
            **{field: getattr(self.record, field, None) for field in _COPY_FIELDS}
        )
        try:
            db = object_session(self.record)
        except UnmappedInstanceError:
            db = None
        sessions = get_optional_session_local()
        if db is not None:
            bind = db.get_bind()
            if isinstance(bind, Connection):
                bind = bind.engine
            sessions = sessionmaker(bind=bind)
            if not release_db_connection_if_clean(db):
                # Existing bytes require no publication or transaction change.
                if (
                    kwargs.get("allow_existing_local", True)
                    and self.local_path.is_file()
                ):
                    return self.local_path
                raise RuntimeError("Local publication requires a clean transaction")
        clone = copy.copy(self)
        clone.record = snapshot
        try:
            with file_cleanup_lock(str(snapshot.file_id)):
                _validate_publication(str(snapshot.file_id), sessions, snapshot)
                return operation(clone, *args, **kwargs)
        except Timeout as exc:
            from .managed_file_ref import DurableStorageOperationError

            raise DurableStorageOperationError(
                "File publication is temporarily unavailable",
                storage_key=snapshot.storage_key,
            ) from exc

    return cast(_Operation, publish)


@contextmanager
def _preview_guard(file_id: str | None) -> Iterator[None]:
    if not file_id:
        yield
        return
    with file_cleanup_lock(file_id):
        _validate_publication(file_id, get_optional_session_local())
        yield


def guard_preview_publication(operation: _Operation) -> _Operation:
    if asyncio.iscoroutinefunction(operation):

        @wraps(operation)
        async def asynchronous(path: Any, file_id: str | None = None) -> Any:
            async def publish() -> Any:
                guard = _preview_guard(file_id)
                await asyncio.to_thread(guard.__enter__)
                try:
                    return await operation(path, file_id)
                finally:
                    await asyncio.to_thread(guard.__exit__, None, None, None)

            try:
                return await _drain(publish())
            except Timeout:
                # The async converter already uses None for retryable failure.
                return None

        return cast(_Operation, asynchronous)

    @wraps(operation)
    def synchronous(path: Any, file_id: str | None = None) -> Any:
        try:
            with _preview_guard(file_id):
                return operation(path, file_id)
        except Timeout as exc:
            from .managed_file_ref import DurableStorageOperationError

            raise DurableStorageOperationError(
                "File preview is temporarily unavailable", storage_key=None
            ) from exc

    return cast(_Operation, synchronous)
