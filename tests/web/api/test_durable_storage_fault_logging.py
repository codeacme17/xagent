"""Every durable-storage fault must reach the log with its provider cause.

The envelopes these paths return are deliberately detail-free, and neither
FastAPI's ``HTTPException`` handler nor the ``/v1/*`` error handler logs a
traceback for the exception it translates -- so the log line is the *only*
record of what actually failed. ``ManagedFileRef`` wraps provider faults into
``DurableStorageOperationError`` carrying just the storage key, which means a
log line without ``exc_info`` leaves an operator unable to tell a throttle from
a timeout from rejected credentials. That is the gap that blocked an incident
investigation in #1467.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Iterator, cast

import pytest
from fastapi import HTTPException
from fastapi.datastructures import UploadFile

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.api import files as files_api
from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.api.v1.errors import V1ApiError
from xagent.web.models.user import User
from xagent.web.services.managed_file_ref import (
    DurableStorageOperationError,
    ManagedFileRef,
)

from .conftest import _direct_db_session, _setup_admin

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture()
def isolated_upload_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Path]:
    """Point staging and the object store at ``tmp_path``.

    Kept local rather than shared with ``test_upload_connection_boundary``:
    this suite's conftest deliberately exposes helpers by explicit import and
    keeps fixtures out of it, and no fixture-sharing module exists for the
    upload paths yet. The durable write is patched to fail before it reaches
    the object store, but the store is still redirected so a regression that
    lets the write through cannot touch a real backend.
    """
    upload_root = tmp_path / "uploads"
    object_root = tmp_path / "objects"
    upload_root.mkdir()
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", object_root.as_uri())
    get_unscoped_file_storage.cache_clear()
    monkeypatch.setattr(files_api, "get_uploads_dir", lambda: upload_root)
    try:
        yield upload_root
    finally:
        get_unscoped_file_storage.cache_clear()


def _stage_under(root: Path, user_id: int):
    """A deterministic staging-path chooser scoped to ``root``."""

    def get_upload_path(
        filename: str,
        task_id: str | None,
        folder: str | None,
        requested_user_id: int,
    ) -> Path:
        del task_id, folder
        assert requested_user_id == user_id
        path = root / f"user_{user_id}" / Path(filename).name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    return get_upload_path


_FILENAME = "quarterly-report.txt"
_STORAGE_KEY = f"users/7/uploads/8ac1f2/{_FILENAME}"
_PROVIDER_MESSAGE = "SlowDown: Please reduce your request rate (status 503)"
_UNAVAILABLE_DETAIL = "Durable storage is temporarily unavailable"


class _ProviderThrottled(RuntimeError):
    """Stand-in for the boto/S3 error class the wrap discards."""


def _wrapped_fault() -> DurableStorageOperationError:
    """Build a fault shaped exactly like ``ManagedFileRef``'s wraps.

    The message carries the storage key and nothing else; everything an
    operator needs to classify the failure lives in ``__cause__``. Assigning
    ``__cause__`` directly is what ``raise ... from exc`` does, and it survives
    a later bare ``raise`` of this object.
    """
    fault = DurableStorageOperationError(
        f"Failed to write durable object: {_STORAGE_KEY}"
    )
    fault.__cause__ = _ProviderThrottled(_PROVIDER_MESSAGE)
    return fault


def _rendered(record: logging.LogRecord) -> str:
    """Render a record the way a handler would, ``exc_info`` included."""
    return logging.Formatter("%(message)s").format(record)


def _warnings(caplog: pytest.LogCaptureFixture, logger_name: str) -> list[str]:
    return [
        _rendered(record)
        for record in caplog.records
        if record.name == logger_name and record.levelno == logging.WARNING
    ]


def _sole_warning(caplog: pytest.LogCaptureFixture, logger_name: str) -> str:
    """The one warning a direct helper call may emit."""
    rendered = _warnings(caplog, logger_name)
    assert len(rendered) == 1, f"expected exactly one warning, got {rendered}"
    return rendered[0]


def _warning_matching(
    caplog: pytest.LogCaptureFixture, logger_name: str, needle: str
) -> str:
    """Pick the fault line out of an endpoint's warnings.

    Not ``_sole_warning``: the request paths under test also run best-effort
    cleanup that logs to this same logger when it cannot remove a staged file
    (``_delete_staged_upload``), and an unrelated second warning must not read
    as a failure of the line this test is about.
    """
    matches = [line for line in _warnings(caplog, logger_name) if needle in line]
    assert len(matches) == 1, f"expected one warning matching {needle!r}: {matches}"
    return matches[0]


def _assert_cause_chain_recorded(rendered: str, *, wrap_message: bool = True) -> None:
    """The provider fault -- class and message -- must be in the log text.

    ``wrap_message=False`` for the delete path, whose exception was never
    wrapped by ``ManagedFileRef`` and so carries no storage key of its own.
    """
    assert _ProviderThrottled.__name__ in rendered
    assert _PROVIDER_MESSAGE in rendered
    if wrap_message:
        # The wrap's own message (and with it the storage key) is the anchor an
        # operator greps for; it must not be dropped in favour of the cause.
        assert _STORAGE_KEY in rendered


def test_durable_storage_unavailable_logs_cause_and_keeps_body_detail_free(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The shared 503 helper is where all nine file-API sites get their log.

    Asserting on the helper rather than on each endpoint is deliberate: the
    exception is a required positional argument and the helper is ``NoReturn``,
    so a call site can neither skip the cause nor log without raising. That
    leaves the helper's own body as the only place the chain can be dropped.
    """
    fault = _wrapped_fault()

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException) as raised:
            files_api._raise_durable_storage_unavailable(fault, "download")

    assert raised.value.status_code == 503
    # Scope segments in the key can encode end-user identity, so the body
    # stays the fixed message -- the detail is server-side only.
    assert raised.value.detail == _UNAVAILABLE_DETAIL
    assert _STORAGE_KEY not in str(raised.value.detail)
    assert _PROVIDER_MESSAGE not in str(raised.value.detail)
    assert raised.value.__cause__ is fault

    rendered = _sole_warning(caplog, files_api.logger.name)
    assert "Durable storage unavailable during download" in rendered
    _assert_cause_chain_recorded(rendered)


def test_durable_storage_unavailable_accepts_an_unwrapped_provider_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The delete path hands over the raw provider exception, not a wrap.

    ``delete_file`` catches ``Exception`` around the durable cleanup rather
    than a ``DurableStorageOperationError``, so the helper has to log something
    that was never wrapped by ``ManagedFileRef``. Before #1467 that site logged
    the storage key and discarded the exception entirely.
    """
    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException) as raised:
            files_api._raise_durable_storage_unavailable(
                _ProviderThrottled(_PROVIDER_MESSAGE),
                "durable cleanup before row delete",
                storage_key=_STORAGE_KEY,
            )

    assert raised.value.status_code == 503
    rendered = _sole_warning(caplog, files_api.logger.name)
    assert "durable cleanup before row delete" in rendered
    # The key is a named field, not part of the bounded operation label.
    assert f"storage_key={_STORAGE_KEY}" in rendered
    _assert_cause_chain_recorded(rendered, wrap_message=False)


@pytest.mark.asyncio
async def test_upload_durable_write_failure_logs_the_provider_cause(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    isolated_upload_storage: Path,
) -> None:
    """End to end on the incident path: a failed durable write, one 503, one log.

    ``store_uploaded_files`` backs ``/api/files/upload`` and both public-chat
    upload entry points (share and widget), so this covers every route where
    the reported 503 bursts were observed.
    """
    upload_root = isolated_upload_storage
    # Side effect only: lays down the admin row the upload is attributed to.
    _setup_admin()
    db = _direct_db_session()
    try:
        user_id = int(db.query(User.id).filter(User.username == "admin").scalar())
    finally:
        db.close()
    monkeypatch.setattr(
        files_api, "get_upload_path", _stage_under(upload_root, user_id)
    )

    def fail_sync(_self: ManagedFileRef, *_args: Any, **_kwargs: Any) -> None:
        raise _wrapped_fault()

    monkeypatch.setattr(ManagedFileRef, "sync_to_durable", fail_sync)
    upload = UploadFile(
        filename=_FILENAME,
        file=io.BytesIO(b"payload"),
        headers={"content-type": "text/plain"},
    )

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException) as raised:
            await files_api.store_uploaded_files(
                upload_items=[upload],
                task_type="general",
                task_id=None,
                folder=None,
                user_id=user_id,
                single_file_mode=True,
            )

    assert raised.value.status_code == 503
    assert raised.value.detail == _UNAVAILABLE_DETAIL
    # The cause chain still reaches the client-facing exception too.
    assert isinstance(raised.value.__cause__, DurableStorageOperationError)

    rendered = _warning_matching(
        caplog, files_api.logger.name, "Durable storage unavailable during upload"
    )
    _assert_cause_chain_recorded(rendered)


def test_v1_turn_attachment_durable_fault_logs_the_provider_cause(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The SDK path returns its own 503 envelope and needs its own log line."""

    def fail_resolve(**_kwargs: Any) -> None:
        raise _wrapped_fault()

    monkeypatch.setattr(v1_tasks, "resolve_turn_file_infos", fail_resolve)

    with caplog.at_level(logging.WARNING, logger=v1_tasks.logger.name):
        with pytest.raises(V1ApiError) as raised:
            v1_tasks._resolve_turn_files_or_400(
                file_ids=["8ac1f2"],
                owner_user_id=7,
                db=cast(Any, None),
                task_id=42,
            )

    assert raised.value.http_status == 503
    assert _STORAGE_KEY not in raised.value.message
    assert _PROVIDER_MESSAGE not in raised.value.message

    rendered = _warning_matching(
        caplog, v1_tasks.logger.name, "during turn attachment resolution"
    )
    assert "task_id=42" in rendered
    # The create path has task_id=None, so these carry identification there.
    assert "owner_user_id=7" in rendered
    assert "file_ids=8ac1f2" in rendered
    _assert_cause_chain_recorded(rendered)
