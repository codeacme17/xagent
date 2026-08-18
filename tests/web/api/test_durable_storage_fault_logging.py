"""Every durable-storage fault must reach the log with its provider cause.

The envelopes these paths return are deliberately detail-free, and neither
FastAPI's ``HTTPException`` handler nor the ``/v1/*`` error handler logs a
traceback for the exception it translates -- so the log line is the *only*
record of what actually failed. ``ManagedFileRef`` wraps provider faults into
``DurableStorageOperationError`` carrying just the storage key, which means a
log line without ``exc_info`` leaves an operator unable to tell a throttle from
a timeout from rejected credentials. That is the gap that blocked an incident
investigation in #1467.

The same helper also renders the *classified* provider fault as ``key=value``
fields, so a burst of these logs can be aggregated by cause rather than grepped
as text. Classification itself is unit-tested in
``tests/core/test_storage_provider_faults.py``; what is pinned here is that the
fields reach the request paths' log lines, and that ``retryable=False`` in them
does not change the 503 the client receives.
"""

from __future__ import annotations

import ast
import errno
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

# Not module-wide: only the end-to-end upload test touches the database. The
# other tests drive the helper directly and would pay a schema create/drop for
# nothing.


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


_EIO = errno.EIO
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


def _s3_throttle_fault() -> DurableStorageOperationError:
    """The incident's real chain: wrap -> s3fs OSError -> botocore ClientError.

    s3fs maps unrecognized codes onto ``OSError(EIO, ...)`` and hangs the
    original ``ClientError`` off ``__cause__``, so the provider code sits two
    links below the wrap. Built by hand rather than with botocore so the test
    does not depend on an optional dependency being importable.
    """

    class ClientError(Exception):
        def __init__(self) -> None:
            super().__init__("An error occurred (SlowDown)")
            self.response = {
                "Error": {"Code": "SlowDown", "Message": _PROVIDER_MESSAGE},
                "ResponseMetadata": {"HTTPStatusCode": 503},
            }

    translated = OSError(_EIO, _PROVIDER_MESSAGE)
    translated.__cause__ = ClientError()
    fault = DurableStorageOperationError(
        f"Failed to write durable object: {_STORAGE_KEY}"
    )
    fault.__cause__ = translated
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
@pytest.mark.usefixtures("_test_db")
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


# Every ``_raise_durable_storage_unavailable`` call site in files.py, with the
# fields it is expected to carry. N2 in review -- the signed-redirect site
# shipping with no identifier -- was invisible because only two sites had
# assertions; this sweep is what makes the set itself the contract.
_FAULT_SITES = (
    ("signed durable redirect", ("file_id",)),
    ("upload", ()),
    ("download", ("file_id",)),
    ("preview", ("file_id",)),
    ("pptx preview", ("file_id",)),
    ("public download", ("file_id",)),
    ("public preview", ("file_id",)),
    ("public preview task asset", ("file_id",)),
    ("durable cleanup before row delete", ("file_id", "storage_key")),
)


def test_every_fault_site_label_is_bounded_and_reaches_the_log() -> None:
    """The nine labels are a closed set of bounded, aggregatable values.

    ``upload`` carries no identifier by design: it is a batch-registration path
    with no single file_id. Every other site identifies its subject.
    """
    tree = ast.parse(Path(files_api.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_raise_durable_storage_unavailable"
    ]
    # Parsed rather than grepped: a substring count also matches the ``def``
    # line, and a label is only a contract if the count is exact.
    assert len(calls) == len(_FAULT_SITES), (
        f"{len(calls)} call sites but {len(_FAULT_SITES)} labels declared -- "
        "add the new site to _FAULT_SITES with the fields it should carry"
    )

    declared = {label for label, _ in _FAULT_SITES}
    passed = {
        node.args[1].value
        for node in calls
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
    }
    # Every label is a plain literal, so it stays a bounded, aggregatable value
    # -- an f-string here is how the storage key first became part of a label.
    assert passed == declared, f"labels drifted: {passed ^ declared}"

    for node in calls:
        expected = dict(_FAULT_SITES)[node.args[1].value]
        assert tuple(kw.arg for kw in node.keywords) == expected, (
            f"site {node.args[1].value!r} passes "
            f"{[kw.arg for kw in node.keywords]}, expected {list(expected)}"
        )


@pytest.mark.parametrize(("label", "expected_fields"), _FAULT_SITES)
def test_fault_site_renders_its_label_and_fields(
    caplog: pytest.LogCaptureFixture,
    label: str,
    expected_fields: tuple[str, ...],
) -> None:
    """Each site's label and field set must survive into the log line."""
    fields = {name: f"value-for-{name}" for name in expected_fields}

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException) as raised:
            files_api._raise_durable_storage_unavailable(
                _wrapped_fault(), label, **fields
            )

    assert raised.value.status_code == 503
    rendered = _sole_warning(caplog, files_api.logger.name)
    assert f"during {label}" in rendered
    for name in expected_fields:
        assert f"{name}=value-for-{name}" in rendered
    _assert_cause_chain_recorded(rendered)


def test_field_values_cannot_forge_a_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A newline in a client-supplied field must not start a new record.

    ``/v1/*`` renders the request's ``files`` list, an unvalidated ``list[str]``,
    so a caller cannot be relied on to have checked (CWE-117).
    """
    forged = "ok,\n2026-01-01 00:00:00 ERROR    xagent.web - FABRICATED entry"

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException):
            files_api._raise_durable_storage_unavailable(
                _wrapped_fault(), "upload", file_ids=forged
            )

    rendered = _sole_warning(caplog, files_api.logger.name)
    message_line = rendered.splitlines()[0]
    assert "FABRICATED" in message_line, "the value must survive, escaped"
    assert "\\n" in message_line
    assert not any(line.startswith("2026-01-01") for line in rendered.splitlines()), (
        "the injected text must not stand as its own record"
    )


def test_an_overlong_field_value_is_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One field must not be able to crowd out the rest of the line."""
    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException):
            files_api._raise_durable_storage_unavailable(
                _wrapped_fault(), "upload", file_ids="x" * 5000
            )

    rendered = _sole_warning(caplog, files_api.logger.name)
    assert "...[truncated]" in rendered
    assert len(rendered.splitlines()[0]) < 1000


@pytest.mark.asyncio
@pytest.mark.usefixtures("_test_db")
async def test_upload_log_carries_the_classified_provider_fault(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    isolated_upload_storage: Path,
) -> None:
    """The fields an operator aggregates on must reach the upload log line.

    This is the burst from #1467: intermittent 503s whose cause could not be
    named. With the chain classified, one query groups them by
    ``provider_code`` instead of eyeballing tracebacks.
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
        raise _s3_throttle_fault()

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
    rendered = _warning_matching(
        caplog, files_api.logger.name, "Durable storage unavailable during upload"
    )
    assert "provider_code=SlowDown" in rendered
    assert "provider_http_status=503" in rendered
    assert "retryable=True" in rendered
    # The traceback still accompanies the fields; they add to it, not replace it.
    assert _PROVIDER_MESSAGE in rendered
    assert _STORAGE_KEY in rendered


def test_a_permanent_fault_is_labelled_but_still_answered_503(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Classification is diagnostic only -- it must not reroute the status.

    Mapping permanent causes onto a non-retryable status is a deliberate
    follow-up: it would change the SDK retry contract and the widget error path,
    so this change stops at naming the cause.
    """

    class ClientError(Exception):
        def __init__(self) -> None:
            super().__init__("An error occurred (InvalidAccessKeyId)")
            self.response = {
                "Error": {"Code": "InvalidAccessKeyId"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            }

    fault = DurableStorageOperationError(
        f"Failed to write durable object: {_STORAGE_KEY}"
    )
    fault.__cause__ = ClientError()

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException) as raised:
            files_api._raise_durable_storage_unavailable(fault, "upload")

    assert raised.value.status_code == 503
    assert raised.value.detail == _UNAVAILABLE_DETAIL
    rendered = _sole_warning(caplog, files_api.logger.name)
    assert "provider_code=InvalidAccessKeyId" in rendered
    assert "retryable=False" in rendered
