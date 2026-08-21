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

import ast
import io
import logging
from pathlib import Path
from typing import Any, Iterator, cast

import pytest
from fastapi import HTTPException
from fastapi.datastructures import UploadFile

from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.web.api import files as files_api
from xagent.web.api import websocket as websocket_api
from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.api.v1.errors import V1ApiError
from xagent.web.models.user import User
from xagent.web.services.managed_file_ref import (
    _MAX_LOG_VALUE_LENGTH,
    DurableStorageOperationError,
    ManagedFileRef,
    log_durable_storage_fault,
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


_FILENAME = "quarterly-report.txt"
_STORAGE_KEY = f"users/7/uploads/8ac1f2/{_FILENAME}"
_PROVIDER_MESSAGE = "SlowDown: Please reduce your request rate (status 503)"
_UNAVAILABLE_DETAIL = "Durable storage is temporarily unavailable"


class _ProviderThrottled(RuntimeError):
    """Stand-in for the boto/S3 error class the wrap discards."""


def _wrapped_fault() -> DurableStorageOperationError:
    """Build a fault shaped exactly like ``ManagedFileRef``'s wraps.

    The key rides on ``storage_key``, not in the message, because ``str(exc)``
    escapes to places the raise site does not control. Keeping this replica in
    the production shape is the point: with the key in the message it would test
    a shape nothing raises, and the log assertions would pass for the wrong
    reason -- from the message text rather than from the rendered field.

    Everything an operator needs to classify the failure lives in ``__cause__``.
    Assigning it directly is what ``raise ... from exc`` does, and it survives a
    later bare ``raise`` of this object.
    """
    fault = DurableStorageOperationError(
        "Failed to write durable object", storage_key=_STORAGE_KEY
    )
    fault.__cause__ = _ProviderThrottled(_PROVIDER_MESSAGE)
    return fault


def test_the_wrap_keeps_the_storage_key_out_of_its_own_message() -> None:
    """``str(exc)`` is the value that escapes; it must not carry the key.

    A bare ``raise`` from a WebSocket fault arm carries this exception into a
    task-wide broadcast and a persisted command row, and broad
    ``except RuntimeError`` arms interpolate it into client-facing text. Those
    egresses are pre-existing code that no arm in this PR can reach, so the
    invariant has to hold at the exception rather than at each of them.
    """
    fault = _wrapped_fault()

    assert _STORAGE_KEY not in str(fault)
    assert "users/" not in str(fault)
    assert fault.storage_key == _STORAGE_KEY

    # Every real wrap site, not just this replica.
    for path in (
        Path(files_api.__file__).parent.parent / "services" / "managed_file_ref.py",
    ):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id
                in {"DurableStorageOperationError", "DurableObjectIntegrityError"}
            ):
                continue
            message = node.args[0] if node.args else None
            assert not isinstance(message, ast.JoinedStr), (
                f"{path.name}:{node.lineno} interpolates into the message; the "
                "identifier belongs in storage_key= so str(exc) stays safe"
            )


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


def _assert_cause_chain_recorded(rendered: str, *, wrap_key: bool = True) -> None:
    """The provider fault -- class and message -- must be in the log text.

    ``wrap_key=False`` where the wrap's own storage key should not appear: the
    delete path hands over a raw provider exception that has none, and a caller
    passing an explicit ``storage_key`` field deliberately overrides it.
    """
    assert _ProviderThrottled.__name__ in rendered
    assert _PROVIDER_MESSAGE in rendered
    if wrap_key:
        # The key is the anchor an operator greps for. It is no longer in the
        # message -- ``str(exc)`` escapes to clients -- so this asserts the
        # helper renders it from the exception instead of losing it.
        assert f"storage_key={_STORAGE_KEY}" in rendered


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
    _assert_cause_chain_recorded(rendered, wrap_key=False)


def test_one_wrap_yields_one_record_however_many_arms_report_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A fault crossing several handlers is recorded once.

    Two arms legitimately report the same fault -- the request-scoped one that
    answers the client, and the endpoint-scoped one that catches whatever
    escaped -- and neither can know whether the other ran. So the wrap carries
    the fact that it has been logged. Without this, sustained-outage logs
    duplicate every fault, and the duplicate looks like a second failure.
    """
    fault = _wrapped_fault()

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        log_durable_storage_fault(
            files_api.logger, "websocket chat turn preparation", fault, task_id=42
        )
        log_durable_storage_fault(
            files_api.logger, "websocket chat turn", fault, task_id=42
        )

    # The first arm to report wins, which is the innermost -- the one that knows
    # what it was doing. The coarser endpoint label is the one dropped.
    rendered = _sole_warning(caplog, files_api.logger.name)
    assert "during websocket chat turn preparation" in rendered
    _assert_cause_chain_recorded(rendered)


def test_an_unwrapped_provider_error_is_not_marked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dedup is scoped to this module's wraps, and only they carry the mark.

    Setting a private attribute on a foreign exception is not safe in principle
    -- it may reject the write, and reading it back is not guaranteed either --
    and it buys nothing: an unwrapped provider error reaches exactly one
    reporting site, so there is nothing to deduplicate.
    """
    raw = _ProviderThrottled(_PROVIDER_MESSAGE)

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        log_durable_storage_fault(files_api.logger, "download", raw)
        log_durable_storage_fault(files_api.logger, "preview", raw)

    assert len(_warnings(caplog, files_api.logger.name)) == 2
    assert not hasattr(raw, "_durable_fault_logged")


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


def test_v1_turn_attachment_integrity_fault_is_not_reported_as_an_outage(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The integrity subclass through the real cascade, not just the parent.

    The test above injects only ``DurableStorageOperationError``, so it would
    pass with the two arms swapped -- the parent would catch the subclass and
    this site would report permanent corruption as a retryable outage. That is
    the safety property the ordering exists for, and injecting the subclass is
    what actually exercises it.
    """
    from xagent.web.services.managed_file_ref import (
        FILE_INTEGRITY_REUPLOAD_MESSAGE,
        DurableObjectIntegrityError,
    )

    def fail_resolve(**_kwargs: Any) -> None:
        raise DurableObjectIntegrityError(FILE_INTEGRITY_REUPLOAD_MESSAGE)

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
    # The envelope is deliberately unchanged; what must not happen is a second,
    # contradicting record calling permanent corruption a transient outage.
    assert not [
        line
        for line in _warnings(caplog, v1_tasks.logger.name)
        if "Durable storage unavailable" in line
    ], "an integrity fault emitted an outage warning -- the arms are misordered"


# Every ``_raise_durable_storage_unavailable`` call site in files.py, with the
# fields it is expected to carry. N2 in review -- the signed-redirect site
# shipping with no identifier -- was invisible because only two sites had
# assertions; this sweep is what makes the set itself the contract.
_FAULT_SITES = (
    ("signed durable redirect", ("file_id",)),
    ("upload", ("user_id", "task_id")),
    ("download", ("file_id",)),
    ("preview", ("file_id",)),
    ("pptx preview", ("file_id",)),
    ("public download", ("file_id",)),
    ("public preview", ("file_id",)),
    ("public preview task asset", ("file_id",)),
    ("durable cleanup before row delete", ("file_id", "storage_key")),
)


# --- shared AST primitives -------------------------------------------------
#
# Several contracts in this file can only be checked against source: which
# call sites exist, what they bind, which arms come first, and whether a
# handler re-raises. Each of those started as its own hand-rolled walk, and
# four near-identical parse-and-search blocks were three too many. These are
# the pieces they share; the checks themselves stay separate because they
# assert different things.


def _module_ast(module: Any) -> ast.Module:
    """Parse an imported module's own source."""
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _function_named(tree: ast.Module, name: str) -> ast.AST:
    """The (async) function definition called ``name``, anywhere in ``tree``."""
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name == name
    )


def _handler_types(handler: ast.ExceptHandler) -> list[str]:
    """The exception class names an ``except`` arm names, tuple or not."""
    if isinstance(handler.type, ast.Name):
        return [handler.type.id]
    if isinstance(handler.type, ast.Tuple):
        return [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
    return []


def _durable_arm_pairs(tree: ast.Module) -> list[ast.Try]:
    """Every ``try`` that handles both the integrity subclass and its parent."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            "DurableObjectIntegrityError" in _handler_types(h) for h in node.handlers
        )
        and any(
            "DurableStorageOperationError" in _handler_types(h) for h in node.handlers
        )
    ]


def _identifiers(expression: ast.expr) -> set[str]:
    """Every name and attribute the expression reads.

    ``file_ref.record.file_id`` yields ``file_ref``, ``record`` and ``file_id``,
    so a field may be bound to a bare variable or reached through attributes.
    """
    found: set[str] = set()
    for node in ast.walk(expression):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
    return found


def _binds_a_matching_name(field: str, expression: ast.expr) -> bool:
    """Whether the expression reads an identifier plausibly holding ``field``.

    A qualifier prefix is allowed -- ``task_id=parsed_task_id`` and
    ``user_id=owner_user_id`` name the thing they carry -- while an unrelated
    name is rejected, which is what catches ``file_id=storage_key``.

    The prefix must end at an underscore. A bare suffix test would admit
    ``file_id=profile_id``, since ``"profile_id".endswith("file_id")`` -- an
    unrelated identifier passing on a coincidental substring, which is the
    opposite of the point.

    **This checks spelling, not referent, and that is a real limit.** The
    "public preview task asset" site shipped with ``file_id=file_id`` where the
    failing object was ``asset_record`` -- a different row from the route's
    ``file_id``, so the one line meant to identify the failure named an object
    that was fine. This check passed it, because the spelling was right. Only a
    test that drives the endpoint distinguishes those, which is #1522; do not
    read a green run here as "every site logs the right object".
    """
    return any(
        identifier == field or identifier.endswith(f"_{field}")
        for identifier in _identifiers(expression)
    )


@pytest.mark.parametrize(
    ("field", "source", "accepted"),
    [
        ("file_id", "file_id", True),
        ("file_id", "file_ref.record.file_id", True),
        ("task_id", "parsed_task_id", True),
        ("user_id", "owner_user_id", True),
        ("file_id", "storage_key", False),
        # A coincidental suffix, not a qualified name: the underscore boundary
        # is the whole difference, and without it this passes.
        ("file_id", "profile_id", False),
    ],
)
def test_the_binding_check_accepts_qualified_names_and_rejects_unrelated_ones(
    field: str, source: str, accepted: bool
) -> None:
    """The rule the site sweep leans on, pinned on its own.

    Asserted here rather than only through the sweep because the sweep can only
    fail on the sites that exist: a rule too loose to reject anything would
    still pass it. ``profile_id`` is the case that made this necessary.
    """
    expression = ast.parse(source, mode="eval").body
    assert _binds_a_matching_name(field, expression) is accepted


def test_every_fault_site_label_is_bounded_and_reaches_the_log() -> None:
    """The nine labels are a closed set of bounded, aggregatable values.

    ``upload`` carries no ``file_id`` by design -- it is a batch-registration
    path, and any file in the batch may be the one that failed -- but it does
    carry tenant and task, which is what correlates a 503 burst. Every other
    site identifies its subject.
    """
    tree = _module_ast(files_api)
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
        label = node.args[1].value
        expected = dict(_FAULT_SITES)[label]
        assert tuple(kw.arg for kw in node.keywords) == expected, (
            f"site {label!r} passes "
            f"{[kw.arg for kw in node.keywords]}, expected {list(expected)}"
        )
        # Names alone are not the contract. ``file_id=storage_key`` declares the
        # right field and renders the wrong value, which is invisible both here
        # and to the sweep below -- that one supplies its own placeholder values
        # and never reads the call site. Requiring the bound expression to read
        # something of the same name is what ties the label to the variable.
        for keyword in node.keywords:
            assert keyword.arg is not None, f"site {label!r} uses **kwargs"
            assert _binds_a_matching_name(keyword.arg, keyword.value), (
                f"site {label!r} binds {keyword.arg}= to "
                f"{ast.unparse(keyword.value)!r}, which never reads "
                f"{keyword.arg} -- the field name and the value must agree"
            )


def test_no_client_supplied_message_type_can_become_an_operation_label() -> None:
    """The label must be a literal from the map, never built from the input.

    ``operation`` is rendered into the message and, unlike the fields beside it,
    is not escaped or bounded (#1520), so the received ``type`` reaching it would
    reintroduce the injection this PR closed for fields.

    Asserted against the **call site**, not by re-evaluating the lookup: an
    earlier version of this test called ``_DISPATCH_OPERATIONS.get(hostile, ...)``
    itself and claimed that pinned the property. It did not -- rewriting the call
    site to ``f"websocket {message_data.get('type')}"`` would have left it green,
    which is the exact leak class it claimed to guard.
    """
    endpoint = _function_named(_module_ast(websocket_api), "websocket_chat_endpoint")

    assignments = [
        node
        for node in ast.walk(endpoint)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "dispatching"
            for target in node.targets
        )
    ]
    assert assignments, "no assignment to `dispatching` -- the label plumbing moved"

    for node in assignments:
        value = node.value
        if isinstance(value, ast.Constant):
            assert isinstance(value.value, str)
            continue
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute), (
            f"line {node.lineno}: `dispatching` is built by "
            f"{ast.unparse(value)!r}; it must be a literal or a lookup in "
            "_DISPATCH_OPERATIONS, never composed from the received type"
        )
        assert value.func.attr == "get"
        assert (
            isinstance(value.func.value, ast.Name)
            and value.func.value.id == "_DISPATCH_OPERATIONS"
        ), f"line {node.lineno}: lookup is not against _DISPATCH_OPERATIONS"
        fallback = value.args[1] if len(value.args) > 1 else None
        assert isinstance(fallback, ast.Name) and fallback.id == (
            "_UNKNOWN_DISPATCH_OPERATION"
        ), f"line {node.lineno}: unknown types must fall back to a fixed label"

    # And the values a client can select between are bounded literals.
    labels = set(websocket_api._DISPATCH_OPERATIONS.values())
    labels.add(websocket_api._UNKNOWN_DISPATCH_OPERATION)
    for label in labels:
        assert label == label.strip()
        assert not any(char in label for char in "\n\r\t")
        assert len(label) < 64


def test_every_dispatched_message_type_has_a_label() -> None:
    """The map and the dispatch cascade must not drift apart.

    They are two switches on the same value, kept in step by hand: a seventh
    handler added to the cascade without a label would log its faults as
    "unknown message type", which is silent and exactly the mislabelling the
    map was added to fix.
    """
    endpoint = _function_named(_module_ast(websocket_api), "websocket_chat_endpoint")
    dispatched = {
        comparator.value
        for node in ast.walk(endpoint)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Call)
        and isinstance(node.left.func, ast.Attribute)
        and node.left.func.attr == "get"
        and [getattr(arg, "value", None) for arg in node.left.args] == ["type"]
        for comparator in node.comparators
        if isinstance(comparator, ast.Constant)
    }

    assert dispatched, "found no dispatch branches -- the parse assumption broke"
    labelled = set(websocket_api._DISPATCH_OPERATIONS)
    unlabelled = _SWALLOWED_DISPATCH_TYPES | _INNER_REPORTED_DISPATCH_TYPES
    assert dispatched == labelled | unlabelled, (
        "dispatch cascade and label map disagree: "
        f"{dispatched ^ (labelled | unlabelled)}"
    )
    assert not (labelled & unlabelled), (
        "a type cannot be both labelled and declared unreachable: "
        f"{labelled & unlabelled}"
    )


# Message types deliberately absent from the label map because their handler
# swallows the fault before it can reach the endpoint arm. Declared, not
# assumed: the test below reads the handlers to confirm it.
_SWALLOWED_DISPATCH_TYPES = {"execute_task", "intervention"}

# Absent for a different reason: the handler propagates, but its own fault arm
# reports first and the shared logger marks the instance, so the endpoint-level
# call is a no-op. ``test_a_reported_chat_fault_needs_no_endpoint_label`` pins
# that behaviour rather than trusting this comment.
_INNER_REPORTED_DISPATCH_TYPES = {"chat"}
_SWALLOWING_HANDLERS = {
    "execute_task": "handle_execute_task",
    "intervention": "handle_intervention",
}


def test_a_reported_chat_fault_needs_no_endpoint_label(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Why ``chat`` is not in the label map, asserted rather than asserted-in-prose.

    Its own fault arm reports before re-raising, and the shared logger marks the
    instance, so the endpoint-level call that would render a ``chat`` label is a
    no-op. Giving it one would name a line that is never emitted -- which is what
    it had, and what a reviewer found by tracing the marker.

    If the marker or the arm ever changes so a second record *is* produced, this
    fails and the label goes back.
    """
    fault = _wrapped_fault()

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        # The arm inside the chat handler.
        log_durable_storage_fault(
            files_api.logger, "websocket chat turn preparation", fault, task_id=42
        )
        # The endpoint-level arm, reached by the bare ``raise`` that follows it.
        log_durable_storage_fault(
            files_api.logger, "websocket chat turn", fault, task_id=42
        )

    rendered = _sole_warning(caplog, files_api.logger.name)
    assert "during websocket chat turn preparation" in rendered
    assert "chat" in _INNER_REPORTED_DISPATCH_TYPES


@pytest.mark.parametrize(
    ("dispatch_type", "handler"), sorted(_SWALLOWING_HANDLERS.items())
)
def test_a_type_is_unlabelled_only_because_its_handler_swallows(
    dispatch_type: str, handler: str
) -> None:
    """The omission above must stay true of the code, not just of a comment.

    Leaving these two out of the label map is only correct while their handlers
    end in ``except RuntimeError`` without re-raising, because that is what stops
    a durable fault ever reaching the arm that would use the label. If someone
    adds a re-raise -- or a durable arm, which is the #1515 work -- the type
    becomes reachable and needs a label, and this is what says so.
    """
    func = _function_named(_module_ast(websocket_api), handler)
    runtime_arms = [
        arm
        for arm in (
            handler_arm
            for try_node in ast.walk(func)
            if isinstance(try_node, ast.Try)
            for handler_arm in try_node.handlers
        )
        if isinstance(arm.type, ast.Name) and arm.type.id == "RuntimeError"
    ]

    assert runtime_arms, f"{handler} no longer has an except RuntimeError arm"
    for arm in runtime_arms:
        reraises = any(
            isinstance(node, ast.Raise) and node.exc is None for node in ast.walk(arm)
        )
        assert not reraises, (
            f"{handler}'s RuntimeError arm now re-raises, so {dispatch_type!r} "
            "can reach the endpoint fault arm and needs a label in "
            "_DISPATCH_OPERATIONS"
        )


# Every module carrying an integrity/parent arm pair. The ordering below is a
# safety property, not a style rule: ``DurableObjectIntegrityError`` subclasses
# ``DurableStorageOperationError``, so a parent arm placed first makes the child
# arm dead and reports permanent corruption as a retryable outage -- loud enough
# to trip the alerts that watch for one, and burying the real diagnosis.
_MODULES_WITH_DURABLE_ARM_PAIRS = (
    ("files.py", lambda: files_api),
    ("websocket.py", lambda: websocket_api),
    ("v1/tasks.py", lambda: v1_tasks),
    ("file_ingestion_tool.py", lambda: _file_ingestion_tool()),
)


def _file_ingestion_tool() -> Any:
    from xagent.core.tools.adapters.vibe import file_ingestion_tool

    return file_ingestion_tool


@pytest.mark.parametrize(("label", "loader"), _MODULES_WITH_DURABLE_ARM_PAIRS)
def test_the_integrity_arm_precedes_its_parent_at_every_site(
    label: str, loader: Any
) -> None:
    """Checked at all twelve pairs, because only one has real end-to-end cover.

    The integrity-vs-outage distinction is asserted through a real call path in
    exactly one place (``file_ingestion_tool``, via ``test_kb_creation_tools``).
    Everywhere else -- seven pairs in ``files.py``, three in ``websocket.py``,
    one in ``v1/tasks.py`` -- the tests either drive the shared helper directly
    with a pre-built exception or inject only the parent class, so a swapped
    ordering would change behaviour silently at ten of the twelve.

    This does not replace those tests: it proves ordering, not that each arm
    produces the right answer. What it does is make the one regression that
    silently breaks the property at every site impossible to land. Real
    end-to-end coverage for the ``files.py`` sites is #1522.
    """
    tree = _module_ast(loader())
    pairs = _durable_arm_pairs(tree)
    assert pairs, f"{label}: expected at least one integrity/parent pair"

    for node in pairs:
        integrity_at = min(
            handler.lineno
            for handler in node.handlers
            if "DurableObjectIntegrityError" in _handler_types(handler)
        )
        parent_at = min(
            handler.lineno
            for handler in node.handlers
            if "DurableStorageOperationError" in _handler_types(handler)
        )
        assert integrity_at < parent_at, (
            f"{label}: the try at line {node.lineno} catches "
            f"DurableStorageOperationError (line {parent_at}) before its "
            f"subclass DurableObjectIntegrityError (line {integrity_at}), which "
            "makes the integrity arm dead and reports corruption as an outage"
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
    # A site that names its own storage_key overrides the wrap's, so the wrap's
    # key is legitimately absent there -- that precedence is the point.
    _assert_cause_chain_recorded(
        rendered, wrap_key="storage_key" not in expected_fields
    )


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


@pytest.mark.parametrize("length", [_MAX_LOG_VALUE_LENGTH - 1, _MAX_LOG_VALUE_LENGTH])
def test_a_field_value_at_or_under_the_bound_is_kept_whole(
    caplog: pytest.LogCaptureFixture, length: int
) -> None:
    """The bound is inclusive, so neither of these may be marked truncated."""
    value = "v" * length

    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException):
            files_api._raise_durable_storage_unavailable(
                _wrapped_fault(), "upload", file_ids=value
            )

    rendered = _sole_warning(caplog, files_api.logger.name)
    assert f"file_ids={value} " in f"{rendered.splitlines()[0]} "
    assert "truncated" not in rendered.splitlines()[0]


def test_an_overlong_field_value_is_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One field must not be able to crowd out the rest of the line.

    Pins the boundary rather than only the marker: with the length unasserted
    this passed for any bound at all, so a change from 256 to 4096 would have
    kept a green run while one field again took over the line.
    """
    with caplog.at_level(logging.WARNING, logger=files_api.logger.name):
        with pytest.raises(HTTPException):
            files_api._raise_durable_storage_unavailable(
                _wrapped_fault(), "upload", file_ids="x" * 5000
            )

    message_line = _sole_warning(caplog, files_api.logger.name).splitlines()[0]
    # The retained part is the intact leading prefix, at exactly the bound --
    # one character more is the off-by-one this pins.
    assert f"file_ids={'x' * _MAX_LOG_VALUE_LENGTH}...[truncated]" in message_line
    assert "x" * (_MAX_LOG_VALUE_LENGTH + 1) not in message_line
    assert len(message_line) < 1000
    # The magnitude is part of the contract too, not just the mechanics: the
    # cap exists so one field cannot crowd out the rest of the line, and the
    # assertions above are all written against the constant, so they would stay
    # green if it were raised to a value that defeats that. A ceiling rather
    # than an equality, so the number stays tunable within its purpose.
    assert _MAX_LOG_VALUE_LENGTH <= 512
