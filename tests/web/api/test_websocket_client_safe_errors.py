"""Exception text must not reach chat clients (PR #1472 review finding N3)."""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator

from .conftest import _direct_db_session

SECRET = "/srv/xagent/secrets/prod.key"


def _client_payloads(connection_manager: MagicMock) -> list[dict]:
    return [
        call.args[0]
        for call in (
            connection_manager.send_personal_message.await_args_list
            + connection_manager.broadcast_to_task.await_args_list
        )
        if call.args and isinstance(call.args[0], dict)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised",
    [
        ValueError(f"invalid payload while reading {SECRET}"),
        KeyError(f"missing key near {SECRET}"),
        TypeError(f"bad type from {SECRET}"),
    ],
    ids=["value", "key", "type"],
)
async def test_execute_task_redacts_an_incidental_validation_error(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
) -> None:
    """A widget visitor gets the fixed string, never the exception's text."""
    db = _direct_db_session()
    try:
        user = User(
            username=f"safe-error-owner-{type(raised).__name__}",
            password_hash="hash",
        )
        db.add(user)
        db.commit()
        task = Task(
            user_id=int(user.id),
            title="Client safe errors",
            description="Run the existing task",
            status=TaskStatus.PENDING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task_id = int(task.id)
        user_id = int(user.id)
    finally:
        db.close()

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    async def raise_at_schedule(**_kwargs: object) -> asyncio.Task:
        raise raised

    monkeypatch.setattr(
        TaskTurnOrchestrator,
        "schedule_existing_task_execution",
        AsyncMock(side_effect=raise_at_schedule),
        raising=False,
    )

    await websocket_api.handle_execute_task(
        MagicMock(),
        task_id,
        {"user": SimpleNamespace(id=user_id, is_admin=False)},
    )

    payloads = _client_payloads(connection_manager)
    assert payloads, "the handler must tell the client something"
    serialized = repr(payloads)
    assert SECRET not in serialized
    assert str(raised) not in serialized
    assert any(
        payload.get("message") == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
        for payload in payloads
    )


@pytest.mark.asyncio
async def test_execute_task_keeps_a_message_written_for_the_sender(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validation failure raised as client-visible keeps its own wording."""
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    # No authenticated actor: the handler raises its own curated message.
    await websocket_api.handle_execute_task(MagicMock(), 1, {})

    payloads = _client_payloads(connection_manager)
    assert any(
        "authentication required" in str(payload.get("message", "")).lower()
        for payload in payloads
    )


FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

# arg name -> positional index of the client-visible message
PRODUCERS: dict[str, int | None] = {
    "finish_delivery_failure": 0,
    "finish_delivery": 1,
    "send_message_delivery": None,  # keyword-only
}

# The one deliberate exception: agent RuntimeError text is surfaced to the
# sender by an existing, tested contract. Narrowing it is a product decision,
# tracked in #1479 - not a hole to be quietly widened.
#
# Anchored to (enclosing function, expression), not to the expression alone:
# keying on source text blesses the string everywhere it is ever pasted,
# including into a *validation* branch, which is not what #1479 describes.
ALLOWED_RAW_MESSAGES = {
    ("_handle_chat_message_unserialized", "f'Runtime error: {str(e)}'"),
    ("handle_execute_task", "f'Runtime error: {str(e)}'"),
    ("handle_intervention", "f'Runtime error: {str(e)}'"),
    ("_handle_pause_task_unserialized", "f'Runtime error: {str(e)}'"),
    ("_handle_resume_task_unserialized", "f'Runtime error: {str(e)}'"),
}


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_functions(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> list[ast.AST]:
    """Innermost-first chain of functions a node can read locals from."""
    chain: list[ast.AST] = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, FUNCTION_NODES):
            chain.append(current)
        current = parents.get(current)
    return chain


def _message_expression(node: ast.Call, index: int | None) -> ast.expr | None:
    for keyword in node.keywords:
        if keyword.arg == "message":
            return keyword.value
    if index is not None and len(node.args) > index:
        return node.args[index]
    return None


# ``send_text`` takes a serialized payload, so the dict sits one call deeper.
ERROR_PAYLOAD_SINKS = {"send_personal_message", "broadcast_to_task", "send_text"}


def _unwrap_serializer(expr: ast.expr) -> ast.expr:
    """``json.dumps(payload)`` -> ``payload``; anything else unchanged."""
    if isinstance(expr, ast.Call) and _called_name(expr) == "dumps" and expr.args:
        return expr.args[0]
    return expr


def _error_payload_message(node: ast.Call) -> ast.expr | None:
    """The message of an ``{"type": "error", ...}`` payload, if this is one."""
    if _called_name(node) not in ERROR_PAYLOAD_SINKS:
        return None
    for raw in (*node.args, *(kw.value for kw in node.keywords)):
        argument = _unwrap_serializer(raw)
        if not isinstance(argument, ast.Dict):
            continue
        keys = {
            key.value: value
            for key, value in zip(argument.keys, argument.values)
            if isinstance(key, ast.Constant)
        }
        kind = keys.get("type")
        if isinstance(kind, ast.Constant) and kind.value == "error":
            return keys.get("message")
    return None


def _called_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_parameter(scopes: list[ast.AST], name: str) -> bool:
    """A forwarded parameter is vetted at the wrapper's own call sites."""
    for scope in scopes:
        if not isinstance(scope, FUNCTION_NODES):
            continue
        arguments = scope.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            if argument.arg == name:
                return True
    return False


def _local_assignments(scopes: list[ast.AST], name: str) -> list[ast.expr]:
    """Values assigned to `name` inside the given scopes only."""
    values: list[ast.expr] = []
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        values.append(node.value)
    return values


def _is_client_safe(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return True
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        return expr.func.id == "client_safe_error_message"
    # `_TURN_REJECTION_MESSAGES.get(reason, "<literal>")` - a curated table.
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        return expr.func.attr == "get" and all(
            isinstance(arg, ast.Constant) or isinstance(arg, ast.Attribute)
            for arg in expr.args[1:]
        )
    if isinstance(expr, ast.BoolOp):
        return all(_is_client_safe(value) for value in expr.values)
    return False


def test_no_delivery_producer_can_bypass_the_client_safe_message() -> None:
    """Exception text may not reach a client through the *recognized* shapes.

    Scope, stated honestly: this walks direct calls to the producers in
    ``PRODUCERS`` and ``{"type": "error", ...}`` dict *literals* passed to
    ``send_personal_message``/``broadcast_to_task``. It does NOT follow
    dict-spread payloads, payloads built by a helper and passed as a call, or
    wrapper functions that forward a raw argument into a producer. Those
    shapes leak today and are tracked in #1497 - do not read a passing run as
    "nothing can reach a client raw".
    """
    # Explicit encoding: this module carries non-ASCII prose, and the
    # platform default would decode it as cp1252/GBK on a Windows runner.
    source = Path(websocket_api.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents = _parents(tree)

    checked_producers = 0
    checked_error_payloads = 0
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        is_producer = name in PRODUCERS
        if is_producer:
            expr = _message_expression(node, PRODUCERS[name])
        else:
            # The error bubble renders in the same conversation as the
            # rejection ack, so it is the same disclosure surface.
            expr = _error_payload_message(node)
            name = f"{name}(error payload)"
        if expr is None:
            continue
        if is_producer:
            checked_producers += 1
        else:
            checked_error_payloads += 1
        scopes = _enclosing_functions(node, parents)
        if isinstance(expr, ast.Name) and _is_parameter(scopes, expr.id):
            continue
        candidates = (
            _local_assignments(scopes, expr.id)
            if isinstance(expr, ast.Name)
            else [expr]
        )
        if not candidates:
            offenders.append(f"{name}:{node.lineno} passes an unresolvable name")
            continue
        enclosing = next(
            (scope.name for scope in scopes if isinstance(scope, FUNCTION_NODES)),
            "<module>",
        )
        for candidate in candidates:
            if _is_client_safe(candidate):
                continue
            if (enclosing, ast.unparse(candidate)) in ALLOWED_RAW_MESSAGES:
                continue
            offenders.append(
                f"{name}:{node.lineno} may send {ast.unparse(candidate)!r}"
            )

    # Pinned near the actual counts (23 / 23 at the time of writing) so that
    # a producer disappearing from the sweep trips this before it trips a user.
    assert checked_producers >= 21, (
        f"the producers moved; only {checked_producers} matched"
    )
    assert checked_error_payloads >= 21, (
        f"the error payloads moved; only {checked_error_payloads} matched"
    )
    assert not offenders, (
        "raw text can reach a chat client; route it through "
        "client_safe_error_message: " + "; ".join(sorted(set(offenders)))
    )


WEBSOCKET_LOGGER = "xagent.web.api.websocket"

# Every handler that turns an enqueue refusal into client-visible text. Each
# entry is (handler, extra message_data) - the ack in handle_chat_message only
# fires when the client supplied an id, so that one needs the extra key.
ENQUEUE_FAILURE_HANDLERS = [
    ("handle_chat_message", {"client_message_id": "cmid-1"}),
    ("handle_pause_task", {}),
    ("handle_resume_task", {}),
]


def test_missing_task_keeps_its_wording_for_the_sender(_test_db: None) -> None:
    """A missing task is the sender's own answer, so redaction must spare it.

    ``execute_task_background`` already raises this as client-visible; the
    pause/resume enqueue path raised a bare ``ValueError``, which the redaction
    turned into the generic string and left the sender with nothing to act on.
    """
    with pytest.raises(ValueError) as raised:
        websocket_api._enqueue_websocket_task_command_sync(
            task_id=424242,
            actor_user_id=1,
            actor_is_admin=False,
            command_id="pause:missing-task",
            kind=websocket_api.TaskCommandKind.PAUSE,
            payload={},
            allow_missing_task=False,
        )

    assert isinstance(raised.value, websocket_api.ClientVisibleValidationError)
    assert (
        websocket_api.client_safe_error_message(raised.value) == "Task 424242 not found"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "extra_message_data"),
    ENQUEUE_FAILURE_HANDLERS,
    ids=[name for name, _ in ENQUEUE_FAILURE_HANDLERS],
)
async def test_redacted_enqueue_failure_still_reaches_the_log(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    handler_name: str,
    extra_message_data: dict,
) -> None:
    """Redacting the client's copy must not delete the operator's copy.

    These handlers previously leaked ``str(exc)`` to the client and logged
    nothing; the leak was the only record. With the text redacted, an
    incidental failure would otherwise vanish without a trace.
    """
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "_enqueue_websocket_task_command",
        AsyncMock(side_effect=ValueError(f"enqueue failed reading {SECRET}")),
    )

    handler = getattr(websocket_api, handler_name)
    with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
        await handler(
            MagicMock(),
            7,
            {"user": SimpleNamespace(id=1, is_admin=False), **extra_message_data},
        )

    payloads = _client_payloads(connection_manager)
    assert payloads, "the handler must tell the client something"
    assert SECRET not in repr(payloads)
    assert any(
        payload.get("message") == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
        for payload in payloads
    )

    records = [record for record in caplog.records if record.name == WEBSOCKET_LOGGER]
    assert records, f"{handler_name} redacted the failure without logging it"
    assert any(SECRET in record.getMessage() for record in records)
    assert any(record.exc_info is not None for record in records)


@pytest.mark.parametrize(
    ("error", "expected_level", "expects_traceback"),
    [
        (
            websocket_api.ClientVisibleValidationError("User authentication required"),
            logging.WARNING,
            False,
        ),
        (ValueError(f"incidental fault at {SECRET}"), logging.ERROR, True),
    ],
    ids=["curated", "incidental"],
)
def test_log_level_follows_the_marker_not_the_call_site(
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    expected_level: int,
    expects_traceback: bool,
) -> None:
    """A curated refusal is routine; only an incidental fault earns a traceback.

    Without the split, any visitor could make the server dump a stack on
    demand by sending an unauthenticated frame in a loop.
    """
    with caplog.at_level(logging.DEBUG, logger=WEBSOCKET_LOGGER):
        websocket_api.log_client_facing_failure(
            error, "Pause command rejected for task %s: %s", 7
        )

    (record,) = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert record.levelno == expected_level
    assert (record.exc_info is not None) is expects_traceback
    assert "task 7" in record.getMessage()
