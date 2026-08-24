"""Exception text must not reach chat clients (PR #1472 review finding N3)."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

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


def _sent_text_payloads(websocket: MagicMock) -> list[dict]:
    """Payloads written straight to the socket, bypassing ``manager``.

    ``handle_builder_chat`` uses this sink, so a helper that only reads the
    manager mock cannot see it - which is why that handler's leak survived
    until the AST sweep learned to recognize ``send_text``.
    """
    payloads = []
    for call in websocket.send_text.await_args_list:
        if not call.args:
            continue
        try:
            decoded = json.loads(call.args[0])
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, dict):
            payloads.append(decoded)
    return payloads


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

# The one deliberate exception: agent RuntimeError text is passed through to
# the INITIATING SENDER - the rejection ack and the personal error bubble.
# Narrowing that wording is the product decision tracked in #1479.
#
# The broadcast half of the passthrough is closed (maintainer scope ruling on
# #1514): the task-wide broadcast reaches every connection under the task_id,
# anonymous widget and share visitors included, and DurableStorageOperation-
# Error subclasses RuntimeError with tenant-scope text in its message, so
# broadcasts carry CLIENT_SAFE_TASK_FAILURE instead - pinned by the two
# audience-boundary tests at the end of this file. Still #1479: whether the
# sender copy should also be narrowed when the initiator is an anonymous
# public connection.
#
# Anchored to (enclosing function, expression) rather than to the expression
# alone, which blessed the string in every function it appeared in. This stops
# reuse in a *different* function only: `_local_assignments` unions every
# assignment in the enclosing function regardless of branch, so moving the
# string between branches of an allowlisted function is NOT caught. #1547.
ALLOWED_RAW_MESSAGES = {
    ("_handle_chat_message_unserialized", "f'Runtime error: {str(e)}'"),
    # The same #1479 flow seen through the closure scope chain: the outer
    # function's runtime-error string reaches these closures' ``message``
    # parameter at their call sites. Surfaced when the parameter short-circuit
    # stopped hiding rebound names; not a new leak.
    ("finish_delivery", "f'Runtime error: {str(e)}'"),
    ("finish_delivery_failure", "f'Runtime error: {str(e)}'"),
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

# Both render in the client's conversation, so both are the same disclosure
# surface. ``agent_error`` was missing until review found a producer using it.
ERROR_PAYLOAD_TYPES = {"error", "agent_error"}

# The only functions allowed to mint client-visible text from an exception.
SAFE_MESSAGE_BUILDERS = {
    "client_safe_error_message",
    "client_safe_task_command_failure",
}


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
        if isinstance(kind, ast.Constant) and kind.value in ERROR_PAYLOAD_TYPES:
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
        if expr.func.id == "client_safe_error_message":
            return True
        if expr.func.id == "client_safe_task_command_failure":
            # The prefix argument must be attribute access on server state
            # (``command.kind``), never a literal or a bare name a caller
            # could point at untrusted text.
            return bool(expr.args) and isinstance(expr.args[0], ast.Attribute)
        return False
    # `_TURN_REJECTION_MESSAGES.get(reason, "<literal>")` - a curated table.
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        return expr.func.attr == "get" and all(
            isinstance(arg, ast.Constant) or isinstance(arg, ast.Attribute)
            for arg in expr.args[1:]
        )
    if isinstance(expr, ast.BoolOp):
        return all(_is_client_safe(value) for value in expr.values)
    return False


class _ScanResult(NamedTuple):
    offenders: list[str]
    producers: int
    error_payloads: int
    used_allowlist: set[tuple[str, str]]


def _scan(tree: ast.Module) -> _ScanResult:
    """The one copy of the sweep's recognition logic.

    Both the production sweep and the snippet-based regression tests run
    this same function, so a change to the analysis cannot pass the snippet
    tests while silently not applying to the real module (or vice versa).
    """
    parents = _parents(tree)
    producers = 0
    error_payloads = 0
    offenders: list[str] = []
    used_allowlist: set[tuple[str, str]] = set()
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
            producers += 1
        else:
            error_payloads += 1
        scopes = _enclosing_functions(node, parents)
        candidates = (
            _local_assignments(scopes, expr.id)
            if isinstance(expr, ast.Name)
            else [expr]
        )
        if not candidates:
            # Only a name nothing rebinds is a genuinely forwarded parameter,
            # vetted at the wrapper's own call sites. A parameter rebound by a
            # plain assignment lands in the candidates instead; rebinding via
            # AnnAssign/AugAssign/walrus/tuple targets is still invisible to
            # ``_local_assignments`` and tracked in #1547.
            if isinstance(expr, ast.Name) and _is_parameter(scopes, expr.id):
                continue
            offenders.append(f"{name}:{node.lineno} passes an unresolvable name")
            continue
        enclosing = next(
            (scope.name for scope in scopes if isinstance(scope, FUNCTION_NODES)),
            "<module>",
        )
        for candidate in candidates:
            if _is_client_safe(candidate):
                continue
            key = (enclosing, ast.unparse(candidate))
            if key in ALLOWED_RAW_MESSAGES:
                used_allowlist.add(key)
                continue
            offenders.append(
                f"{name}:{node.lineno} may send {ast.unparse(candidate)!r}"
            )
    return _ScanResult(offenders, producers, error_payloads, used_allowlist)


def test_no_delivery_producer_can_bypass_the_client_safe_message() -> None:
    """Exception text may not reach a client through the *recognized* shapes.

    Scope, stated honestly: this walks direct calls to the producers in
    ``PRODUCERS`` and dict *literals* whose ``type`` is one of
    ``ERROR_PAYLOAD_TYPES``, passed to one of ``ERROR_PAYLOAD_SINKS``.

    It does NOT follow dict-spread payloads, payloads built by a helper and
    passed as a call, wrapper functions that forward a raw argument into a
    producer (all #1497), or payloads whose ``type`` is a variable rather than
    a string literal (#1547). ``agent_error`` was added only after review
    found ``_broadcast_terminal_command_error`` escaping on that dimension
    alone - the type set is a maintained list, not a derived invariant.

    Those shapes leak today. Do not read a passing run as "nothing can reach a
    client raw"; the xfail tests below pin the ones we know about.
    """
    # Explicit encoding: this module carries non-ASCII prose, and the
    # platform default would decode it as cp1252/GBK on a Windows runner.
    source = Path(websocket_api.__file__).read_text(encoding="utf-8")
    result = _scan(ast.parse(source))

    for builder in SAFE_MESSAGE_BUILDERS:
        assert callable(getattr(websocket_api, builder, None)), (
            f"SAFE_MESSAGE_BUILDERS blesses {builder!r}, which does not exist"
        )

    # Actual counts at the time of writing: 23 producers, 30 error payloads.
    # These floors sit below that, so a minority of sites can still vanish
    # silently; tightening them to exact equality is tracked in #1547.
    assert result.producers >= 21, (
        f"the producers moved; only {result.producers} matched"
    )
    assert result.error_payloads >= 21, (
        f"the error payloads moved; only {result.error_payloads} matched"
    )
    # Every allowlist entry must be earned by a live call site: a stale entry
    # is a standing exemption nothing uses, and an unused closure entry is
    # exactly what a reverted parameter-rebinding fix would leave behind.
    assert result.used_allowlist == ALLOWED_RAW_MESSAGES, (
        "stale allowlist entries: "
        f"{sorted(ALLOWED_RAW_MESSAGES - result.used_allowlist)}"
    )
    assert not result.offenders, (
        "raw text can reach a chat client; route it through "
        "client_safe_error_message: " + "; ".join(sorted(set(result.offenders)))
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


@pytest.mark.asyncio
async def test_builder_chat_redacts_through_its_own_socket_sink(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The leak found by self-review, pinned at runtime rather than by AST alone.

    ``handle_builder_chat`` answers on ``websocket.send_text`` instead of going
    through ``manager``, which is why it escaped both the original sweep and
    every behavioural test in this file.
    """
    from xagent.web.services import builder_chat_runtime

    monkeypatch.setattr(
        builder_chat_runtime,
        "load_builder_chat_runtime_inputs",
        AsyncMock(side_effect=ValueError(f"builder fault at {SECRET}")),
    )

    websocket = MagicMock()
    websocket.send_text = AsyncMock()
    websocket.state = SimpleNamespace()

    with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
        await websocket_api.handle_builder_chat(
            websocket,
            {"message": "build me an agent"},
            SimpleNamespace(id=1, is_admin=False),
        )

    records = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert any(SECRET in r.getMessage() for r in records), (
        "the operator half of the contract: redacting the client's copy must "
        "not delete the server's"
    )

    payloads = _sent_text_payloads(websocket)
    errors = [p for p in payloads if p.get("type") == "error"]
    assert errors, "the handler must answer the builder client"
    assert SECRET not in repr(payloads)
    assert errors[-1]["message"] == websocket_api.CLIENT_SAFE_VALIDATION_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_name", ["handle_pause_task", "handle_resume_task"], ids=["pause", "resume"]
)
async def test_permission_wording_survives_redaction_in_every_handler(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    handler_name: str,
) -> None:
    """All three handlers share one raise site; only one of them was asserted.

    ``test_websocket_error_payload.py`` pins this wording for
    ``handle_chat_message``. A regression in either sibling's ``except`` - one
    that redacted the refusal to the generic string, or leaked something else
    through it - would have gone unnoticed.
    """
    db = _direct_db_session()
    try:
        owner = User(username=f"owner-{handler_name}", password_hash="hash")
        intruder = User(username=f"intruder-{handler_name}", password_hash="hash")
        db.add_all([owner, intruder])
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Someone else's task",
            description="Not yours",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task_id, intruder_id = int(task.id), int(intruder.id)
    finally:
        db.close()

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    await getattr(websocket_api, handler_name)(
        MagicMock(),
        task_id,
        {"user": SimpleNamespace(id=intruder_id, is_admin=False)},
    )

    payloads = _client_payloads(connection_manager)
    assert payloads, "the handler must refuse the intruder out loud"
    assert any(
        payload.get("message")
        == f"Access denied: Task {task_id} does not belong to you"
        for payload in payloads
    ), payloads


# Each entry is a bypass shape this module admits it does not cover: a source
# snippet the guard reports clean even though raw exception text reaches a
# client. Each mirrors a real site rather than a minimal repro, so the xfail
# cannot flip on a shape nothing actually uses - dict-spread copies
# ``execute_task_background`` (text under ``error``, type inherited from the
# spread), helper-built copies ``send_historical_data_as_stream``, and
# wrapper-forwarded copies ``notify_deferred_delivery``. They are pinned as strict xfails so the day the guard learns a shape,
# its test flips to a failure and says so, instead of the hole quietly
# outliving the issue that tracks it.
BYPASS_SHAPES = [
    pytest.param(
        """
async def leak(websocket, task_id):
    try:
        pass
    except Exception as e:
        terminal_payload = _terminal_task_error_payload(task_id, str(e))
        message = str(e)
        await manager.broadcast_to_task(
            {
                **terminal_payload,
                "task_id": task_id,
                "error": message,
                "timestamp": 0,
            },
            task_id,
        )
""",
        id="dict-spread",
    ),
    pytest.param(
        """
async def leak(websocket, task_id):
    try:
        pass
    except Exception as e:
        await manager.broadcast_to_task(
            create_stream_event("error", task_id, {"message": str(e)}), task_id
        )
""",
        id="helper-built",
    ),
    pytest.param(
        """
async def forward(websocket, raw):
    await send_message_delivery(
        websocket,
        client_message_id="c",
        turn_id="t",
        accepted=False,
        message=raw,
        rejection_outcome="not_accepted",
    )


async def leak(websocket):
    try:
        pass
    except Exception as e:
        await forward(websocket, str(e))
""",
        id="wrapper-forwarded",
    ),
]

# Deliberately NOT listed above: ``message_data["_durable_command_error"] =
# str(e)``. Earlier rounds of this PR described that as reaching clients via
# _broadcast_terminal_command_error, which is wrong - TaskCommandRejected is
# re-raised without broadcasting (websocket.py, execute_durable_task_command),
# and the two branches that do broadcast now go through the chokepoint.
#
# The text lands in the TaskExecutionCommand.error column. That column IS
# read back to a client - a2a.py returns it verbatim as a 500 internal_error
# body - so the reason this particular channel is not a client leak is
# narrower than "nothing reads the column": a2a only ever enqueues CANCEL, so
# the pause/resume text written here cannot reach that read path. Widening a2a
# to another command kind would turn this into a real leak, which is why the
# dependency is written down rather than left implicit.


def _guard_offenders(source: str) -> list[str]:
    """Run the one sweep implementation over an arbitrary module source."""
    return _scan(ast.parse(source)).offenders


@pytest.mark.xfail(
    strict=True,
    reason="Known guard blind spots, all tracked in #1497. When one is closed "
    "this flips to a failure - fix the issue, then delete its param.",
)
@pytest.mark.parametrize("source", BYPASS_SHAPES)
def test_known_bypass_shapes_are_still_invisible_to_the_guard(source: str) -> None:
    """A passing sweep does not mean no raw text can reach a client.

    Every snippet here puts ``str(e)`` in front of a client and the guard says
    nothing. Asserting that out loud is the difference between a documented
    gap and a forgotten one.
    """
    assert _guard_offenders(source), (
        "the guard now sees this shape - remove it from BYPASS_SHAPES and "
        "close the tracking issue"
    )


@pytest.mark.asyncio
async def test_unresolvable_task_answers_the_sender_instead_of_dropping_them(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence is a worse answer than redaction.

    This raise site was a bare ``Exception`` while its sibling a few lines
    above already carried the marker. Being untyped, it escaped every typed
    handler, reached the connection-level ``finally: manager.disconnect`` and
    left the client with nothing at all - not the text, not even the generic
    string.
    """
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    missing_task_id = 987654
    with caplog.at_level(logging.DEBUG, logger=WEBSOCKET_LOGGER):
        await websocket_api.handle_execute_task(
            MagicMock(),
            missing_task_id,
            {"user": SimpleNamespace(id=1, is_admin=False)},
        )

    payloads = _client_payloads(connection_manager)
    assert any(
        payload.get("message") == f"Task {missing_task_id} not found or access denied"
        for payload in payloads
    ), payloads

    # Marking this raise made it reachable by a typed handler, which is the
    # point - but that handler must not hand an anonymous visitor a way to
    # make the server dump a stack for every task id they guess.
    records = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert records, "the refusal still has to be recorded"
    assert all(r.exc_info is None for r in records), (
        "a curated refusal is routine; only an incidental fault earns a stack"
    )
    assert all(r.levelno <= logging.WARNING for r in records)


# --- Round 5: the parameter short-circuit must not hide rebound names -------

REBOUND_PARAMETER_SHAPES = [
    pytest.param(
        """
async def leak(websocket, message):
    try:
        pass
    except Exception as e:
        message = str(e)
        await send_message_delivery(
            websocket,
            client_message_id="c",
            turn_id="t",
            accepted=False,
            message=message,
            rejection_outcome="not_accepted",
        )
""",
        id="same-scope-rebind",
    ),
    pytest.param(
        """
async def outer(websocket, message):
    async def inner(e):
        message = str(e)
        await send_message_delivery(
            websocket,
            client_message_id="c",
            turn_id="t",
            accepted=False,
            message=message,
            rejection_outcome="not_accepted",
        )
""",
        id="nested-shadow",
    ),
]


@pytest.mark.parametrize("source", REBOUND_PARAMETER_SHAPES)
def test_guard_catches_a_rebound_parameter(source: str) -> None:
    """A name is only "vetted at the caller" while nothing in scope rebinds it.

    The short-circuit used to fire on the bare parameter match, before local
    assignments were even collected, so ``message = str(e)`` shadowing a
    ``message`` argument sailed through.
    """
    assert _guard_offenders(source), "the rebound parameter must be flagged"


def test_guard_still_trusts_a_genuinely_forwarded_parameter() -> None:
    """The wrapper shape stays clean; only its callers are judged."""
    source = """
async def forward(websocket, message):
    await send_message_delivery(
        websocket,
        client_message_id="c",
        turn_id="t",
        accepted=False,
        message=message,
        rejection_outcome="not_accepted",
    )
"""
    assert not _guard_offenders(source)


# The concurrent-delete race (TaskCommandTaskMissing between lookup and
# enqueue) is pinned in tests/web/services/test_task_command_transport.py:
# recovery-allowed returns None, and the strict path converts to
# ClientVisibleValidationError with "Task N not found" preserved.


# --- Round 5: runtime payload contracts for the changed egresses ------------


@pytest.mark.asyncio
async def test_terminal_command_failure_keeps_context_and_redacts_detail() -> None:
    """The kind prefix is ours; the exception text is not.

    The frontend renders ``message`` verbatim for ``agent_error``, so the
    redaction must not also delete the command context the client used to
    get, and the secret must not ride along in any field.
    """
    connection_manager = MagicMock()
    connection_manager.broadcast_to_task = AsyncMock()
    command = SimpleNamespace(
        kind=websocket_api.TaskCommandKind.PAUSE, task_id=7, command_id="cmd-7"
    )
    with patch.object(websocket_api, "manager", connection_manager):
        await websocket_api._broadcast_terminal_command_error(
            command, RuntimeError(f"lease lost at {SECRET}")
        )
    (payload, task_id) = connection_manager.broadcast_to_task.await_args.args
    assert task_id == 7
    assert SECRET not in repr(payload)
    assert payload["message"] == (
        f"Task command pause failed: {websocket_api.CLIENT_SAFE_VALIDATION_ERROR}"
    )
    assert payload["command_kind"] == "pause"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_name",
    ["_handle_pause_task_unserialized", "_handle_resume_task_unserialized"],
    ids=["pause", "resume"],
)
async def test_inner_command_validation_redacts_the_client_payload(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    handler_name: str,
) -> None:
    """The unserialized handlers' validation branch is a client egress too."""
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "run_db_io_cancellation_safe",
        AsyncMock(side_effect=ValueError(f"snapshot fault at {SECRET}")),
    )

    await getattr(websocket_api, handler_name)(
        MagicMock(),
        7,
        {"user": SimpleNamespace(id=1, is_admin=False)},
    )

    payloads = _client_payloads(connection_manager)
    assert payloads, "the handler must answer the client"
    assert SECRET not in repr(payloads)
    assert any(
        payload.get("message") == websocket_api.CLIENT_SAFE_VALIDATION_ERROR
        for payload in payloads
    )


@pytest.mark.asyncio
async def test_intervention_validation_redacts_the_client_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The intervention validation branch is asserted, not just swept."""
    connection_manager = MagicMock()
    connection_manager.broadcast_to_task = AsyncMock(
        side_effect=ValueError(f"intervention fault at {SECRET}")
    )
    connection_manager.send_personal_message = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    await websocket_api.handle_intervention(
        MagicMock(), 7, {"step_id": "s1", "action": "approve"}
    )

    sent = [c.args[0] for c in connection_manager.send_personal_message.await_args_list]
    assert sent, "the handler must answer the sender"
    assert SECRET not in repr(sent)
    assert sent[-1]["message"] == websocket_api.CLIENT_SAFE_VALIDATION_ERROR


@pytest.mark.asyncio
async def test_unexpected_execute_error_keeps_its_traceback(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """handle_execute_task re-raises into callers that log no stack."""

    class _Unexpected(Exception):
        pass

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    db = _direct_db_session()
    try:
        user = User(username="unexpected-owner", password_hash="hash")
        db.add(user)
        db.commit()
        task = Task(
            user_id=int(user.id),
            title="Unexpected",
            description="generic branch",
            status=TaskStatus.PENDING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task_id, user_id = int(task.id), int(user.id)
    finally:
        db.close()
    monkeypatch.setattr(
        TaskTurnOrchestrator,
        "schedule_existing_task_execution",
        AsyncMock(side_effect=_Unexpected("boom")),
    )

    with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
        with pytest.raises(_Unexpected):
            await websocket_api.handle_execute_task(
                MagicMock(),
                task_id,
                {"user": SimpleNamespace(id=user_id, is_admin=False)},
            )

    records = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert any(r.exc_info is not None for r in records), (
        "the traceback must be recorded where the detail is known"
    )


@pytest.mark.asyncio
async def test_unexpected_intervention_error_keeps_its_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """handle_intervention re-raises into public endpoints that swallow."""

    class _Unexpected(Exception):
        pass

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock(side_effect=_Unexpected("boom"))
    monkeypatch.setattr(websocket_api, "manager", connection_manager)

    with caplog.at_level(logging.ERROR, logger=WEBSOCKET_LOGGER):
        with pytest.raises(_Unexpected):
            await websocket_api.handle_intervention(
                MagicMock(), 7, {"step_id": "s1", "action": "approve"}
            )

    records = [r for r in caplog.records if r.name == WEBSOCKET_LOGGER]
    assert any(r.exc_info is not None for r in records), (
        "the traceback must be recorded where the detail is known"
    )


@pytest.mark.asyncio
async def test_chat_validation_redacts_both_the_ack_and_the_broadcast(
    _test_db: None,
) -> None:
    """The inner chat validation branch answers on two sinks; assert both.

    The rejection ack goes to the sender and the task broadcast goes to every
    subscriber through a dict-spread payload the AST guard cannot follow, so
    this is runtime-only coverage: reverting the branch to ``str(e)`` must
    fail here even though the sweep stays green.
    """
    db = _direct_db_session()
    try:
        owner = User(username="chat-validation-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Live control",
            description="validation branch",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task.runner_id = "rt5-runner"
        task.run_id = "rt5-run"
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(
        side_effect=ValueError(f"validation fault at {SECRET}")
    )
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.reserve_resume.return_value = True

    def _fake_error_payload(task_id: int, message: str, **kwargs: object) -> dict:
        return {"type": "agent_error", "message": message, "task_id": task_id}

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            MagicMock(),
        ),
        patch(
            "xagent.web.api.websocket._read_task_error_payload_isolated",
            MagicMock(side_effect=_fake_error_payload),
        ),
    ):
        await websocket_api._handle_chat_message_unserialized(
            MagicMock(),
            task_id,
            {
                "message": "trip the validation branch",
                "client_message_id": "chat-validation-secret",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [],
            },
        )

    personal = [c.args[0] for c in ws_manager.send_personal_message.await_args_list]
    broadcast = [c.args[0] for c in ws_manager.broadcast_to_task.await_args_list]
    everything = personal + broadcast
    assert everything, "the failure must be answered somewhere"
    assert SECRET not in repr(everything), everything

    rejected = [p for p in personal if p.get("type") == "message_rejected"]
    assert rejected and rejected[0]["message"] == (
        websocket_api.CLIENT_SAFE_VALIDATION_ERROR
    )
    task_errors = [b for b in broadcast if b.get("type") == "agent_error"]
    assert task_errors and task_errors[0]["message"] == (
        websocket_api.CLIENT_SAFE_VALIDATION_ERROR
    )


def _chat_runtime_error_harness(secret_error: Exception):
    """Shared live-control setup that raises ``secret_error`` at injection."""
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(side_effect=secret_error)
    mgr = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )
    bg_mgr = MagicMock()
    bg_mgr.reserve_resume.return_value = True

    def _fake_error_payload(task_id: int, message: str, **kwargs: object) -> dict:
        return {"type": "agent_error", "message": message, "task_id": task_id}

    return mgr, ws_manager, bg_mgr, _fake_error_payload


@pytest.mark.asyncio
async def test_runtime_error_broadcast_is_redacted_but_the_sender_keeps_the_detail(
    _test_db: None,
) -> None:
    """The audience boundary of the #1479 passthrough (maintainer ruling).

    The initiating sender keeps ``Runtime error: ...`` in the rejection ack -
    that is the existing contract and #1479 owns narrowing it. The task-wide
    broadcast reaches every subscriber, anonymous widget/share connections
    included, so it must carry the fixed string and never the exception text.
    """
    db = _direct_db_session()
    try:
        owner = User(username="runtime-boundary-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Audience boundary",
            description="runtime branch",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task.runner_id = "rt6-runner"
        task.run_id = "rt6-run"
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    raised = RuntimeError(f"durable object scope={SECRET}")
    mgr, ws_manager, bg_mgr, fake_payload = _chat_runtime_error_harness(raised)

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch(
            "xagent.web.api.websocket.mark_user_message_delivery_sync",
            MagicMock(),
        ),
        patch(
            "xagent.web.api.websocket._read_task_error_payload_isolated",
            MagicMock(side_effect=fake_payload),
        ),
    ):
        await websocket_api._handle_chat_message_unserialized(
            MagicMock(),
            task_id,
            {
                "message": "trip the runtime branch",
                "client_message_id": "runtime-boundary",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [],
            },
        )

    broadcast = [c.args[0] for c in ws_manager.broadcast_to_task.await_args_list]
    assert broadcast, "the task-wide notification must still go out"
    assert SECRET not in repr(broadcast), broadcast
    task_errors = [b for b in broadcast if b.get("type") == "agent_error"]
    assert task_errors and task_errors[0]["message"] == (
        websocket_api.CLIENT_SAFE_TASK_FAILURE
    )

    personal = [c.args[0] for c in ws_manager.send_personal_message.await_args_list]
    rejected = [p for p in personal if p.get("type") == "message_rejected"]
    assert rejected, "the sender still gets the rejection ack"
    assert rejected[0]["message"] == f"Runtime error: {raised}", (
        "the sender's copy is the existing #1479 contract and must survive"
    )


@pytest.mark.asyncio
async def test_execute_runtime_error_broadcast_is_redacted(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same audience boundary through handle_execute_task's runtime branch."""
    db = _direct_db_session()
    try:
        owner = User(username="exec-runtime-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Exec audience boundary",
            description="runtime branch",
            status=TaskStatus.PENDING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    monkeypatch.setattr(
        websocket_api,
        "_read_task_error_payload_isolated",
        MagicMock(
            side_effect=lambda task_id, message, **kwargs: {
                "type": "agent_error",
                "message": message,
                "task_id": task_id,
            }
        ),
    )
    monkeypatch.setattr(
        TaskTurnOrchestrator,
        "schedule_existing_task_execution",
        AsyncMock(side_effect=RuntimeError(f"storage prefix {SECRET}")),
    )

    await websocket_api.handle_execute_task(
        MagicMock(),
        task_id,
        {"user": SimpleNamespace(id=owner_id, is_admin=False)},
    )

    broadcast = [
        c.args[0] for c in connection_manager.broadcast_to_task.await_args_list
    ]
    assert broadcast, "the task-wide notification must still go out"
    assert SECRET not in repr(broadcast), broadcast
    assert any(
        b.get("message") == websocket_api.CLIENT_SAFE_TASK_FAILURE for b in broadcast
    )
