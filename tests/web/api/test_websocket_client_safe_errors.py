"""Exception text must not reach chat clients (PR #1472 review finding N3)."""

from __future__ import annotations

import ast
import asyncio
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
# tracked in #1479 - not a hole to be quietly widened. Any *other* raw
# expression reaching a producer must fail this test.
ALLOWED_RAW_MESSAGE_SOURCES = {"f'Runtime error: {str(e)}'"}  # ast.unparse form


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


def _error_payload_message(node: ast.Call) -> ast.expr | None:
    """The message of an ``{"type": "error", ...}`` payload, if this is one."""
    if _called_name(node) not in {"send_personal_message", "broadcast_to_task"}:
        return None
    for argument in (*node.args, *(kw.value for kw in node.keywords)):
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
    source = Path(websocket_api.__file__).read_text()
    tree = ast.parse(source)
    parents = _parents(tree)

    checked = 0
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name in PRODUCERS:
            expr = _message_expression(node, PRODUCERS[name])
        else:
            # The error bubble renders in the same conversation as the
            # rejection ack, so it is the same disclosure surface.
            expr = _error_payload_message(node)
            name = f"{name}(error payload)"
        if expr is None:
            continue
        checked += 1
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
        for candidate in candidates:
            if _is_client_safe(candidate):
                continue
            if ast.unparse(candidate) in ALLOWED_RAW_MESSAGE_SOURCES:
                continue
            offenders.append(
                f"{name}:{node.lineno} may send {ast.unparse(candidate)!r}"
            )

    assert checked >= 20, f"the producers moved; only {checked} call sites matched"
    assert not offenders, (
        "raw text can reach a chat client; route it through "
        "client_safe_error_message: " + "; ".join(sorted(set(offenders)))
    )
