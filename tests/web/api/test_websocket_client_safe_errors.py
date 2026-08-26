"""Exception text must not reach chat clients (PR #1472 review finding N3)."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, NamedTuple
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
READABLE_SCOPE_NODES = (*FUNCTION_NODES, ast.Lambda)

# arg name -> positional index of the client-visible message
PRODUCERS: dict[str, int | None] = {
    "finish_delivery_failure": 0,
    "finish_delivery": 1,
    "notify_deferred_delivery": 1,
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
        if isinstance(current, READABLE_SCOPE_NODES):
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
ERROR_PAYLOAD_TYPES = {"error", "agent_error", "task_error"}
SENSITIVE_PAYLOAD_FIELDS = {"type", "message", "error"}
NON_ERROR_STREAM_EVENT_BUILDERS = {
    "_agent_outbound_event_type": None,
    "_waiting_or_paused_event_fields": 0,
}
DICT_ERROR_PAYLOAD_BUILDERS = {
    "_read_task_error_payload_offloop": "error",
    "_task_error_payload": "error",
    "_terminal_task_error_payload": "agent_error",
    "create_terminal_task_error_event": "task_error",
}

# The only functions allowed to mint client-visible text from an exception.
SAFE_MESSAGE_BUILDERS = {
    "client_safe_error_message",
    "client_safe_task_command_failure",
}
SAFE_MESSAGE_CONSTANTS = {
    "CLIENT_SAFE_TASK_FAILURE",
    "CLIENT_SAFE_VALIDATION_ERROR",
}


def _unwrap_serializer(expr: ast.expr) -> ast.expr:
    """``json.dumps(payload)`` -> ``payload``; anything else unchanged."""
    if isinstance(expr, ast.Call) and _called_name(expr) == "dumps" and expr.args:
        return expr.args[0]
    return expr


def _call_argument(
    node: ast.Call,
    position: int,
    keyword: str,
) -> ast.expr | None:
    """Resolve one argument passed either positionally or by exact keyword."""
    if len(node.args) > position:
        return node.args[position]
    return next(
        (candidate.value for candidate in node.keywords if candidate.arg == keyword),
        None,
    )


def _dict_variants(
    expr: ast.expr,
    scopes: list[ast.AST],
    module_helpers: set[str],
    resolving: frozenset[str] = frozenset(),
) -> list[tuple[dict[str, ast.expr], set[str]]]:
    """Resolve possible effective fields from dict and local-name spreads."""
    if isinstance(expr, ast.Await):
        return _dict_variants(expr.value, scopes, module_helpers, resolving)
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and (helper := expr.func.id) in DICT_ERROR_PAYLOAD_BUILDERS
        and helper in module_helpers
        and not _has_local_binding(scopes, helper)
    ):
        message_position = 2 if helper == "_task_error_payload" else 1
        message = _call_argument(expr, message_position, "message")
        if message is None:
            return [({}, {"type", "message", "error"})]
        event_type = next(
            (keyword.value for keyword in expr.keywords if keyword.arg == "event_type"),
            ast.Constant(DICT_ERROR_PAYLOAD_BUILDERS[helper]),
        )
        fields = {"type": event_type, "message": message}
        if helper == "create_terminal_task_error_event":
            fields["error"] = message
        unresolved = set() if isinstance(event_type, ast.Constant) else {"type"}
        return [(fields, unresolved)]
    if isinstance(expr, ast.Name):
        if expr.id in resolving:
            return [({}, SENSITIVE_PAYLOAD_FIELDS.copy())]
        assignments = _local_assignments(scopes, expr.id)
        if not assignments:
            return [({}, SENSITIVE_PAYLOAD_FIELDS.copy())]
        return [
            variant
            for assignment in assignments
            for variant in _dict_variants(
                assignment,
                scopes,
                module_helpers,
                resolving | {expr.id},
            )
        ]
    if not isinstance(expr, ast.Dict):
        return [({}, SENSITIVE_PAYLOAD_FIELDS.copy())]

    variants: list[tuple[dict[str, ast.expr], set[str]]] = [({}, set())]
    for key, value in zip(expr.keys, expr.values):
        if key is None:
            spread_variants = _dict_variants(value, scopes, module_helpers, resolving)
            merged_variants = []
            for fields, unresolved_fields in variants:
                for spread_fields, spread_unresolved in spread_variants:
                    merged_fields = fields.copy()
                    merged_unresolved = unresolved_fields.copy()
                    for field in spread_unresolved:
                        merged_fields.pop(field, None)
                    merged_unresolved.update(spread_unresolved)
                    merged_unresolved.difference_update(
                        spread_fields.keys() - spread_unresolved
                    )
                    merged_fields.update(spread_fields)
                    merged_variants.append((merged_fields, merged_unresolved))
            variants = merged_variants
        elif isinstance(key, ast.Constant) and isinstance(key.value, str):
            for fields, unresolved_fields in variants:
                fields[key.value] = value
                unresolved_fields.discard(key.value)
                if key.value == "type" and not isinstance(value, ast.Constant):
                    unresolved_fields.add("type")
        elif not isinstance(key, ast.Constant):
            for fields, unresolved_fields in variants:
                for field in SENSITIVE_PAYLOAD_FIELDS:
                    fields.pop(field, None)
                unresolved_fields.update(SENSITIVE_PAYLOAD_FIELDS)
    return variants


def _error_payload_messages(
    node: ast.Call,
    parents: dict[ast.AST, ast.AST],
    module_helpers: set[str],
) -> list[ast.expr]:
    """Client-visible text fields of a recognized error payload."""
    if _called_name(node) not in ERROR_PAYLOAD_SINKS:
        return []
    messages: list[ast.expr] = []
    scopes = _enclosing_functions(node, parents)
    for raw in (*node.args, *(kw.value for kw in node.keywords)):
        argument = _unwrap_serializer(raw)
        helper_builds_error = False
        if isinstance(argument, ast.Call):
            helper = argument.func.id if isinstance(argument.func, ast.Name) else None
            if helper == "create_terminal_task_error_event":
                if helper not in module_helpers or _has_local_binding(scopes, helper):
                    messages.append(argument)
                    continue
                message = _call_argument(argument, 1, "message")
                # A recognized error helper must never disappear from the
                # sweep merely because its call shape cannot be resolved.
                messages.append(message if message is not None else argument)
                continue
            if helper == "create_stream_event":
                if helper not in module_helpers or _has_local_binding(scopes, helper):
                    messages.append(argument)
                    continue
                event_type = _call_argument(argument, 0, "event_type")
                if isinstance(event_type, ast.Constant):
                    if event_type.value not in ERROR_PAYLOAD_TYPES:
                        # A literal non-error stream event is outside this sink.
                        continue
                elif _is_known_non_error_event_type(
                    event_type,
                    scopes,
                    module_helpers,
                ):
                    # Known non-error stream events are not error payloads.
                    continue
                else:
                    messages.append(argument)
                    continue
                data = _call_argument(argument, 2, "data")
                if data is None:
                    messages.append(argument)
                    continue
                if not isinstance(data, ast.Dict):
                    messages.append(data)
                    continue
                argument = data
                helper_builds_error = True
        if not isinstance(argument, ast.Dict):
            continue
        for keys, unresolved_fields in _dict_variants(
            argument,
            scopes,
            module_helpers,
        ):
            kind = keys.get("type")
            is_error_payload = not _is_known_non_error_event_type(
                kind,
                scopes,
                module_helpers,
            ) and (
                (isinstance(kind, ast.Constant) and kind.value in ERROR_PAYLOAD_TYPES)
                or helper_builds_error
                or "type" in unresolved_fields
            )
            if is_error_payload:
                messages.extend(
                    value
                    for field in ("message", "error")
                    if (value := keys.get(field)) is not None
                )
                if unresolved_fields.intersection({"message", "error"}):
                    messages.append(argument)
    return messages


def _called_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_parameter(scopes: list[ast.AST], name: str) -> bool:
    """A forwarded parameter is vetted at the wrapper's own call sites."""
    for scope in scopes:
        if not isinstance(scope, READABLE_SCOPE_NODES):
            continue
        arguments = scope.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            arguments.vararg,
            arguments.kwarg,
        ):
            if argument is not None and argument.arg == name:
                return True
    return False


def _local_assignments(scopes: list[ast.AST], name: str) -> list[ast.expr]:
    """Values assigned to `name` inside the given scopes only."""
    values: list[ast.expr] = []
    for scope in _readable_local_scopes(scopes, name):
        for node in _scope_nodes(scope):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        values.append(node.value)
    return values


def _readable_local_scopes(scopes: list[ast.AST], name: str) -> list[ast.AST]:
    """Scopes whose locals remain visible before a ``global`` declaration."""
    readable: list[ast.AST] = []
    for scope in scopes:
        if any(
            isinstance(node, ast.Global) and name in node.names
            for node in _scope_nodes(scope)
        ):
            return readable
        readable.append(scope)
    return readable


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Walk one function scope without borrowing bindings from child scopes."""
    pending = (
        [scope.body]
        if isinstance(scope, ast.Lambda)
        else list(getattr(scope, "body", ()))
    )
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (*READABLE_SCOPE_NODES, ast.ClassDef)):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _binds_name_without_store(node: ast.AST, name: str) -> bool:
    if isinstance(
        node,
        (*FUNCTION_NODES, ast.ClassDef, ast.ExceptHandler, ast.MatchAs, ast.MatchStar),
    ):
        return node.name == name
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return any(
            (alias.asname or alias.name.split(".")[0]) == name for alias in node.names
        )
    return isinstance(node, ast.MatchMapping) and node.rest == name


def _has_local_binding(scopes: list[ast.AST], name: str) -> bool:
    """Whether a trusted module helper name is shadowed in a readable scope."""
    scopes = _readable_local_scopes(scopes, name)
    if _is_parameter(scopes, name):
        return True
    for scope in scopes:
        for node in _scope_nodes(scope):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id == name
            ):
                return True
            if _binds_name_without_store(node, name):
                return True
    return False


def _is_known_non_error_event_type(
    expr: ast.expr | None,
    scopes: list[ast.AST],
    module_builders: set[str],
) -> bool:
    """Recognize only module helpers that return a non-error event type."""

    def trusted_builder(candidate: ast.expr, result_index: int | None) -> bool:
        return (
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id in module_builders
            and NON_ERROR_STREAM_EVENT_BUILDERS[candidate.func.id] == result_index
            and not _has_local_binding(scopes, candidate.func.id)
        )

    if not isinstance(expr, ast.Name):
        return isinstance(expr, ast.expr) and trusted_builder(expr, None)
    scopes = _readable_local_scopes(scopes, expr.id)
    if not scopes:
        return False
    if _is_parameter(scopes, expr.id):
        return False

    bindings: list[tuple[ast.expr, int | None]] = []
    assignment_stores: set[int] = set()
    nodes = [node for scope in scopes for node in _scope_nodes(scope)]
    for node in nodes:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == expr.id:
                assignment_stores.add(id(target))
                bindings.append((node.value, None))
            elif isinstance(target, ast.Tuple):
                for index, element in enumerate(target.elts):
                    if isinstance(element, ast.Name) and element.id == expr.id:
                        assignment_stores.add(id(element))
                        bindings.append((node.value, index))

    if any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == expr.id
        and id(node) not in assignment_stores
        for node in nodes
    ):
        return False
    if any(_binds_name_without_store(node, expr.id) for node in nodes):
        return False
    return bool(bindings) and all(
        trusted_builder(value, result_index) for value, result_index in bindings
    )


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


def _trusted_module_helpers(tree: ast.Module) -> set[str]:
    """Helpers with one real module implementation and no other binding."""
    expected = {
        *DICT_ERROR_PAYLOAD_BUILDERS,
        *NON_ERROR_STREAM_EVENT_BUILDERS,
        "create_stream_event",
    }
    trusted: set[str] = set()
    nodes = list(_scope_nodes(tree))
    overload_imports = [
        (node, alias)
        for node in nodes
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if (alias.asname or alias.name.split(".")[0]) == "overload"
    ]
    canonical_overload = (
        len(overload_imports) == 1
        and isinstance(overload_imports[0][0], ast.ImportFrom)
        and overload_imports[0][0].level == 0
        and overload_imports[0][0].module == "typing"
        and overload_imports[0][1].name == "overload"
        and overload_imports[0][1].asname in {None, "overload"}
        and not any(
            id(node) != id(overload_imports[0][0])
            and (
                _binds_name_without_store(node, "overload")
                or (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Store)
                    and node.id == "overload"
                )
            )
            for node in nodes
        )
    )
    for name in expected:
        definitions = [
            node
            for node in nodes
            if isinstance(node, FUNCTION_NODES) and node.name == name
        ]
        overload_stubs = [
            node
            for node in definitions
            if canonical_overload
            and len(node.decorator_list) == 1
            and isinstance(node.decorator_list[0], ast.Name)
            and node.decorator_list[0].id == "overload"
        ]
        implementations = [node for node in definitions if not node.decorator_list]
        unsafe_decorated = len(definitions) != len(overload_stubs) + len(
            implementations
        )
        definition_ids = {id(node) for node in definitions}
        has_other_binding = any(
            id(node) not in definition_ids
            and (
                _binds_name_without_store(node, name)
                or (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Store)
                    and node.id == name
                )
            )
            for node in nodes
        )
        if len(implementations) == 1 and not unsafe_decorated and not has_other_binding:
            trusted.add(name)
    return trusted


def _scan(tree: ast.Module) -> _ScanResult:
    """The one copy of the sweep's recognition logic.

    Both the production sweep and the snippet-based regression tests run
    this same function, so a change to the analysis cannot pass the snippet
    tests while silently not applying to the real module (or vice versa).
    """
    parents = _parents(tree)
    imported_safe_constants = {
        alias.name
        for statement in tree.body
        if isinstance(statement, ast.ImportFrom)
        and statement.level == 2
        and statement.module == "services.client_error_messages"
        for alias in statement.names
        if alias.asname is None and alias.name in SAFE_MESSAGE_CONSTANTS
    }
    module_helpers = _trusted_module_helpers(tree)
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
            message = _message_expression(node, PRODUCERS[name])
            expressions = [message] if message is not None else []
        else:
            # The error bubble renders in the same conversation as the
            # rejection ack, so it is the same disclosure surface.
            expressions = _error_payload_messages(
                node,
                parents,
                module_helpers,
            )
            name = f"{name}(error payload)"
        if not expressions:
            continue
        if is_producer:
            producers += 1
        else:
            error_payloads += len(expressions)
        scopes = _enclosing_functions(node, parents)
        for expr in expressions:
            candidates = (
                _local_assignments(scopes, expr.id)
                if isinstance(expr, ast.Name)
                else [expr]
            )
            is_unshadowed_safe_constant = (
                isinstance(expr, ast.Name)
                and expr.id in imported_safe_constants
                and not candidates
                and not _is_parameter(scopes, expr.id)
            )
            if is_unshadowed_safe_constant or _is_client_safe(expr):
                continue
            if not candidates:
                # Only a name nothing rebinds is a genuinely forwarded parameter,
                # vetted at the wrapper's own call sites. Known client-facing
                # wrappers are also scanned as producers at their call sites.
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

    Scope, stated honestly: this walks the direct producers and error payload
    sinks used by this module. It understands the task-error and stream-event
    helpers, explicit overrides on dict-spread payloads, and the listed
    deferred-delivery wrapper.

    It is not general interprocedural data-flow analysis. Dynamic payload types
    fail closed unless they come from the listed module helpers, and the type
    set is maintained rather than derived (#1547). Do not read a passing run as
    proof that arbitrary Python data flow cannot reach a client.
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


# Each entry mirrors an egress shape fixed for #1696: dict-spread copies
# ``execute_task_background`` (text under ``error``, type inherited from the
# spread), helper-built copies ``send_historical_data_as_stream``, and
# wrapper-forwarded copies ``notify_deferred_delivery``. These regression
# fixtures ensure the static guard continues to reject all three.
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
        terminal_payload = _terminal_task_error_payload(task_id, str(e))
        await manager.broadcast_to_task(
            {
                **terminal_payload,
                "message": str(e),
            },
            task_id,
        )
""",
        id="dict-spread-message",
    ),
    pytest.param(
        """
async def leak(websocket, task_id):
    try:
        pass
    except Exception as e:
        CLIENT_SAFE_TASK_FAILURE = str(e)
        await manager.broadcast_to_task(
            {
                "type": "task_error",
                "message": CLIENT_SAFE_TASK_FAILURE,
            },
            task_id,
        )
""",
        id="shadowed-safe-message-constant",
    ),
    pytest.param(
        """
from untrusted import raw_message as CLIENT_SAFE_TASK_FAILURE


async def leak(websocket, task_id):
    await manager.broadcast_to_task(
        {
            "type": "task_error",
            "message": CLIENT_SAFE_TASK_FAILURE,
        },
        task_id,
    )
""",
        id="safe-message-constant-wrong-import",
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
async def leak(websocket, task_id):
    try:
        pass
    except Exception as e:
        await manager.broadcast_to_task(
            create_terminal_task_error_event(
                task_id=task_id,
                message=str(e),
            ),
            task_id,
        )
""",
        id="terminal-helper-keywords",
    ),
    pytest.param(
        """
async def leak(websocket, task_id):
    try:
        pass
    except Exception as e:
        await manager.send_personal_message(
            create_stream_event(
                event_type="error",
                task_id=task_id,
                data={"message": str(e)},
            ),
            websocket,
        )
""",
        id="stream-helper-keywords",
    ),
    pytest.param(
        """
async def leak(websocket, task_id, event_type):
    try:
        pass
    except Exception as e:
        await manager.send_personal_message(
            create_stream_event(
                event_type=event_type,
                task_id=task_id,
                data={"message": str(e)},
            ),
            websocket,
        )
""",
        id="stream-helper-dynamic-event-type",
    ),
    pytest.param(
        """
async def leak(websocket, task_id):
    try:
        pass
    except Exception as e:
        await manager.send_personal_message(
            create_stream_event(
                task_id=task_id,
                data={"message": str(e)},
            ),
            websocket,
        )
""",
        id="stream-helper-missing-event-type",
    ),
    pytest.param(
        """
async def leak(websocket, task_id):
    try:
        pass
    except Exception as e:
        await manager.send_personal_message(
            create_stream_event(
                event_type=untrusted._agent_outbound_event_type(task_id),
                task_id=task_id,
                data={"message": str(e)},
            ),
            websocket,
        )
""",
        id="stream-helper-non-error-builder-impostor",
    ),
    pytest.param(
        """
def _agent_outbound_event_type(payload):
    return "agent_progress"


async def leak(websocket, task_id, _agent_outbound_event_type):
    try:
        pass
    except Exception as e:
        await manager.send_personal_message(
            create_stream_event(
                event_type=_agent_outbound_event_type(task_id),
                task_id=task_id,
                data={"message": str(e)},
            ),
            websocket,
        )
""",
        id="stream-helper-shadowed-non-error-builder",
    ),
    pytest.param(
        """
async def notify_deferred_delivery(accepted, raw):
    await send_message_delivery(
        object(),
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
        await notify_deferred_delivery(False, str(e))
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


def _broadcast_source(payload: str, args: str = "task_id", setup: str = "") -> str:
    """Build compact snippets while keeping bindings in the sink's scope."""
    return (
        f"async def leak({args}):\n"
        f"    {setup}await manager.broadcast_to_task({payload}, task_id)\n"
    )


def _match_capture_source(pattern: str) -> str:
    return f"""def _agent_outbound_event_type(payload):
    return "agent_progress"
async def leak(task_id, value):
    match value:
        case {pattern}:
            pass
    await manager.broadcast_to_task(create_stream_event(
        _agent_outbound_event_type(value), task_id,
        {{"message": str(RuntimeError("raw"))}}), task_id)
"""


RAW_ERROR = 'str(RuntimeError("raw"))'
SPREAD_PAYLOAD = '{**payload, "timestamp": 0}'

# Dense by design: these are data rows, not executable logic. Fields are
# id, payload, function arguments, setup statement(s), expected offenders.
# fmt: off
BROADCAST_GUARD_SPECS = [
    ("dynamic-explicit-type", f'{{"type": event_type, "message": {RAW_ERROR}}}', "task_id, event_type", "", True),
    ("dynamic-key-after-non-error-type", f'{{"type": "task_completed", key: {RAW_ERROR}}}', "task_id, key", "", True),
    ("explicit-fields-override-dynamic-key", f'{{key: {RAW_ERROR}, "type": "task_completed", "message": "safe", "error": "safe"}}', "task_id, key", "", False),
    ("local-message", SPREAD_PAYLOAD, "task_id", f'payload = {{"type": "task_error", "message": {RAW_ERROR}}}; ', True),
    ("local-error", SPREAD_PAYLOAD, "task_id", f'payload = {{"type": "task_error", "error": {RAW_ERROR}}}; ', True),
    ("nested-direct", f'{{**{{**{{"type": "task_error", "message": {RAW_ERROR}}}}}, "timestamp": 0}}', "task_id", "", True),
    ("recognized-helper", SPREAD_PAYLOAD, "task_id", f"payload = create_terminal_task_error_event(task_id, {RAW_ERROR}); ", True),
    ("unresolved", '{**build_untrusted_payload(), "timestamp": 0}', "task_id", "", True),
    ("unresolved-after-error-type", '{"type": "task_error", **build_untrusted_payload()}', "task_id", "", True),
    ("dynamic-helper-type", SPREAD_PAYLOAD, "task_id, event_type", f"payload = await _read_task_error_payload_offloop(task_id, {RAW_ERROR}, event_type=event_type); ", True),
    ("safe-known-helper", SPREAD_PAYLOAD, "task_id", 'safe = client_safe_error_message(RuntimeError("detail")); payload = await _read_task_error_payload_offloop(task_id, safe); ', False),
    ("explicit-final-non-error-type", '{**build_control_state(), "type": "task_completed"}', "task_id", "", False),
]
# fmt: on

DICT_SPREAD_GUARD_CASES = {
    case: (_broadcast_source(payload, args, setup), expected)
    for case, payload, args, setup, expected in BROADCAST_GUARD_SPECS
}
safe_helper_source, _ = DICT_SPREAD_GUARD_CASES["safe-known-helper"]
DICT_SPREAD_GUARD_CASES["safe-known-helper"] = (
    """async def _read_task_error_payload_offloop(task_id, message):
    return {"type": "task_error", "message": message}
"""
    + safe_helper_source,
    False,
)
DICT_SPREAD_GUARD_CASES.update(
    {
        "local-function-helper-impostor": (
            """async def leak(task_id):
    def _read_task_error_payload_offloop(_task_id, _message):
        return {"type": "task_error", "message": str(RuntimeError("raw"))}
    payload = _read_task_error_payload_offloop(task_id, "looks safe")
    await manager.broadcast_to_task({**payload, "timestamp": 0}, task_id)
""",
            True,
        ),
        "local-class-terminal-helper-impostor": (
            """async def leak(task_id):
    class create_terminal_task_error_event:
        pass
    await manager.broadcast_to_task(
        create_terminal_task_error_event(task_id, "looks safe"), task_id)
""",
            True,
        ),
        "local-import-stream-helper-impostor": (
            _broadcast_source(
                'create_stream_event("task_completed", task_id, {"message": "safe"})',
                setup="from untrusted import create_stream_event; ",
            ),
            True,
        ),
        "trusted-helper-then-rebound": (
            """def _waiting_or_paused_event_fields(status):
    return "task_paused", "safe"
async def leak(task_id, status, raw_type):
    event_type, message = _waiting_or_paused_event_fields(status)
    event_type = raw_type
    await manager.broadcast_to_task(
        {"type": event_type, "message": str(RuntimeError("raw"))}, task_id)
""",
            True,
        ),
        "nested-safe-assignment-does-not-hide-global-payload": (
            """payload = {"type": "task_error", "message": str(RuntimeError("raw"))}
async def leak(task_id):
    def unrelated():
        payload = {"type": "task_completed", "message": "safe"}
    await manager.broadcast_to_task({**payload}, task_id)
""",
            True,
        ),
        "module-assignment-shadows-spread-helper": (
            """def _read_task_error_payload_offloop(task_id, message):
    return {"type": "task_error", "message": message}
_read_task_error_payload_offloop = untrusted
async def leak(task_id):
    await manager.broadcast_to_task(
        {**_read_task_error_payload_offloop(task_id, "safe")}, task_id)
""",
            True,
        ),
        "module-import-shadows-direct-helper": (
            """def create_stream_event(event_type, task_id, data):
    return {"type": event_type, **data}
from untrusted import create_stream_event
async def leak(task_id):
    await manager.broadcast_to_task(
        create_stream_event("task_completed", task_id, {"message": "safe"}), task_id)
""",
            True,
        ),
        "module-redefinition-shadows-non-error-helper": (
            """def create_stream_event(event_type, task_id, data):
    return {"type": event_type, **data}
def _agent_outbound_event_type(payload):
    return "task_completed"
def _agent_outbound_event_type(payload):
    return "error"
async def leak(task_id, payload):
    await manager.broadcast_to_task(create_stream_event(
        _agent_outbound_event_type(payload), task_id,
        {"message": str(RuntimeError("raw"))}), task_id)
""",
            True,
        ),
        "decorated-direct-helper-is-not-canonical": (
            """def replace(helper):
    return untrusted
@replace
def create_stream_event(event_type, task_id, data):
    return {"type": event_type, **data}
async def leak(task_id):
    await manager.broadcast_to_task(
        create_stream_event("task_completed", task_id, {"message": "safe"}), task_id)
""",
            True,
        ),
        "decorated-non-error-helper-is-not-canonical": (
            """def replace(helper):
    return untrusted
def create_stream_event(event_type, task_id, data):
    return {"type": event_type, **data}
@replace
def _agent_outbound_event_type(payload):
    return "task_completed"
async def leak(task_id, payload):
    await manager.broadcast_to_task(create_stream_event(
        _agent_outbound_event_type(payload), task_id,
        {"message": str(RuntimeError("raw"))}), task_id)
""",
            True,
        ),
        "same-import-overwrites-overload": (
            """from typing import overload, no_type_check as overload
@overload
def _read_task_error_payload_offloop(task_id, message): ...
def _read_task_error_payload_offloop(task_id, message):
    return {"type": "task_error", "message": message}
async def leak(task_id):
    await manager.broadcast_to_task(
        {**_read_task_error_payload_offloop(task_id, "safe")}, task_id)
""",
            True,
        ),
        "relative-typing-overload-is-not-canonical": (
            """from .typing import overload
@overload
def _read_task_error_payload_offloop(task_id, message): ...
def _read_task_error_payload_offloop(task_id, message):
    return {"type": "task_error", "message": message}
async def leak(task_id):
    await manager.broadcast_to_task(
        {**_read_task_error_payload_offloop(task_id, "safe")}, task_id)
""",
            True,
        ),
        "global-payload-cuts-off-outer-safe-binding": (
            """payload = {"type": "task_error", "message": str(RuntimeError("raw"))}
async def outer(task_id):
    payload = {"type": "task_completed", "message": "safe"}
    async def leak():
        global payload
        await manager.broadcast_to_task({**payload}, task_id)
""",
            True,
        ),
        "global-event-type-cuts-off-outer-trusted-binding": (
            """def _waiting_or_paused_event_fields(status):
    return "task_paused", "safe"
async def outer(task_id, status):
    event_type, message = _waiting_or_paused_event_fields(status)
    async def leak():
        global event_type
        await manager.broadcast_to_task(
            {"type": event_type, "message": str(RuntimeError("raw"))}, task_id)
""",
            True,
        ),
        "outer-global-keeps-inner-helper-shadow": (
            """async def _read_task_error_payload_offloop(task_id, message):
    return {"type": "task_error", "message": message}
async def outer(task_id):
    global _read_task_error_payload_offloop
    async def leak():
        def _read_task_error_payload_offloop(_task_id, _message):
            return {"type": "task_error", "message": str(RuntimeError("raw"))}
        await manager.broadcast_to_task(
            {**_read_task_error_payload_offloop(task_id, "safe")}, task_id)
""",
            True,
        ),
        **{
            f"{kind}-helper-parameter-impostor": (
                source,
                True,
            )
            for kind, source in {
                "lambda": """leak = lambda task_id, _read_task_error_payload_offloop: manager.broadcast_to_task(
    {**_read_task_error_payload_offloop(task_id, "safe")}, task_id)
""",
                "vararg": """async def leak(task_id, *_read_task_error_payload_offloop):
    await manager.broadcast_to_task(
        {**_read_task_error_payload_offloop(task_id, "safe")}, task_id)
""",
                "kwarg": """async def leak(task_id, **_read_task_error_payload_offloop):
    await manager.broadcast_to_task(
        {**_read_task_error_payload_offloop(task_id, "safe")}, task_id)
""",
            }.items()
        },
        **{
            f"{capture}-capture-shadows-non-error-builder": (
                _match_capture_source(pattern),
                True,
            )
            for capture, pattern in (
                ("match-as", '{"kind": _agent_outbound_event_type}'),
                ("match-star", "[*_agent_outbound_event_type]"),
                ("match-rest", '{"kind": kind, **_agent_outbound_event_type}'),
            )
        },
    }
)


@pytest.mark.parametrize(
    ("source", "has_offenders"),
    DICT_SPREAD_GUARD_CASES.values(),
    ids=DICT_SPREAD_GUARD_CASES,
)
def test_dict_spread_guard_cases(source: str, has_offenders: bool) -> None:
    """Spread resolution rejects raw/opaque error shapes without false positives."""
    assert bool(_guard_offenders(source)) is has_offenders


@pytest.mark.parametrize("source", BYPASS_SHAPES)
def test_known_bypass_shapes_are_rejected_by_the_guard(source: str) -> None:
    """The guard rejects every producer shape fixed for #1696."""
    assert _guard_offenders(source), (
        "the guard missed a client-facing raw exception shape fixed for #1696"
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


# --- Round 6: the durable command origin registry ---------------------------
#
# Personal detail from durable execution goes to the exact socket that
# submitted the command, verified to still be connected to that task - or
# nowhere. Origin is never inferred from task membership, actor id, or
# connection order.


@pytest.fixture()
def _clean_origins() -> Iterator[None]:
    saved = dict(websocket_api._command_origins._origins)
    websocket_api._command_origins._origins.clear()
    yield
    websocket_api._command_origins._origins.clear()
    websocket_api._command_origins._origins.update(saved)


def _pause_command(
    task_id: int = 7, command_id: str = "pause:origin"
) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        task_id=task_id,
        actor_user_id=1,
        command_id=command_id,
        kind=websocket_api.TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
        target_run_id=None,
        attempt_count=1,
        failure_count=0,
        defer_count=0,
    )


def _origin_test_manager(registered: set) -> MagicMock:
    """A manager mock whose registration check is membership in ``registered``."""
    m = MagicMock()
    m.send_personal_message = AsyncMock()
    m.broadcast_to_task = AsyncMock()
    m.is_connection_registered = MagicMock(
        side_effect=lambda ws, task_id: ws in registered
    )
    return m


async def _run_pause_to_runtime_error(
    monkeypatch: pytest.MonkeyPatch, manager_mock: MagicMock, command
) -> None:
    monkeypatch.setattr(websocket_api, "manager", manager_mock)
    monkeypatch.setattr(
        websocket_api,
        "_load_command_actor",
        lambda actor_user_id: SimpleNamespace(id=actor_user_id or 1, is_admin=False),
    )
    monkeypatch.setattr(
        websocket_api, "task_has_live_foreign_runner", lambda task_id: False
    )
    import xagent.web.services.task_setup_snapshot as snapshot_module

    monkeypatch.setattr(
        snapshot_module,
        "load_task_setup_snapshot_sync",
        MagicMock(side_effect=RuntimeError(f"storage fault at {SECRET}")),
    )
    with pytest.raises(RuntimeError):
        await websocket_api._execute_durable_task_command(command)


def _personal_targets(manager_mock: MagicMock) -> list[tuple[dict, object]]:
    return [
        (c.args[0], c.args[1])
        for c in manager_mock.send_personal_message.await_args_list
        if c.args and isinstance(c.args[0], dict)
    ]


@pytest.mark.asyncio
async def test_durable_raw_detail_reaches_only_the_verified_origin(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    _clean_origins: None,
) -> None:
    """Reviewer-specified ordering: [public, authenticated-origin, broadcast-only].

    Before the registry, the executor picked the first ordinary socket, so
    the public visitor received the raw RuntimeError text. Now the raw
    detail goes to the registered origin regardless of order, and the
    public socket gets nothing personal.
    """
    public, owner_origin, sse = (
        MagicMock(name="public"),
        MagicMock(name="origin"),
        MagicMock(name="sse"),
    )
    sse.is_broadcast_only = True
    manager_mock = _origin_test_manager(registered={public, owner_origin, sse})
    command = _pause_command()
    websocket_api._command_origins.register(
        command.command_id, owner_origin, command.task_id
    )

    await _run_pause_to_runtime_error(monkeypatch, manager_mock, command)

    raw_sends = [
        (payload, ws)
        for payload, ws in _personal_targets(manager_mock)
        if SECRET in repr(payload)
    ]
    assert raw_sends, "the verified origin must still receive the detail"
    assert all(ws is owner_origin for _, ws in raw_sends), raw_sends
    assert not any(ws is public for _, ws in _personal_targets(manager_mock)), (
        "the public socket must receive nothing personal"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "degrade_case",
    ["no-registration", "origin-disconnected", "wrong-task"],
)
async def test_durable_raw_detail_degrades_when_origin_is_unverifiable(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    _clean_origins: None,
    degrade_case: str,
) -> None:
    """Worker restart/handoff, disconnect, and task mismatch all degrade safely.

    No registration entry (a different worker executed the command), an
    origin that has since disconnected, or an entry recorded for another
    task: in every case the raw text reaches no socket at all.
    """
    public, owner_origin = MagicMock(name="public"), MagicMock(name="origin")
    registered = {public, owner_origin}
    command = _pause_command()
    if degrade_case == "origin-disconnected":
        websocket_api._command_origins.register(
            command.command_id, owner_origin, command.task_id
        )
        registered = {public}
    elif degrade_case == "wrong-task":
        websocket_api._command_origins.register(
            command.command_id, owner_origin, command.task_id + 1
        )
    manager_mock = _origin_test_manager(registered=registered)

    await _run_pause_to_runtime_error(monkeypatch, manager_mock, command)

    # The handler still emits its personal reply, but the executor gave it a
    # discarding stub, so the raw text reaches no real socket. Anything else
    # as the target - the public socket in particular - is a rerouted leak.
    raw_targets = [
        ws for payload, ws in _personal_targets(manager_mock) if SECRET in repr(payload)
    ]
    assert all(
        isinstance(ws, websocket_api._DiscardingCommandWebSocket) for ws in raw_targets
    ), f"raw detail rerouted to a real socket: {raw_targets}"


@pytest.mark.asyncio
async def test_origin_entry_dies_with_its_command_or_socket(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    _clean_origins: None,
) -> None:
    """Lifecycle: terminal outcomes and disconnects both clear the entry."""
    origins = websocket_api._command_origins

    # Terminal rejection clears it.
    command = _pause_command(command_id="pause:cleanup")
    socket = MagicMock(name="origin")
    origins.register(command.command_id, socket, command.task_id)
    monkeypatch.setattr(
        websocket_api,
        "_execute_durable_task_command",
        AsyncMock(side_effect=websocket_api.TaskCommandRejected("done")),
    )
    with pytest.raises(websocket_api.TaskCommandRejected):
        await websocket_api.execute_durable_task_command(command)
    assert origins.resolve(command.command_id, command.task_id) is None
    assert not origins.has(command.command_id, command.task_id)

    # A deferral that will retry keeps it; exhaustion clears it.
    origins.register(command.command_id, socket, command.task_id)
    monkeypatch.setattr(
        websocket_api,
        "_execute_durable_task_command",
        AsyncMock(side_effect=websocket_api.ClientVisibleTaskCommandDeferred("wait")),
    )
    monkeypatch.setattr(websocket_api, "manager", _origin_test_manager({socket}))
    with pytest.raises(websocket_api.TaskCommandDeferred):
        await websocket_api.execute_durable_task_command(command)
    assert origins.has(command.command_id, command.task_id), (
        "retrying deferral keeps the origin"
    )
    exhausted = _pause_command(command_id="pause:cleanup")
    exhausted.defer_count = websocket_api.MAX_COMMAND_DEFERS
    with pytest.raises(websocket_api.TaskCommandDeferred):
        await websocket_api.execute_durable_task_command(exhausted)
    assert not origins.has(command.command_id, command.task_id)

    # Disconnect clears every entry for that socket.
    origins.register("a:1", socket, 7)
    origins.register("b:2", socket, 8)
    real_manager = websocket_api.ConnectionManager()
    real_manager.disconnect(socket)
    assert not origins.has("a:1", 7) and not origins.has("b:2", 8), (
        "disconnect must clear the socket's entries"
    )


@pytest.mark.asyncio
async def test_durable_chat_detail_reaches_the_verified_origin_only(
    _test_db: None,
) -> None:
    """G18: on the durable path the ack is suppressed, so the detail bubble
    to the verified origin is the sender's only copy - and the broadcast that
    everyone else sees stays generic."""
    db = _direct_db_session()
    try:
        owner = User(username="durable-detail-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Durable detail",
            description="runtime branch, durable path",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task.runner_id = "rt7-runner"
        task.run_id = "rt7-run"
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    raised = RuntimeError(f"durable object scope={SECRET}")
    mgr, ws_manager, bg_mgr, fake_payload = _chat_runtime_error_harness(raised)
    origin_socket = MagicMock(name="verified-origin")

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
            origin_socket,
            task_id,
            {
                "message": "durable runtime failure",
                "client_message_id": "durable-detail",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [],
                "_durable_ack_sent": True,
            },
        )

    broadcast = [c.args[0] for c in ws_manager.broadcast_to_task.await_args_list]
    assert broadcast and SECRET not in repr(broadcast), broadcast
    personal = [
        (c.args[0], c.args[1]) for c in ws_manager.send_personal_message.await_args_list
    ]
    # The suppressed ack means no message_rejected; the detail bubble is the
    # sender's only copy and goes to the socket the executor resolved.
    assert not any(p.get("type") == "message_rejected" for p, _ in personal)
    detail = [(p, ws) for p, ws in personal if SECRET in repr(p)]
    assert detail, "the verified origin must receive the detail bubble"
    assert all(ws is origin_socket for _, ws in detail), detail


@pytest.mark.asyncio
async def test_execute_detail_reaches_the_ingress_socket(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G18: legacy execute keeps its real ingress socket, so the sender gets
    the detail personally while the broadcast stays generic."""
    db = _direct_db_session()
    try:
        owner = User(username="exec-detail-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="Exec detail",
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

    ingress = MagicMock(name="ingress")
    await websocket_api.handle_execute_task(
        ingress,
        task_id,
        {"user": SimpleNamespace(id=owner_id, is_admin=False)},
    )

    broadcast = [
        c.args[0] for c in connection_manager.broadcast_to_task.await_args_list
    ]
    assert broadcast and SECRET not in repr(broadcast), broadcast
    detail = [
        (c.args[0], c.args[1])
        for c in connection_manager.send_personal_message.await_args_list
        if SECRET in repr(c.args[0])
    ]
    assert detail, "the ingress socket must receive the detail personally"
    assert all(ws is ingress for _, ws in detail), detail


@pytest.mark.asyncio
async def test_preview_unknown_message_answers_on_the_wire_without_echo() -> None:
    """G16: endpoint-level coverage of the unknown-message response.

    Feeds an unknown type through the real receive loop and decodes the
    actual send_text JSON, so a deleted branch, wrong wiring, or a
    reintroduced echo of the client's message type all fail here.
    """
    from fastapi import WebSocketDisconnect

    mock_websocket = AsyncMock()
    mock_websocket.state = MagicMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    hostile_type = f"nope-{SECRET}"
    mock_websocket.receive_text.side_effect = [
        json.dumps({"type": hostile_type}),
        WebSocketDisconnect(),
    ]

    with patch(
        "xagent.web.api.websocket.get_authenticated_user", return_value=mock_user
    ):
        await websocket_api.websocket_build_preview_endpoint(mock_websocket)

    sent = [json.loads(c.args[0]) for c in mock_websocket.send_text.call_args_list]
    errors = [p for p in sent if p.get("type") == "error"]
    assert errors == [{"type": "error", "message": "Unknown message type"}] or (
        len(errors) == 1 and errors[0]["message"] == "Unknown message type"
    ), errors
    assert hostile_type not in repr(sent), "the client's type must not echo back"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "extra"),
    [
        ("handle_chat_message", {"client_message_id": "reg-1"}),
        ("handle_pause_task", {}),
        ("handle_resume_task", {}),
    ],
    ids=["chat", "pause", "resume"],
)
async def test_ingress_handlers_register_the_command_origin(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    _clean_origins: None,
    handler_name: str,
    extra: dict,
) -> None:
    """The creating ingress binds; deleting the line degrades senders silently."""
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    connection_manager.is_connection_registered = MagicMock(return_value=True)
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    enqueued = SimpleNamespace(
        command_id=41,
        client_command_id="cmd:origin-reg",
        payload_matches=True,
        status="claimed",
        created=True,
    )
    monkeypatch.setattr(
        websocket_api,
        "_enqueue_websocket_task_command",
        AsyncMock(return_value=enqueued),
    )
    monkeypatch.setattr(websocket_api, "dispatch_task_command_promptly", AsyncMock())

    ingress = MagicMock(name="ingress")
    await getattr(websocket_api, handler_name)(
        ingress,
        7,
        {"user": SimpleNamespace(id=1, is_admin=False), **extra},
    )

    assert websocket_api._command_origins.has("cmd:origin-reg", 7), (
        f"{handler_name} must register its origin"
    )
    assert websocket_api._command_origins.resolve("cmd:origin-reg", 7) is ingress


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "extra"),
    [
        ("handle_chat_message", {"client_message_id": "dup-1"}),
        ("handle_pause_task", {}),
        ("handle_resume_task", {}),
    ],
    ids=["chat", "pause", "resume"],
)
async def test_a_duplicate_enqueue_never_binds_the_origin(
    _test_db: None,
    monkeypatch: pytest.MonkeyPatch,
    _clean_origins: None,
    handler_name: str,
    extra: dict,
) -> None:
    """A payload-matching duplicate (created=False) must not acquire origin.

    This is the P1 blocker: a duplicate - a co-tenant resubmission, one after
    the creator disconnected, or one handled on another worker - reaches these
    handlers with a valid command_id but created=False. It still dispatches
    (idempotent), but it must never register, or the durable executor could
    resolve it and send the creator's raw detail to the wrong socket.
    """
    connection_manager = MagicMock()
    connection_manager.send_personal_message = AsyncMock()
    connection_manager.broadcast_to_task = AsyncMock()
    connection_manager.is_connection_registered = MagicMock(return_value=True)
    monkeypatch.setattr(websocket_api, "manager", connection_manager)
    dispatch = AsyncMock()
    monkeypatch.setattr(websocket_api, "dispatch_task_command_promptly", dispatch)
    monkeypatch.setattr(
        websocket_api,
        "_enqueue_websocket_task_command",
        AsyncMock(
            return_value=SimpleNamespace(
                command_id=41,
                client_command_id="dup:cmd",
                payload_matches=True,
                status="claimed",
                created=False,
            )
        ),
    )

    duplicate = MagicMock(name="duplicate-ingress")
    await getattr(websocket_api, handler_name)(
        duplicate,
        7,
        {"user": SimpleNamespace(id=1, is_admin=False), **extra},
    )

    assert not websocket_api._command_origins.has("dup:cmd", 7), (
        "a duplicate must never bind the origin"
    )
    assert dispatch.await_count == 1, "the duplicate still dispatches idempotently"


def test_a_resubmitted_command_id_cannot_capture_another_senders_origin(
    _clean_origins: None,
) -> None:
    """First registration wins (preflight PoC: co-tenant origin hijack).

    On a public/share task every visitor carries the owner principal, so the
    enqueue dedupe returns the in-flight row for a resubmission of the same
    command_id and the second connection would otherwise reach `register` and
    overwrite the origin - redirecting the first sender's error detail to the
    attacker. The registry must keep the original.
    """
    origins = websocket_api._command_origins
    victim, attacker = MagicMock(name="victim"), MagicMock(name="attacker")

    origins.register("shared-cmd", victim, 7)
    origins.register("shared-cmd", attacker, 7)  # resubmission on the same task

    with patch.object(
        websocket_api.manager, "is_connection_registered", return_value=True
    ):
        assert origins.resolve("shared-cmd", 7) is victim, (
            "the attacker must not capture the victim's origin"
        )

    # Re-registering the same socket stays idempotent.
    origins.register("shared-cmd", victim, 7)
    with patch.object(
        websocket_api.manager, "is_connection_registered", return_value=True
    ):
        assert origins.resolve("shared-cmd", 7) is victim


def test_same_command_id_on_two_tasks_is_isolated(_clean_origins: None) -> None:
    """command_id is unique only per task, so the key is (task_id, command_id).

    A shared id must not let one task's registration void or answer the
    other's - the DB carries a (task_id, command_id) uniqueness constraint,
    not command_id alone.
    """
    origins = websocket_api._command_origins
    sock_a, sock_b = MagicMock(name="task-a"), MagicMock(name="task-b")
    origins.register("1", sock_a, 100)
    origins.register("1", sock_b, 200)

    with patch.object(
        websocket_api.manager, "is_connection_registered", return_value=True
    ):
        assert origins.resolve("1", 100) is sock_a
        assert origins.resolve("1", 200) is sock_b

    # Discarding one task's entry leaves the other intact.
    origins.discard_command("1", 100)
    assert not origins.has("1", 100)
    assert origins.has("1", 200)


@pytest.mark.asyncio
async def test_live_chat_runtime_error_does_not_double_send_the_detail(
    _test_db: None,
) -> None:
    """On the live path the rejection ack already carries the detail, so the
    origin bubble must not fire a second copy (preflight side-effect)."""
    db = _direct_db_session()
    try:
        owner = User(username="no-double-owner", password_hash="hash")
        db.add(owner)
        db.commit()
        task = Task(
            user_id=int(owner.id),
            title="No double",
            description="live runtime branch",
            status=TaskStatus.RUNNING,
            execution_mode="balanced",
            source="internal",
        )
        db.add(task)
        db.commit()
        task.runner_id = "rt8-runner"
        task.run_id = "rt8-run"
        db.commit()
        task_id, owner_id = int(task.id), int(owner.id)
    finally:
        db.close()

    raised = RuntimeError(f"live fault {SECRET}")
    mgr, ws_manager, bg_mgr, fake_payload = _chat_runtime_error_harness(raised)
    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=mgr),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
        patch("xagent.web.api.websocket.mark_user_message_delivery_sync", MagicMock()),
        patch(
            "xagent.web.api.websocket._read_task_error_payload_isolated",
            MagicMock(side_effect=fake_payload),
        ),
    ):
        await websocket_api._handle_chat_message_unserialized(
            MagicMock(name="live-sender"),
            task_id,
            {
                "message": "live runtime failure",
                "client_message_id": "no-double",
                "user": SimpleNamespace(id=owner_id, is_admin=False),
                "files": [],
                # no _durable_ack_sent: this is the live path
            },
        )

    personal = [
        c.args[0]
        for c in ws_manager.send_personal_message.await_args_list
        if isinstance(c.args[0], dict)
    ]
    with_detail = [p for p in personal if SECRET in repr(p)]
    assert len(with_detail) == 1, (
        f"live path must carry the detail exactly once, got {with_detail}"
    )
    assert with_detail[0].get("type") == "message_rejected"


def test_a_later_duplicate_cannot_rebind_after_the_creator_disconnects(
    _clean_origins: None,
) -> None:
    """P1 disconnect/rebind: once the creator's entry is gone, no bind at all.

    First-registration-wins protects a live entry, but the sharper case is the
    creator disconnecting (its entry cleared) and a later duplicate arriving.
    Because only the creating ingress registers, the duplicate never calls
    register, so resolve stays empty rather than pointing at the late arrival.
    """
    origins = websocket_api._command_origins
    creator = MagicMock(name="creator")
    origins.register("cmd", creator, 7)

    # creator disconnects
    real_manager = websocket_api.ConnectionManager()
    real_manager.disconnect(creator)
    assert not origins.has("cmd", 7)

    # a later duplicate is created=False at the handler, so it never registers;
    # the registry stays empty and the executor will safe-discard.
    assert not origins.has("cmd", 7)
    with patch.object(
        websocket_api.manager, "is_connection_registered", return_value=True
    ):
        assert origins.resolve("cmd", 7) is None


def test_registry_is_bounded_by_lru_eviction(_clean_origins: None) -> None:
    """P2: entries whose commands run on another worker cannot grow unbounded.

    A long-lived socket whose commands are always claimed elsewhere would never
    get local cleanup. The store is an LRU capped at _MAX_ORIGINS; the oldest
    entry is evicted on overflow, which only makes resolve miss (safe discard),
    never reroutes detail.
    """
    origins = websocket_api._command_origins
    cap = websocket_api._CommandOriginRegistry._MAX_ORIGINS
    socket = MagicMock(name="long-lived")

    for i in range(cap + 50):
        origins.register(f"cmd-{i}", socket, 7)

    assert len(origins._origins) == cap, "the store must not grow past the cap"
    # oldest evicted, newest retained
    assert not origins.has("cmd-0", 7)
    assert origins.has(f"cmd-{cap + 49}", 7)
    with patch.object(
        websocket_api.manager, "is_connection_registered", return_value=True
    ):
        assert origins.resolve("cmd-0", 7) is None  # evicted -> safe discard
