"""Exception text must not reach chat clients (PR #1472 review finding N3)."""

from __future__ import annotations

import asyncio
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
