"""Resume-slot contention on a durable message defers instead of failing.

A form response arrives as a durable ``message`` command. When the live-control
injection path found the resume reservation already held, it finished the
delivery as failed, which the durable executor turned into an immediately
terminal rejection: the command row reached ``failed`` in seconds and nothing
ever re-dispatched it, so the task stayed in ``WAITING_FOR_USER`` with the
pending tool call unrun (#1469).

Contention is not one condition. A held reservation and a draining process both
clear on their own; a running resume coordinator lasts as long as the execution
does. Only the first two are worth waiting for.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    BackgroundTaskManager,
    ResumeReservationOutcome,
    _execute_durable_task_command,
    _handle_chat_message_unserialized,
    execute_durable_task_command,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.task_command_transport import (
    MAX_COMMAND_DEFER_RETRY_SECONDS,
    MAX_COMMAND_DEFERS,
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    contended_defer_retry_seconds,
)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'resume_contention.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _user(db, username: str) -> User:
    user = User(username=username, password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _live_task(db, owner_id: int) -> Task:
    task = Task(
        user_id=owner_id,
        title="t",
        description="d",
        status=TaskStatus.RUNNING,
        execution_mode="balanced",
        source="sdk",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    task.runner_id = "live-runner"
    task.run_id = "live-run"
    db.commit()
    return task


def _contended_manager(outcome: ResumeReservationOutcome) -> MagicMock:
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = outcome
    bg_mgr.resume_reservation_age.return_value = 3.5
    bg_mgr.running_tasks.get.return_value = None
    return bg_mgr


async def _send_live_message(task: Task, owner: User, message_data: dict) -> MagicMock:
    """Drive the handler down to the live-control branch and return ws manager."""

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(return_value=True)
    agent_manager = MagicMock(get_agent_for_task=AsyncMock(return_value=agent))
    ws_manager = MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )

    with (
        patch("xagent.web.api.chat.get_agent_manager", return_value=agent_manager),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch("xagent.web.api.websocket.execute_resume_background", AsyncMock()),
    ):
        await _handle_chat_message_unserialized(
            MagicMock(),
            int(task.id),
            message_data,
        )
    agent.post_user_message.assert_not_awaited()
    return ws_manager


@pytest.mark.asyncio
async def test_try_reserve_resume_separates_the_three_contention_causes() -> None:
    manager = BackgroundTaskManager()

    assert manager.try_reserve_resume(1) is ResumeReservationOutcome.RESERVED
    age = manager.resume_reservation_age(1)
    assert age is not None and age >= 0.0

    # Held by an injection that has not registered its coordinator yet.
    assert manager.try_reserve_resume(1) is ResumeReservationOutcome.RESERVATION_HELD
    assert not manager.reserve_resume(1)

    running = asyncio.get_running_loop().create_future()
    coordinator = asyncio.ensure_future(running)
    try:
        manager.register_reserved_resume(1, coordinator)
        # The reservation is consumed, so there is no holder age to report.
        assert manager.resume_reservation_age(1) is None
        assert (
            manager.try_reserve_resume(1)
            is ResumeReservationOutcome.COORDINATOR_RUNNING
        )

        manager._shutting_down = True
        assert manager.try_reserve_resume(2) is ResumeReservationOutcome.SHUTTING_DOWN
    finally:
        running.cancel()
        await asyncio.gather(coordinator, return_exceptions=True)


@pytest.mark.asyncio
async def test_held_reservation_defers_the_durable_message_instead_of_failing_it(
    db_session,
) -> None:
    """The #1469 regression: a transient hold must not burn the failure budget."""

    owner = _user(db_session, "contention-owner")
    task = _live_task(db_session, int(owner.id))
    message_data = {
        "message": "here is the form response",
        "client_message_id": "form-turn",
        "user": owner,
        "files": [],
        "_durable_ack_sent": True,
    }

    with patch(
        "xagent.web.api.websocket.background_task_manager",
        _contended_manager(ResumeReservationOutcome.RESERVATION_HELD),
    ):
        ws_manager = await _send_live_message(task, owner, message_data)

    assert message_data["_durable_command_defer"] == "form-turn"
    # A durable error is what made the command terminal within seconds.
    assert "_durable_command_error" not in message_data
    assert not [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") in {"message_accepted", "message_rejected"}
    ]


@pytest.mark.asyncio
async def test_a_draining_process_defers_so_another_runner_can_take_the_message(
    db_session,
) -> None:
    owner = _user(db_session, "draining-owner")
    task = _live_task(db_session, int(owner.id))
    message_data = {
        "message": "sent while this process drains",
        "client_message_id": "drain-turn",
        "user": owner,
        "files": [],
        "_durable_ack_sent": True,
    }

    with patch(
        "xagent.web.api.websocket.background_task_manager",
        _contended_manager(ResumeReservationOutcome.SHUTTING_DOWN),
    ):
        await _send_live_message(task, owner, message_data)

    assert message_data["_durable_command_defer"] == "drain-turn"
    assert "_durable_command_error" not in message_data


@pytest.mark.asyncio
async def test_a_running_coordinator_still_rejects_and_says_it_was_not_applied(
    db_session,
) -> None:
    """Waiting on a live execution is not a bounded retry, so the sender is told."""

    owner = _user(db_session, "coordinator-owner")
    task = _live_task(db_session, int(owner.id))
    message_data = {
        "message": "arrives mid-execution",
        "client_message_id": "late-turn",
        "user": owner,
        "files": [],
        "_durable_ack_sent": True,
    }

    with patch(
        "xagent.web.api.websocket.background_task_manager",
        _contended_manager(ResumeReservationOutcome.COORDINATOR_RUNNING),
    ):
        await _send_live_message(task, owner, message_data)

    assert "_durable_command_defer" not in message_data
    assert "was not applied" in message_data["_durable_command_error"]
    assert "already running" in message_data["_durable_command_error"]


@pytest.mark.asyncio
async def test_a_message_without_a_durable_command_is_rejected_not_deferred(
    db_session,
) -> None:
    """Nothing would re-dispatch a legacy socket send, so it keeps its rejection."""

    owner = _user(db_session, "legacy-socket-owner")
    task = _live_task(db_session, int(owner.id))
    message_data = {
        "message": "legacy recovery send",
        "client_message_id": "legacy-turn",
        "user": owner,
        "files": [],
    }

    with patch(
        "xagent.web.api.websocket.background_task_manager",
        _contended_manager(ResumeReservationOutcome.RESERVATION_HELD),
    ):
        ws_manager = await _send_live_message(task, owner, message_data)

    assert "_durable_command_defer" not in message_data
    rejections = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert len(rejections) == 1
    assert rejections[0]["rejection_outcome"] == "not_accepted"
    assert "was not applied" in rejections[0]["message"]
    # Nothing is running yet -- only a held reservation, so saying the task is
    # already running would name the wrong cause.
    assert "busy applying an earlier message" in rejections[0]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("defer_count", [0, 1, 4, 20])
async def test_the_executor_reschedules_a_contended_message_with_backoff(
    db_session,
    defer_count: int,
) -> None:
    owner = _user(db_session, f"executor-owner-{defer_count}")
    task = _live_task(db_session, int(owner.id))
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id="contended-message",
        kind=TaskCommandKind.MESSAGE,
        payload={"type": "chat_message", "client_message_id": "contended-message"},
        target_run_id="live-run",
        attempt_count=defer_count + 1,
        defer_count=defer_count,
    )

    async def mark_deferred(_websocket, _task_id, message_data: dict) -> None:
        message_data["_durable_command_defer"] = "contended-message"
        message_data["_durable_command_defer_reason"] = "resume slot held"

    with (
        patch.object(
            websocket_api.manager,
            "connections_for_task",
            return_value=[SimpleNamespace()],
        ),
        patch(
            "xagent.web.api.websocket._handle_chat_message_unserialized",
            new=AsyncMock(side_effect=mark_deferred),
        ),
        pytest.raises(TaskCommandDeferred) as exc_info,
    ):
        await _execute_durable_task_command(command)

    assert "resume slot held" in str(exc_info.value)
    assert exc_info.value.retry_after_seconds == contended_defer_retry_seconds(
        defer_count
    )
    # The wait grows with each deferral rather than re-polling every second.
    assert contended_defer_retry_seconds(defer_count) >= 2.0
    assert contended_defer_retry_seconds(defer_count) <= 30.0


@pytest.mark.asyncio
async def test_a_draining_process_waits_the_full_window_before_retrying(
    db_session,
) -> None:
    """Ramping up from a short retry only queues the task's later commands."""

    owner = _user(db_session, "drain-window-owner")
    task = _live_task(db_session, int(owner.id))
    message_data = {
        "message": "sent while this process drains",
        "client_message_id": "drain-window-turn",
        "user": owner,
        "files": [],
        "_durable_ack_sent": True,
    }

    with patch(
        "xagent.web.api.websocket.background_task_manager",
        _contended_manager(ResumeReservationOutcome.SHUTTING_DOWN),
    ):
        await _send_live_message(task, owner, message_data)

    assert (
        message_data["_durable_command_defer_after"] == MAX_COMMAND_DEFER_RETRY_SECONDS
    )

    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id="drain-window-turn",
        kind=TaskCommandKind.MESSAGE,
        payload=dict(message_data),
        target_run_id="live-run",
        attempt_count=1,
    )

    with (
        patch.object(
            websocket_api.manager,
            "connections_for_task",
            return_value=[SimpleNamespace()],
        ),
        patch(
            "xagent.web.api.websocket._handle_chat_message_unserialized",
            new=AsyncMock(),
        ),
        pytest.raises(TaskCommandDeferred) as exc_info,
    ):
        await _execute_durable_task_command(command)

    assert exc_info.value.retry_after_seconds == MAX_COMMAND_DEFER_RETRY_SECONDS


@pytest.mark.asyncio
async def test_the_last_deferral_tells_the_sender_the_message_was_dropped(
    db_session,
) -> None:
    """A wait is the wrong thing to report once nothing will retry again."""

    owner = _user(db_session, "exhausted-owner")
    task = _live_task(db_session, int(owner.id))
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id="exhausted-message",
        kind=TaskCommandKind.MESSAGE,
        payload={"type": "chat_message", "client_message_id": "exhausted-message"},
        target_run_id="live-run",
        attempt_count=MAX_COMMAND_DEFERS,
        defer_count=MAX_COMMAND_DEFERS - 1,
    )
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.api.websocket._execute_durable_task_command",
            new=AsyncMock(side_effect=TaskCommandDeferred("resume slot held")),
        ),
        pytest.raises(TaskCommandDeferred),
    ):
        await execute_durable_task_command(command)

    broadcast = ws_manager.broadcast_to_task.await_args.args[0]
    assert broadcast["type"] == "agent_error"
    assert "was not applied" in broadcast["message"]
    assert "Please send it again" in broadcast["message"]
    # The internal wait reason must not be what the sender is left with.
    assert "resume slot held" not in broadcast["message"]
