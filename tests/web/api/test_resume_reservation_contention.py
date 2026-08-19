"""Resume-slot contention on a durable message defers instead of failing.

A form response arrives as a durable ``message`` command. When the live-control
injection path found the resume reservation already held, it finished the
delivery as failed, which the durable executor turned into an immediately
terminal rejection: the command row reached ``failed`` in seconds and nothing
ever re-dispatched it, so the task stayed in ``WAITING_FOR_USER`` with the
pending tool call unrun (#1469).

Contention is not one condition. A held reservation and a draining process both
clear on their own and never reached an injection. A running resume coordinator
lasts as long as the execution does, and may be applying this very turn, so it
is neither deferrable nor safe to tell the sender to resend.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    MAX_COMMAND_DEFERS,
    ClaimedTaskCommand,
    TaskCommandDeferred,
    TaskCommandKind,
    TaskCommandRejected,
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
    bg_mgr.running_tasks.get.return_value = None
    return bg_mgr


def _message_command(
    task: Task, owner: User, command_id: str, **kwargs
) -> ClaimedTaskCommand:
    return ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id=command_id,
        kind=TaskCommandKind.MESSAGE,
        payload={"type": "chat_message", "client_message_id": command_id},
        target_run_id="live-run",
        attempt_count=kwargs.pop("attempt_count", 1),
        **kwargs,
    )


def _durable_payload(owner: User, turn_id: str) -> dict:
    return {
        "message": "here is the form response",
        "client_message_id": turn_id,
        "user": owner,
        "files": [],
        "_durable_ack_sent": True,
    }


async def _send_live_message(task: Task, message_data: dict) -> MagicMock:
    """Drive the handler down to the live-control branch; return the ws manager."""

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

    # Held by an injection that has not registered its coordinator yet.
    assert manager.try_reserve_resume(1) is ResumeReservationOutcome.RESERVATION_HELD
    assert not manager.reserve_resume(1)

    running = asyncio.get_running_loop().create_future()
    coordinator = asyncio.ensure_future(running)
    try:
        manager.register_reserved_resume(1, coordinator)
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
@pytest.mark.parametrize(
    "outcome",
    [
        ResumeReservationOutcome.RESERVATION_HELD,
        ResumeReservationOutcome.SHUTTING_DOWN,
    ],
)
async def test_self_clearing_contention_defers_the_durable_message(
    db_session,
    outcome: ResumeReservationOutcome,
) -> None:
    """The #1469 regression: a transient hold must not burn the failure budget."""

    owner = _user(db_session, f"contention-owner-{outcome.value}")
    task = _live_task(db_session, int(owner.id))
    message_data = _durable_payload(owner, "form-turn")

    with patch(
        "xagent.web.api.websocket.background_task_manager",
        _contended_manager(outcome),
    ):
        await _send_live_message(task, message_data)

    assert message_data["_durable_command_defer"] == "form-turn"
    # A durable error is what made the command terminal within seconds, and is
    # what distinguishes a deferral from a rejection here.
    assert "_durable_command_error" not in message_data


@pytest.mark.asyncio
async def test_a_contended_message_defers_through_the_real_executor(
    db_session,
) -> None:
    """Handler and executor wired together, rather than each mocking the other."""

    owner = _user(db_session, "integration-owner")
    task = _live_task(db_session, int(owner.id))
    command = _message_command(task, owner, "form-turn")
    command.payload.update(
        {"message": "here is the form response", "files": [], "_durable_ack_sent": True}
    )

    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(return_value=True)

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=MagicMock(get_agent_for_task=AsyncMock(return_value=agent)),
        ),
        patch(
            "xagent.web.api.websocket.manager",
            MagicMock(
                broadcast_to_task=AsyncMock(),
                send_personal_message=AsyncMock(),
                connections_for_task=MagicMock(return_value=[SimpleNamespace()]),
            ),
        ),
        patch("xagent.web.api.websocket.execute_resume_background", AsyncMock()),
        patch(
            "xagent.web.api.websocket.background_task_manager",
            _contended_manager(ResumeReservationOutcome.RESERVATION_HELD),
        ),
        pytest.raises(TaskCommandDeferred) as exc_info,
    ):
        await _execute_durable_task_command(command)

    assert "resume slot" in str(exc_info.value)
    # Nothing was injected, so an exhausted budget may ask for a resend.
    assert exc_info.value.resend_safe is True
    agent.post_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_running_coordinator_is_rejected_without_inviting_a_resend(
    db_session,
) -> None:
    """The coordinator may be applying this same turn, so a resend duplicates."""

    owner = _user(db_session, "coordinator-owner")
    task = _live_task(db_session, int(owner.id))
    message_data = _durable_payload(owner, "late-turn")

    with patch(
        "xagent.web.api.websocket.background_task_manager",
        _contended_manager(ResumeReservationOutcome.COORDINATOR_RUNNING),
    ):
        await _send_live_message(task, message_data)

    assert "_durable_command_defer" not in message_data
    error = message_data["_durable_command_error"]
    assert "still applying an earlier message" in error
    assert "rather than sending again" in error
    assert "was not applied" not in error


@pytest.mark.asyncio
async def test_a_rejected_message_carries_a_notice_instead_of_failing_silently(
    db_session,
) -> None:
    """The handler's own reply cannot reach the client on the durable path.

    ``finish_delivery`` suppresses the socket send once the enqueue acked, so
    without this the sender is left with a message that never applied and no
    signal at all. The executor only marks the text: the dispatcher emits it
    once the terminal write is confirmed, because a claim lost to another
    runner is still going to be retried.
    """

    owner = _user(db_session, "silent-reject-owner")
    task = _live_task(db_session, int(owner.id))
    command = _message_command(task, owner, "rejected-message")
    ws_manager = MagicMock(broadcast_to_task=AsyncMock())

    with (
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.api.websocket._execute_durable_task_command",
            new=AsyncMock(side_effect=TaskCommandRejected("could not be applied")),
        ),
        pytest.raises(TaskCommandRejected) as exc_info,
    ):
        await execute_durable_task_command(command)

    ws_manager.broadcast_to_task.assert_not_awaited()
    assert "could not be applied" in str(exc_info.value.terminal_client_message)


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
        ws_manager = await _send_live_message(task, message_data)

    assert "_durable_command_defer" not in message_data
    rejections = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert len(rejections) == 1
    assert rejections[0]["rejection_outcome"] == "not_accepted"
    # Nothing is running yet -- only a held reservation, so naming a running
    # task would report the wrong cause.
    assert "busy starting an earlier message" in rejections[0]["message"]


@pytest.mark.asyncio
async def test_a_draining_process_tells_the_legacy_sender_to_retry_shortly(
    db_session,
) -> None:
    owner = _user(db_session, "draining-legacy-owner")
    task = _live_task(db_session, int(owner.id))
    message_data = {
        "message": "sent while this process drains",
        "client_message_id": "drain-turn",
        "user": owner,
        "files": [],
    }

    with patch(
        "xagent.web.api.websocket.background_task_manager",
        _contended_manager(ResumeReservationOutcome.SHUTTING_DOWN),
    ):
        ws_manager = await _send_live_message(task, message_data)

    rejection = next(
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    )
    assert "server is restarting" in rejection["message"]


@pytest.mark.asyncio
async def test_only_a_resend_safe_deferral_asks_the_sender_to_resend(
    db_session,
) -> None:
    """A message can also defer while an injection it started is in flight.

    Telling that sender to resend would duplicate work already on its way,
    because a resend mints a fresh id rather than matching the pending one.
    """

    owner = _user(db_session, "exhausted-owner")
    task = _live_task(db_session, int(owner.id))

    async def notice_for(exc: TaskCommandDeferred, **overrides) -> str | None:
        command = _message_command(
            task,
            owner,
            "exhausted-message",
            defer_count=overrides.pop("defer_count", MAX_COMMAND_DEFERS - 1),
        )
        with (
            patch(
                "xagent.web.api.websocket._execute_durable_task_command",
                new=AsyncMock(side_effect=exc),
            ),
            pytest.raises(TaskCommandDeferred) as exc_info,
        ):
            await execute_durable_task_command(command)
        return exc_info.value.terminal_client_message

    resend_safe = await notice_for(
        TaskCommandDeferred("resume slot held", resend_safe=True)
    )
    assert resend_safe is not None
    assert "was not applied" in resend_safe
    assert "Please send it again" in resend_safe
    # The internal wait reason must not be what the sender is left with.
    assert "resume slot held" not in resend_safe

    in_flight = await notice_for(TaskCommandDeferred("waiting for injection"))
    assert in_flight is not None
    assert "waiting for injection" in in_flight
    assert "send it again" not in in_flight

    # Below the budget nothing is terminal yet, so nothing is announced.
    assert (
        await notice_for(
            TaskCommandDeferred("resume slot held", resend_safe=True),
            defer_count=0,
        )
        is None
    )


@pytest.mark.asyncio
async def test_a_non_message_deferral_keeps_its_own_exhaustion_text(
    db_session,
) -> None:
    """Only a message send has a resend for the sender to perform."""

    owner = _user(db_session, "pause-exhausted-owner")
    task = _live_task(db_session, int(owner.id))
    command = ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id="pause-command",
        kind=TaskCommandKind.PAUSE,
        payload={"type": "pause_task"},
        target_run_id="live-run",
        attempt_count=1,
        defer_count=MAX_COMMAND_DEFERS - 1,
    )

    with (
        patch(
            "xagent.web.api.websocket._execute_durable_task_command",
            new=AsyncMock(
                side_effect=TaskCommandDeferred("waiting for the lease owner")
            ),
        ),
        pytest.raises(TaskCommandDeferred) as exc_info,
    ):
        await execute_durable_task_command(command)

    notice = exc_info.value.terminal_client_message
    assert notice is not None
    assert "waiting for the lease owner" in notice
    assert "send it again" not in notice


@pytest.mark.asyncio
async def test_a_deferred_message_is_delivered_once_the_reservation_clears(
    db_session,
) -> None:
    """The whole point of #1469: the retry has to actually deliver.

    Every other test here stops at the deferral. This one lets the reservation
    clear and asserts the second attempt injects the message.
    """

    owner = _user(db_session, "eventual-delivery-owner")
    task = _live_task(db_session, int(owner.id))

    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.side_effect = [
        ResumeReservationOutcome.RESERVATION_HELD,
        ResumeReservationOutcome.RESERVED,
    ]
    bg_mgr.running_tasks.get.return_value = None

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
        patch("xagent.web.api.websocket.background_task_manager", bg_mgr),
    ):
        first = _durable_payload(owner, "form-turn")
        await _handle_chat_message_unserialized(MagicMock(), int(task.id), first)
        assert first["_durable_command_defer"] == "form-turn"
        agent.post_user_message.assert_not_awaited()

        second = _durable_payload(owner, "form-turn")
        await _handle_chat_message_unserialized(MagicMock(), int(task.id), second)
        await asyncio.sleep(0)

    assert "_durable_command_defer" not in second
    assert "_durable_command_error" not in second
    agent.post_user_message.assert_awaited_once()
    assert agent.post_user_message.await_args.kwargs["turn_id"] == "form-turn"
