"""Resume-slot contention on a durable message defers instead of failing.

A reply to a waiting clarification form is delivered as a durable ``message``
task command. ``BackgroundTaskManager.reserve_resume`` used to collapse three
unrelated contention causes into one ``False``, and the handler turned that
into ``_durable_command_error`` -- which ``_execute_durable_task_command``
converts into a terminal ``TaskCommandRejected`` (#1469).

That conversion sits *after* the executor's delivery-status checks, so a
still-``DELIVERY_PENDING`` row preempted it and the command deferred anyway.
The rejection only became terminal on the two shapes where nothing preempted
it:

* no delivery row for this command yet -- contention is observed before the
  row is claimed, so the reply was dropped on the spot;
* a row already marked ``DELIVERY_DISPATCHED`` -- the coordinator had applied
  the message, and the sender was still told it failed, with an
  ``agent_error`` broadcast to every connection on the task.

Contention is not one condition. A held reservation and a draining process
both self-clear and never reached an injection, so waiting is always safe. A
running resume coordinator lasts as long as the whole execution, so it stays a
rejection -- except alongside a recovered delivery, which proves an earlier
attempt of this same command already claimed this exact payload and left it
pending. That arm is what covers the dispatched-row race above.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    BackgroundTaskManager,
    ClientVisibleTaskCommandDeferred,
    ResumeReservationOutcome,
    _execute_durable_task_command,
    _handle_chat_message_unserialized,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.chat_history_service import (
    DELIVERY_DISPATCHED,
    DELIVERY_PENDING,
    mark_user_message_delivery_sync,
)
from xagent.web.services.task_command_transport import (
    ClaimedTaskCommand,
    TaskCommandKind,
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


def _waiting_task(db, owner_id: int) -> Task:
    """A task parked in WAITING_FOR_USER, as it is while a clarification
    form is outstanding. ``run_id``/``runner_id`` mirror a task that was
    RUNNING before it parked; they are not what gates ``live_task_lease``
    here (that is ``status == RUNNING``), but keep the row realistic.
    """
    task = Task(
        user_id=owner_id,
        title="t",
        description="d",
        status=TaskStatus.WAITING_FOR_USER,
        execution_mode="balanced",
        source="sdk",
        runner_id="waiting-runner",
        run_id="waiting-run",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _message_command(
    task: Task, owner: User, command_id: str, **kwargs
) -> ClaimedTaskCommand:
    return ClaimedTaskCommand(
        id=1,
        task_id=int(task.id),
        actor_user_id=int(owner.id),
        command_id=command_id,
        kind=TaskCommandKind.MESSAGE,
        payload={"message": "test", "client_message_id": command_id, "files": []},
        target_run_id=kwargs.pop("target_run_id", None),
        attempt_count=kwargs.pop("attempt_count", 1),
        **kwargs,
    )


def _live_control_agent() -> MagicMock:
    agent = MagicMock()
    agent.supports_live_control.return_value = True
    agent.get_dag_pattern.return_value = None
    agent.post_user_message = AsyncMock(return_value=True)
    return agent


def _agent_manager_returning(agent: MagicMock) -> MagicMock:
    return MagicMock(get_agent_for_task=AsyncMock(return_value=agent))


def _ws_manager() -> MagicMock:
    return MagicMock(
        broadcast_to_task=AsyncMock(),
        send_personal_message=AsyncMock(),
    )


def _contended_manager(outcome: ResumeReservationOutcome) -> MagicMock:
    """A stand-in BackgroundTaskManager whose reservation always resolves to
    ``outcome``. Used only for the narrow decision-table tests (3-5) below,
    which do not need the real reservation state machine -- that is covered
    by test 1 (the enum classification) and test 2 (the incident itself,
    driven through the real ``BackgroundTaskManager``).
    """
    bg_mgr = MagicMock()
    bg_mgr.try_reserve_resume.return_value = outcome
    bg_mgr.running_tasks.get.return_value = None
    return bg_mgr


@pytest.mark.asyncio
async def test_try_reserve_resume_classifies_every_contention_cause() -> None:
    """A real ``BackgroundTaskManager``, not a mock: the four outcomes and
    ``reserve_resume``'s bool projection must agree for each."""
    manager = BackgroundTaskManager()

    reserved = manager.try_reserve_resume(1)
    assert reserved is ResumeReservationOutcome.RESERVED
    assert manager.reserve_resume(2) is True  # a fresh id behaves the same

    held = manager.try_reserve_resume(1)
    assert held is ResumeReservationOutcome.RESERVATION_HELD
    assert manager.reserve_resume(1) is False

    coordinator = asyncio.ensure_future(asyncio.Event().wait())
    try:
        manager.register_reserved_resume(1, coordinator)
        running = manager.try_reserve_resume(1)
        assert running is ResumeReservationOutcome.COORDINATOR_RUNNING
        assert manager.reserve_resume(1) is False
    finally:
        coordinator.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator

    manager._shutting_down = True
    shutting = manager.try_reserve_resume(3)
    assert shutting is ResumeReservationOutcome.SHUTTING_DOWN
    assert manager.reserve_resume(3) is False


@pytest.mark.asyncio
async def test_a_recovered_delivery_defers_through_the_real_reservation_state(
    db_session,
) -> None:
    """The #1469 regression, driven through the real production shape.

    Attempt 1 reserves the resume slot and registers a coordinator that is
    still applying the earlier turn (mirroring a task parked at
    WAITING_FOR_USER, whose ``live_task_lease`` is None, so injection is
    deferred to the coordinator rather than posted synchronously). Attempt 2
    is a durable retry of the SAME command (``durable_attempt_count > 1``):
    it finds that same coordinator still running, but because
    ``recovered_delivery`` proves an earlier attempt of this exact command
    already claimed this exact pending payload, it must defer instead of
    rejecting -- the coordinator in flight is the one that will apply it.
    Once the coordinator marks the delivery DELIVERY_DISPATCHED (as it does
    on a successful injection), the next attempt must complete through the
    existing-delivery short-circuit rather than reaching the live-control
    branch again.
    """
    owner = _user(db_session, "incident-owner")
    task = _waiting_task(db_session, owner.id)
    task_id = int(task.id)
    turn_id = "form-turn"
    agent = _live_control_agent()
    ws_manager = _ws_manager()
    coordinator_release = asyncio.Event()

    async def coordinator_stub(**_kwargs: object) -> None:
        await coordinator_release.wait()

    def message_data(attempt: int) -> dict:
        return {
            "message": "here is the form response",
            "client_message_id": turn_id,
            "user": owner,
            "files": [],
            "_durable_ack_sent": True,
            "_durable_attempt_count": attempt,
        }

    try:
        with (
            patch(
                "xagent.web.api.chat.get_agent_manager",
                return_value=_agent_manager_returning(agent),
            ),
            patch("xagent.web.api.websocket.manager", ws_manager),
            patch(
                "xagent.web.api.websocket.execute_resume_background",
                side_effect=coordinator_stub,
            ),
        ):
            # Attempt 1: nothing is contended yet, so this reserves cleanly
            # and registers the coordinator (still running, per the stub).
            attempt_1 = message_data(1)
            await _handle_chat_message_unserialized(MagicMock(), task_id, attempt_1)
            assert "_durable_command_defer" not in attempt_1

            db_session.expire_all()
            stored = (
                db_session.query(TaskChatMessage)
                .filter(TaskChatMessage.turn_id == turn_id)
                .one()
            )
            assert stored.delivery_status == DELIVERY_PENDING

            coordinator = websocket_api.background_task_manager.resume_tasks.get(
                task_id
            )
            assert coordinator is not None
            assert not coordinator.done()

            # Attempt 2: same command, re-dispatched. The coordinator from
            # attempt 1 is still running. The row is still pending here, so
            # before the fix the executor's DELIVERY_PENDING check preempted
            # the contention rejection and this deferred too -- what changes
            # is that the handler now says so directly, which is what keeps
            # the dispatched-row race below from going terminal.
            attempt_2 = message_data(2)
            await _handle_chat_message_unserialized(MagicMock(), task_id, attempt_2)
            assert attempt_2["_durable_command_defer"] == turn_id
            assert "_durable_command_error" not in attempt_2
            agent.post_user_message.assert_not_awaited()

            db_session.expire_all()
            rows = (
                db_session.query(TaskChatMessage)
                .filter(TaskChatMessage.turn_id == turn_id)
                .all()
            )
            # No second claim, no re-injection: exactly the one row from
            # attempt 1, still pending.
            assert len(rows) == 1
            assert rows[0].delivery_status == DELIVERY_PENDING

            # The coordinator injects and marks the delivery dispatched, the
            # way a successful ``execute_resume_background`` injection does.
            mark_user_message_delivery_sync(task_id, turn_id, DELIVERY_DISPATCHED)

            # Attempt 3 must not touch the reservation at all: it resolves
            # through the existing-delivery short-circuit before the
            # live-control branch is ever reached.
            attempt_3 = message_data(3)
            with patch.object(
                websocket_api.background_task_manager,
                "try_reserve_resume",
                wraps=websocket_api.background_task_manager.try_reserve_resume,
            ) as reserve_spy:
                await _handle_chat_message_unserialized(MagicMock(), task_id, attempt_3)
            reserve_spy.assert_not_called()
            assert "_durable_command_defer" not in attempt_3
            assert "_durable_command_error" not in attempt_3
            agent.post_user_message.assert_not_awaited()
    finally:
        coordinator_release.set()
        leftover = websocket_api.background_task_manager.resume_tasks.pop(task_id, None)
        websocket_api.background_task_manager._resume_reservations.discard(task_id)
        if leftover is not None:
            if not leftover.done():
                leftover.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await leftover


@pytest.mark.asyncio
async def test_coordinator_running_without_a_recovered_delivery_still_rejects(
    db_session,
) -> None:
    """The narrowing must stay a narrowing: a running coordinator alone is
    not proof it owns THIS turn's delivery. Without a recovered delivery,
    deferring would hold the per-task command queue for the coordinator's
    whole run while only postponing the same eventual rejection."""
    owner = _user(db_session, "coordinator-no-recovery-owner")
    task = _waiting_task(db_session, owner.id)
    agent = _live_control_agent()
    ws_manager = _ws_manager()
    message_data = {
        "message": "first attempt, already contended",
        "client_message_id": "no-recovery-turn",
        "user": owner,
        "files": [],
        "_durable_ack_sent": True,
    }

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=_agent_manager_returning(agent),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.api.websocket.background_task_manager",
            _contended_manager(ResumeReservationOutcome.COORDINATOR_RUNNING),
        ),
    ):
        await _handle_chat_message_unserialized(MagicMock(), int(task.id), message_data)

    assert "_durable_command_defer" not in message_data
    error = message_data["_durable_command_error"]
    assert "still applying an earlier message" in error
    assert "rather than sending again" in error
    agent.post_user_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [ResumeReservationOutcome.RESERVATION_HELD, ResumeReservationOutcome.SHUTTING_DOWN],
)
async def test_self_clearing_contention_defers_the_durable_message(
    db_session,
    outcome: ResumeReservationOutcome,
) -> None:
    """A held reservation and a draining process both self-clear and never
    reached an injection, so the durable path may always wait on them."""
    owner = _user(db_session, f"self-clearing-owner-{outcome.value}")
    task = _waiting_task(db_session, owner.id)
    agent = _live_control_agent()
    ws_manager = _ws_manager()
    message_data = {
        "message": "please wait for the slot to clear",
        "client_message_id": f"defer-turn-{outcome.value}",
        "user": owner,
        "files": [],
        "_durable_ack_sent": True,
    }

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=_agent_manager_returning(agent),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.api.websocket.background_task_manager",
            _contended_manager(outcome),
        ),
    ):
        await _handle_chat_message_unserialized(MagicMock(), int(task.id), message_data)

    assert message_data["_durable_command_defer"] == f"defer-turn-{outcome.value}"
    assert "_durable_command_error" not in message_data
    agent.post_user_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_snippet"),
    [
        (
            ResumeReservationOutcome.RESERVATION_HELD,
            "busy starting an earlier message",
        ),
        (ResumeReservationOutcome.SHUTTING_DOWN, "server is restarting"),
        (
            ResumeReservationOutcome.COORDINATOR_RUNNING,
            "still applying an earlier message",
        ),
    ],
)
async def test_legacy_socket_path_rejects_every_cause_with_its_own_wording(
    db_session,
    outcome: ResumeReservationOutcome,
    expected_snippet: str,
) -> None:
    """A live socket sender is not durable -- nothing re-dispatches it, so
    contention always rejects on this path, whatever the cause. The wording
    still distinguishes causes, and a running coordinator never invites a
    resend (a resend would duplicate work the coordinator may already be
    applying)."""
    owner = _user(db_session, f"legacy-owner-{outcome.value}")
    task = _waiting_task(db_session, owner.id)
    agent = _live_control_agent()
    ws_manager = _ws_manager()
    message_data = {
        "message": "legacy socket send",
        "client_message_id": f"legacy-turn-{outcome.value}",
        "user": owner,
        "files": [],
    }

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=_agent_manager_returning(agent),
        ),
        patch("xagent.web.api.websocket.manager", ws_manager),
        patch(
            "xagent.web.api.websocket.background_task_manager",
            _contended_manager(outcome),
        ),
    ):
        await _handle_chat_message_unserialized(MagicMock(), int(task.id), message_data)

    assert "_durable_command_defer" not in message_data
    rejected = [
        call.args[0]
        for call in ws_manager.send_personal_message.call_args_list
        if call.args[0].get("type") == "message_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0]["rejection_outcome"] == "not_accepted"
    assert expected_snippet in rejected[0]["message"]
    if outcome is ResumeReservationOutcome.COORDINATOR_RUNNING:
        assert "rather than sending again" in rejected[0]["message"]
    agent.post_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_executor_defers_when_the_marker_matches_this_command(
    db_session,
) -> None:
    """``_execute_durable_task_command`` turns the handler's defer marker
    into ``ClientVisibleTaskCommandDeferred`` when it names this command."""
    owner = _user(db_session, "executor-marker-owner")
    task = _waiting_task(db_session, owner.id)
    command = _message_command(task, owner, "matching-command")

    async def set_matching_marker(_websocket, _task_id, message_data):
        message_data["_durable_command_defer"] = "matching-command"

    with (
        patch(
            "xagent.web.api.websocket._handle_chat_message_unserialized",
            side_effect=set_matching_marker,
        ),
        pytest.raises(
            ClientVisibleTaskCommandDeferred,
            match="waiting for the live-control resume slot",
        ),
    ):
        await _execute_durable_task_command(command)


@pytest.mark.asyncio
async def test_executor_does_not_defer_for_a_different_commands_marker(
    db_session,
) -> None:
    """A marker left over for a different command id must not defer this
    one -- the check is scoped to ``command.command_id``, not "any marker
    present"."""
    owner = _user(db_session, "executor-marker-mismatch-owner")
    task = _waiting_task(db_session, owner.id)
    command = _message_command(task, owner, "this-command")

    async def set_stale_marker(_websocket, _task_id, message_data):
        message_data["_durable_command_defer"] = "some-other-command"

    with patch(
        "xagent.web.api.websocket._handle_chat_message_unserialized",
        side_effect=set_stale_marker,
    ):
        result = await _execute_durable_task_command(command)

    assert result is not None
    assert result["command_id"] == "this-command"


@pytest.mark.asyncio
async def test_contention_without_a_delivery_row_defers_at_the_executor(
    db_session,
) -> None:
    """Contention is observed before the delivery row is claimed.

    This is one of the two shapes that actually went terminal: with no row,
    ``_load_command_message_delivery_status`` returns ``None``, so neither the
    DELIVERY_PENDING nor the DELIVERY_FAILED check fires and nothing preempts
    the handler's verdict. Before the fix the contention text reached the
    ``_durable_command_error`` conversion and the reply was rejected outright.

    Asserted at the executor rather than on the handler's marker, because
    without the marker the run would fall past both status checks to the
    success return -- a command persisted COMPLETED with nothing applied,
    which is worse than the rejection being replaced.
    """
    owner = _user(db_session, "no-row-owner")
    task = _waiting_task(db_session, owner.id)
    command = _message_command(task, owner, "no-row-command")
    agent = _live_control_agent()

    with (
        patch(
            "xagent.web.api.chat.get_agent_manager",
            return_value=_agent_manager_returning(agent),
        ),
        patch("xagent.web.api.websocket.manager", _ws_manager()),
        patch(
            "xagent.web.api.websocket.background_task_manager",
            _contended_manager(ResumeReservationOutcome.RESERVATION_HELD),
        ),
        pytest.raises(
            ClientVisibleTaskCommandDeferred,
            match="waiting for the live-control resume slot",
        ),
    ):
        await _execute_durable_task_command(command)

    agent.post_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_dispatched_row_stops_reporting_an_applied_message_as_failed(
    db_session,
) -> None:
    """The coordinator winning the race must not become a false failure.

    ``DELIVERY_DISPATCHED`` lands between the handler's reservation check and
    the executor's status read, so the DELIVERY_PENDING check that preempts
    the contention rejection no longer fires. Before the fix the run reached
    the ``_durable_command_error`` conversion and told the sender -- and every
    other connection on the task, through ``agent_error`` -- that a message
    the coordinator had just applied was dropped.

    The dispatch write is interposed right after the real handler returns,
    which is exactly that window.
    """
    owner = _user(db_session, "race-owner")
    task = _waiting_task(db_session, owner.id)
    task_id = int(task.id)
    turn_id = "race-turn"
    agent = _live_control_agent()
    coordinator_release = asyncio.Event()

    async def coordinator_stub(**_kwargs: object) -> None:
        await coordinator_release.wait()

    real_handler = websocket_api._handle_chat_message_unserialized

    async def handler_then_coordinator_dispatch(ws, tid, md) -> None:
        await real_handler(ws, tid, md)
        # The coordinator's own DELIVERY_DISPATCHED write, landing after the
        # handler has already decided and before the executor reads status.
        mark_user_message_delivery_sync(tid, turn_id, DELIVERY_DISPATCHED)

    try:
        with (
            patch(
                "xagent.web.api.chat.get_agent_manager",
                return_value=_agent_manager_returning(agent),
            ),
            patch("xagent.web.api.websocket.manager", _ws_manager()),
            patch(
                "xagent.web.api.websocket.execute_resume_background",
                side_effect=coordinator_stub,
            ),
        ):
            # Attempt 1 reserves, registers the coordinator, and leaves the
            # row pending.
            await _handle_chat_message_unserialized(
                MagicMock(),
                task_id,
                {
                    # Must match ``_message_command``'s payload exactly:
                    # ``recovered_delivery`` requires ``payload_matches``.
                    "message": "test",
                    "client_message_id": turn_id,
                    "user": owner,
                    "files": [],
                    "_durable_ack_sent": True,
                    "_durable_attempt_count": 1,
                },
            )
            db_session.expire_all()
            assert (
                db_session.query(TaskChatMessage)
                .filter(TaskChatMessage.turn_id == turn_id)
                .one()
                .delivery_status
                == DELIVERY_PENDING
            )

            # Attempt 2 recovers that pending claim, sees the coordinator, and
            # must defer even though the row turns dispatched underneath it.
            with (
                patch(
                    "xagent.web.api.websocket._handle_chat_message_unserialized",
                    side_effect=handler_then_coordinator_dispatch,
                ),
                pytest.raises(
                    ClientVisibleTaskCommandDeferred,
                    match="waiting for the live-control resume slot",
                ),
            ):
                await _execute_durable_task_command(
                    _message_command(task, owner, turn_id, attempt_count=2)
                )

            db_session.expire_all()
            assert (
                db_session.query(TaskChatMessage)
                .filter(TaskChatMessage.turn_id == turn_id)
                .one()
                .delivery_status
                == DELIVERY_DISPATCHED
            )
    finally:
        coordinator_release.set()
        leftover = websocket_api.background_task_manager.resume_tasks.pop(task_id, None)
        websocket_api.background_task_manager._resume_reservations.discard(task_id)
        if leftover is not None:
            if not leftover.done():
                leftover.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await leftover
