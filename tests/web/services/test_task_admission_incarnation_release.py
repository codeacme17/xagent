"""A drained incarnation releases its own ticket while the task keeps running.

Tickets carry no execution identity: ownership is the task lease, which every
incarnation of one task shares. A RESUME classified into another bucket
therefore kept the previous bucket's slot for the whole resumed run, because
both release paths run only while the task is not RUNNING
(xorbitsai/xagent#2777). The coordinator now releases exactly the ticket of
the execution handle that finished, keeping the idle release as the backstop.
"""

import asyncio

from tests.web.services.test_task_admission_execution_slots import (
    RunExecution,
    enqueue,
    stamped_tickets,
    task_of,
)
from tests.web.services.test_task_execution_admission import engine as engine_fixture
from tests.web.services.test_task_execution_admission import host as host_fixture
from xagent.web.models.task import Task, TaskStatus
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services import task_execution_admission as admission
from xagent.web.services.task_admission_observation import read_admission_snapshot

engine = engine_fixture
host = host_fixture


def classify_by_kind(db, command):
    """START work belongs to ``batch``; every continuation to ``interactive``."""
    bucket = "batch" if command.kind == "start" else "interactive"
    return admission.AdmissionPolicy(bucket, 1, 20)


def active(host, bucket):
    with host.sessions() as db:
        rows = read_admission_snapshot(db, [bucket])
    return rows[0].active if rows else 0


async def eventually_active(host, bucket, expected):
    async with asyncio.timeout(5):
        while active(host, bucket) != expected:
            await asyncio.sleep(0.01)


async def pause_then_resume_across_buckets(host):
    """Task A pauses in ``batch`` and resumes in ``interactive``.

    Returns the paused and resumed executions; the paused cleanup is still held
    when this returns, so both buckets hold A's slot.
    """
    admission.set_task_admission_hook(classify_by_kind)
    first = enqueue(host)
    task_id = task_of(host, first)
    paused = RunExecution(host, settled=TaskStatus.PAUSED)
    assert await transport.dispatch_one_task_command(paused)
    paused.finish.set()
    await asyncio.wait_for(paused.terminal.wait(), 5)
    resume = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.RESUME)
    resumed = RunExecution(host, new_run=False)
    assert await transport.dispatch_one_task_command(
        resumed, command_db_id=resume.command_id
    )
    assert resumed.started == [resume.command_id]
    assert stamped_tickets(host, task_id) == [first.command_id, resume.command_id]
    assert (active(host, "batch"), active(host, "interactive")) == (1, 1)
    return paused, resumed


async def test_previous_bucket_slot_is_released_once_its_incarnation_drains(host):
    paused, resumed = await pause_then_resume_across_buckets(host)
    other = enqueue(host)
    other_execution = RunExecution(host)
    # Both slots stay held while the previous incarnation is still draining.
    assert not await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )

    paused.cleanup.set()
    await eventually_active(host, "batch", 0)
    assert active(host, "interactive") == 1
    assert not resumed.terminal.is_set()
    # The resumed run keeps only the slot of the lane it now runs in.
    assert await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )
    assert other_execution.started == [other.command_id]

    resumed.finish.set()
    resumed.cleanup.set()
    other_execution.finish.set()
    other_execution.cleanup.set()
    await eventually_active(host, "interactive", 0)
    await eventually_active(host, "batch", 0)


async def test_cancelled_successor_never_releases_the_predecessor_slot(host):
    paused, resumed = await pause_then_resume_across_buckets(host)
    handle = resumed.handles[0]
    handle.cancel()
    await asyncio.gather(handle, return_exceptions=True)
    await eventually_active(host, "interactive", 0)
    # The previous incarnation still holds its own slot until it drains.
    assert active(host, "batch") == 1
    other = enqueue(host)
    other_execution = RunExecution(host)
    assert not await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )

    paused.cleanup.set()
    await eventually_active(host, "batch", 0)
    assert await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )
    other_execution.finish.set()
    other_execution.cleanup.set()


async def test_same_bucket_resume_keeps_one_slot_after_the_predecessor_drains(host):
    admission.set_task_admission_hook(
        lambda db, command: admission.AdmissionPolicy("turns", 1, 20)
    )
    first = enqueue(host)
    task_id = task_of(host, first)
    paused = RunExecution(host, settled=TaskStatus.PAUSED)
    assert await transport.dispatch_one_task_command(paused)
    paused.finish.set()
    await asyncio.wait_for(paused.terminal.wait(), 5)
    resume = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.RESUME)
    resumed = RunExecution(host, new_run=False)
    assert await transport.dispatch_one_task_command(
        resumed, command_db_id=resume.command_id
    )
    assert active(host, "turns") == 1

    paused.cleanup.set()
    async with asyncio.timeout(5):
        while stamped_tickets(host, task_id) != [resume.command_id]:
            await asyncio.sleep(0.01)
    # Releasing the drained START ticket leaves the RESUME ticket holding the slot.
    assert active(host, "turns") == 1
    other = enqueue(host)
    other_execution = RunExecution(host)
    assert not await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )

    resumed.finish.set()
    resumed.cleanup.set()
    await eventually_active(host, "turns", 0)
    assert await transport.dispatch_one_task_command(
        other_execution, command_db_id=other.command_id
    )
    other_execution.finish.set()
    other_execution.cleanup.set()


async def test_drained_handle_of_a_superseded_owner_releases_nothing(host):
    """A handle draining after another owner took the task frees no slot."""
    admission.set_task_admission_hook(classify_by_kind)
    first = enqueue(host)
    task_id = task_of(host, first)
    paused = RunExecution(host, settled=TaskStatus.PAUSED)
    assert await transport.dispatch_one_task_command(paused)
    paused.finish.set()
    await asyncio.wait_for(paused.terminal.wait(), 5)
    # Another worker takes the task over and admits its own RESUME while the
    # previous owner's cleanup is still held.
    successor = ownership.TaskLease(
        task_id=task_id, runner_id="worker-2", attempt_id="successor-attempt"
    )
    resume = enqueue(host, task_id=task_id, kind=transport.TaskCommandKind.RESUME)
    with host.sessions() as db, db.begin():
        task = db.get(Task, task_id)
        task.runner_id = successor.runner_id
        task.lease_attempt_id = successor.attempt_id
        db.flush()
        assert admission.reserve_task_admission(db, resume.command_id, successor)
    assert active(host, "interactive") == 1
    assert stamped_tickets(host, task_id) == [first.command_id, resume.command_id]

    paused.cleanup.set()
    await asyncio.gather(paused.handles[0], return_exceptions=True)
    await asyncio.sleep(0.2)
    assert stamped_tickets(host, task_id) == [first.command_id, resume.command_id]
    assert active(host, "interactive") == 1


def test_stale_owner_cannot_release_the_successor_ticket(host):
    """The single-ticket release is fenced by the ticket's current owner."""
    admission.set_task_admission_hook(classify_by_kind)
    first = enqueue(host)
    task_id = task_of(host, first)
    with host.sessions() as db, db.begin():
        current = ownership.acquire_task_lease_no_commit(db, task_id, runner_id="w1")
        assert current is not None
        assert admission.reserve_task_admission(db, first.command_id, current)
    stale = ownership.TaskLease(
        task_id=task_id, runner_id="w0", attempt_id="stale-attempt"
    )
    with host.sessions() as db, db.begin():
        admission.release_task_admission(db, stale, first.command_id)
    assert stamped_tickets(host, task_id) == [first.command_id]
    assert active(host, "batch") == 1
    with host.sessions() as db, db.begin():
        admission.release_task_admission(db, current, first.command_id)
    assert stamped_tickets(host, task_id) == []
    assert active(host, "batch") == 0
