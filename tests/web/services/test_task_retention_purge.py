"""Row-level semantics of the retention purge (#2563).

Runs against both SQLite and a disposable PostgreSQL through the shared
``engine`` fixture, the same way ``test_task_retention.py`` does, because the
two paths differ by dialect in ways that matter: foreign keys are enforced on
both here (the SQLite engine turns the pragma on), but ``ON DELETE SET NULL``
against ``ck_task_interaction_requests_active_anchor`` and the ``DateTime``
round-trip are only *identical* on both if something checks.

What is deliberately **not** here: the proof that the purge's row lock fences
a concurrent command insert. That needs two live sessions against a real
server and lives in ``test_task_retention_purge_postgresql.py``. This file
covers what one session can observe.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_interaction_schema_shared import (
    make_row as make_interaction_row,
)
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.database import Base
from xagent.web.models.task import (
    DAGExecution,
    Task,
    TaskStatus,
    TraceCheckpointBlob,
    TraceEvent,
    TraceMessageBlob,
)
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.task_retention_purge import (
    RetentionPurgeAction,
    RetentionPurgeReport,
    RetentionPurgeUnsupported,
    ensure_retention_purge_supported,
    purge_task,
    retention_purge_configured,
    run_retention_purge_batch,
    run_retention_purge_loop,
    select_purge_candidates,
)

engine = engine_fixture

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)
CONVERSATION_DAYS = 365
TRACE_DAYS = 90


@pytest.fixture
def sessions(engine) -> sessionmaker[Session]:
    """Sessions shaped like production's: ``autoflush=False``, per models/database.py."""
    Base.metadata.create_all(engine)
    return sa.orm.sessionmaker(bind=engine, autoflush=False)


def _age(days: int) -> datetime:
    """An anchor exactly ``days`` old relative to :data:`NOW`."""
    return NOW - timedelta(days=days)


def _make_user(db: Session, username: str) -> int:
    user = User(username=username, password_hash="unused")
    db.add(user)
    db.flush()
    return int(user.id)


def _make_task(
    db: Session,
    *,
    user_id: int,
    anchor: datetime,
    status: TaskStatus = TaskStatus.COMPLETED,
    lease_expires_at: datetime | None = None,
) -> int:
    task = Task(
        user_id=user_id,
        title="retention purge fixture",
        status=status,
        last_activity_at=anchor,
        lease_expires_at=lease_expires_at,
    )
    db.add(task)
    db.flush()
    return int(task.id)


def _make_trace(db: Session, *, task_id: int, suffix: str = "1") -> int:
    event = TraceEvent(
        task_id=task_id,
        event_id=f"evt-{task_id}-{suffix}",
        event_type="agent_execution_checkpoint",
        timestamp=NOW,
        data={},
    )
    db.add(event)
    db.flush()
    return int(event.id)


def _seed_full_task(
    db: Session,
    *,
    username: str,
    anchor: datetime,
    status: TaskStatus = TaskStatus.COMPLETED,
) -> int:
    """A task carrying one of everything the two paths distinguish between."""
    user_id = _make_user(db, username)
    task_id = _make_task(db, user_id=user_id, anchor=anchor, status=status)
    trace_id = _make_trace(db, task_id=task_id)
    db.add_all(
        [
            TraceMessageBlob(
                task_id=task_id,
                execution_id="exec-1",
                message_hash=f"m{task_id}",
                message_data={},
                message_bytes=2,
            ),
            TraceCheckpointBlob(
                task_id=task_id,
                execution_id="exec-1",
                blob_kind="snapshot",
                blob_hash=f"c{task_id}",
                blob_data={},
                blob_bytes=2,
            ),
            DAGExecution(task_id=task_id),
            TaskChatMessage(
                task_id=task_id,
                user_id=user_id,
                role="user",
                content="hello",
                message_type="text",
            ),
        ]
    )
    db.execute(
        sa.update(Task)
        .where(Task.id == task_id)
        .values(
            last_checkpoint_event_id=f"evt-{task_id}-1",
            last_checkpoint_trace_event_id=trace_id,
        )
    )
    db.commit()
    return task_id


def _counts(db: Session, task_id: int) -> dict[str, int]:
    def count(model, column) -> int:
        return int(
            db.execute(
                sa.select(sa.func.count()).select_from(model).where(column == task_id)
            ).scalar_one()
        )

    return {
        "tasks": count(Task, Task.id),
        "trace_events": count(TraceEvent, TraceEvent.task_id),
        "message_blobs": count(TraceMessageBlob, TraceMessageBlob.task_id),
        "checkpoint_blobs": count(TraceCheckpointBlob, TraceCheckpointBlob.task_id),
        "dag_executions": count(DAGExecution, DAGExecution.task_id),
        "chat_messages": count(TaskChatMessage, TaskChatMessage.task_id),
    }


def _purge(
    sessions: sessionmaker[Session],
    task_id: int,
    *,
    conversation_days: int | None = CONVERSATION_DAYS,
    trace_days: int | None = TRACE_DAYS,
    dry_run: bool = False,
) -> RetentionPurgeAction:
    with sessions() as db:
        return purge_task(
            db,
            task_id,
            now=NOW,
            conversation_days=conversation_days,
            trace_days=trace_days,
            dry_run=dry_run,
        )


# ---------------------------------------------------------------------------
# Conversation expiry: the whole task goes.
# ---------------------------------------------------------------------------


def test_conversation_expiry_removes_the_task_and_its_rows(sessions) -> None:
    with sessions() as db:
        task_id = _seed_full_task(db, username="c1", anchor=_age(400))

    assert _purge(sessions, task_id) is RetentionPurgeAction.PURGED_CONVERSATION

    with sessions() as db:
        assert _counts(db, task_id) == {
            "tasks": 0,
            "trace_events": 0,
            "message_blobs": 0,
            "checkpoint_blobs": 0,
            "dag_executions": 0,
            "chat_messages": 0,
        }


def test_conversation_expiry_detaches_uploads_rather_than_deleting_them(
    sessions,
) -> None:
    """``purge_task_rows``' documented behaviour, pinned at this caller.

    Reclaiming the rows and their bytes is #1086's, and a purge that started
    deleting them here would take that decision without the object-storage
    cleanup that makes it safe.
    """
    with sessions() as db:
        task_id = _seed_full_task(db, username="c2", anchor=_age(400))
        owner_id = int(
            db.execute(sa.select(Task.user_id).where(Task.id == task_id)).scalar_one()
        )
        db.add(
            UploadedFile(
                file_id="upload-1",
                filename="a.txt",
                storage_path="/tmp/a.txt",
                file_size=1,
                user_id=owner_id,
                task_id=task_id,
            )
        )
        db.commit()

    assert _purge(sessions, task_id) is RetentionPurgeAction.PURGED_CONVERSATION

    with sessions() as db:
        rows = db.execute(
            sa.select(UploadedFile.task_id).where(UploadedFile.file_id == "upload-1")
        ).all()
    # One row, and its task_id cleared -- not zero rows, which is what the
    # column's ON DELETE CASCADE would have produced had the unit of work not
    # detached it first.
    assert rows == [(None,)]


def test_conversation_expiry_wins_over_a_longer_trace_period(sessions) -> None:
    """A trace period longer than the conversation period must not retain a task.

    ``RetentionDisposition.CONVERSATION_EXPIRED`` subsumes traces; this pins
    that the purge acts on that and does not, say, take the narrower path
    because the trace period has not elapsed.
    """
    with sessions() as db:
        task_id = _seed_full_task(db, username="c3", anchor=_age(400))

    action = _purge(sessions, task_id, conversation_days=365, trace_days=3650)

    assert action is RetentionPurgeAction.PURGED_CONVERSATION
    with sessions() as db:
        assert _counts(db, task_id)["tasks"] == 0


# ---------------------------------------------------------------------------
# Trace expiry: the conversation stays.
# ---------------------------------------------------------------------------


def test_trace_expiry_keeps_the_conversation_and_clears_both_pointers(
    sessions,
) -> None:
    with sessions() as db:
        task_id = _seed_full_task(db, username="t1", anchor=_age(100))

    assert _purge(sessions, task_id) is RetentionPurgeAction.PURGED_TRACES

    with sessions() as db:
        assert _counts(db, task_id) == {
            "tasks": 1,
            "trace_events": 0,
            "message_blobs": 0,
            "checkpoint_blobs": 0,
            "dag_executions": 1,
            "chat_messages": 1,
        }
        pointers = db.execute(
            sa.select(
                Task.last_checkpoint_event_id, Task.last_checkpoint_trace_event_id
            ).where(Task.id == task_id)
        ).one()
        assert pointers == (None, None)


def test_trace_expiry_does_not_advance_updated_at(sessions) -> None:
    """Expiring a trace is maintenance, not execution activity.

    ``tasks.updated_at`` carries ``onupdate=func.now()``, so the pointer-
    clearing UPDATE would bump it unless the column is named in the SET
    clause. #2557's side-effect review lists maintenance writes that advance
    it as a hazard of their own.
    """
    with sessions() as db:
        task_id = _seed_full_task(db, username="t2", anchor=_age(100))
        before = db.execute(
            sa.select(Task.updated_at).where(Task.id == task_id)
        ).scalar_one()

    assert _purge(sessions, task_id) is RetentionPurgeAction.PURGED_TRACES

    with sessions() as db:
        after = db.execute(
            sa.select(Task.updated_at).where(Task.id == task_id)
        ).scalar_one()
    assert after == before


def test_trace_expiry_does_not_move_the_retention_anchor(sessions) -> None:
    """A purged trace must not postpone the task's own conversation expiry."""
    with sessions() as db:
        task_id = _seed_full_task(db, username="t3", anchor=_age(100))

    assert _purge(sessions, task_id) is RetentionPurgeAction.PURGED_TRACES

    with sessions() as db:
        anchor = db.execute(
            sa.select(Task.last_activity_at).where(Task.id == task_id)
        ).scalar_one()
    assert _as_utc(anchor) == _age(100)


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite returns ``DateTime(timezone=True)`` naive; PostgreSQL aware."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def test_trace_expiry_skips_a_task_holding_an_active_interaction_row(
    sessions,
) -> None:
    """The CHECK that behaves like RESTRICT for active rows.

    ``task_interaction_requests.resume_trace_event_id`` is ON DELETE SET NULL
    and ``ck_task_interaction_requests_active_anchor`` forbids a NULL anchor
    on an active row, so deleting this task's trace_events would fail the
    CHECK and roll back the batch. The whole-task path never meets this
    because it deletes the interaction rows first.
    """
    with sessions() as db:
        task_id = _seed_full_task(db, username="t4", anchor=_age(100))
        anchor_id = _make_trace(db, task_id=task_id, suffix="anchor")
        db.add(
            TaskInteractionRequest(
                **make_interaction_row(
                    task_id=task_id,
                    resume_trace_event_id=anchor_id,
                    status="active",
                )
            )
        )
        db.commit()

    action = _purge(sessions, task_id)

    assert action is RetentionPurgeAction.SKIPPED_ACTIVE_INTERACTION
    with sessions() as db:
        assert _counts(db, task_id)["trace_events"] == 2, "nothing was deleted"


def test_trace_expiry_proceeds_past_a_terminated_interaction_row(sessions) -> None:
    """A terminal row's anchor may be cleared -- that asymmetry is by design."""
    with sessions() as db:
        task_id = _seed_full_task(db, username="t5", anchor=_age(100))
        anchor_id = _make_trace(db, task_id=task_id, suffix="anchor")
        db.add(
            TaskInteractionRequest(
                **make_interaction_row(
                    task_id=task_id,
                    resume_trace_event_id=anchor_id,
                    status="terminated",
                )
            )
        )
        db.commit()

    assert _purge(sessions, task_id) is RetentionPurgeAction.PURGED_TRACES

    with sessions() as db:
        assert _counts(db, task_id)["trace_events"] == 0
        row = db.execute(
            sa.select(TaskInteractionRequest.resume_trace_event_id).where(
                TaskInteractionRequest.task_id == task_id
            )
        ).scalar_one()
        assert row is None, "SET NULL applied to the surviving terminal row"


# ---------------------------------------------------------------------------
# Refusals.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [TaskStatus.RUNNING, TaskStatus.PENDING, TaskStatus.WAITING_FOR_USER]
)
def test_a_non_terminal_task_is_never_purged(sessions, status) -> None:
    with sessions() as db:
        task_id = _seed_full_task(
            db, username=f"n-{status.name}", anchor=_age(400), status=status
        )

    assert _purge(sessions, task_id) is RetentionPurgeAction.SKIPPED_BUSY

    with sessions() as db:
        assert _counts(db, task_id)["tasks"] == 1


def test_a_task_with_a_pending_command_is_never_purged(sessions) -> None:
    """Accepted work the user cannot see must not be deleted underneath them."""
    with sessions() as db:
        task_id = _seed_full_task(db, username="p1", anchor=_age(400))
        db.add(
            TaskExecutionCommand(
                task_id=task_id,
                actor_user_id=None,
                command_id="cmd-1",
                kind="append",
                payload={},
                status="pending",
            )
        )
        db.commit()

    assert _purge(sessions, task_id) is RetentionPurgeAction.SKIPPED_BUSY

    with sessions() as db:
        assert _counts(db, task_id)["tasks"] == 1


def test_a_task_younger_than_both_periods_is_not_purged(sessions) -> None:
    with sessions() as db:
        task_id = _seed_full_task(db, username="y1", anchor=_age(10))

    assert _purge(sessions, task_id) is RetentionPurgeAction.SKIPPED_BUSY

    with sessions() as db:
        assert _counts(db, task_id)["trace_events"] == 1


def test_unlimited_retention_purges_nothing(sessions) -> None:
    with sessions() as db:
        task_id = _seed_full_task(db, username="u1", anchor=_age(10_000))

    action = _purge(sessions, task_id, conversation_days=None, trace_days=None)

    assert action is RetentionPurgeAction.SKIPPED_BUSY
    with sessions() as db:
        assert _counts(db, task_id)["tasks"] == 1


# ---------------------------------------------------------------------------
# Dry run.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("anchor_days", "expected"),
    [
        (400, RetentionPurgeAction.PURGED_CONVERSATION),
        (100, RetentionPurgeAction.PURGED_TRACES),
    ],
)
def test_dry_run_reports_the_real_action_and_writes_nothing(
    sessions, anchor_days, expected
) -> None:
    with sessions() as db:
        task_id = _seed_full_task(
            db, username=f"d{anchor_days}", anchor=_age(anchor_days)
        )
        before = _counts(db, task_id)

    assert _purge(sessions, task_id, dry_run=True) is expected

    with sessions() as db:
        assert _counts(db, task_id) == before


def test_dry_run_is_idempotent(sessions) -> None:
    with sessions() as db:
        task_id = _seed_full_task(db, username="d-idem", anchor=_age(400))
        before = _counts(db, task_id)

    first = _purge(sessions, task_id, dry_run=True)
    second = _purge(sessions, task_id, dry_run=True)

    assert first is second is RetentionPurgeAction.PURGED_CONVERSATION
    with sessions() as db:
        assert _counts(db, task_id) == before


# ---------------------------------------------------------------------------
# Candidate selection.
# ---------------------------------------------------------------------------


def test_candidates_exclude_live_tasks_and_respect_the_limit(sessions) -> None:
    with sessions() as db:
        expired = [
            _seed_full_task(db, username=f"s{i}", anchor=_age(400)) for i in range(3)
        ]
        _seed_full_task(db, username="s-young", anchor=_age(1))
        _seed_full_task(
            db, username="s-running", anchor=_age(400), status=TaskStatus.RUNNING
        )

        assert select_purge_candidates(
            db,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
            limit=10,
        ) == sorted(expired)
        assert (
            len(
                select_purge_candidates(
                    db,
                    now=NOW,
                    conversation_days=CONVERSATION_DAYS,
                    trace_days=TRACE_DAYS,
                    limit=2,
                )
            )
            == 2
        )


def test_candidates_use_the_shorter_of_the_two_periods(sessions) -> None:
    """A task due only for trace expiry must still be scanned in.

    Selecting on the conversation period alone would leave every
    trace-expiry-only task invisible to the sweep -- the batch would look
    empty while the shorter period silently never applied.
    """
    with sessions() as db:
        trace_only = _seed_full_task(db, username="w1", anchor=_age(100))

        assert select_purge_candidates(
            db,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
            limit=10,
        ) == [trace_only]
        assert (
            select_purge_candidates(
                db,
                now=NOW,
                conversation_days=CONVERSATION_DAYS,
                trace_days=None,
                limit=10,
            )
            == []
        )


def test_candidates_resume_after_a_cursor(sessions) -> None:
    """The scan is resumable, which is what keeps an undeletable page moving."""
    with sessions() as db:
        first = _seed_full_task(db, username="cur1", anchor=_age(400))
        second = _seed_full_task(db, username="cur2", anchor=_age(400))

        assert select_purge_candidates(
            db,
            now=NOW,
            conversation_days=CONVERSATION_DAYS,
            trace_days=TRACE_DAYS,
            limit=10,
            after_task_id=first,
        ) == [second]


def test_a_page_of_undeletable_tasks_does_not_starve_the_ones_behind_it(
    sessions,
    monkeypatch,
) -> None:
    """The regression that made the cursor necessary.

    A task holding an ``active`` interaction row is declined every time, and
    nothing in this loop ever resolves that. Before the cursor, a page filled
    with such tasks was re-read on every sweep: the tasks behind them -- here
    a conversation-expired one -- were never reached, and because a full page
    also shortens the pause, the loop span on it rather than merely stalling.

    Two sweeps, because that is exactly what the bug survived: the first page
    is all refusals, and only a cursor carried into the second reaches the
    victim.
    """
    monkeypatch.setenv("XAGENT_CONVERSATION_RETENTION_DAYS", str(CONVERSATION_DAYS))
    monkeypatch.setenv("XAGENT_TRACE_RETENTION_DAYS", str(TRACE_DAYS))
    with sessions() as db:
        for index in range(3):
            blocked = _seed_full_task(db, username=f"starve{index}", anchor=_age(100))
            anchor_id = _make_trace(db, task_id=blocked, suffix="anchor")
            db.add(
                TaskInteractionRequest(
                    **make_interaction_row(
                        task_id=blocked,
                        resume_trace_event_id=anchor_id,
                        status="active",
                    )
                )
            )
        db.commit()
        victim = _seed_full_task(db, username="starve-victim", anchor=_age(400))

    first = _run_batch(sessions, limit=3)
    assert first.skipped_active_interaction == 3
    assert first.last_task_id is not None

    second = _run_batch(sessions, limit=3, after_task_id=first.last_task_id)

    assert second.purged_conversations == 1
    with sessions() as db:
        assert _counts(db, victim)["tasks"] == 0


def test_no_configured_period_selects_nothing(sessions) -> None:
    with sessions() as db:
        _seed_full_task(db, username="z1", anchor=_age(10_000))

        assert (
            select_purge_candidates(
                db, now=NOW, conversation_days=None, trace_days=None, limit=10
            )
            == []
        )


# ---------------------------------------------------------------------------
# Batch, kill switch, and the dialect gate.
# ---------------------------------------------------------------------------


def test_batch_purges_every_candidate_and_logs_one_audit_line(
    sessions, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("XAGENT_CONVERSATION_RETENTION_DAYS", str(CONVERSATION_DAYS))
    monkeypatch.setenv("XAGENT_TRACE_RETENTION_DAYS", str(TRACE_DAYS))
    with sessions() as db:
        conversation = _seed_full_task(db, username="b1", anchor=_age(400))
        traces = _seed_full_task(db, username="b2", anchor=_age(100))
        _seed_full_task(db, username="b3", anchor=_age(1))

    with caplog.at_level(
        logging.INFO, logger="xagent.web.services.task_retention_purge"
    ):
        report = _run_batch(sessions)

    assert (report.purged_conversations, report.purged_traces) == (1, 1)
    assert report.eligible == 2
    audit = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("retention purge ")
    ]
    assert len(audit) == 1
    assert "purged_conversations=1" in audit[0] and "purged_traces=1" in audit[0]
    with sessions() as db:
        assert _counts(db, conversation)["tasks"] == 0
        assert _counts(db, traces)["trace_events"] == 0


def test_batch_stops_mid_batch_when_the_kill_switch_is_pulled(
    sessions, monkeypatch
) -> None:
    """Already-purged tasks stay purged; the rest are simply not started."""
    monkeypatch.setenv("XAGENT_CONVERSATION_RETENTION_DAYS", str(CONVERSATION_DAYS))
    with sessions() as db:
        first = _seed_full_task(db, username="k1", anchor=_age(400))
        second = _seed_full_task(db, username="k2", anchor=_age(400))

    import xagent.web.services.task_retention_purge as purge_module

    calls = {"n": 0}
    real = purge_module.get_retention_enabled

    def flip_after_one() -> bool:
        calls["n"] += 1
        return calls["n"] <= 1 and real()

    monkeypatch.setattr(purge_module, "get_retention_enabled", flip_after_one)

    report = _run_batch(sessions)

    assert report.purged_conversations == 1
    with sessions() as db:
        remaining = _counts(db, first)["tasks"] + _counts(db, second)["tasks"]
    assert remaining == 1


def test_batch_refuses_a_store_that_cannot_fence(sessions, engine) -> None:
    if engine.dialect.name == "postgresql":
        pytest.skip("this test is about the refusal, which PostgreSQL does not get")
    with pytest.raises(RetentionPurgeUnsupported) as excinfo:
        _run_batch(sessions, require_supported=True)
    assert "postgresql" in str(excinfo.value)


def test_ensure_supported_accepts_postgresql_only(sessions, engine) -> None:
    with sessions() as db:
        if engine.dialect.name == "postgresql":
            ensure_retention_purge_supported(db)
        else:
            with pytest.raises(RetentionPurgeUnsupported):
                ensure_retention_purge_supported(db)


def _run_batch(
    sessions: sessionmaker[Session],
    *,
    require_supported: bool = False,
    **kwargs: object,
) -> RetentionPurgeReport:
    """Run a batch, bypassing the dialect gate unless the test is about it.

    The gate belongs to the job, not to the row semantics (see the module
    docstring of ``task_retention_purge``), so every other batch test here
    would otherwise only ever run on PostgreSQL. The PostgreSQL parameter of
    the shared ``engine`` fixture runs these same tests through the real gate.
    """
    if require_supported:
        return run_retention_purge_batch(sessions, now=NOW, **kwargs)  # type: ignore[arg-type]
    import xagent.web.services.task_retention_purge as purge_module

    original = purge_module.ensure_retention_purge_supported
    purge_module.ensure_retention_purge_supported = lambda db: None  # type: ignore[assignment]
    try:
        return run_retention_purge_batch(sessions, now=NOW, **kwargs)  # type: ignore[arg-type]
    finally:
        purge_module.ensure_retention_purge_supported = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Report and configuration plumbing.
# ---------------------------------------------------------------------------


def test_every_action_increments_exactly_one_counter() -> None:
    """A new action without a counter would go silently uncounted."""
    for action in RetentionPurgeAction:
        before = RetentionPurgeReport()
        after = before.with_action(action)
        changed = [
            name
            for name in (
                "purged_conversations",
                "purged_traces",
                "skipped_busy",
                "skipped_active_interaction",
            )
            if getattr(after, name) != getattr(before, name)
        ]
        assert changed and len(changed) == 1, f"{action} updates {changed}"


def test_audit_line_marks_a_dry_run() -> None:
    assert "dry-run" in RetentionPurgeReport(dry_run=True).audit_line()
    assert "dry-run" not in RetentionPurgeReport().audit_line()


def test_retention_purge_configured_follows_the_switches(monkeypatch) -> None:
    monkeypatch.delenv("XAGENT_CONVERSATION_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("XAGENT_TRACE_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("XAGENT_RETENTION_ENABLED", raising=False)
    assert retention_purge_configured() is False

    monkeypatch.setenv("XAGENT_CONVERSATION_RETENTION_DAYS", "365")
    assert retention_purge_configured() is True

    monkeypatch.setenv("XAGENT_RETENTION_ENABLED", "false")
    assert retention_purge_configured() is False


def test_trace_only_configuration_is_enough_to_run(monkeypatch) -> None:
    """Keeping conversations forever while expiring traces is a supported shape."""
    monkeypatch.delenv("XAGENT_CONVERSATION_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("XAGENT_RETENTION_ENABLED", raising=False)
    monkeypatch.setenv("XAGENT_TRACE_RETENTION_DAYS", "90")
    assert retention_purge_configured() is True


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_loop_gives_up_on_an_unsupported_store(sessions, engine) -> None:
    """A dialect the lock does not fence is configuration, not weather.

    Retrying would log the same refusal forever and, worse, imply that the
    next attempt might succeed.
    """
    if engine.dialect.name == "postgresql":
        pytest.skip("the refusal is what is under test")

    await asyncio.wait_for(run_retention_purge_loop(sessions), timeout=5)


@pytest.mark.asyncio
async def test_the_loop_stops_when_signalled(sessions, monkeypatch) -> None:
    """Shutdown must not wait out a sweep interval measured in days."""
    import xagent.web.services.task_retention_purge as purge_module

    monkeypatch.setattr(
        purge_module, "ensure_retention_purge_supported", lambda db: None
    )
    monkeypatch.setenv("XAGENT_RETENTION_SWEEP_INTERVAL_SECONDS", "86400")

    stop_event = asyncio.Event()
    batches = {"n": 0}
    real_batch = purge_module.run_retention_purge_batch

    def counting_batch(*args, **kwargs):
        batches["n"] += 1
        stop_event.set()
        return real_batch(*args, **kwargs)

    monkeypatch.setattr(purge_module, "run_retention_purge_batch", counting_batch)

    await asyncio.wait_for(
        run_retention_purge_loop(sessions, stop_event=stop_event), timeout=5
    )

    assert batches["n"] == 1, "the loop must not start another sweep after stopping"


@pytest.mark.asyncio
async def test_the_loop_carries_its_cursor_across_sweeps(sessions, monkeypatch) -> None:
    """The starvation regression, at the level that actually ships.

    ``test_a_page_of_undeletable_tasks_does_not_starve_the_ones_behind_it``
    pins the parameter; this pins the loop passing it. Without that, the
    parameter exists and nothing uses it, which is exactly the shape the bug
    had.
    """
    import xagent.web.services.task_retention_purge as purge_module

    monkeypatch.setattr(
        purge_module, "ensure_retention_purge_supported", lambda db: None
    )
    monkeypatch.setenv("XAGENT_CONVERSATION_RETENTION_DAYS", str(CONVERSATION_DAYS))
    monkeypatch.setenv("XAGENT_TRACE_RETENTION_DAYS", str(TRACE_DAYS))
    monkeypatch.setenv("XAGENT_RETENTION_BATCH_SIZE", "3")
    monkeypatch.setenv("XAGENT_RETENTION_BATCH_PAUSE_SECONDS", "0.01")
    monkeypatch.setenv("XAGENT_RETENTION_SWEEP_INTERVAL_SECONDS", "0.01")

    with sessions() as db:
        for index in range(3):
            blocked = _seed_full_task(
                db, username=f"loopstarve{index}", anchor=_age(100)
            )
            anchor_id = _make_trace(db, task_id=blocked, suffix="anchor")
            db.add(
                TaskInteractionRequest(
                    **make_interaction_row(
                        task_id=blocked,
                        resume_trace_event_id=anchor_id,
                        status="active",
                    )
                )
            )
        db.commit()
        victim = _seed_full_task(db, username="loopstarve-victim", anchor=_age(400))

    stop_event = asyncio.Event()
    sweeps = {"n": 0}
    real_batch = purge_module.run_retention_purge_batch

    def counting_batch(*args, **kwargs):
        sweeps["n"] += 1
        if sweeps["n"] >= 4:
            stop_event.set()
        return real_batch(*args, **kwargs)

    monkeypatch.setattr(purge_module, "run_retention_purge_batch", counting_batch)

    await asyncio.wait_for(
        run_retention_purge_loop(sessions, stop_event=stop_event), timeout=10
    )

    with sessions() as db:
        assert _counts(db, victim)["tasks"] == 0, (
            "the loop never got past the page of undeletable tasks"
        )


@pytest.mark.asyncio
async def test_the_loop_survives_a_failing_batch(sessions, monkeypatch) -> None:
    """An unattended sweep has no other surface: one bad batch must not end it."""
    import xagent.web.services.task_retention_purge as purge_module

    monkeypatch.setattr(
        purge_module, "ensure_retention_purge_supported", lambda db: None
    )
    monkeypatch.setenv("XAGENT_RETENTION_SWEEP_INTERVAL_SECONDS", "60")

    stop_event = asyncio.Event()
    calls = {"n": 0}

    def failing_batch(*args, **kwargs):
        calls["n"] += 1
        stop_event.set()
        raise RuntimeError("batch exploded")

    monkeypatch.setattr(purge_module, "run_retention_purge_batch", failing_batch)

    await asyncio.wait_for(
        run_retention_purge_loop(sessions, stop_event=stop_event), timeout=5
    )

    assert calls["n"] == 1
