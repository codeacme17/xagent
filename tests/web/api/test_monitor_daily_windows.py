"""Daily-window boundaries in ``/monitor/stats`` and ``/monitor/dashboard-stats``.

``TraceEvent.timestamp`` and ``Task.updated_at`` are timezone-aware columns
that production writes aware UTC into, but the "today" boundaries in
``monitor.py`` were built from naive local wall-clock time (#1256). On any
host east or west of UTC the daily windows shift by the UTC offset: on a
UTC+8 server "today's calls" covered 08:00 yesterday through 08:00 today,
and the same skew moved ``activeModels`` and ``activeAgents``.

These tests pin the UTC frame. The server clock is frozen at an instant
where the local calendar day (UTC+8) has already rolled past the UTC day,
so a boundary built from naive local time lands a full day ahead of the
UTC data -- every window then counts zero, which cannot be mistaken for an
off-by-one at the boundary. Rows are seeded just before and just after UTC
midnight so the correct window is told apart from both the naive-local one
and a window that simply counts everything.

The SQLite path is enough here: the sqlite dialect drops the UTC offset at
bind time on both the stored rows and the query boundary, so the comparison
happens in UTC wall-clock on both sides exactly when the boundary is built
in UTC -- and lands a day off when it is built from local time. The
session-``TimeZone``-dependent coercion PostgreSQL applies to a naive
literal lives in the sibling ``test_monitor_postgresql.py`` suite's domain.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from xagent.web.api import monitor as monitor_module
from xagent.web.api.monitor import get_dashboard_stats, get_monitoring_stats
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User

from .conftest import _direct_db_session

pytestmark = pytest.mark.usefixtures("_test_db")

# 18:00 UTC on a UTC+8 server: the local wall clock reads 02:00 on the
# *next* calendar day. A naive-local "today" boundary is therefore
# 2026-08-18 00:00 while the correct UTC boundary is 2026-08-17 00:00 --
# a full-day gap no seeded row can straddle by accident.
_NOW_UTC = datetime(2026, 8, 17, 18, 0, tzinfo=timezone.utc)
_NOW_LOCAL_NAIVE = datetime(2026, 8, 18, 2, 0)

_UTC_TODAY_EARLY = datetime(2026, 8, 17, 0, 1, tzinfo=timezone.utc)
_UTC_TODAY_LATE = datetime(2026, 8, 17, 17, 0, tzinfo=timezone.utc)
_UTC_YESTERDAY_LATE = datetime(2026, 8, 16, 23, 59, tzinfo=timezone.utc)


class _FrozenDatetime(datetime):
    """``datetime`` whose ``now()`` is pinned to ``_NOW_UTC``.

    The naive branch returns what a UTC+8 server's wall clock would show at
    that instant, so the pre-fix call sites reproduce the skew
    deterministically instead of only when the test happens to run between
    16:00 and 24:00 UTC.
    """

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if tz is None:
            return _NOW_LOCAL_NAIVE
        return _NOW_UTC.astimezone(tz)


@pytest.fixture
def _utc_plus_8_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monitor_module, "datetime", _FrozenDatetime)


def _seed_admin(db: Session) -> User:
    admin = User(username="window-admin", password_hash="hash", is_admin=True)
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return admin


def _seed_task(db: Session, admin: User, *, title: str, updated_at: datetime) -> Task:
    task = Task(
        user_id=admin.id,
        title=title,
        status=TaskStatus.RUNNING,
        updated_at=updated_at,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _seed_boundary_straddling_events(db: Session, task: Task) -> None:
    """Three events: one just before UTC midnight, two after.

    The two "today" events use different event types so both arms of the
    ``event_type.in_([...])`` filter contribute, and only the LLM call
    carries a model name so ``activeModels`` counts exactly one distinct
    model when the window is right and zero when it is a day ahead.
    """
    for event_id, event_type, timestamp, data in [
        (
            "yesterday-llm",
            "llm_call_start",
            _UTC_YESTERDAY_LATE,
            {"model_name": "stale-model"},
        ),
        (
            "today-llm",
            "llm_call_start",
            _UTC_TODAY_EARLY,
            {"model_name": "fresh-model"},
        ),
        ("today-tool", "tool_execution_start", _UTC_TODAY_LATE, None),
    ]:
        db.add(
            TraceEvent(
                task_id=task.id,
                event_id=event_id,
                event_type=event_type,
                timestamp=timestamp,
                data=data,
            )
        )
    db.commit()


async def test_stats_today_window_is_the_utc_day(_utc_plus_8_server: None) -> None:
    """``todayCalls``/``activeModels`` count from UTC midnight, not local.

    Expected: the two events after 2026-08-17 00:00 UTC, and the one model
    they name. The naive-local boundary (2026-08-18 00:00) yields 0 for
    both; a window ignoring the boundary altogether yields 3 and 2.
    """
    db = _direct_db_session()
    try:
        admin = _seed_admin(db)
        task = _seed_task(db, admin, title="stats-task", updated_at=_UTC_TODAY_LATE)
        _seed_boundary_straddling_events(db, task)

        stats = await get_monitoring_stats(db=db, current_user=admin)

        assert stats["todayCalls"] == 2
        assert stats["activeModels"] == 1
    finally:
        db.close()


async def test_dashboard_windows_are_the_utc_day(_utc_plus_8_server: None) -> None:
    """``todayCalls`` and ``activeAgents`` use the UTC day boundary.

    One RUNNING task last updated late yesterday (UTC) and one updated
    today: only the latter is active. Both tasks' events straddle the same
    boundary as the stats test, so ``todayCalls`` again expects 2.
    """
    db = _direct_db_session()
    try:
        admin = _seed_admin(db)
        stale_task = _seed_task(
            db, admin, title="stale-task", updated_at=_UTC_YESTERDAY_LATE
        )
        fresh_task = _seed_task(
            db, admin, title="fresh-task", updated_at=_UTC_TODAY_LATE
        )
        del stale_task  # seeded for its updated_at; nothing else to assert on
        _seed_boundary_straddling_events(db, fresh_task)

        stats = await get_dashboard_stats(db=db, current_user=admin)

        assert stats["todayCalls"] == 2
        assert stats["activeAgents"] == 1
    finally:
        db.close()
