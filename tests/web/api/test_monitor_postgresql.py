"""Monitoring endpoints against a real PostgreSQL database.

``monitor.py`` reads fields out of ``TraceEvent.data`` (a ``Column(JSON)``)
through a per-dialect helper. Its PostgreSQL branch is the one branch the
SQLite suite never executes, and it shipped ``~?`` -- an operator PostgreSQL
does not have for ``json``, ``jsonb`` or ``text``. Every query through that
branch therefore failed on PostgreSQL while SQLite CI stayed green, and each
call site swallowed the error into an empty result (#1149).

The two symptoms differ by endpoint, which is why all three are covered here:

- ``/monitor/stats`` raised. Its ``except`` substitutes ``active_models = 0``,
  but the failed statement leaves the transaction aborted, so the *next*
  query in the same session (the ``llm_call_end`` scan) raised
  ``PendingRollbackError`` outside any local handler and the request became a
  500.
- ``/monitor/popular-tools`` and ``/monitor/model-stats`` issue no further
  query after their handler, so they returned HTTP 200 with an empty list --
  a dashboard of zeros with nothing but a log line to show for it.

Fixture pattern copied from
``tests/web/services/test_task_status_storage_postgresql.py`` (skip-if-unset
via ``XAGENT_TEST_POSTGRES_URL``; CI provides it in the PostgreSQL job).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Iterator

import pytest
from sqlalchemy.orm import Session

from xagent.web.api.monitor import (
    get_model_stats,
    get_monitoring_stats,
    get_popular_tools,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.user import User

# A JSON string value PostgreSQL accepts into a ``json`` column but refuses to
# convert to text: ``json ->> key`` raises "unsupported Unicode escape
# sequence" for it. This is the payload shape the dialect branch's guard
# exists to keep out of ``->>``.
NUL_PAYLOAD = "\x00"


@pytest.fixture()
def pg_session() -> Iterator[Session]:
    """Session against a real PostgreSQL, where the json operators are real.

    SQLite resolves the helper's other branch, so a dialect-specific operator
    error can only surface here.
    """
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def _seed_admin_with_trace_events(db: Session) -> User:
    """One admin, one task, and the trace events the three endpoints count.

    Every payload carrying ``NUL_PAYLOAD`` is one the guard must drop without
    failing the surrounding query.
    """
    admin = User(username="monitor-pg-admin", password_hash="hash", is_admin=True)
    db.add(admin)
    db.commit()
    db.refresh(admin)

    task = Task(user_id=admin.id, title="monitor-pg-task")
    db.add(task)
    db.commit()
    db.refresh(task)

    now = datetime.now()
    payloads: list[tuple[str, dict[str, Any]]] = [
        ("llm_call_start", {"model_name": "gpt-4o", "step_id": "s1", "attempt": 1}),
        ("llm_call_start", {"model_name": "gpt-4o", "step_id": "s2", "attempt": 1}),
        ("llm_call_start", {"model_name": "claude-opus", "step_id": "s3"}),
        ("llm_call_start", {"model_name": "nul-model", "note": NUL_PAYLOAD}),
        ("tool_execution_start", {"tool_name": "calculator"}),
        ("tool_execution_start", {"tool_name": "calculator"}),
        ("tool_execution_start", {"tool_name": "web_search"}),
        ("tool_execution_start", {"tool_name": "nul-tool", "note": NUL_PAYLOAD}),
    ]
    for index, (event_type, data) in enumerate(payloads):
        db.add(
            TraceEvent(
                task_id=task.id,
                event_id=f"monitor-pg-{index}",
                event_type=event_type,
                timestamp=now,
                data=data,
            )
        )
    db.commit()
    return admin


@pytest.mark.postgresql
async def test_monitoring_stats_counts_active_models_on_postgresql(
    pg_session: Session,
) -> None:
    """/monitor/stats reports the real model count instead of failing.

    Before the fix this raised a 500: the rejected operator aborted the
    transaction and the next query in the handler could not run.
    """
    admin = _seed_admin_with_trace_events(pg_session)

    stats = await get_monitoring_stats(db=pg_session, current_user=admin)

    # gpt-4o and claude-opus; the NUL-carrying payload is dropped by the guard.
    assert stats["activeModels"] == 2


@pytest.mark.postgresql
async def test_popular_tools_returns_usage_counts_on_postgresql(
    pg_session: Session,
) -> None:
    """/monitor/popular-tools returns real rows instead of an empty list."""
    admin = _seed_admin_with_trace_events(pg_session)

    tools = await get_popular_tools(db=pg_session, current_user=admin)

    assert [(entry["name"], entry["usage_count"]) for entry in tools] == [
        ("calculator", 2),
        ("web_search", 1),
    ]


@pytest.mark.postgresql
async def test_model_stats_returns_per_model_calls_on_postgresql(
    pg_session: Session,
) -> None:
    """/monitor/model-stats returns real rows instead of an empty list."""
    admin = _seed_admin_with_trace_events(pg_session)

    stats = await get_model_stats(db=pg_session, current_user=admin)

    assert {entry["name"]: entry["total_tasks"] for entry in stats} == {
        "gpt-4o": 2,
        "claude-opus": 1,
    }
