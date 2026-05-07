from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.user import User
from xagent.web.services.task_execution_context_service import (
    load_task_execution_context_messages,
    load_task_execution_recovery_state,
    summarize_tool_event,
)


def _create_db_session():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return SessionLocal()


def _create_task(db_session):
    user = User(username="tester", password_hash="hashed_password", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    task = Task(
        user_id=int(user.id),
        title="Execution context task",
        description="Task execution context",
        status=TaskStatus.COMPLETED,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


def test_load_task_execution_context_messages_summarizes_list_files_results():
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        db_session.add(
            TraceEvent(
                task_id=int(task.id),
                build_id=None,
                event_id="tool-event-1",
                event_type="tool_execution_end",
                timestamp=datetime.now(timezone.utc),
                step_id="step_1",
                parent_event_id=None,
                data={
                    "tool_name": "list_files",
                    "success": True,
                    "result": {
                        "input": [
                            {"filename": "foo.docx"},
                            {"filename": "bar.pdf"},
                        ],
                        "output": [],
                        "temp": [],
                        "workspace": [],
                    },
                },
            )
        )
        db_session.commit()

        messages = load_task_execution_context_messages(db_session, int(task.id))

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "Tool list_files previously returned" in messages[0]["content"]
        assert '"input"' in messages[0]["content"]
        assert "foo.docx" in messages[0]["content"]
        assert "bar.pdf" in messages[0]["content"]
    finally:
        db_session.close()


def test_summarize_tool_event_uses_generic_fallback_for_unknown_tools():
    summary = summarize_tool_event(
        {
            "tool_name": "web_search",
            "success": True,
            "result": {"top_result": "https://example.com", "title": "Example"},
        }
    )

    assert summary is not None
    assert summary.startswith("- Tool web_search previously returned: ")
    assert "top_result" in summary


@pytest.mark.asyncio
async def test_load_task_execution_recovery_state_recovers_skill_context(monkeypatch):
    db_session = _create_db_session()
    try:
        task = _create_task(db_session)
        db_session.add(
            TraceEvent(
                task_id=int(task.id),
                build_id=None,
                event_id="skill-event-1",
                event_type="skill_select_end",
                timestamp=datetime.now(timezone.utc),
                step_id=None,
                parent_event_id=None,
                data={
                    "selected": True,
                    "skill_name": "translator",
                },
            )
        )
        db_session.commit()

        async def fake_load_skill_context_by_name(skill_name: str):
            assert skill_name == "translator"
            return "## Available Skill: translator\n\nUse translation workflow."

        monkeypatch.setattr(
            "xagent.web.services.task_execution_context_service._load_skill_context_by_name",
            fake_load_skill_context_by_name,
        )

        recovery_state = await load_task_execution_recovery_state(
            db_session, int(task.id)
        )

        assert recovery_state["skill_context"] == (
            "## Available Skill: translator\n\nUse translation workflow."
        )
        assert recovery_state["messages"] == []
    finally:
        db_session.close()
