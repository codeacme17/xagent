"""Workforce trigger deployment channel (#950).

Covers the /api/workforces/{workforce_id}/triggers route group, the firing
path that routes workforce triggers through create_workforce_run_record (so
validate_workforce_for_run applies), and the defense-in-depth guard that
keeps manager-agent triggers from ever constructing a plain Task.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.trigger import TriggerRun, TriggerRunStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import WorkforceRun
from xagent.web.services.connector_runtime import (
    set_connector_runtime_resolver_for_testing,
)
from xagent.web.services.trigger_providers import sign_webhook_payload
from xagent.web.services.triggers import (
    TriggerRunPreparationError,
    dispatch_pending_trigger_runs,
    prepare_trigger_run,
    scan_due_scheduled_triggers,
)

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def reset_connector_runtime_resolver():
    set_connector_runtime_resolver_for_testing(None)
    yield
    set_connector_runtime_resolver_for_testing(None)


@pytest.fixture(autouse=True)
def mock_bg_scheduler():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _user_id(username: str = "admin") -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).first()
        assert user is not None
        return int(user.id)
    finally:
        db.close()


def _create_published_agent(user_id: int, name: str) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=user_id,
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _create_workforce(
    headers: dict[str, str],
    *,
    name: str = "Trigger Workforce",
    username: str = "admin",
    publish: bool = False,
) -> int:
    owner_id = _user_id(username)
    manager_agent_id = _create_published_agent(owner_id, f"{name} Manager")
    worker_agent_id = _create_published_agent(owner_id, f"{name} Worker")
    response = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "Coordinates triggered work",
            "manager_agent_id": manager_agent_id,
            "workers": [
                {
                    "source_type": "existing",
                    "agent_id": worker_agent_id,
                    "alias": "worker-1",
                    "assignment_instructions": "Handle trigger events",
                    "enabled": True,
                    "sort_order": 1,
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    workforce_id = int(response.json()["id"])
    if publish:
        published = client.post(
            f"/api/workforces/{workforce_id}/publish", headers=headers
        )
        assert published.status_code == 200, published.text
    return workforce_id


def _create_workforce_webhook_trigger(
    headers: dict[str, str], workforce_id: int, **overrides
) -> dict:
    payload = {"type": "webhook", "name": "Workforce webhook", **overrides}
    response = client.post(
        f"/api/workforces/{workforce_id}/triggers",
        headers=headers,
        json=payload,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _signed_webhook_headers(
    secret: str, raw_body: bytes, *, event_id: str | None = None
) -> dict[str, str]:
    timestamp = str(int(time.time()))
    headers = {
        "x-xagent-signature": sign_webhook_payload(secret, timestamp, raw_body),
        "x-xagent-timestamp": timestamp,
    }
    if event_id:
        headers["x-xagent-event-id"] = event_id
    return headers


def test_workforce_trigger_crud_roundtrip() -> None:
    headers = _admin_headers()
    workforce_id = _create_workforce(headers)

    created = _create_workforce_webhook_trigger(headers, workforce_id)
    assert created["workforce_id"] == workforce_id
    assert created["agent_id"] is None
    assert created["callback_id"]
    assert created["webhook_secret"]

    listed = client.get(
        f"/api/workforces/{workforce_id}/triggers", headers=headers
    )
    assert listed.status_code == 200
    rows = listed.json()
    assert [row["id"] for row in rows] == [created["id"]]

    updated = client.patch(
        f"/api/workforces/{workforce_id}/triggers/{created['id']}",
        headers=headers,
        json={"name": "Renamed", "enabled": False},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "Renamed"
    assert updated.json()["enabled"] is False

    runs = client.get(
        f"/api/workforces/{workforce_id}/triggers/{created['id']}/runs",
        headers=headers,
    )
    assert runs.status_code == 200
    assert runs.json() == []

    deleted = client.delete(
        f"/api/workforces/{workforce_id}/triggers/{created['id']}",
        headers=headers,
    )
    assert deleted.status_code == 200
    assert (
        client.get(f"/api/workforces/{workforce_id}/triggers", headers=headers).json()
        == []
    )


def test_workforce_trigger_routes_reject_other_user() -> None:
    headers = _admin_headers()
    workforce_id = _create_workforce(headers)
    trigger = _create_workforce_webhook_trigger(headers, workforce_id)

    other_headers = _register_second_user()
    assert (
        client.get(
            f"/api/workforces/{workforce_id}/triggers", headers=other_headers
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/workforces/{workforce_id}/triggers",
            headers=other_headers,
            json={"type": "webhook"},
        ).status_code
        == 403
    )
    assert (
        client.patch(
            f"/api/workforces/{workforce_id}/triggers/{trigger['id']}",
            headers=other_headers,
            json={"name": "hijack"},
        ).status_code
        == 403
    )
    assert (
        client.delete(
            f"/api/workforces/{workforce_id}/triggers/{trigger['id']}",
            headers=other_headers,
        ).status_code
        == 403
    )


def test_workforce_trigger_scoped_to_its_workforce() -> None:
    headers = _admin_headers()
    workforce_a = _create_workforce(headers, name="Workforce A")
    workforce_b = _create_workforce(headers, name="Workforce B")
    trigger = _create_workforce_webhook_trigger(headers, workforce_a)

    cross = client.patch(
        f"/api/workforces/{workforce_b}/triggers/{trigger['id']}",
        headers=headers,
        json={"name": "cross"},
    )
    assert cross.status_code == 404
    assert (
        client.delete(
            f"/api/workforces/{workforce_b}/triggers/{trigger['id']}",
            headers=headers,
        ).status_code
        == 404
    )


def test_workforce_trigger_test_fire_creates_workforce_run(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    workforce_id = _create_workforce(headers, publish=True)
    trigger = _create_workforce_webhook_trigger(headers, workforce_id)

    fired = client.post(
        f"/api/workforces/{workforce_id}/triggers/{trigger['id']}/test",
        headers=headers,
        json={"payload": {"subject": "hello"}},
    )
    assert fired.status_code == 200, fired.text
    body = fired.json()
    assert body["duplicate"] is False
    run_body = body["trigger_run"]
    assert run_body["task_id"] is not None

    db = _direct_db_session()
    try:
        run = db.query(TriggerRun).filter(TriggerRun.id == run_body["id"]).one()
        task = db.query(Task).filter(Task.id == int(run.task_id)).one()
        workforce_run = (
            db.query(WorkforceRun)
            .filter(WorkforceRun.workforce_id == workforce_id)
            .one()
        )
        assert str(task.source) == "trigger"
        assert bool(task.is_visible) is False
        assert int(workforce_run.task_id) == int(task.id)
        assert str(workforce_run.idempotency_key) == f"trigger:{int(run.id)}"
        config = dict(task.agent_config or {})
        # Trigger execution context is merged on top of (not instead of) the
        # workforce task config: losing workforce_run_id drops delegation.
        assert config.get("workforce_run_id") == int(workforce_run.id)
        assert config.get("trigger_id") == trigger["id"]
        assert config.get("trigger_run_id") == int(run.id)
        assert config.get("trigger_test") is True
    finally:
        db.close()


def test_workforce_webhook_callback_fires_and_deduplicates(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    workforce_id = _create_workforce(headers, publish=True)
    trigger = _create_workforce_webhook_trigger(headers, workforce_id)

    url = f"/api/triggers/callback/webhook/{trigger['callback_id']}"
    raw_body = b'{"subject": "hello"}'

    unsigned = client.post(url, content=raw_body)
    assert unsigned.status_code == 401

    event_headers = _signed_webhook_headers(
        trigger["webhook_secret"], raw_body, event_id="evt-wf-1"
    )
    first = client.post(url, headers=event_headers, content=raw_body)
    assert first.status_code == 200, first.text
    assert first.json()["outcome"] == "accepted"
    assert len(first.json()["trigger_run_ids"]) == 1

    second = client.post(url, headers=event_headers, content=raw_body)
    assert second.status_code == 200, second.text
    assert second.json()["duplicates"] == 1

    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 1
        assert (
            db.query(WorkforceRun)
            .filter(WorkforceRun.workforce_id == workforce_id)
            .count()
            == 1
        )
        task = db.query(Task).filter(Task.source == "trigger").one()
        assert dict(task.agent_config or {}).get("workforce_run_id") is not None
    finally:
        db.close()


def test_workforce_trigger_fire_fails_when_workforce_not_active() -> None:
    headers = _admin_headers()
    workforce_id = _create_workforce(headers)  # stays draft
    trigger = _create_workforce_webhook_trigger(headers, workforce_id)

    fired = client.post(
        f"/api/workforces/{workforce_id}/triggers/{trigger['id']}/test",
        headers=headers,
        json={"payload": {"subject": "hello"}},
    )
    # validate_workforce_for_run rejected the run; the FAILED trigger run is
    # kept so an idempotent redelivery can repair it after publish.
    assert fired.status_code == 500
    assert "active" in fired.json()["detail"]

    db = _direct_db_session()
    try:
        run = db.query(TriggerRun).one()
        assert str(run.status) == TriggerRunStatus.FAILED.value
        assert run.task_id is None
        assert (
            db.query(WorkforceRun)
            .filter(WorkforceRun.workforce_id == workforce_id)
            .count()
            == 0
        )
    finally:
        db.close()


def test_manager_agent_trigger_cannot_fire_defense_in_depth() -> None:
    """A trigger row bound to a generated manager agent must never build a Task."""
    headers = _admin_headers()
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "Soon Manager",
            "description": "test",
            "instructions": "You are a test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    agent_id = int(resp.json()["id"])
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Manager webhook"},
    )
    assert created.status_code == 200, created.text
    trigger_id = int(created.json()["id"])

    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        setattr(agent, "origin", AgentOrigin.WORKFORCE_GENERATED_MANAGER.value)
        db.commit()

        from xagent.web.models.trigger import AgentTrigger

        trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        )
        with pytest.raises(TriggerRunPreparationError) as excinfo:
            prepare_trigger_run(
                db,
                trigger=trigger,
                event_payload={"subject": "hello"},
                source_event_id="evt-manager-1",
            )
        assert "workforce manager agent" in str(excinfo.value)

        run = db.query(TriggerRun).one()
        assert str(run.status) == TriggerRunStatus.FAILED.value
        assert run.task_id is None
        assert db.query(Task).count() == 0
    finally:
        db.close()


def test_scheduled_workforce_trigger_scan_and_dispatch(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    workforce_id = _create_workforce(headers, publish=True)
    created = client.post(
        f"/api/workforces/{workforce_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Workforce schedule",
            "config": {
                "interval_seconds": 3600,
                "next_run_at": "2020-01-01T00:00:00+00:00",
            },
        },
    )
    assert created.status_code == 200, created.text

    db = _direct_db_session()
    try:
        runs = scan_due_scheduled_triggers(db)
        assert len(runs) == 1
        run = runs[0]
        assert run.task_id is not None
        assert str(run.status) == TriggerRunStatus.PENDING.value

        workforce_run = (
            db.query(WorkforceRun)
            .filter(WorkforceRun.workforce_id == workforce_id)
            .one()
        )
        assert int(workforce_run.task_id) == int(run.task_id)

        assert asyncio.run(dispatch_pending_trigger_runs(db)) == 1
        db.expire_all()
        started = db.query(TriggerRun).filter(TriggerRun.id == int(run.id)).one()
        assert str(started.status) == TriggerRunStatus.RUNNING.value
        task = db.query(Task).filter(Task.id == int(run.task_id)).one()
        assert task.status == TaskStatus.RUNNING
    finally:
        db.close()
