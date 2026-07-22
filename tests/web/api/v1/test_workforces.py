"""Integration tests for the /v1/workforces/{id}/runs SDK endpoint (#949).

Covers workforce-bound API-key run creation, the shared
/v1/chat/tasks/{task_id} surface reached through a workforce key,
owner-type isolation (agent key vs workforce key), idempotency replay,
and usage metering.

The background-execution kickoff is mocked (``_schedule_bg``) so the
suite exercises HTTP shape + DB rows + the orchestrator's atomic claim
without spinning up an LLM -- same approach as test_tasks.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def mock_schedule_bg():
    """Stub the lease-aware bg scheduler so the orchestrator's atomic
    claim + transcript persist still runs against the real DB; only the
    asyncio.create_task / agent execution is skipped."""
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def _user_id(username: str = "admin") -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User.id).filter(User.username == username).scalar())
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


def _create_active_workforce(
    headers: dict[str, str], name: str = "SDK Workforce", username: str = "admin"
) -> int:
    """Create + publish an active workforce; return its id."""
    owner_id = _user_id(username)
    manager_agent_id = _create_published_agent(owner_id, f"{name} Manager")
    worker_agent_id = _create_published_agent(owner_id, f"{name} Worker")
    resp = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "Coordinates SDK tests",
            "manager_agent_id": manager_agent_id,
            "workers": [
                {
                    "source_type": "existing",
                    "agent_id": worker_agent_id,
                    "alias": "worker-1",
                    "assignment_instructions": "Handle everything",
                    "enabled": True,
                    "sort_order": 1,
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    workforce_id = int(resp.json()["id"])
    published = client.post(f"/api/workforces/{workforce_id}/publish", headers=headers)
    assert published.status_code == 200, published.text
    return workforce_id


def _create_workforce_key(headers: dict[str, str], workforce_id: int) -> str:
    resp = client.post(
        "/api/agent-api-keys",
        headers=headers,
        json={"workforce_id": workforce_id, "label": "sdk"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["full_key"]


def _create_agent_key(headers: dict[str, str]) -> tuple[int, str]:
    agent_resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": "v1 wf test agent",
            "description": "test",
            "instructions": "you are a test agent",
            "execution_mode": "balanced",
        },
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]
    key_resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert key_resp.status_code == 200, key_resp.text
    return agent_id, key_resp.json()["full_key"]


def _manager_agent_id(workforce_id: int) -> int:
    db = _direct_db_session()
    try:
        wf = db.query(Workforce).filter(Workforce.id == workforce_id).one()
        return int(wf.manager_agent_id)
    finally:
        db.close()


# ===== POST /v1/workforces/{id}/runs =====


def test_create_run_happy_path():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    full_key = _create_workforce_key(headers, workforce_id)

    resp = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "coordinate the work"}},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["workforce_id"] == workforce_id
    assert body["agent_id"] == _manager_agent_id(workforce_id)
    assert body["created"] is True
    assert body["status"] in ("pending", "running")
    assert body["run_id"]
    assert body["control_state"] == "running"
    task_id = body["task_id"]

    # DB: manager-agent task created with source='sdk', bound to a run.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.source == "sdk"
        assert task.is_visible is False
        assert int(task.agent_id) == _manager_agent_id(workforce_id)
        from xagent.web.models.workforce import WorkforceRun

        run = (
            db.query(WorkforceRun)
            .filter(WorkforceRun.id == body["workforce_run_id"])
            .one()
        )
        assert int(run.task_id) == task_id
        assert int(run.workforce_id) == workforce_id
    finally:
        db.close()


def test_bound_task_reachable_via_chat_tasks_with_workforce_key():
    """The run's task_id is pollable / has steps through the shared
    /v1/chat/tasks surface using the same workforce key."""
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    full_key = _create_workforce_key(headers, workforce_id)

    run = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "go"}},
    ).json()
    task_id = run["task_id"]

    got = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(full_key))
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["task_id"] == task_id
    assert body["workforce_id"] == workforce_id
    assert body["agent_id"] == _manager_agent_id(workforce_id)

    steps = client.get(f"/v1/chat/tasks/{task_id}/steps", headers=_bearer(full_key))
    assert steps.status_code == 200, steps.text
    assert steps.json()["task_id"] == task_id


def test_agent_key_cannot_create_workforce_run():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    _agent_id, agent_key = _create_agent_key(headers)

    resp = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(agent_key),
        json={"message": {"role": "user", "content": "go"}},
    )
    # get_workforce_from_api_key rejects an agent-bound key with the same
    # opaque 401 as any other auth failure.
    assert resp.status_code == 401, resp.text
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_workforce_key_cannot_create_agent_task():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    full_key = _create_workforce_key(headers, workforce_id)
    manager_id = _manager_agent_id(workforce_id)

    resp = client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={
            "agent_id": manager_id,
            "message": {"role": "user", "content": "direct"},
        },
    )
    # A workforce key resolves to no agent, so agent_id never matches.
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "agent_not_found"


def test_path_workforce_id_mismatch_returns_404():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    other_workforce_id = _create_active_workforce(headers, name="Other Workforce")
    full_key = _create_workforce_key(headers, workforce_id)

    resp = client.post(
        f"/v1/workforces/{other_workforce_id}/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "go"}},
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "workforce_not_found"


def test_missing_authorization_returns_401():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    resp = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        json={"message": {"role": "user", "content": "go"}},
    )
    assert resp.status_code == 401, resp.text
    assert resp.json()["error"]["code"] == "invalid_api_key"


def test_empty_message_returns_422():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    full_key = _create_workforce_key(headers, workforce_id)
    resp = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": ""}},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "invalid_input"


def test_idempotency_replay_returns_same_run_without_double_metering():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    full_key = _create_workforce_key(headers, workforce_id)

    first = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(full_key),
        json={
            "message": {"role": "user", "content": "go"},
            "idempotency_key": "abc-123",
        },
    )
    assert first.status_code == 202, first.text
    assert first.json()["created"] is True
    first_run_id = first.json()["workforce_run_id"]

    second = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(full_key),
        json={
            "message": {"role": "user", "content": "go again"},
            "idempotency_key": "abc-123",
        },
    )
    assert second.status_code == 202, second.text
    assert second.json()["created"] is False
    assert second.json()["workforce_run_id"] == first_run_id

    # Only the real creation counted; the replay must not double-bill.
    stats = client.get("/api/agent-api-keys/stats", headers=headers).json()
    assert stats["calls_this_month"] == 1


def test_run_creation_records_key_usage():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    full_key = _create_workforce_key(headers, workforce_id)

    before = client.get("/api/agent-api-keys/stats", headers=headers).json()
    assert before["calls_this_month"] == 0

    client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "go"}},
    )
    after = client.get("/api/agent-api-keys/stats", headers=headers).json()
    assert after["calls_this_month"] == 1


def test_append_message_through_workforce_key():
    """After the first turn ends, a workforce key can append a next turn
    to the manager task through /v1/chat/tasks/{id}/messages."""
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    full_key = _create_workforce_key(headers, workforce_id)

    run = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "first"}},
    ).json()
    task_id = run["task_id"]

    # Drive the task to a terminal state so append is allowed (append
    # requires COMPLETED/FAILED; RUNNING/PENDING both 409).
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = TaskStatus.COMPLETED
        db.commit()
    finally:
        db.close()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={
            "workforce_id": workforce_id,
            "message": {"role": "user", "content": "next"},
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["task_id"] == task_id
    assert body["workforce_id"] == workforce_id


def test_append_rejects_mismatched_workforce_id():
    headers = _admin_headers()
    workforce_id = _create_active_workforce(headers)
    full_key = _create_workforce_key(headers, workforce_id)

    run = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "first"}},
    ).json()
    task_id = run["task_id"]

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = TaskStatus.COMPLETED
        db.commit()
    finally:
        db.close()

    resp = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(full_key),
        json={"workforce_id": 999999, "message": {"role": "user", "content": "next"}},
    )
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "workforce_not_found"


def test_cross_workforce_task_isolation():
    """A workforce key cannot read a task belonging to another workforce."""
    headers = _admin_headers()
    wf_a = _create_active_workforce(headers, name="Workforce A")
    wf_b = _create_active_workforce(headers, name="Workforce B")
    key_a = _create_workforce_key(headers, wf_a)
    key_b = _create_workforce_key(headers, wf_b)

    run_a = client.post(
        f"/v1/workforces/{wf_a}/runs",
        headers=_bearer(key_a),
        json={"message": {"role": "user", "content": "a"}},
    ).json()
    task_a = run_a["task_id"]

    # key_b must not see workforce A's task.
    resp = client.get(f"/v1/chat/tasks/{task_a}", headers=_bearer(key_b))
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["code"] == "task_not_found"
