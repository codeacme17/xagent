"""Per-guest isolation for public share links (#973, PR1).

A single share link can be opened by many anonymous visitors. Each
``POST /api/share/auth`` mints a fresh, server-owned ``guest_id`` signed into
the guest JWT; every task created by a guest is stamped with that id, and
``get_task_for_share_context`` requires the caller's ``guest_id`` to match the
task's. This prevents guest A — holding a perfectly valid share JWT for the
same shared entity — from reading or continuing guest B's conversation.

Covers both the agent-share and workforce-share paths, plus fail-closed
rejection of legacy tokens that predate the ``guest_id`` claim.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.web.api.public_chat_access import create_public_chat_access_token
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.services import workforce_runs as workforce_runs_service

from .conftest import _admin_headers, _direct_db_session, _setup_admin, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _user_id(username: str = "admin") -> int:
    _setup_admin()
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == username).one().id)
    finally:
        db.close()


def _create_published_agent(name: str, share_token: str) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=_user_id(),
            name=name,
            description="d",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            share_enabled=True,
            share_token=share_token,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _create_workforce(name: str) -> int:
    headers = _admin_headers()
    manager_agent_id = _create_published_agent(f"{name} Manager", f"{name}-mgr-tok")
    worker_agent_id = _create_published_agent(f"{name} Worker", f"{name}-wrk-tok")
    response = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "isolation tests",
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
    assert response.status_code == 200, response.text
    workforce_id = int(response.json()["id"])
    published = client.post(f"/api/workforces/{workforce_id}/publish", headers=headers)
    assert published.status_code == 200, published.text
    return workforce_id


def _enable_workforce_share(workforce_id: int) -> str:
    response = client.post(
        f"/api/workforces/{workforce_id}/share-link", headers=_admin_headers()
    )
    assert response.status_code == 200, response.text
    return str(response.json()["share_token"])


def _authenticate_share_guest(share_token: str) -> dict[str, str]:
    response = client.post("/api/share/auth", json={"share_token": share_token})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _stub_begin_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(**_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(background_task=None)

    monkeypatch.setattr(
        workforce_runs_service.TaskTurnOrchestrator, "begin_turn", _stub
    )


def _upload_to_task(headers: dict[str, str], task_id: int) -> Any:
    return client.post(
        "/api/share/files/upload",
        headers=headers,
        data={"task_type": "task", "task_id": str(task_id)},
        files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
    )


# ===== distinct guests get distinct server-minted ids =====


def test_share_auth_mints_distinct_guest_ids_per_call() -> None:
    """Two auths of the *same* link are two independent anonymous guests: the
    server mints a fresh guest id each time (never client-supplied)."""
    agent_id = _create_published_agent("Distinct Guest Agent", "distinct-tok")
    assert agent_id

    first = client.post("/api/share/auth", json={"share_token": "distinct-tok"})
    second = client.post("/api/share/auth", json={"share_token": "distinct-tok"})
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["access_token"] != second.json()["access_token"]


# ===== agent-share cross-guest isolation =====


def test_agent_share_guest_cannot_touch_other_guests_task() -> None:
    agent_id = _create_published_agent("Iso Agent", "iso-agent-tok")
    guest_a = _authenticate_share_guest("iso-agent-tok")
    guest_b = _authenticate_share_guest("iso-agent-tok")

    created = client.post(
        "/api/share/chat/task/create",
        headers=guest_b,
        json={"title": "b task", "description": "b task"},
    )
    assert created.status_code == 200, created.text
    task_b = int(created.json()["task_id"])

    # Guest B stamped the task; the config carries B's guest id.
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_b).one()
        assert int(task.agent_id) == agent_id
        assert isinstance(task.agent_config.get("guest_id"), str)
    finally:
        db.close()

    # Guest A holds a valid share JWT for the same agent, but must not reach B.
    assert _upload_to_task(guest_a, task_b).status_code == 403
    # Guest B still reaches its own task.
    assert _upload_to_task(guest_b, task_b).status_code == 200


# ===== workforce-share cross-guest isolation =====


def test_workforce_share_guest_cannot_touch_other_guests_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_workforce("Iso WF")
    token = _enable_workforce_share(workforce_id)
    guest_a = _authenticate_share_guest(token)
    guest_b = _authenticate_share_guest(token)
    _stub_begin_turn(monkeypatch)

    created = client.post(
        "/api/share/chat/task/create",
        headers=guest_b,
        json={"title": "b run", "description": "b run"},
    )
    assert created.status_code == 200, created.text
    task_b = int(created.json()["task_id"])

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_b).one()
        assert isinstance(task.agent_config.get("guest_id"), str)
    finally:
        db.close()

    assert _upload_to_task(guest_a, task_b).status_code == 403
    assert _upload_to_task(guest_b, task_b).status_code == 200


# ===== fail-closed on legacy tokens without a guest_id claim =====


def test_legacy_agent_share_token_without_guest_id_is_rejected() -> None:
    agent_id = _create_published_agent("Legacy Agent", "legacy-agent-tok")
    legacy = create_public_chat_access_token(
        {
            "sub": "admin",
            "user_id": _user_id(),
            "auth_mode": "share",
            "share_agent_id": agent_id,
            "share_token": "legacy-agent-tok",
        }
    )
    response = client.post(
        "/api/share/chat/task/create",
        headers={"Authorization": f"Bearer {legacy}"},
        json={"title": "hi", "description": "hi"},
    )
    assert response.status_code == 401, response.text


def test_legacy_workforce_share_token_without_guest_id_is_rejected() -> None:
    workforce_id = _create_workforce("Legacy WF")
    token = _enable_workforce_share(workforce_id)
    legacy = create_public_chat_access_token(
        {
            "sub": "admin",
            "user_id": _user_id(),
            "auth_mode": "share",
            "share_workforce_id": workforce_id,
            "share_token": token,
        }
    )
    response = client.post(
        "/api/share/chat/task/create",
        headers={"Authorization": f"Bearer {legacy}"},
        json={"title": "hi", "description": "hi"},
    )
    assert response.status_code == 401, response.text
