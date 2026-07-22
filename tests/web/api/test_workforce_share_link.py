"""Tests for the workforce shareable-link deployment channel (#947).

Covers: owner-side share-link management endpoints (enable / rotate /
disable, gated on ``status == "active"``), public share auth resolving a
workforce deployment token, guest task creation entering through
``create_workforce_run(source="shared_link")``, share-context task scoping,
and the WS APPEND path flipping the run back to ``running``.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.deployment import Deployment, DeploymentOwnerType
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceRun
from xagent.web.services import workforce_runs as workforce_runs_service

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    client,
)

pytestmark = pytest.mark.usefixtures("_test_db")


def _user_id(username: str = "admin") -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).one()
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


def _create_workforce(name: str, *, publish: bool = True) -> int:
    headers = _admin_headers()
    owner_id = _user_id()
    manager_agent_id = _create_published_agent(owner_id, f"{name} Manager")
    worker_agent_id = _create_published_agent(owner_id, f"{name} Worker")
    response = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "Coordinates share-link tests",
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
    if publish:
        published = client.post(
            f"/api/workforces/{workforce_id}/publish", headers=headers
        )
        assert published.status_code == 200, published.text
    return workforce_id


def _enable_share(workforce_id: int) -> str:
    response = client.post(
        f"/api/workforces/{workforce_id}/share-link", headers=_admin_headers()
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["share_enabled"] is True
    token = body["share_token"]
    assert isinstance(token, str) and token
    return str(token)


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


# ===== Share-link management endpoints =====


def test_share_link_requires_active_workforce() -> None:
    workforce_id = _create_workforce("Draft Share Workforce", publish=False)

    response = client.post(
        f"/api/workforces/{workforce_id}/share-link", headers=_admin_headers()
    )
    assert response.status_code == 400, response.text

    rotated = client.post(
        f"/api/workforces/{workforce_id}/share-link/rotate", headers=_admin_headers()
    )
    assert rotated.status_code == 400, rotated.text


def test_share_link_enable_rotate_disable_lifecycle() -> None:
    workforce_id = _create_workforce("Lifecycle Share Workforce")
    headers = _admin_headers()

    # Default state: share is opt-in and starts disabled with no token.
    initial = client.get(f"/api/workforces/{workforce_id}/share-link", headers=headers)
    assert initial.status_code == 200, initial.text
    assert initial.json() == {
        "workforce_id": workforce_id,
        "share_enabled": False,
        "share_token": None,
        "share_updated_at": None,
    }

    token = _enable_share(workforce_id)

    db = _direct_db_session()
    try:
        deployment = (
            db.query(Deployment)
            .filter(
                Deployment.owner_type == DeploymentOwnerType.WORKFORCE.value,
                Deployment.owner_id == workforce_id,
            )
            .one()
        )
        assert deployment.share_enabled is True
        assert deployment.share_token == token
        assert deployment.share_updated_at is not None
    finally:
        db.close()

    fetched = client.get(f"/api/workforces/{workforce_id}/share-link", headers=headers)
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["share_token"] == token

    # Re-enabling keeps the existing token stable.
    assert _enable_share(workforce_id) == token

    rotated = client.post(
        f"/api/workforces/{workforce_id}/share-link/rotate", headers=headers
    )
    assert rotated.status_code == 200, rotated.text
    rotated_token = rotated.json()["share_token"]
    assert rotated_token and rotated_token != token

    disabled = client.delete(
        f"/api/workforces/{workforce_id}/share-link", headers=headers
    )
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["share_enabled"] is False
    assert disabled.json()["share_token"] is None


def test_share_link_disable_without_deployment_row_is_noop() -> None:
    workforce_id = _create_workforce("Never Shared Workforce")

    disabled = client.delete(
        f"/api/workforces/{workforce_id}/share-link", headers=_admin_headers()
    )
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["share_enabled"] is False
    assert disabled.json()["share_token"] is None


def test_share_link_requires_workforce_owner() -> None:
    workforce_id = _create_workforce("Owner Only Share Workforce")
    other_headers = _register_second_user()

    for method, url in (
        ("get", f"/api/workforces/{workforce_id}/share-link"),
        ("post", f"/api/workforces/{workforce_id}/share-link"),
        ("post", f"/api/workforces/{workforce_id}/share-link/rotate"),
        ("delete", f"/api/workforces/{workforce_id}/share-link"),
    ):
        response = getattr(client, method)(url, headers=other_headers)
        assert response.status_code == 403, response.text


def test_share_link_mutations_rejected_for_archived_workforce() -> None:
    workforce_id = _create_workforce("Archived Share Workforce")
    headers = _admin_headers()
    _enable_share(workforce_id)

    archived = client.delete(f"/api/workforces/{workforce_id}", headers=headers)
    assert archived.status_code == 200, archived.text

    enabled = client.post(f"/api/workforces/{workforce_id}/share-link", headers=headers)
    assert enabled.status_code == 409, enabled.text


# ===== Public share auth =====


def test_share_auth_resolves_workforce_token() -> None:
    workforce_id = _create_workforce("Auth Share Workforce")
    token = _enable_share(workforce_id)

    response = client.post("/api/share/auth", json={"share_token": token})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["workforce_id"] == workforce_id
    assert body["agent_id"] is None
    assert body["agent_name"] == "Auth Share Workforce"
    assert body["access_token"]


def test_share_auth_rejects_unknown_or_disabled_workforce_token() -> None:
    workforce_id = _create_workforce("Disabled Auth Workforce")
    token = _enable_share(workforce_id)

    unknown = client.post("/api/share/auth", json={"share_token": "no-such-token"})
    assert unknown.status_code == 404, unknown.text

    disabled = client.delete(
        f"/api/workforces/{workforce_id}/share-link", headers=_admin_headers()
    )
    assert disabled.status_code == 200, disabled.text

    response = client.post("/api/share/auth", json={"share_token": token})
    assert response.status_code == 404, response.text


def test_share_auth_rejects_inactive_workforce() -> None:
    workforce_id = _create_workforce("Inactive Auth Workforce")
    token = _enable_share(workforce_id)

    unpublished = client.post(
        f"/api/workforces/{workforce_id}/unpublish", headers=_admin_headers()
    )
    assert unpublished.status_code == 200, unpublished.text

    response = client.post("/api/share/auth", json={"share_token": token})
    assert response.status_code == 403, response.text


def test_share_auth_rejects_archived_workforce() -> None:
    workforce_id = _create_workforce("Archived Auth Workforce")
    token = _enable_share(workforce_id)

    archived = client.delete(
        f"/api/workforces/{workforce_id}", headers=_admin_headers()
    )
    assert archived.status_code == 200, archived.text

    response = client.post("/api/share/auth", json={"share_token": token})
    assert response.status_code == 403, response.text


def test_rotated_workforce_share_token_invalidates_existing_guest_tokens() -> None:
    workforce_id = _create_workforce("Rotate Auth Workforce")
    token = _enable_share(workforce_id)
    guest_headers = _authenticate_share_guest(token)

    rotated = client.post(
        f"/api/workforces/{workforce_id}/share-link/rotate", headers=_admin_headers()
    )
    assert rotated.status_code == 200, rotated.text

    # Old raw token no longer authenticates.
    stale = client.post("/api/share/auth", json={"share_token": token})
    assert stale.status_code == 404, stale.text

    # Previously issued guest JWTs are invalidated by the token mismatch.
    response = client.post(
        "/api/share/chat/task/create",
        headers=guest_headers,
        json={"title": "hello", "description": "hello"},
    )
    assert response.status_code == 403, response.text


# ===== Guest task creation → create_workforce_run =====


def test_share_task_create_starts_workforce_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_workforce("Guest Run Workforce")
    token = _enable_share(workforce_id)
    guest_headers = _authenticate_share_guest(token)
    _stub_begin_turn(monkeypatch)

    response = client.post(
        "/api/share/chat/task/create",
        headers=guest_headers,
        json={"title": "hello workforce", "description": "hello workforce"},
    )
    assert response.status_code == 200, response.text
    task_id = int(response.json()["task_id"])

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.source == "shared_link"
        assert bool(task.is_visible) is False
        assert int(task.user_id) == _user_id()

        run = db.query(WorkforceRun).filter(WorkforceRun.task_id == task_id).one()
        assert int(run.workforce_id) == workforce_id
        assert bool(run.is_preview) is False
    finally:
        db.close()

    # The run shows up in the owner's runs history with source=shared_link.
    listed = client.get(
        f"/api/workforces/{workforce_id}/runs", headers=_admin_headers()
    )
    assert listed.status_code == 200, listed.text
    items = listed.json()["items"]
    assert items and items[0]["source"] == "shared_link"


def test_share_task_create_requires_message() -> None:
    workforce_id = _create_workforce("Empty Message Workforce")
    token = _enable_share(workforce_id)
    guest_headers = _authenticate_share_guest(token)

    response = client.post(
        "/api/share/chat/task/create",
        headers=guest_headers,
        json={"title": "hello", "description": "   "},
    )
    assert response.status_code == 400, response.text


def test_share_task_create_rejects_foreign_agent_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_workforce("Foreign Agent Workforce")
    token = _enable_share(workforce_id)
    guest_headers = _authenticate_share_guest(token)
    _stub_begin_turn(monkeypatch)

    other_agent_id = _create_published_agent(_user_id(), "Unrelated Agent")
    response = client.post(
        "/api/share/chat/task/create",
        headers=guest_headers,
        json={
            "title": "hello",
            "description": "hello",
            "agent_id": other_agent_id,
        },
    )
    assert response.status_code == 403, response.text


# ===== Share-context task scoping =====


async def test_share_guest_cannot_touch_internal_run_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_workforce("Scoping Workforce")
    token = _enable_share(workforce_id)
    guest_headers = _authenticate_share_guest(token)
    _stub_begin_turn(monkeypatch)

    # An owner-initiated (internal) run of the same workforce.
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        workforce = db.query(Workforce).filter(Workforce.id == workforce_id).one()
        internal = await workforce_runs_service.create_workforce_run(
            db, user, workforce, message="internal run"
        )
        internal_task_id = int(internal.task.id)
    finally:
        db.close()

    upload = client.post(
        "/api/share/files/upload",
        headers=guest_headers,
        data={"task_type": "task", "task_id": str(internal_task_id)},
        files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert upload.status_code == 403, upload.text


def test_share_guest_can_upload_to_own_shared_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_workforce("Upload Workforce")
    token = _enable_share(workforce_id)
    guest_headers = _authenticate_share_guest(token)
    _stub_begin_turn(monkeypatch)

    created = client.post(
        "/api/share/chat/task/create",
        headers=guest_headers,
        json={"title": "hello", "description": "hello"},
    )
    assert created.status_code == 200, created.text
    task_id = int(created.json()["task_id"])

    upload = client.post(
        "/api/share/files/upload",
        headers=guest_headers,
        data={"task_type": "task", "task_id": str(task_id)},
        files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert upload.status_code == 200, upload.text


def test_agent_share_guest_cannot_access_workforce_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_workforce("Cross Guest Workforce")
    token = _enable_share(workforce_id)
    guest_headers = _authenticate_share_guest(token)
    _stub_begin_turn(monkeypatch)

    created = client.post(
        "/api/share/chat/task/create",
        headers=guest_headers,
        json={"title": "hello", "description": "hello"},
    )
    assert created.status_code == 200, created.text
    workforce_task_id = int(created.json()["task_id"])

    # A share guest of a plain agent must not reach the workforce task.
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=_user_id(),
            name="Plain Share Agent",
            description="d",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            share_enabled=True,
            share_token="plain-agent-share-token",
        )
        db.add(agent)
        db.commit()
    finally:
        db.close()
    agent_guest_headers = _authenticate_share_guest("plain-agent-share-token")

    upload = client.post(
        "/api/share/files/upload",
        headers=agent_guest_headers,
        data={"task_type": "task", "task_id": str(workforce_task_id)},
        files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert upload.status_code == 403, upload.text


# ===== WS APPEND flips the run back to running =====


async def test_ws_append_syncs_workforce_run_back_to_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a multi-turn APPEND claims the task, the WS path must project
    the task status onto the WorkforceRun so runs history shows ``running``
    again instead of staying ``completed``."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from sqlalchemy.orm import selectinload

    from xagent.web.api.websocket import handle_chat_message
    from xagent.web.models.workforce import WorkforceAgent
    from xagent.web.services.workforce_snapshot import build_workforce_snapshot

    workforce_id = _create_workforce("Append Sync Workforce")

    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        workforce = (
            db.query(Workforce)
            .options(
                selectinload(Workforce.manager_agent),
                selectinload(Workforce.workers).selectinload(WorkforceAgent.agent),
            )
            .filter(Workforce.id == workforce_id)
            .one()
        )
        snapshot = build_workforce_snapshot(db, user, workforce)
        task = Task(
            user_id=int(user.id),
            title="Workforce shared session",
            description="hello",
            status=TaskStatus.COMPLETED,
            agent_id=int(workforce.manager_agent_id),
            source="shared_link",
        )
        db.add(task)
        db.flush()
        run = WorkforceRun(
            workforce_id=workforce_id,
            task_id=int(task.id),
            user_id=int(user.id),
            status="completed",
            snapshot=snapshot,
        )
        db.add(run)
        db.flush()
        task.agent_config = {"workforce_run_id": int(run.id)}
        db.commit()
        task_id, run_id = int(task.id), int(run.id)
    finally:
        db.close()

    from xagent.web.services import task_orchestrator as task_orchestrator_service

    async def _claiming_begin_turn(**kwargs: Any) -> SimpleNamespace:
        # Simulate the orchestrator's atomic claim flipping the task RUNNING.
        claim_db = _direct_db_session()
        try:
            claim_db.query(Task).filter(Task.id == int(kwargs["task_id"])).update(
                {"status": TaskStatus.RUNNING}
            )
            claim_db.commit()
        finally:
            claim_db.close()
        return SimpleNamespace(background_task=None)

    monkeypatch.setattr(
        task_orchestrator_service.TaskTurnOrchestrator,
        "begin_turn",
        _claiming_begin_turn,
    )

    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        ws_manager = MagicMock(
            broadcast_to_task=AsyncMock(), send_personal_message=AsyncMock()
        )
        with patch("xagent.web.api.websocket.manager", ws_manager):
            await handle_chat_message(
                MagicMock(),
                task_id,
                {"message": "follow-up", "user": user, "files": []},
            )
    finally:
        db.close()

    db = _direct_db_session()
    try:
        run = db.query(WorkforceRun).filter(WorkforceRun.id == run_id).one()
        assert run.status == "running"
    finally:
        db.close()
