from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.utils.encryption import decrypt_value
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.trigger import AgentTrigger, TriggerRun, TriggerRunStatus
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.task_orchestrator import TurnStarted, finish_turn
from xagent.web.services.trigger_providers import sign_webhook_payload
from xagent.web.services.triggers import (
    _compute_next_run_at,
    _start_prepared_trigger_run_id,
    dispatch_pending_trigger_runs,
    scan_due_scheduled_triggers,
)

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture(autouse=True)
def mock_bg_scheduler():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _create_agent(headers: dict[str, str], name: str = "Trigger Agent") -> int:
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "test",
            "instructions": "You are a trigger test agent.",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    return int(resp.json()["id"])


def _connect_gmail_account(
    username: str = "admin",
    *,
    email: str = "owner@gmail.example",
    provider: str = "gmail",
) -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).one()
        account = UserOAuth(
            user_id=int(user.id),
            provider=provider,
            access_token="access-token",
            email=email,
        )
        db.add(account)
        db.commit()
        db.refresh(account)
        return int(account.id)
    finally:
        db.close()


def test_webhook_trigger_crud_returns_secret_once() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Inbound webhook",
            "prompt_template": "payload={{payload}}",
            "config": {"source": "crm"},
        },
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["type"] == "webhook"
    assert body["callback_id"]
    assert body["webhook_secret"]

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == body["id"]).one()
        # New rows store only the encrypted HMAC secret; no bcrypt hash.
        assert trigger.secret_hash is None
        assert trigger.secret_encrypted
        assert trigger.secret_encrypted != body["webhook_secret"]
        assert decrypt_value(str(trigger.secret_encrypted)) == body["webhook_secret"]
        assert trigger.provider == "webhook"
        assert trigger.callback_id == body["callback_id"]
    finally:
        db.close()

    listed = client.get(f"/api/agents/{agent_id}/triggers", headers=headers)
    assert listed.status_code == 200, listed.text
    assert len(listed.json()) == 1
    assert listed.json()[0]["webhook_secret"] is None

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{body['id']}",
        headers=headers,
        json={"name": "Renamed webhook", "rotate_secret": True},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "Renamed webhook"
    assert patched.json()["webhook_secret"]


def test_trigger_test_run_creates_hidden_agent_task(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Test webhook",
            "prompt_template": "Handle {{payload}}",
        },
    )
    trigger_id = created.json()["id"]

    fired = client.post(
        f"/api/agents/{agent_id}/triggers/{trigger_id}/test",
        headers=headers,
        json={"payload": {"subject": "hello"}, "source_event_id": "test-event"},
    )
    assert fired.status_code == 200, fired.text
    run_body = fired.json()["trigger_run"]
    assert run_body["status"] == TriggerRunStatus.RUNNING.value
    assert run_body["task_id"]
    assert fired.json()["duplicate"] is False
    assert mock_bg_scheduler.call_count == 1

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == run_body["task_id"]).one()
        assert task.agent_id == agent_id
        assert task.source == "trigger"
        assert task.is_visible is False
        assert task.status == TaskStatus.RUNNING
        assert "hello" in (task.description or "")
    finally:
        db.close()


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


def test_public_webhook_validates_signature_and_deduplicates(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Public webhook"},
    )
    body = created.json()
    url = f"/api/triggers/callback/webhook/{body['callback_id']}"
    raw_body = b'{"subject": "hello"}'

    unsigned = client.post(url, content=raw_body)
    assert unsigned.status_code == 401

    forged_headers = _signed_webhook_headers("wrong-secret", raw_body, event_id="evt-1")
    forged = client.post(url, headers=forged_headers, content=raw_body)
    assert forged.status_code == 401

    event_headers = _signed_webhook_headers(
        body["webhook_secret"], raw_body, event_id="evt-1"
    )
    first = client.post(url, headers=event_headers, content=raw_body)
    assert first.status_code == 200, first.text
    assert first.json()["outcome"] == "accepted"
    assert len(first.json()["trigger_run_ids"]) == 1
    assert first.json()["duplicates"] == 0

    second = client.post(url, headers=event_headers, content=raw_body)
    assert second.status_code == 200, second.text
    assert second.json()["duplicates"] == 1
    assert second.json()["trigger_run_ids"] == []
    assert mock_bg_scheduler.call_count == 1

    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 1
        assert db.query(Task).filter(Task.source == "trigger").count() == 1
    finally:
        db.close()


def test_public_webhook_rejects_stale_timestamp(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Replay webhook"},
    )
    body = created.json()
    url = f"/api/triggers/callback/webhook/{body['callback_id']}"
    raw_body = b'{"subject": "hello"}'

    stale_timestamp = str(int(time.time()) - 3600)
    stale = client.post(
        url,
        headers={
            "x-xagent-signature": sign_webhook_payload(
                body["webhook_secret"], stale_timestamp, raw_body
            ),
            "x-xagent-timestamp": stale_timestamp,
        },
        content=raw_body,
    )
    assert stale.status_code == 401
    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 0
    finally:
        db.close()


def test_public_callback_unknown_provider_and_callback_are_controlled(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Known webhook"},
    )
    body = created.json()
    raw_body = b"{}"

    unknown_provider = client.post(
        f"/api/triggers/callback/ghost/{body['callback_id']}",
        content=raw_body,
    )
    assert unknown_provider.status_code == 404
    assert unknown_provider.json()["outcome"] == "unknown_provider"

    unknown_callback = client.post(
        "/api/triggers/callback/webhook/does-not-exist",
        content=raw_body,
    )
    assert unknown_callback.status_code == 404
    assert unknown_callback.json()["outcome"] == "unknown_callback"


def test_public_callback_disabled_trigger_is_rejected_after_verification(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Disabled webhook", "enabled": False},
    )
    body = created.json()
    raw_body = b'{"subject": "hello"}'

    fired = client.post(
        f"/api/triggers/callback/webhook/{body['callback_id']}",
        headers=_signed_webhook_headers(body["webhook_secret"], raw_body),
        content=raw_body,
    )
    assert fired.status_code == 409
    assert fired.json()["outcome"] == "rejected_disabled"


def test_public_callback_filters_events_against_allow_list(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "webhook",
            "name": "Filtered webhook",
            "config": {"event_types": ["order.created"]},
        },
    )
    body = created.json()
    url = f"/api/triggers/callback/webhook/{body['callback_id']}"

    ignored_body = b'{"event_type": "order.deleted", "id": "evt-a"}'
    ignored = client.post(
        url,
        headers=_signed_webhook_headers(body["webhook_secret"], ignored_body),
        content=ignored_body,
    )
    assert ignored.status_code == 200, ignored.text
    assert ignored.json()["trigger_run_ids"] == []

    matched_body = b'{"event_type": "order.created", "id": "evt-b"}'
    matched = client.post(
        url,
        headers=_signed_webhook_headers(body["webhook_secret"], matched_body),
        content=matched_body,
    )
    assert matched.status_code == 200, matched.text
    assert len(matched.json()["trigger_run_ids"]) == 1

    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 1
    finally:
        db.close()


def test_public_webhook_invalid_utf8_body_is_a_controlled_parse_failure(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Invalid UTF-8 webhook"},
    )
    body = created.json()
    url = f"/api/triggers/callback/webhook/{body['callback_id']}"
    raw_body = b'\xff{"subject":"hello"}'

    fired = client.post(
        url,
        headers=_signed_webhook_headers(body["webhook_secret"], raw_body),
        content=raw_body,
    )
    assert fired.status_code == 400
    assert fired.json()["outcome"] == "execution_failure"

    db = _direct_db_session()
    try:
        assert db.query(TriggerRun).count() == 0
    finally:
        db.close()
    assert mock_bg_scheduler.call_count == 0


def test_trigger_name_validation_rejects_empty_and_oversized() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    empty = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "   "},
    )
    assert empty.status_code == 400

    oversized = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "x" * 201},
    )
    assert oversized.status_code == 422

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Valid"},
    )
    assert created.status_code == 200, created.text

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{created.json()['id']}",
        headers=headers,
        json={"name": " "},
    )
    assert patched.status_code == 400


def test_gmail_trigger_crud_persists_filters() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    account_id = _connect_gmail_account(email="Owner@Gmail.Example")

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Support inbox",
            "config": {
                "watch_label": "INBOX",
                "sender_filter": "boss@company.com",
                "subject_keyword": "urgent",
                "oauth_account_id": account_id,
            },
            "prompt_template": "Handle Gmail message {{payload}}",
        },
    )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["type"] == "gmail"
    assert body["webhook_token"] is None
    assert body["webhook_secret"] is None
    assert body["config"] == {
        "watch_label": "INBOX",
        "sender_filter": "boss@company.com",
        "subject_keyword": "urgent",
        "oauth_account_id": account_id,
    }

    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == body["id"]).one()
        assert trigger.provider == "gmail"
        assert trigger.resource_id == "owner@gmail.example"
    finally:
        db.close()

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{body['id']}",
        headers=headers,
        json={
            "config": {
                "watch_label": "CATEGORY_PRIMARY",
                "sender_filter": "",
                "subject_keyword": "invoice",
                "oauth_account_id": account_id,
            }
        },
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["config"] == {
        "watch_label": "CATEGORY_PRIMARY",
        "sender_filter": "",
        "subject_keyword": "invoice",
        "oauth_account_id": account_id,
    }


def test_gmail_trigger_requires_oauth_account() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "No account",
            "config": {"watch_label": "INBOX"},
        },
    )
    assert created.status_code == 400
    assert "oauth_account_id" in created.json()["detail"]


def test_gmail_trigger_rejects_foreign_oauth_account() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    from .conftest import _register_second_user

    _register_second_user()
    foreign_account_id = _connect_gmail_account("bob", email="bob@gmail.example")

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Foreign account",
            "config": {"watch_label": "INBOX", "oauth_account_id": foreign_account_id},
        },
    )
    assert created.status_code == 400
    assert "not found" in created.json()["detail"].lower()


def test_gmail_trigger_rejects_non_gmail_oauth_account() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    drive_account_id = _connect_gmail_account(
        email="owner@gmail.example", provider="google-drive"
    )

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Wrong provider",
            "config": {"watch_label": "INBOX", "oauth_account_id": drive_account_id},
        },
    )
    assert created.status_code == 400
    assert "not a gmail account" in created.json()["detail"].lower()


def test_enabled_gmail_trigger_create_best_effort_registers_watch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_ensure_gmail_watches_for_user(_db, *, user_id: int):
        calls.append(user_id)
        raise RuntimeError("watch registration unavailable")

    monkeypatch.setattr(
        "xagent.web.services.gmail_triggers.ensure_gmail_watches_for_user",
        fake_ensure_gmail_watches_for_user,
        raising=False,
    )
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    account_id = _connect_gmail_account()

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Support inbox",
            "config": {"watch_label": "INBOX", "oauth_account_id": account_id},
        },
    )

    assert created.status_code == 200, created.text
    assert (
        created.json()["last_error"]
        == "Gmail watch registration failed: watch registration unavailable"
    )
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        assert calls == [int(admin.id)]
    finally:
        db.close()


def test_enabling_existing_gmail_trigger_best_effort_registers_watch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_ensure_gmail_watches_for_user(_db, *, user_id: int):
        calls.append(user_id)
        raise RuntimeError("watch registration unavailable")

    monkeypatch.setattr(
        "xagent.web.services.gmail_triggers.ensure_gmail_watches_for_user",
        fake_ensure_gmail_watches_for_user,
        raising=False,
    )
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    account_id = _connect_gmail_account()
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "enabled": False,
            "name": "Support inbox",
            "config": {"watch_label": "INBOX", "oauth_account_id": account_id},
        },
    )
    assert created.status_code == 200, created.text
    assert calls == []

    patched = client.patch(
        f"/api/agents/{agent_id}/triggers/{created.json()['id']}",
        headers=headers,
        json={"enabled": True},
    )

    assert patched.status_code == 200, patched.text
    assert (
        patched.json()["last_error"]
        == "Gmail watch registration failed: watch registration unavailable"
    )
    db = _direct_db_session()
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        assert calls == [int(admin.id)]
    finally:
        db.close()


def test_gmail_trigger_requires_watch_label() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)

    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Missing label",
            "config": {"sender_filter": "boss@company.com"},
        },
    )
    assert created.status_code == 400
    assert "watch_label" in created.json()["detail"]


def test_gmail_trigger_test_run_creates_hidden_agent_task(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    account_id = _connect_gmail_account()
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "gmail",
            "name": "Gmail support",
            "config": {"watch_label": "INBOX", "oauth_account_id": account_id},
            "prompt_template": "Triage this email: {{payload}}",
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    fired = client.post(
        f"/api/agents/{agent_id}/triggers/{trigger_id}/test",
        headers=headers,
        json={
            "payload": {
                "from": "boss@company.com",
                "subject": "urgent invoice",
                "snippet": "please review",
            },
            "source_event_id": "gmail-msg-1",
        },
    )
    assert fired.status_code == 200, fired.text
    run_body = fired.json()["trigger_run"]
    assert run_body["status"] == TriggerRunStatus.RUNNING.value
    assert run_body["task_id"]
    assert fired.json()["duplicate"] is False
    assert mock_bg_scheduler.call_count == 1

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == run_body["task_id"]).one()
        assert task.agent_id == agent_id
        assert task.source == "trigger"
        assert task.is_visible is False
        assert task.status == TaskStatus.RUNNING
        assert "urgent invoice" in (task.description or "")
    finally:
        db.close()


def test_scheduled_next_run_skips_stale_intervals_without_iteration() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stale_due_at = now - timedelta(days=3650)

    next_run_at = _compute_next_run_at(
        {"interval_seconds": 1},
        from_time=now,
        previous_due_at=stale_due_at,
        include_explicit=False,
    )

    assert next_run_at == now + timedelta(seconds=1)


def test_scheduled_scan_fires_due_trigger_and_advances_next_run(
    mock_bg_scheduler,
) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Every minute",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    due_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    db = _direct_db_session()
    try:
        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        trigger.next_run_at = due_at
        db.add(trigger)
        db.commit()

        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1
        db.refresh(trigger)
        assert trigger.next_run_at is not None
        next_run_at = trigger.next_run_at
        if next_run_at.tzinfo is None:
            next_run_at = next_run_at.replace(tzinfo=timezone.utc)
        assert next_run_at > due_at
        run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
        assert run.status == TriggerRunStatus.PENDING.value
        assert run.task_id is not None
        task = db.query(Task).filter(Task.id == run.task_id).one()
        assert task.agent_id == agent_id
        assert task.source == "trigger"
        assert task.is_visible is False
        assert task.status == TaskStatus.PENDING

        assert mock_bg_scheduler.call_count == 0
        assert asyncio.run(dispatch_pending_trigger_runs(db)) == 1
        db.refresh(run)
        db.refresh(task)
        assert run.status == TriggerRunStatus.RUNNING.value
        assert task.status == TaskStatus.RUNNING
    finally:
        db.close()

    assert mock_bg_scheduler.call_count == 1


def test_dispatch_claims_pending_trigger_run_once_under_concurrency() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "Concurrent scheduled",
            "config": {"interval_seconds": 60},
        },
    )
    assert created.status_code == 200, created.text

    db = _direct_db_session()
    try:
        trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == created.json()["id"]).one()
        )
        trigger.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=5)
        db.add(trigger)
        db.commit()
        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1
        run_id = int(runs[0].id)
    finally:
        db.close()

    begin_calls = 0

    async def fake_begin_turn(**kwargs):
        nonlocal begin_calls
        begin_calls += 1
        await asyncio.sleep(0.05)

        async def done() -> None:
            return None

        return TurnStarted(
            task_id=int(kwargs["task_id"]),
            status=TaskStatus.RUNNING,
            updated_at=None,
            before_message_id=None,
            task_source="trigger",
            background_task=asyncio.create_task(done()),
        )

    async def start_twice() -> list[bool]:
        first, second = await asyncio.gather(
            _start_prepared_trigger_run_id(run_id),
            _start_prepared_trigger_run_id(run_id),
        )
        return [first, second]

    with patch(
        "xagent.web.services.triggers.TaskTurnOrchestrator.begin_turn",
        new=fake_begin_turn,
    ):
        results = asyncio.run(start_twice())

    assert results.count(True) == 1
    assert begin_calls == 1


def test_scheduled_scan_disables_one_shot_trigger(mock_bg_scheduler) -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    due_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={
            "type": "scheduled",
            "name": "One shot",
            "config": {"next_run_at": due_at.isoformat()},
        },
    )
    assert created.status_code == 200, created.text
    trigger_id = created.json()["id"]

    db = _direct_db_session()
    try:
        runs = scan_due_scheduled_triggers(db, now=datetime.now(timezone.utc))
        assert len(runs) == 1

        trigger = db.query(AgentTrigger).filter(AgentTrigger.id == trigger_id).one()
        assert trigger.enabled is False
        assert trigger.next_run_at is None
        run = db.query(TriggerRun).filter(TriggerRun.id == runs[0].id).one()
        assert run.status == TriggerRunStatus.PENDING.value
    finally:
        db.close()

    assert mock_bg_scheduler.call_count == 0


def test_finish_turn_syncs_trigger_run_status() -> None:
    headers = _admin_headers()
    agent_id = _create_agent(headers)
    created = client.post(
        f"/api/agents/{agent_id}/triggers",
        headers=headers,
        json={"type": "webhook", "name": "Completion webhook"},
    )
    trigger_id = created.json()["id"]

    fired = client.post(
        f"/api/agents/{agent_id}/triggers/{trigger_id}/test",
        headers=headers,
        json={"payload": {"subject": "done"}},
    )
    run_body = fired.json()["trigger_run"]

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == run_body["task_id"]).one()
        task.status = TaskStatus.COMPLETED
        db.add(
            TaskChatMessage(
                task_id=int(task.id),
                user_id=int(task.user_id),
                role="assistant",
                content="done",
                message_type="assistant_message",
            )
        )
        db.add(task)
        db.commit()

        finish_turn(db, int(task.id))

        run = db.query(TriggerRun).filter(TriggerRun.id == run_body["id"]).one()
        assert run.status == TriggerRunStatus.COMPLETED.value
        assert run.finished_at is not None
        assert run.error_message is None
    finally:
        db.close()
