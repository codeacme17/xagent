from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from xagent.config import (
    get_gmail_pubsub_push_token,
    get_gmail_pubsub_topic_name,
    get_gmail_watch_enabled,
    get_gmail_watch_renewal_interval_seconds,
    get_gmail_watch_renewal_lead_seconds,
)
from xagent.web.models.agent import Agent
from xagent.web.models.gmail_watch import GmailWatchState
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.trigger import AgentTrigger, TriggerRun, TriggerType
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.gmail_triggers import (
    GmailPubsubNotification,
    GmailPubsubProcessResult,
    _get_google_oauth_config,
    ensure_gmail_watches_for_user,
    process_gmail_pubsub_notification,
    register_gmail_watch_for_account,
    scan_due_gmail_watch_renewals,
)

from .conftest import _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


class _FakeExecutable:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def execute(self) -> dict[str, object]:
        return self.payload


class _FakeHistoryResource:
    def __init__(self, history_response: dict[str, object]) -> None:
        self._history_response = history_response
        self.calls: list[dict[str, object]] = []

    def list(self, **kwargs: object) -> _FakeExecutable:
        self.calls.append(dict(kwargs))
        return _FakeExecutable(self._history_response)


class _FakeMessagesResource:
    def __init__(self, messages: dict[str, dict[str, object]]) -> None:
        self._messages = messages
        self.calls: list[dict[str, object]] = []

    def get(self, **kwargs: object) -> _FakeExecutable:
        self.calls.append(dict(kwargs))
        return _FakeExecutable(self._messages[str(kwargs["id"])])


class _FakeUsersResource:
    def __init__(
        self,
        response: dict[str, object],
        calls: list[dict[str, object]],
        history: _FakeHistoryResource,
        messages: _FakeMessagesResource,
    ):
        self._watch_response = response
        self._calls = calls
        self._history = history
        self._messages = messages

    def watch(self, *, userId: str, body: dict[str, object]) -> _FakeExecutable:
        self._calls.append({"userId": userId, "body": body})
        return _FakeExecutable(self._watch_response)

    def history(self) -> _FakeHistoryResource:
        return self._history

    def messages(self) -> _FakeMessagesResource:
        return self._messages


class _FakeGmailService:
    def __init__(
        self,
        response: dict[str, object] | None = None,
        *,
        history_response: dict[str, object] | None = None,
        messages: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response or {}
        self.history_resource = _FakeHistoryResource(history_response or {})
        self.messages_resource = _FakeMessagesResource(messages or {})

    def users(self) -> _FakeUsersResource:
        return _FakeUsersResource(
            self._response,
            self.calls,
            self.history_resource,
            self.messages_resource,
        )


@pytest.fixture
def mock_bg_scheduler():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg",
        new=MagicMock(),
    ) as mocked:
        yield mocked


def _create_user(db, username: str = "gmail-watch-user") -> User:
    user = User(
        username=username,
        email=f"{username}@example.com",
        password_hash="hash",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_gmail_oauth(db, user: User) -> UserOAuth:
    oauth = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        access_token="access-token",
        refresh_token="refresh-token",
        provider_user_id="provider-user",
        email="codeacme17@gmail.com",
    )
    db.add(oauth)
    db.commit()
    db.refresh(oauth)
    return oauth


def _create_gmail_trigger(db, user: User, *, enabled: bool = True) -> AgentTrigger:
    agent = Agent(
        user_id=int(user.id),
        name="Gmail trigger agent",
        description="test",
        instructions="Handle Gmail.",
        execution_mode="balanced",
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)

    trigger = AgentTrigger(
        user_id=int(user.id),
        agent_id=int(agent.id),
        type=TriggerType.GMAIL.value,
        name="Gmail inbox",
        enabled=enabled,
        config={"watch_label": "INBOX"},
        prompt_template="Handle {{payload}}",
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger


def test_gmail_watch_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_TOPIC", raising=False)
    monkeypatch.delenv("XAGENT_GMAIL_PUBSUB_PUSH_TOKEN", raising=False)
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_ENABLED", raising=False)
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_RENEWAL_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_RENEWAL_LEAD_SECONDS", raising=False)

    assert get_gmail_pubsub_topic_name() is None
    assert get_gmail_pubsub_push_token() is None
    assert get_gmail_watch_enabled() is False
    assert get_gmail_watch_renewal_interval_seconds() == 3600
    assert get_gmail_watch_renewal_lead_seconds() == 24 * 60 * 60


def test_gmail_watch_config_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TOPIC", "projects/demo/topics/xagent-gmail")
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PUSH_TOKEN", "push-secret")
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_RENEWAL_INTERVAL_SECONDS", "120")
    monkeypatch.setenv("XAGENT_GMAIL_WATCH_RENEWAL_LEAD_SECONDS", "60")

    assert get_gmail_pubsub_topic_name() == "projects/demo/topics/xagent-gmail"
    assert get_gmail_pubsub_push_token() == "push-secret"
    assert get_gmail_watch_enabled() is True
    assert get_gmail_watch_renewal_interval_seconds() == 120
    assert get_gmail_watch_renewal_lead_seconds() == 60


def test_gmail_oauth_config_falls_back_to_env_when_db_provider_is_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-client-secret")
    db = _direct_db_session()
    try:
        provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == "google")
            .one()
        )
        provider.client_id = ""
        provider.client_secret = ""
        db.add(provider)
        db.commit()

        assert _get_google_oauth_config(db) == ("env-client-id", "env-client-secret")
    finally:
        db.close()


def test_gmail_watch_state_model_can_persist() -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db)
        oauth = _create_gmail_oauth(db, user)

        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            watch_expiration=datetime(2026, 7, 1, tzinfo=timezone.utc),
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        db.refresh(state)

        assert state.id is not None
        assert state.history_id == "100"
        assert state.oauth_account_id == int(oauth.id)
    finally:
        db.close()


def test_register_gmail_watch_for_account_persists_google_watch_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TOPIC", "projects/demo/topics/xagent-gmail")
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-watch-registration")
        oauth = _create_gmail_oauth(db, user)
        fake_service = _FakeGmailService(
            {"historyId": "321", "expiration": "1782864000000"}
        )

        state = register_gmail_watch_for_account(
            db,
            oauth,
            service_factory=lambda _db, _oauth: fake_service,
        )

        assert fake_service.calls == [
            {
                "userId": "me",
                "body": {
                    "topicName": "projects/demo/topics/xagent-gmail",
                    "labelIds": ["INBOX"],
                },
            }
        ]
        assert state.history_id == "321"
        assert state.email == "codeacme17@gmail.com"
        assert state.topic_name == "projects/demo/topics/xagent-gmail"
        assert state.watch_expiration is not None
        assert state.watch_expiration.replace(tzinfo=timezone.utc) == datetime(
            2026, 7, 1, tzinfo=timezone.utc
        )
    finally:
        db.close()


def test_ensure_gmail_watches_for_user_registers_only_when_gmail_trigger_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TOPIC", "projects/demo/topics/xagent-gmail")
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-watch-needed")
        _create_gmail_oauth(db, user)
        fake_service = _FakeGmailService({"historyId": "456"})

        assert (
            ensure_gmail_watches_for_user(
                db,
                user_id=int(user.id),
                service_factory=lambda _db, _oauth: fake_service,
            )
            == []
        )
        assert fake_service.calls == []

        _create_gmail_trigger(db, user)
        states = ensure_gmail_watches_for_user(
            db,
            user_id=int(user.id),
            service_factory=lambda _db, _oauth: fake_service,
        )

        assert len(states) == 1
        assert states[0].history_id == "456"
        assert fake_service.calls == [
            {
                "userId": "me",
                "body": {
                    "topicName": "projects/demo/topics/xagent-gmail",
                    "labelIds": ["INBOX"],
                },
            }
        ]
    finally:
        db.close()


def test_process_gmail_pubsub_notification_fires_matching_gmail_trigger(
    mock_bg_scheduler,
) -> None:
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-history-user")
        oauth = _create_gmail_oauth(db, user)
        trigger = _create_gmail_trigger(db, user)
        state = GmailWatchState(
            user_id=int(user.id),
            oauth_account_id=int(oauth.id),
            email="codeacme17@gmail.com",
            history_id="100",
            topic_name="projects/demo/topics/xagent-gmail",
        )
        db.add(state)
        db.commit()
        fake_service = _FakeGmailService(
            history_response={
                "history": [
                    {
                        "messagesAdded": [
                            {"message": {"id": "msg-1", "threadId": "thread-1"}}
                        ]
                    }
                ]
            },
            messages={
                "msg-1": {
                    "id": "msg-1",
                    "threadId": "thread-1",
                    "labelIds": ["INBOX"],
                    "snippet": "Please reply exactly GMAIL_TRIGGER_OK",
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "boss@company.com"},
                            {"name": "Subject", "value": "urgent: e2e"},
                        ]
                    },
                }
            },
        )

        result = asyncio.run(
            process_gmail_pubsub_notification(
                db,
                GmailPubsubNotification(
                    email_address="codeacme17@gmail.com",
                    history_id="222",
                    pubsub_message_id="pubsub-1",
                ),
                service_factory=lambda _db, _oauth: fake_service,
            )
        )

        assert result.processed == 1
        assert result.duplicates == 0
        assert fake_service.history_resource.calls == [
            {
                "userId": "me",
                "startHistoryId": "100",
                "historyTypes": ["messageAdded"],
            }
        ]
        run = (
            db.query(TriggerRun).filter(TriggerRun.trigger_id == int(trigger.id)).one()
        )
        assert run.source_event_id == "gmail:msg-1"
        assert run.payload_snapshot["from"] == "boss@company.com"
        assert run.payload_snapshot["subject"] == "urgent: e2e"
        db.refresh(state)
        assert state.history_id == "222"
        assert mock_bg_scheduler.call_count == 1
    finally:
        db.close()


def test_gmail_pubsub_endpoint_validates_token_and_decodes_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PUSH_TOKEN", "push-secret")
    seen: dict[str, GmailPubsubNotification] = {}

    async def fake_process(
        _db,
        notification: GmailPubsubNotification,
    ) -> GmailPubsubProcessResult:
        seen["notification"] = notification
        return GmailPubsubProcessResult(processed=1, duplicates=0, skipped=0)

    monkeypatch.setattr(
        "xagent.web.api.triggers.process_gmail_pubsub_notification",
        fake_process,
        raising=False,
    )
    data = base64.b64encode(
        json.dumps({"emailAddress": "codeacme17@gmail.com", "historyId": "222"}).encode(
            "utf-8"
        )
    ).decode("ascii")
    payload = {"message": {"data": data, "messageId": "pubsub-1"}}

    rejected = client.post("/api/triggers/gmail/pubsub", json=payload)
    assert rejected.status_code == 401

    accepted_with_header = client.post(
        "/api/triggers/gmail/pubsub",
        headers={"x-xagent-gmail-pubsub-token": "push-secret"},
        json=payload,
    )

    assert accepted_with_header.status_code == 200, accepted_with_header.text
    assert accepted_with_header.json() == {
        "processed": 1,
        "duplicates": 0,
        "skipped": 0,
    }

    accepted_with_query_token = client.post(
        "/api/triggers/gmail/pubsub?token=push-secret",
        json=payload,
    )

    assert accepted_with_query_token.status_code == 200, accepted_with_query_token.text
    assert accepted_with_query_token.json() == {
        "processed": 1,
        "duplicates": 0,
        "skipped": 0,
    }
    assert seen["notification"].email_address == "codeacme17@gmail.com"
    assert seen["notification"].history_id == "222"
    assert seen["notification"].pubsub_message_id == "pubsub-1"


def test_scan_due_gmail_watch_renewals_respects_enabled_flag_and_expiration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TOPIC", "projects/demo/topics/xagent-gmail")
    monkeypatch.delenv("XAGENT_GMAIL_WATCH_ENABLED", raising=False)
    db = _direct_db_session()
    try:
        user = _create_user(db, "gmail-renewal-user")
        oauth = _create_gmail_oauth(db, user)
        _create_gmail_trigger(db, user)
        fake_service = _FakeGmailService(
            {"historyId": "789", "expiration": "1782864000000"}
        )

        assert (
            scan_due_gmail_watch_renewals(
                db,
                now=datetime(2026, 6, 29, tzinfo=timezone.utc),
                service_factory=lambda _db, _oauth: fake_service,
            )
            == 0
        )

        monkeypatch.setenv("XAGENT_GMAIL_WATCH_ENABLED", "true")
        assert (
            scan_due_gmail_watch_renewals(
                db,
                now=datetime(2026, 6, 29, tzinfo=timezone.utc),
                service_factory=lambda _db, _oauth: fake_service,
            )
            == 1
        )
        state = (
            db.query(GmailWatchState)
            .filter(GmailWatchState.oauth_account_id == int(oauth.id))
            .one()
        )
        assert state.history_id == "789"

        assert (
            scan_due_gmail_watch_renewals(
                db,
                now=datetime(2026, 6, 29, tzinfo=timezone.utc),
                service_factory=lambda _db, _oauth: fake_service,
            )
            == 0
        )

        state.watch_expiration = datetime(2026, 6, 29, 12, tzinfo=timezone.utc)
        db.add(state)
        db.commit()
        assert (
            scan_due_gmail_watch_renewals(
                db,
                now=datetime(2026, 6, 29, tzinfo=timezone.utc),
                service_factory=lambda _db, _oauth: fake_service,
            )
            == 1
        )
    finally:
        db.close()
