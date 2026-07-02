from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.gmail_watch import GmailWatchState
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerProvisioningStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.gmail_provisioning import (
    ensure_gmail_mailbox_provisioned,
    gmail_subscription_path,
    gmail_topic_path,
    release_gmail_mailbox_if_unused,
    sweep_gmail_provisioning,
)


class FakeExecutable:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {}

    def execute(self) -> dict[str, Any]:
        return self.response


class FakeGmailUsers:
    def __init__(self, service: FakeGmailService) -> None:
        self.service = service

    def watch(self, *, userId: str, body: dict[str, Any]) -> FakeExecutable:
        self.service.watch_calls.append({"userId": userId, "body": body})
        return FakeExecutable(
            {
                "historyId": self.service.history_id,
                "expiration": self.service.expiration,
            }
        )

    def stop(self, *, userId: str) -> FakeExecutable:
        self.service.stop_calls.append({"userId": userId})
        return FakeExecutable()


class FakeGmailService:
    def __init__(
        self, *, history_id: str = "hist-1", expiration: str = "4102444800000"
    ) -> None:
        self.history_id = history_id
        self.expiration = expiration
        self.watch_calls: list[dict[str, Any]] = []
        self.stop_calls: list[dict[str, Any]] = []

    def users(self) -> FakeGmailUsers:
        return FakeGmailUsers(self)


class FakeBinding:
    def __init__(self, *, role: str, members: list[str]) -> None:
        self.role = role
        self.members = members


class FakeBindings(list[FakeBinding]):
    def add(self, *, role: str, members: list[str]) -> FakeBinding:
        binding = FakeBinding(role=role, members=members)
        self.append(binding)
        return binding


class FakePolicy:
    def __init__(self) -> None:
        self.bindings = FakeBindings()


class FakePublisher:
    def __init__(self) -> None:
        self.topics: set[str] = set()
        self.policies: dict[str, FakePolicy] = {}
        self.deleted_topics: list[str] = []

    def create_topic(self, *, request: dict[str, str]) -> None:
        self.topics.add(request["name"])

    def get_iam_policy(self, *, request: dict[str, str]) -> FakePolicy:
        return self.policies.setdefault(request["resource"], FakePolicy())

    def set_iam_policy(self, *, request: dict[str, Any]) -> None:
        self.policies[request["resource"]] = request["policy"]

    def delete_topic(self, *, request: dict[str, str]) -> None:
        self.deleted_topics.append(request["topic"])
        self.topics.discard(request["topic"])


class FakeSubscriber:
    def __init__(self) -> None:
        self.subscriptions: dict[str, dict[str, Any]] = {}
        self.deleted_subscriptions: list[str] = []

    def create_subscription(self, *, request: dict[str, Any]) -> None:
        self.subscriptions[request["name"]] = request

    def delete_subscription(self, *, request: dict[str, str]) -> None:
        self.deleted_subscriptions.append(request["subscription"])
        self.subscriptions.pop(request["subscription"], None)


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'gmail_provisioning.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture(autouse=True)
def gmail_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_PROJECT_ID", "demo-project")
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_TOPIC_PREFIX", "xagent-gmail")
    monkeypatch.setenv("XAGENT_GMAIL_PUBSUB_SUBSCRIPTION_PREFIX", "xagent-gmail-push")
    monkeypatch.setenv("XAGENT_PUBLIC_API_BASE_URL", "https://api.example.com/")
    monkeypatch.setenv(
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT",
        "pubsub-push@demo-project.iam.gserviceaccount.com",
    )


def _create_user(db: Session) -> User:
    user = User(
        username="owner",
        email="owner@example.com",
        password_hash="hash",
        is_admin=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_agent(db: Session, user: User) -> Agent:
    agent = Agent(
        user_id=int(user.id),
        name="Gmail agent",
        description="test",
        instructions="test",
        status=AgentStatus.DRAFT,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def _create_oauth(db: Session, user: User, *, email: str = "Owner@Gmail.Example"):
    account = UserOAuth(
        user_id=int(user.id),
        provider="gmail",
        access_token="access-token",
        email=email,
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    return account


def _create_gmail_trigger(
    db: Session,
    user: User,
    agent: Agent,
    account: UserOAuth,
    *,
    enabled: bool = True,
) -> AgentTrigger:
    trigger = AgentTrigger(
        user_id=int(user.id),
        agent_id=int(agent.id),
        type=TriggerType.GMAIL.value,
        name="Gmail inbox",
        enabled=enabled,
        provider=TriggerType.GMAIL.value,
        resource_id=str(account.email).lower(),
        config={"watch_label": "INBOX", "oauth_account_id": int(account.id)},
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger


def test_provisioning_creates_deterministic_resources_and_active_state(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService()

    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    email = "owner@gmail.example"
    expected_topic = gmail_topic_path("demo-project", email)
    expected_subscription = gmail_subscription_path("demo-project", email)
    expected_audience = (
        f"https://api.example.com/api/triggers/callback/gmail/{state.callback_id}"
    )
    assert state.status == TriggerProvisioningStatus.ACTIVE.value
    assert state.last_error is None
    assert state.topic_name == expected_topic
    assert state.subscription_name == expected_subscription
    assert state.push_audience == expected_audience
    assert state.history_id == "hist-1"
    assert publisher.topics == {expected_topic}
    assert set(subscriber.subscriptions) == {expected_subscription}
    assert subscriber.subscriptions[expected_subscription]["push_config"] == {
        "push_endpoint": expected_audience,
        "oidc_token": {
            "service_account_email": "pubsub-push@demo-project.iam.gserviceaccount.com",
            "audience": expected_audience,
        },
    }
    assert gmail.watch_calls == [
        {
            "userId": "me",
            "body": {"topicName": expected_topic, "labelIds": ["INBOX"]},
        }
    ]


def test_missing_public_api_base_records_failed_state_without_app_base_fallback(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XAGENT_PUBLIC_API_BASE_URL", raising=False)
    monkeypatch.setenv("XAGENT_APP_BASE_URL", "https://frontend.example.com")
    user = _create_user(db_session)
    account = _create_oauth(db_session, user)

    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: FakeGmailService(),
        publisher_factory=lambda: FakePublisher(),
        subscriber_factory=lambda: FakeSubscriber(),
    )

    assert state.status == TriggerProvisioningStatus.FAILED.value
    assert "XAGENT_PUBLIC_API_BASE_URL" in str(state.last_error)
    assert state.push_audience is None


def test_sweep_retries_stale_failed_referenced_mailbox(db_session: Session) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    _create_gmail_trigger(db_session, user, agent, account)
    state = GmailWatchState(
        user_id=int(user.id),
        oauth_account_id=int(account.id),
        email="owner@gmail.example",
        history_id="",
        topic_name="",
        status=TriggerProvisioningStatus.FAILED.value,
        last_error="old failure",
        updated_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    db_session.add(state)
    db_session.commit()
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService(history_id="hist-retry")

    attempts = sweep_gmail_provisioning(
        db_session,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    refreshed = (
        db_session.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == int(account.id))
        .one()
    )
    assert attempts == 1
    assert refreshed.status == TriggerProvisioningStatus.ACTIVE.value
    assert refreshed.history_id == "hist-retry"
    assert refreshed.last_error is None


def test_unregister_releases_mailbox_only_after_last_enabled_trigger_is_deleted(
    db_session: Session,
) -> None:
    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    first = _create_gmail_trigger(db_session, user, agent, account)
    second = _create_gmail_trigger(db_session, user, agent, account)
    publisher = FakePublisher()
    subscriber = FakeSubscriber()
    gmail = FakeGmailService()
    state = ensure_gmail_mailbox_provisioned(
        db_session,
        account,
        service_factory=lambda _db, _account: gmail,
        publisher_factory=lambda: publisher,
        subscriber_factory=lambda: subscriber,
    )

    assert (
        release_gmail_mailbox_if_unused(
            db_session,
            int(account.id),
            service_factory=lambda _db, _account: gmail,
            publisher_factory=lambda: publisher,
            subscriber_factory=lambda: subscriber,
        )
        is False
    )
    db_session.delete(first)
    db_session.commit()
    assert (
        release_gmail_mailbox_if_unused(
            db_session,
            int(account.id),
            service_factory=lambda _db, _account: gmail,
            publisher_factory=lambda: publisher,
            subscriber_factory=lambda: subscriber,
        )
        is False
    )
    db_session.delete(second)
    db_session.commit()

    assert (
        release_gmail_mailbox_if_unused(
            db_session,
            int(account.id),
            service_factory=lambda _db, _account: gmail,
            publisher_factory=lambda: publisher,
            subscriber_factory=lambda: subscriber,
        )
        is True
    )
    assert gmail.stop_calls == [{"userId": "me"}]
    assert subscriber.deleted_subscriptions == [state.subscription_name]
    assert publisher.deleted_topics == [state.topic_name]
    assert db_session.query(GmailWatchState).count() == 0


async def test_gmail_provider_register_unregister_offload_sync_sdk_work(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.web.services.trigger_providers.gmail import GmailProvider

    user = _create_user(db_session)
    agent = _create_agent(db_session, user)
    account = _create_oauth(db_session, user)
    trigger = _create_gmail_trigger(db_session, user, agent, account)
    calls: list[str] = []

    def fake_provision(_db: Session, provisioned_trigger: AgentTrigger) -> str:
        calls.append(f"provision:{provisioned_trigger.id}")
        setattr(
            provisioned_trigger,
            "provisioning_status",
            TriggerProvisioningStatus.ACTIVE.value,
        )
        setattr(provisioned_trigger, "provisioning_error", None)
        _db.add(provisioned_trigger)
        _db.commit()
        return TriggerProvisioningStatus.ACTIVE.value

    def fake_release(_db: Session, oauth_account_id: int) -> bool:
        calls.append(f"release:{oauth_account_id}")
        return True

    async def fake_to_thread(fn, *args, **kwargs):
        calls.append(f"to_thread:{fn.__name__}")
        return fn(*args, **kwargs)

    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.provision_gmail_trigger",
        fake_provision,
    )
    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.release_gmail_mailbox_if_unused",
        fake_release,
    )
    monkeypatch.setattr(
        "xagent.web.services.trigger_providers.gmail.asyncio.to_thread",
        fake_to_thread,
    )

    provider = GmailProvider()
    result = await provider.register(db_session, trigger, object())
    await provider.unregister(db_session, trigger, object())

    assert result.status == TriggerProvisioningStatus.ACTIVE
    assert calls == [
        "to_thread:fake_provision",
        f"provision:{trigger.id}",
        "to_thread:fake_release",
        f"release:{account.id}",
    ]
