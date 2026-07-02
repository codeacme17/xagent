"""Stub-provider tests for the unified trigger callback pipeline.

These tests exercise the provider protocol, registry, callback pipeline, and
audit trail without depending on webhook or Gmail implementation details.
"""

from __future__ import annotations

import json
from typing import Any, Mapping
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.trigger import (
    AgentTrigger,
    TriggerAudit,
    TriggerAuditOutcome,
    TriggerProvisioningStatus,
    TriggerRun,
)
from xagent.web.models.user import User
from xagent.web.services.trigger_providers import (
    AckPolicy,
    CallbackRequestContext,
    ChallengeResponse,
    NormalizedEvent,
    RegistrationResult,
    TriggerEventParseError,
    UnknownTriggerProviderError,
    VerificationResult,
    get_trigger_provider,
    process_trigger_callback,
    register_trigger_provider,
    registered_trigger_provider_names,
    unregister_trigger_provider,
)
from xagent.web.services.trigger_providers.schemas import parse_trigger_config
from xagent.web.services.triggers import delete_agent_trigger

STUB_PROVIDER = "stub"
GOOD_TOKEN = "good-token"


class StubProvider:
    """Minimal provider used to exercise the pipeline contract."""

    name = STUB_PROVIDER

    def __init__(self, ack_policy: AckPolicy | None = None) -> None:
        self.ack_policy = ack_policy or AckPolicy()
        self.register_calls: list[int] = []
        self.unregister_calls: list[int] = []

    def validate_config(self, config: Mapping[str, Any]) -> Any:
        return parse_trigger_config("webhook", dict(config))

    def locate_trigger(self, db: Session, callback_id: str) -> AgentTrigger | None:
        return (
            db.query(AgentTrigger)
            .filter(
                AgentTrigger.callback_id == callback_id,
                AgentTrigger.provider == self.name,
            )
            .first()
        )

    def handle_challenge(
        self, context: CallbackRequestContext, raw_body: bytes
    ) -> ChallengeResponse | None:
        challenge = context.header("x-stub-challenge")
        if challenge:
            return ChallengeResponse(status_code=200, body=challenge)
        return None

    def authorize_resource(
        self,
        trigger: AgentTrigger,
        attested_resource_id: str | None,
        event: NormalizedEvent,
    ) -> bool:
        if trigger.resource_id is None:
            return True
        return attested_resource_id == trigger.resource_id

    async def verify(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> VerificationResult:
        if context.header("x-stub-token") != GOOD_TOKEN:
            return VerificationResult.reject("bad stub token")
        return VerificationResult.ok(
            attested_resource_id=context.header("x-stub-resource")
        )

    async def register(
        self, db: Session, trigger: AgentTrigger, config: Any
    ) -> RegistrationResult:
        self.register_calls.append(int(trigger.id))
        return RegistrationResult(status=TriggerProvisioningStatus.ACTIVE)

    async def unregister(self, db: Session, trigger: AgentTrigger, config: Any) -> None:
        self.unregister_calls.append(int(trigger.id))

    async def parse_events(
        self,
        context: CallbackRequestContext,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> list[NormalizedEvent]:
        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except ValueError as exc:
            raise TriggerEventParseError("stub body is not JSON") from exc
        events = decoded if isinstance(decoded, list) else [decoded]
        return [
            NormalizedEvent(
                event_type=str(item.get("type", "stub.event")),
                source_event_id=item.get("id"),
                resource_id=item.get("resource"),
                payload=item,
            )
            for item in events
        ]


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'trigger_pipeline.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture()
def stub_provider():
    provider = StubProvider()
    register_trigger_provider(provider, replace=True)
    try:
        yield provider
    finally:
        unregister_trigger_provider(STUB_PROVIDER)


@pytest.fixture(autouse=True)
def _fast_run_start():
    """Skip real agent execution; run/task rows are still created."""

    async def _noop_start(db, *, run, wait_for_completion=False):
        return True

    with patch(
        "xagent.web.services.triggers.start_prepared_trigger_run",
        new=_noop_start,
    ):
        yield


def _create_user(db: Session) -> User:
    user = User(username="pipeline-user", password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _create_agent(db: Session, user: User) -> Agent:
    agent = Agent(
        user_id=user.id,
        name="Pipeline Agent",
        description="pipeline",
        instructions="pipeline agent",
        execution_mode="balanced",
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=AgentStatus.PUBLISHED,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def _create_stub_trigger(
    db: Session,
    user: User,
    agent: Agent,
    *,
    callback_id: str = "cb-stub-1",
    resource_id: str | None = "res-1",
    enabled: bool = True,
    config: dict[str, Any] | None = None,
) -> AgentTrigger:
    trigger = AgentTrigger(
        user_id=user.id,
        agent_id=agent.id,
        type="webhook",
        name="Stub trigger",
        enabled=enabled,
        config=config or {},
        provider=STUB_PROVIDER,
        callback_id=callback_id,
        resource_id=resource_id,
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger


def _context(
    *,
    callback_id: str = "cb-stub-1",
    headers: dict[str, str] | None = None,
    provider: str = STUB_PROVIDER,
) -> CallbackRequestContext:
    return CallbackRequestContext(
        provider=provider,
        callback_id=callback_id,
        headers=headers or {},
        remote_ip="203.0.113.9",
    )


def _good_headers(resource: str | None = "res-1") -> dict[str, str]:
    headers = {"X-Stub-Token": GOOD_TOKEN}
    if resource is not None:
        headers["X-Stub-Resource"] = resource
    return headers


def _event_body(event_id: str = "evt-1", event_type: str = "stub.event") -> bytes:
    return json.dumps({"id": event_id, "type": event_type, "n": 1}).encode()


def _audits(db: Session) -> list[TriggerAudit]:
    return db.query(TriggerAudit).order_by(TriggerAudit.id.asc()).all()


class TestProviderRegistry:
    def test_lookup_and_unknown_provider(self, stub_provider):
        assert get_trigger_provider(STUB_PROVIDER) is stub_provider
        assert STUB_PROVIDER in registered_trigger_provider_names()
        with pytest.raises(UnknownTriggerProviderError):
            get_trigger_provider("nope")

    def test_duplicate_registration_is_rejected(self, stub_provider):
        with pytest.raises(ValueError):
            register_trigger_provider(StubProvider())
        replacement = StubProvider()
        register_trigger_provider(replacement, replace=True)
        assert get_trigger_provider(STUB_PROVIDER) is replacement


class TestCallbackPipeline:
    async def test_unknown_provider_is_audited(self, db_session):
        result = await process_trigger_callback(
            db_session,
            context=_context(provider="ghost"),
            raw_body=b"{}",
        )
        assert result.status_code == 404
        assert result.outcome == TriggerAuditOutcome.UNKNOWN_PROVIDER
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["unknown_provider"]
        assert audits[0].provider == "ghost"
        assert audits[0].callback_id == "cb-stub-1"
        assert audits[0].trigger_id is None
        assert audits[0].remote_ip == "203.0.113.9"

    async def test_challenge_short_circuits_before_verification(
        self, db_session, stub_provider
    ):
        result = await process_trigger_callback(
            db_session,
            context=_context(headers={"X-Stub-Challenge": "pong"}),
            raw_body=b"",
        )
        assert result.challenge is not None
        assert result.challenge.body == "pong"
        assert result.status_code == 200
        assert _audits(db_session) == []
        assert db_session.query(TriggerRun).count() == 0

    async def test_unknown_callback_id_is_audited(self, db_session, stub_provider):
        result = await process_trigger_callback(
            db_session,
            context=_context(callback_id="cb-missing", headers=_good_headers()),
            raw_body=_event_body(),
        )
        assert result.status_code == 404
        assert result.outcome == TriggerAuditOutcome.UNKNOWN_CALLBACK
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["unknown_callback"]
        assert audits[0].callback_id == "cb-missing"

    async def test_ack_policy_separates_http_ack_from_audit_outcome(self, db_session):
        provider = StubProvider(ack_policy=AckPolicy(not_found_status=200))
        register_trigger_provider(provider, replace=True)
        try:
            result = await process_trigger_callback(
                db_session,
                context=_context(callback_id="cb-missing", headers=_good_headers()),
                raw_body=_event_body(),
            )
        finally:
            unregister_trigger_provider(STUB_PROVIDER)
        assert result.status_code == 200
        assert result.outcome == TriggerAuditOutcome.UNKNOWN_CALLBACK
        assert [a.outcome for a in _audits(db_session)] == ["unknown_callback"]

    async def test_failed_verification_is_audited(self, db_session, stub_provider):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent)

        result = await process_trigger_callback(
            db_session,
            context=_context(headers={"X-Stub-Token": "wrong"}),
            raw_body=_event_body(),
        )
        assert result.status_code == 401
        assert result.outcome == TriggerAuditOutcome.REJECTED_SIGNATURE
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["rejected_signature"]
        assert audits[0].trigger_id == trigger.id
        assert audits[0].detail == {"reason": "bad stub token"}
        assert db_session.query(TriggerRun).count() == 0

    async def test_disabled_trigger_is_rejected_after_verification(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent, enabled=False)

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body(),
        )
        assert result.status_code == 409
        assert result.outcome == TriggerAuditOutcome.REJECTED_DISABLED
        assert [a.outcome for a in _audits(db_session)] == ["rejected_disabled"]
        assert db_session.query(TriggerRun).count() == 0

    async def test_unparseable_body_is_a_controlled_failure(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent)

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=b"\x00not-json",
        )
        assert result.status_code == 400
        assert result.outcome == TriggerAuditOutcome.EXECUTION_FAILURE
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["execution_failure"]
        assert audits[0].detail["stage"] == "parse"

    async def test_accepted_event_creates_run_and_audit(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent)

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-42"),
        )
        assert result.status_code == 200
        assert result.outcome == TriggerAuditOutcome.ACCEPTED
        assert len(result.runs) == 1
        run = db_session.query(TriggerRun).one()
        assert run.trigger_id == trigger.id
        assert run.source_event_id == "evt-42"
        assert run.task_id is not None
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["accepted"]
        assert audits[0].detail["run_ids"] == [run.id]

    async def test_redelivery_is_idempotent(self, db_session, stub_provider):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(db_session, user, agent)

        first = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-dup"),
        )
        second = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-dup"),
        )
        assert len(first.runs) == 1
        assert second.status_code == 200
        assert second.runs == []
        assert second.duplicates == 1
        assert db_session.query(TriggerRun).count() == 1

    async def test_event_type_allow_list_filters_events(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        _create_stub_trigger(
            db_session, user, agent, config={"event_types": ["stub.allowed"]}
        )

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-x", event_type="stub.other"),
        )
        assert result.status_code == 200
        assert result.runs == []
        assert result.filtered_events == 1
        assert db_session.query(TriggerRun).count() == 0

    async def test_resource_mismatch_is_rejected_and_audited(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent, resource_id="res-1")

        result = await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers(resource="res-other")),
            raw_body=_event_body(),
        )
        assert result.status_code == 403
        assert result.outcome == TriggerAuditOutcome.REJECTED_RESOURCE
        audits = _audits(db_session)
        assert [a.outcome for a in audits] == ["rejected_resource"]
        assert audits[0].trigger_id == trigger.id
        assert audits[0].detail["attested_resource_id"] == "res-other"
        assert audits[0].detail["trigger_resource_id"] == "res-1"
        assert db_session.query(TriggerRun).count() == 0

    async def test_audit_rows_survive_trigger_deletion(self, db_session, stub_provider):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent)

        await process_trigger_callback(
            db_session,
            context=_context(headers=_good_headers()),
            raw_body=_event_body("evt-audit"),
        )
        assert len(_audits(db_session)) == 1

        delete_agent_trigger(
            db_session,
            user_id=int(user.id),
            agent_id=int(agent.id),
            trigger_id=int(trigger.id),
        )
        assert db_session.query(AgentTrigger).count() == 0
        audits = _audits(db_session)
        assert len(audits) == 1
        assert audits[0].trigger_id is None
        assert audits[0].outcome == "accepted"


class TestProviderRegistration:
    async def test_register_and_unregister_receive_trigger_context(
        self, db_session, stub_provider
    ):
        user = _create_user(db_session)
        agent = _create_agent(db_session, user)
        trigger = _create_stub_trigger(db_session, user, agent)

        config = stub_provider.validate_config({"event_types": ["stub.event"]})
        registration = await stub_provider.register(db_session, trigger, config)
        assert registration.status == TriggerProvisioningStatus.ACTIVE
        await stub_provider.unregister(db_session, trigger, config)
        assert stub_provider.register_calls == [trigger.id]
        assert stub_provider.unregister_calls == [trigger.id]
