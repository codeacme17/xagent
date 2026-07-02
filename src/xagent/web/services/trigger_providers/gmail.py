"""Gmail trigger provider for the unified callback pipeline."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token
from pydantic import ValidationError
from sqlalchemy import case, func
from sqlalchemy.orm import Session, object_session

from ....config import get_gmail_pubsub_push_service_account
from ...models.gmail_watch import GmailWatchState
from ...models.trigger import AgentTrigger, TriggerProvisioningStatus, TriggerType
from ..gmail_provisioning import (
    provision_gmail_trigger,
    release_gmail_mailbox_if_unused,
)
from .base import (
    CallbackRequestContext,
    TriggerConfigError,
    TriggerEventParseError,
)
from .registry import register_trigger_provider
from .schemas import (
    AckPolicy,
    ChallengeResponse,
    NormalizedEvent,
    RegistrationResult,
    VerificationResult,
    parse_trigger_config,
)

if TYPE_CHECKING:
    from ..gmail_triggers import GmailPubsubNotification, GmailServiceFactory

OidcVerifier = Callable[[str, str], Mapping[str, Any]]

GOOGLE_OIDC_ISSUERS = frozenset({"https://accounts.google.com", "accounts.google.com"})


def _gmail_oauth_account_id(trigger: AgentTrigger) -> int | None:
    config: dict[str, Any] = trigger.config if isinstance(trigger.config, dict) else {}
    value = config.get("oauth_account_id")
    return int(value) if value is not None else None


def _watch_state_for_callback(db: Session, callback_id: str) -> GmailWatchState | None:
    return (
        db.query(GmailWatchState)
        .filter(GmailWatchState.callback_id == callback_id)
        .first()
    )


def _normalized_email(value: object) -> str:
    return str(value or "").strip().lower()


def _bearer_token(context: CallbackRequestContext) -> str | None:
    authorization = context.header("authorization") or ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _claim_audience_matches(claim_value: object, expected: str) -> bool:
    if isinstance(claim_value, str):
        return claim_value == expected
    if isinstance(claim_value, list):
        return expected in {str(item) for item in claim_value}
    return False


def _claim_email_verified(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _history_cursor_advances(current: object, incoming: str) -> bool:
    """True when incoming historyId moves the watch cursor forward.

    Gmail history ids are monotonically increasing integers. Rejecting
    non-advancing ids keeps out-of-order Pub/Sub redeliveries - and stale
    notifications processed right after an expired-history re-registration
    reset the cursor - from rolling the cursor backwards.
    """
    try:
        return int(incoming) > int(str(current or "0") or "0")
    except (TypeError, ValueError):
        return True


def verify_google_oidc_token(token: str, audience: str) -> Mapping[str, Any]:
    """Verify a Google OIDC token signature and audience."""
    claims = id_token.verify_oauth2_token(token, GoogleAuthRequest(), audience)
    return claims if isinstance(claims, Mapping) else {}


def _decode_pubsub_notification(
    raw_body: bytes, *, attested_email: str
) -> GmailPubsubNotification:
    from ..gmail_triggers import GmailPubsubNotification

    try:
        envelope = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TriggerEventParseError(
            "Gmail Pub/Sub envelope is not valid JSON"
        ) from exc
    if not isinstance(envelope, dict):
        raise TriggerEventParseError("Gmail Pub/Sub envelope must be an object")

    message = envelope.get("message")
    if not isinstance(message, dict):
        raise TriggerEventParseError("Gmail Pub/Sub envelope missing message")

    data = message.get("data")
    if not isinstance(data, str) or not data.strip():
        raise TriggerEventParseError("Gmail Pub/Sub message missing data")

    try:
        padded = data + ("=" * (-len(data) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TriggerEventParseError("Gmail Pub/Sub message data is invalid") from exc
    if not isinstance(payload, dict):
        raise TriggerEventParseError("Gmail Pub/Sub message data must be an object")

    history_id = payload.get("historyId")
    if history_id in (None, ""):
        raise TriggerEventParseError("Gmail Pub/Sub notification missing historyId")

    pubsub_message_id = message.get("messageId") or message.get("message_id")
    return GmailPubsubNotification(
        email_address=attested_email,
        history_id=str(history_id),
        pubsub_message_id=str(pubsub_message_id) if pubsub_message_id else None,
    )


class GmailProvider:
    """Gmail provider for per-mailbox Pub/Sub push callbacks."""

    name = TriggerType.GMAIL.value
    ack_policy = AckPolicy(
        not_found_status=200,
        rejected_status=200,
        rejected_resource_status=200,
        disabled_status=200,
    )

    def __init__(
        self,
        *,
        service_factory: GmailServiceFactory | None = None,
        oidc_verifier: OidcVerifier | None = None,
    ) -> None:
        self.service_factory = service_factory
        self.oidc_verifier = oidc_verifier or verify_google_oidc_token

    def validate_config(self, config: Mapping[str, Any]) -> Any:
        try:
            return parse_trigger_config(self.name, dict(config))
        except ValidationError as exc:
            raise TriggerConfigError(str(exc)) from exc

    def locate_trigger(self, db: Session, callback_id: str) -> AgentTrigger | None:
        state = _watch_state_for_callback(db, callback_id)
        if state is None:
            return None
        # Prefer triggers bound to this callback's mailbox so the disabled
        # check applies to the right binding; fall back to the user's other
        # Gmail triggers so cross-mailbox events still surface as audited
        # rejected_resource outcomes instead of unknown callbacks.
        mailbox_matches = case(
            (
                func.lower(AgentTrigger.resource_id) == _normalized_email(state.email),
                0,
            ),
            else_=1,
        )
        return (
            db.query(AgentTrigger)
            .filter(
                AgentTrigger.user_id == int(state.user_id),
                AgentTrigger.type == self.name,
                AgentTrigger.provider == self.name,
            )
            .order_by(
                mailbox_matches,
                AgentTrigger.enabled.desc(),
                AgentTrigger.id.asc(),
            )
            .first()
        )

    def handle_challenge(
        self, context: CallbackRequestContext, raw_body: bytes
    ) -> ChallengeResponse | None:
        return None

    def authorize_resource(
        self,
        trigger: AgentTrigger,
        attested_resource_id: str | None,
        event: NormalizedEvent,
    ) -> bool:
        if not trigger.resource_id or not attested_resource_id:
            return False
        return (
            str(trigger.resource_id).strip().lower()
            == attested_resource_id.strip().lower()
        )

    async def verify(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> VerificationResult:
        state = _watch_state_for_callback(db, context.callback_id)
        if state is None:
            return VerificationResult.reject("Unknown Gmail callback")

        audience = str(state.push_audience or "").strip()
        if not audience:
            return VerificationResult.reject(
                "Gmail callback audience is not configured"
            )

        token = _bearer_token(context)
        if token is None:
            return VerificationResult.reject("Missing Gmail OIDC bearer token")

        try:
            claims = self.oidc_verifier(token, audience)
        except Exception as exc:
            return VerificationResult.reject(
                f"Gmail OIDC token verification failed: {type(exc).__name__}"
            )

        issuer = str(claims.get("iss") or "")
        if issuer not in GOOGLE_OIDC_ISSUERS:
            return VerificationResult.reject("Gmail OIDC issuer is not trusted")

        if not _claim_audience_matches(claims.get("aud"), audience):
            return VerificationResult.reject("Gmail OIDC audience does not match")

        expected_service_account = get_gmail_pubsub_push_service_account()
        if expected_service_account:
            claim_email = _normalized_email(claims.get("email"))
            expected_email = expected_service_account.strip().lower()
            if claim_email != expected_email or not _claim_email_verified(
                claims.get("email_verified")
            ):
                return VerificationResult.reject(
                    "Gmail OIDC email claim must match configured service account "
                    "and email_verified must be true"
                )

        attested_email = _normalized_email(state.email)
        if not attested_email:
            return VerificationResult.reject("Gmail watch state email is empty")
        return VerificationResult.ok(attested_resource_id=attested_email)

    async def register(
        self, db: Session, trigger: AgentTrigger, config: Any
    ) -> RegistrationResult:
        status = await asyncio.to_thread(provision_gmail_trigger, db, trigger)
        return RegistrationResult(
            status=TriggerProvisioningStatus(status),
            resource_id=str(trigger.resource_id) if trigger.resource_id else None,
            error=trigger.provisioning_error,
        )

    async def unregister(self, db: Session, trigger: AgentTrigger, config: Any) -> None:
        oauth_account_id = _gmail_oauth_account_id(trigger)
        if oauth_account_id is not None:
            await asyncio.to_thread(
                release_gmail_mailbox_if_unused, db, oauth_account_id
            )

    async def parse_events(
        self,
        context: CallbackRequestContext,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> list[NormalizedEvent]:
        db = object_session(trigger) if trigger is not None else None
        if db is None:
            raise TriggerEventParseError("Gmail trigger is not attached to a session")

        state = _watch_state_for_callback(db, context.callback_id)
        if state is None:
            raise TriggerEventParseError("Gmail callback state was not found")

        attested_email = _normalized_email(state.email)
        notification = _decode_pubsub_notification(
            raw_body,
            attested_email=attested_email,
        )
        from ..gmail_triggers import build_gmail_service, collect_gmail_pubsub_events

        collection = await collect_gmail_pubsub_events(
            db,
            notification,
            service_factory=self.service_factory or build_gmail_service,
            advance_cursor=False,
        )
        return [
            NormalizedEvent(
                event_type=event.event_type,
                source_event_id=event.source_event_id,
                target_trigger_id=event.trigger_id,
                resource_id=event.resource_id,
                payload=event.payload,
            )
            for event in collection.events
        ]

    async def finalize_callback(
        self,
        *,
        db: Session,
        context: CallbackRequestContext,
        trigger: AgentTrigger | None,
        events: list[NormalizedEvent],
        raw_body: bytes,
    ) -> None:
        _ = (trigger, events)
        state = _watch_state_for_callback(db, context.callback_id)
        if state is None:
            return
        notification = _decode_pubsub_notification(
            raw_body,
            attested_email=_normalized_email(state.email),
        )
        if not _history_cursor_advances(state.history_id, notification.history_id):
            return
        setattr(state, "history_id", notification.history_id)
        setattr(state, "last_error", None)
        db.add(state)
        db.commit()


register_trigger_provider(GmailProvider(), replace=True)
