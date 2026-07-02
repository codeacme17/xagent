"""Gmail trigger provider provisioning boundary.

Issue 007 only owns Gmail Pub/Sub provisioning and teardown. OIDC verification
and Gmail Pub/Sub event parsing are implemented by the next slice, so this
provider intentionally exposes registration behavior without registering
itself in the callback registry yet.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from pydantic import ValidationError
from sqlalchemy.orm import Session

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
from .schemas import (
    AckPolicy,
    ChallengeResponse,
    NormalizedEvent,
    RegistrationResult,
    VerificationResult,
    parse_trigger_config,
)


def _gmail_oauth_account_id(trigger: AgentTrigger) -> int | None:
    config: dict[str, Any] = trigger.config if isinstance(trigger.config, dict) else {}
    value = config.get("oauth_account_id")
    return int(value) if value is not None else None


class GmailProvider:
    """Gmail provider slice for async provisioning and teardown."""

    name = TriggerType.GMAIL.value
    ack_policy = AckPolicy(
        not_found_status=200,
        rejected_status=200,
        rejected_resource_status=200,
        disabled_status=200,
    )

    def validate_config(self, config: Mapping[str, Any]) -> Any:
        try:
            return parse_trigger_config(self.name, dict(config))
        except ValidationError as exc:
            raise TriggerConfigError(str(exc)) from exc

    def locate_trigger(self, db: Session, callback_id: str) -> AgentTrigger | None:
        return (
            db.query(AgentTrigger)
            .filter(
                AgentTrigger.callback_id == callback_id,
                AgentTrigger.type == self.name,
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
        return VerificationResult.reject("Gmail OIDC verification is not implemented")

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
        raise TriggerEventParseError("Gmail event parsing is not implemented")
