"""Common schemas shared by all trigger providers.

These models define the provider-agnostic vocabulary of the unified callback
pipeline: normalized events, verification results, acknowledgement policy,
registration results, and the typed trigger config union.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from ...models.trigger import TriggerProvisioningStatus, TriggerType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NormalizedEvent(BaseModel):
    """One provider event normalized into the shared pipeline shape."""

    event_type: str
    source_event_id: str | None = None
    resource_id: str | None = None
    """Payload-claimed resource identity. Never trusted for authorization."""
    payload: dict[str, Any] = Field(default_factory=dict)
    received_at: datetime = Field(default_factory=_utcnow)


class VerificationResult(BaseModel):
    """Outcome of provider-specific callback authentication."""

    verified: bool
    attested_resource_id: str | None = None
    """Resource identity proven by the provider's trust model (for example a
    mailbox derived from a verified OIDC token), as opposed to any identity
    claimed inside the payload."""
    reason: str | None = None

    @classmethod
    def ok(cls, *, attested_resource_id: str | None = None) -> "VerificationResult":
        return cls(verified=True, attested_resource_id=attested_resource_id)

    @classmethod
    def reject(cls, reason: str) -> "VerificationResult":
        return cls(verified=False, reason=reason)


class ChallengeResponse(BaseModel):
    """Immediate response to a provider handshake/challenge request."""

    status_code: int = 200
    body: str = ""
    media_type: str = "text/plain"


class AckPolicy(BaseModel):
    """HTTP acknowledgement behavior, decoupled from audit outcome.

    Providers with aggressive redelivery (for example Pub/Sub push) can map
    terminal rejections to 2xx to stop redelivery while the audit trail still
    records the real outcome.
    """

    accepted_status: int = 200
    not_found_status: int = 404
    rejected_status: int = 401
    rejected_resource_status: int = 403
    disabled_status: int = 409
    failure_status: int = 500


class RegistrationResult(BaseModel):
    """Result of provisioning provider-side delivery for a trigger."""

    status: TriggerProvisioningStatus
    resource_id: str | None = None
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class BaseTriggerConfig(BaseModel):
    """Fields shared by every typed trigger config."""

    event_types: list[str] | None = None
    """Optional allow-list of normalized event types this trigger fires on."""
    store_full_payload: bool = False
    """Opt-in to encrypted full-payload snapshots on trigger runs."""


class WebhookTriggerConfig(BaseTriggerConfig):
    type: Literal["webhook"] = "webhook"


class ScheduledTriggerConfig(BaseTriggerConfig):
    type: Literal["scheduled"] = "scheduled"
    interval_seconds: int | None = None
    next_run_at: str | None = None

    @field_validator("interval_seconds")
    @classmethod
    def _positive_interval(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("interval_seconds must be positive")
        return value

    @model_validator(mode="after")
    def _require_schedule(self) -> "ScheduledTriggerConfig":
        if self.interval_seconds is None and not (self.next_run_at or "").strip():
            raise ValueError(
                "scheduled trigger requires interval_seconds or next_run_at"
            )
        return self


class GmailTriggerConfig(BaseTriggerConfig):
    type: Literal["gmail"] = "gmail"
    oauth_account_id: int | None = None
    """Connected Gmail OAuth account this trigger is bound to. Optional at the
    schema level during rollout; API-level enforcement lands with the typed
    trigger config slice."""
    watch_label: str
    sender_filter: str | None = None
    subject_keyword: str | None = None

    @field_validator("watch_label")
    @classmethod
    def _non_empty_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("gmail trigger requires watch_label")
        return value


TriggerConfig = Annotated[
    Union[WebhookTriggerConfig, ScheduledTriggerConfig, GmailTriggerConfig],
    Field(discriminator="type"),
]


class _TriggerConfigEnvelope(BaseModel):
    config: TriggerConfig


def parse_trigger_config(trigger_type: str, config: dict[str, Any]) -> Any:
    """Validate a raw config dict against the typed schema for trigger_type.

    The discriminator lives on the trigger row rather than inside the stored
    config JSON, so it is injected here before validation.
    """
    normalized_type = TriggerType(trigger_type).value
    payload = {**config, "type": normalized_type}
    return _TriggerConfigEnvelope(config=payload).config


def dump_trigger_config(config: BaseTriggerConfig) -> dict[str, Any]:
    """Serialize a typed config back to the stored JSON shape (no type key)."""
    data = config.model_dump(exclude_none=True)
    data.pop("type", None)
    return data
