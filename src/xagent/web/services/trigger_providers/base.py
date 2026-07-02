"""TriggerProvider protocol and callback request context.

A trigger provider adapts one third-party callback source (webhook, Gmail
Pub/Sub, ...) to the unified callback pipeline. The pipeline owns ordering,
auditing, and acknowledgement; providers own the provider-specific trust
model: how requests are authenticated, how events are parsed, and how
delivery resources are provisioned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from sqlalchemy.orm import Session

from ...models.trigger import AgentTrigger
from .schemas import (
    AckPolicy,
    ChallengeResponse,
    NormalizedEvent,
    RegistrationResult,
    VerificationResult,
)


class TriggerProviderError(Exception):
    """Base class for provider errors surfaced through the pipeline."""


class TriggerConfigError(TriggerProviderError):
    """Raised when a trigger config fails provider validation."""


class TriggerEventParseError(TriggerProviderError):
    """Raised when a verified callback body cannot be parsed into events."""


@dataclass(frozen=True)
class CallbackRequestContext:
    """Transport-agnostic view of one inbound callback request.

    Providers receive this instead of a framework request object so they can
    be exercised without HTTP. Header keys are matched case-insensitively.
    """

    provider: str
    callback_id: str
    method: str = "POST"
    url_path: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    query_params: Mapping[str, str] = field(default_factory=dict)
    remote_ip: str | None = None

    def header(self, name: str) -> str | None:
        target = name.lower()
        for key, value in self.headers.items():
            if key.lower() == target:
                return value
        return None


@runtime_checkable
class TriggerProvider(Protocol):
    """Contract every trigger callback provider implements.

    Sync methods are pure lookups/validation; async methods perform I/O
    (crypto verification against remote keys, cloud provisioning, event
    ingestion).
    """

    name: str
    ack_policy: AckPolicy

    def validate_config(self, config: Mapping[str, Any]) -> Any:
        """Validate raw trigger config; return the typed config model.

        Raises TriggerConfigError for invalid config.
        """
        ...

    def locate_trigger(self, db: Session, callback_id: str) -> AgentTrigger | None:
        """Resolve the trigger addressed by a non-secret callback locator."""
        ...

    def handle_challenge(
        self, context: CallbackRequestContext, raw_body: bytes
    ) -> ChallengeResponse | None:
        """Short-circuit provider handshake requests (URL verification).

        Return None for normal event deliveries.
        """
        ...

    def authorize_resource(
        self,
        trigger: AgentTrigger,
        attested_resource_id: str | None,
        event: NormalizedEvent,
    ) -> bool:
        """Decide whether an event may fire this trigger.

        Only provider-attested identity may be trusted here; payload-claimed
        identity on the event is informational.
        """
        ...

    async def verify(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> VerificationResult:
        """Authenticate the callback request via provider-specific proof."""
        ...

    async def register(
        self, db: Session, trigger: AgentTrigger, config: Any
    ) -> RegistrationResult:
        """Provision provider-side delivery resources for a trigger."""
        ...

    async def unregister(self, db: Session, trigger: AgentTrigger, config: Any) -> None:
        """Tear down provider-side delivery resources for a trigger."""
        ...

    async def parse_events(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> list[NormalizedEvent]:
        """Parse a verified callback body into normalized events.

        Raises TriggerEventParseError for malformed bodies.
        """
        ...
