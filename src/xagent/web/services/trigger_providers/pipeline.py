"""Unified trigger callback pipeline.

One code path processes every provider callback: provider lookup, challenge
short-circuit, trigger resolution, verification, event parsing, event-type
filtering, resource authorization, trigger execution, auditing, and
acknowledgement. Providers plug in behavior; the pipeline owns ordering and
the audit trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from ...models.trigger import AgentTrigger, TriggerAuditOutcome, TriggerRun
from .audit import record_trigger_audit, record_trigger_audit_best_effort
from .base import CallbackRequestContext, TriggerEventParseError
from .registry import UnknownTriggerProviderError, get_trigger_provider
from .schemas import ChallengeResponse, NormalizedEvent

logger = logging.getLogger(__name__)


@dataclass
class CallbackResult:
    """Terminal state of one callback request through the pipeline."""

    status_code: int
    outcome: TriggerAuditOutcome | None
    detail: str | None = None
    challenge: ChallengeResponse | None = None
    runs: list[TriggerRun] = field(default_factory=list)
    duplicates: int = 0
    rejected_events: int = 0
    filtered_events: int = 0


def _allowed_event_types(trigger: AgentTrigger) -> list[str] | None:
    config: dict[str, Any] = trigger.config if isinstance(trigger.config, dict) else {}
    allow_list = config.get("event_types")
    if isinstance(allow_list, list) and allow_list:
        return [str(item) for item in allow_list]
    return None


def _filter_events(
    trigger: AgentTrigger, events: list[NormalizedEvent]
) -> tuple[list[NormalizedEvent], int]:
    allow_list = _allowed_event_types(trigger)
    if allow_list is None:
        return events, 0
    allowed = [event for event in events if event.event_type in allow_list]
    return allowed, len(events) - len(allowed)


async def process_trigger_callback(
    db: Session,
    *,
    context: CallbackRequestContext,
    raw_body: bytes,
) -> CallbackResult:
    """Run one inbound callback through the unified provider pipeline."""
    provider_name = context.provider
    callback_id = context.callback_id

    try:
        provider = get_trigger_provider(provider_name)
    except UnknownTriggerProviderError:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.UNKNOWN_PROVIDER,
            provider=provider_name,
            callback_id=callback_id,
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=404,
            outcome=TriggerAuditOutcome.UNKNOWN_PROVIDER,
            detail="Unknown trigger provider",
        )

    ack = provider.ack_policy

    challenge = provider.handle_challenge(context, raw_body)
    if challenge is not None:
        return CallbackResult(
            status_code=challenge.status_code, outcome=None, challenge=challenge
        )

    trigger = provider.locate_trigger(db, callback_id)
    if trigger is None:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.UNKNOWN_CALLBACK,
            provider=provider_name,
            callback_id=callback_id,
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.not_found_status,
            outcome=TriggerAuditOutcome.UNKNOWN_CALLBACK,
            detail="Unknown callback",
        )

    verification = await provider.verify(
        context, db=db, trigger=trigger, raw_body=raw_body
    )
    if not verification.verified:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.REJECTED_SIGNATURE,
            provider=provider_name,
            callback_id=callback_id,
            trigger_id=int(trigger.id),
            detail={"reason": verification.reason},
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.rejected_status,
            outcome=TriggerAuditOutcome.REJECTED_SIGNATURE,
            detail=verification.reason or "Verification failed",
        )

    if not trigger.enabled:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.REJECTED_DISABLED,
            provider=provider_name,
            callback_id=callback_id,
            trigger_id=int(trigger.id),
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.disabled_status,
            outcome=TriggerAuditOutcome.REJECTED_DISABLED,
            detail="Trigger is disabled",
        )

    try:
        events = await provider.parse_events(context, trigger, raw_body)
    except TriggerEventParseError as exc:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
            provider=provider_name,
            callback_id=callback_id,
            trigger_id=int(trigger.id),
            detail={"stage": "parse", "error": str(exc)},
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=400,
            outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
            detail=str(exc) or "Malformed callback payload",
        )

    events, filtered_count = _filter_events(trigger, events)

    runs: list[TriggerRun] = []
    duplicates = 0
    rejected_events = 0
    failures: list[str] = []
    for event in events:
        if not provider.authorize_resource(
            trigger, verification.attested_resource_id, event
        ):
            rejected_events += 1
            record_trigger_audit(
                db,
                outcome=TriggerAuditOutcome.REJECTED_RESOURCE,
                provider=provider_name,
                callback_id=callback_id,
                trigger_id=int(trigger.id),
                detail={
                    "attested_resource_id": verification.attested_resource_id,
                    "trigger_resource_id": trigger.resource_id,
                    "event_type": event.event_type,
                },
                remote_ip=context.remote_ip,
            )
            continue

        try:
            run, created = await _fire_event(db, trigger=trigger, event=event)
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            logger.exception("Trigger %s failed to execute callback event", trigger.id)
            record_trigger_audit_best_effort(
                db,
                outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
                provider=provider_name,
                callback_id=callback_id,
                trigger_id=int(trigger.id),
                detail={"stage": "fire", "error": f"{type(exc).__name__}: {exc}"},
                remote_ip=context.remote_ip,
            )
            continue

        if created:
            runs.append(run)
        else:
            duplicates += 1

    if runs or duplicates or not (rejected_events or failures):
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.ACCEPTED,
            provider=provider_name,
            callback_id=callback_id,
            trigger_id=int(trigger.id),
            detail=_accepted_detail(
                runs=runs,
                duplicates=duplicates,
                rejected_events=rejected_events,
                filtered_events=filtered_count,
            ),
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.accepted_status,
            outcome=TriggerAuditOutcome.ACCEPTED,
            runs=runs,
            duplicates=duplicates,
            rejected_events=rejected_events,
            filtered_events=filtered_count,
        )

    if rejected_events:
        return CallbackResult(
            status_code=ack.rejected_resource_status,
            outcome=TriggerAuditOutcome.REJECTED_RESOURCE,
            detail="Event resource does not match this trigger",
            rejected_events=rejected_events,
            filtered_events=filtered_count,
        )

    return CallbackResult(
        status_code=ack.failure_status,
        outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
        detail="; ".join(failures) or "Trigger execution failed",
        filtered_events=filtered_count,
    )


def _accepted_detail(
    *,
    runs: list[TriggerRun],
    duplicates: int,
    rejected_events: int,
    filtered_events: int,
) -> dict[str, Any]:
    return {
        "run_ids": [int(run.id) for run in runs],
        "duplicates": duplicates,
        "rejected_events": rejected_events,
        "filtered_events": filtered_events,
    }


async def _fire_event(
    db: Session,
    *,
    trigger: AgentTrigger,
    event: NormalizedEvent,
) -> tuple[TriggerRun, bool]:
    from ..triggers import fire_trigger

    return await fire_trigger(
        db,
        trigger=trigger,
        event_payload=event.payload,
        source_event_id=event.source_event_id,
    )
