"""Durable audit records for trigger callback and payload access activity."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from ...models.trigger import TriggerAudit, TriggerAuditOutcome

logger = logging.getLogger(__name__)


def record_trigger_audit(
    db: Session,
    *,
    outcome: TriggerAuditOutcome,
    provider: str | None = None,
    callback_id: str | None = None,
    trigger_id: int | None = None,
    detail: dict[str, Any] | None = None,
    remote_ip: str | None = None,
) -> TriggerAudit:
    """Persist one audit row immediately.

    Audit rows are committed as they are written so that a later failure in
    the same request cannot roll the security trail back out.
    """
    audit = TriggerAudit(
        outcome=outcome.value,
        provider=provider,
        callback_id=callback_id,
        trigger_id=trigger_id,
        detail=detail,
        remote_ip=remote_ip,
    )
    db.add(audit)
    db.commit()
    db.refresh(audit)
    return audit


def record_trigger_audit_best_effort(
    db: Session,
    *,
    outcome: TriggerAuditOutcome,
    provider: str | None = None,
    callback_id: str | None = None,
    trigger_id: int | None = None,
    detail: dict[str, Any] | None = None,
    remote_ip: str | None = None,
) -> TriggerAudit | None:
    """Like record_trigger_audit, but never raises.

    Used on failure paths where the original error must win over any audit
    storage problem.
    """
    try:
        return record_trigger_audit(
            db,
            outcome=outcome,
            provider=provider,
            callback_id=callback_id,
            trigger_id=trigger_id,
            detail=detail,
            remote_ip=remote_ip,
        )
    except Exception:
        logger.exception("Failed to write trigger audit row (outcome=%s)", outcome)
        db.rollback()
        return None
