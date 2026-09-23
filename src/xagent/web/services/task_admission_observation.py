"""Bounded admission telemetry and exact database snapshots for host operators."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timezone

from sqlalchemy import and_, case, func, select
from sqlalchemy.orm import Session

from ...core.runtime_performance import runtime_performance
from ..models.task import Task
from ..models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from ..models.task_admission_pacing import TaskAdmissionPacing
from ..models.task_command import TaskExecutionCommand
from .task_admission_pacing import database_time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdmissionSnapshot:
    bucket: str
    lane: str
    capacity: int
    max_pending: int
    active: int
    pending: int
    oldest_pending_seconds: float
    startup_interval_seconds: float | None
    startup_burst: int | None
    startup_delay_seconds: float
    delay_reason: str | None


def read_admission_snapshot(
    db: Session, bucket_keys: Sequence[str]
) -> list[AdmissionSnapshot]:
    """Read explicit host-authorized keys; never expose an unbounded tenant listing."""
    if len(bucket_keys) > 32:
        raise ValueError("An admission snapshot accepts at most 32 bucket keys")
    now = float(db.execute(select(database_time())).scalar_one())
    snapshots = []
    held = func.coalesce(
        and_(
            TaskAdmissionTicket.runner_id == Task.runner_id,
            TaskAdmissionTicket.owner_attempt_id == Task.lease_attempt_id,
        ),
        False,
    )
    waiting = and_(~held, TaskExecutionCommand.status.notin_(("completed", "failed")))
    for key in dict.fromkeys(bucket_keys):
        bucket = db.get(TaskAdmissionBucket, key)
        if bucket is None:
            continue
        active, pending, oldest = db.execute(
            select(
                func.coalesce(func.sum(case((held, 1), else_=0)), 0),
                func.coalesce(func.sum(case((waiting, 1), else_=0)), 0),
                func.min(case((waiting, TaskExecutionCommand.created_at))),
            )
            .select_from(TaskAdmissionTicket)
            .join(Task, Task.id == TaskAdmissionTicket.task_id)
            .join(
                TaskExecutionCommand,
                TaskExecutionCommand.id == TaskAdmissionTicket.command_id,
            )
            .where(TaskAdmissionTicket.bucket_key == key)
        ).one()
        pace = db.get(TaskAdmissionPacing, key)
        delay = (
            0.0
            if pace is None
            else max(
                0.0,
                float(pace.next_start_at)
                - (int(pace.burst) - 1) * float(pace.interval_seconds)
                - now,
            )
        )
        age = (
            0.0
            if oldest is None
            else max(
                0.0,
                now
                - (
                    oldest if oldest.tzinfo else oldest.replace(tzinfo=timezone.utc)
                ).timestamp(),
            )
        )
        snapshots.append(
            AdmissionSnapshot(
                bucket=key,
                lane=str(pace.lane) if pace is not None else "default",
                capacity=int(bucket.capacity),
                max_pending=int(bucket.max_pending),
                active=int(active),
                pending=int(pending),
                oldest_pending_seconds=age,
                startup_interval_seconds=float(pace.interval_seconds)
                if pace is not None
                else None,
                startup_burst=int(pace.burst) if pace is not None else None,
                startup_delay_seconds=delay,
                delay_reason=(
                    "capacity"
                    if active >= bucket.capacity
                    else "startup_pacing"
                    if delay > 0
                    else "dispatch"
                )
                if pending
                else None,
            )
        )
    return snapshots


def record_queue_full(db: Session, bucket_key: str) -> None:
    """Count refusals, even though the acceptance transaction will roll back."""
    try:
        pace = db.get(TaskAdmissionPacing, bucket_key)
        lane = str(pace.lane) if pace is not None else "default"
        runtime_performance.increment(
            "task.admission.queue_full", attributes={"operation": lane}
        )
    except Exception:
        logger.debug("Admission refusal telemetry failed", exc_info=True)


def record_command_admission(db: Session, command: TaskExecutionCommand) -> None:
    """Observe committed claims; IDs remain in scoped snapshots, never metric labels."""
    try:
        ticket = db.get(TaskAdmissionTicket, command.id)
        if ticket is None or ticket.owner_attempt_id is None:
            return
        pace = db.get(TaskAdmissionPacing, ticket.bucket_key)
        lane = str(pace.lane) if pace is not None else "default"
        attributes = {
            "operation": lane,
            "outcome": "initial" if command.attempt_count == 1 else "retry_or_recovery",
        }
        runtime_performance.increment("task.admission.claims", attributes=attributes)
        if command.attempt_count == 1:
            now = float(db.execute(select(database_time())).scalar_one())
            accepted_at = (
                command.created_at
                if command.created_at.tzinfo
                else command.created_at.replace(tzinfo=timezone.utc)
            ).timestamp()
            runtime_performance.observe(
                "task.admission.initial_wait",
                max(0.0, now - accepted_at),
                unit="s",
                attributes={"operation": lane},
            )
    except Exception:
        logger.debug("Admission claim telemetry failed", exc_info=True)
