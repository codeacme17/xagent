"""Atomic persistence for terminal task-command outcomes."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..models.task import Task
from ..models.task_command import TaskExecutionCommand
from ..models.task_command_terminal_event import TaskCommandTerminalEvent


class TerminalTaskEventMessageCode(str, enum.Enum):
    """Closed vocabulary for rendering terminal outcomes without stored text."""

    TASK_COMMAND_FAILED = "task_command_failed"
    TASK_COMMAND_DEFERRED = "task_command_deferred"
    EXTERNAL_CANCEL_NOT_APPLIED = "external_cancel_not_applied"
    EXTERNAL_TURN_INTERRUPTED = "external_turn_interrupted"


class TerminalTaskEventOutcome(str, enum.Enum):
    """Terminal command dispositions that may be projected to a client."""

    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class TerminalTaskEventDraft:
    """Client-safe terminal outcome staged by a command disposition."""

    outcome: TerminalTaskEventOutcome
    message_code: TerminalTaskEventMessageCode | None
    resend_safe: bool
    include_command_identity: bool = True


_DRAFT_ATTRIBUTE = "_xagent_terminal_task_event_draft"
_CANCEL_COMMAND_KIND = "cancel"
_EXTERNAL_COMMAND_SCOPE = "external"


def _is_external_cancel(command: TaskExecutionCommand) -> bool:
    """Match the normalized durable shape used by the WebSocket classifier.

    ``TaskCommandKind`` lives in the transport module that imports this one,
    so importing that enum here would create a cycle. The producer stores the
    enum value, and WebSocket ingress accepts the same exact external scope;
    keeping both checks strict prevents malformed payload values from gaining
    the anonymous-audience disclosure policy.
    """

    return (
        str(command.kind) == _CANCEL_COMMAND_KIND
        and isinstance(command.payload, dict)
        and command.payload.get("scope") == _EXTERNAL_COMMAND_SCOPE
    )


def bind_terminal_event_draft(
    error: BaseException,
    draft: TerminalTaskEventDraft,
) -> None:
    """Attach client-safe presentation metadata without performing delivery."""

    setattr(error, _DRAFT_ATTRIBUTE, draft)


def terminal_event_draft_for_error(
    error: BaseException,
) -> TerminalTaskEventDraft | None:
    """Read presentation metadata previously attached by an executor adapter."""

    draft = getattr(error, _DRAFT_ATTRIBUTE, None)
    return draft if isinstance(draft, TerminalTaskEventDraft) else None


def stage_terminal_event(
    db: Session,
    *,
    command_db_id: int,
    draft: TerminalTaskEventDraft | None = None,
) -> TaskCommandTerminalEvent:
    """Stage one idempotent event without committing the caller's transaction.

    The command must already have a terminal disposition in this transaction.
    The caller owns the commit, which makes disposition and event one recovery
    boundary instead of two best-effort operations. Run correlation is copied
    exclusively from the immutable command-acceptance snapshot; reading the
    task's current run or state version here could associate an old command
    outcome with a newer interaction.
    """

    snapshot = (
        db.query(TaskExecutionCommand, Task)
        .join(Task, Task.id == TaskExecutionCommand.task_id)
        .filter(TaskExecutionCommand.id == command_db_id)
        .populate_existing()
        .one_or_none()
    )
    if snapshot is None:
        raise ValueError(f"Task command {command_db_id} does not exist")
    command, task = snapshot
    if command.status not in {"completed", "failed"}:
        raise ValueError(
            f"Task command {command_db_id} is not terminal: {command.status}"
        )
    outcome_version = int(command.attempt_count or 0)
    if draft is None:
        failed = command.status == "failed"
        draft = TerminalTaskEventDraft(
            outcome=TerminalTaskEventOutcome(str(command.status)),
            message_code=(
                TerminalTaskEventMessageCode.TASK_COMMAND_FAILED if failed else None
            ),
            resend_safe=False,
        )
    if draft.outcome.value != command.status:
        raise ValueError(
            "Terminal event outcome must match the command disposition: "
            f"{draft.outcome.value!r} != {command.status!r}"
        )
    existing = (
        db.query(TaskCommandTerminalEvent)
        .filter(
            TaskCommandTerminalEvent.task_command_id == command_db_id,
            TaskCommandTerminalEvent.outcome_version == outcome_version,
        )
        .one_or_none()
    )
    if existing is not None:
        return existing

    event = TaskCommandTerminalEvent(
        event_id=str(uuid.uuid4()),
        task_command_id=int(command.id),
        task_id=int(command.task_id),
        task_run_id=command.target_run_id,
        task_state_version=(
            int(command.target_state_version)
            if command.target_state_version is not None
            else None
        ),
        command_id=str(command.command_id),
        command_kind=str(command.kind),
        actor_user_id=(
            int(command.actor_user_id) if command.actor_user_id is not None else None
        ),
        task_owner_user_id=int(task.user_id),
        outcome_version=outcome_version,
        outcome=draft.outcome.value,
        message_code=(draft.message_code.value if draft.message_code else None),
        resend_safe=bool(draft.resend_safe),
        include_command_identity=bool(
            draft.include_command_identity
            and command.actor_user_id is not None
            and not _is_external_cancel(command)
        ),
    )
    db.add(event)
    db.flush()
    return event
