"""Leaf-level task-command vocabulary and audience classification."""

from __future__ import annotations

import enum


class TaskCommandKind(str, enum.Enum):
    """Closed vocabulary for commands carried by the durable transport."""

    MESSAGE = "message"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


EXTERNAL_COMMAND_SCOPE = "external"


class TaskCommandAudience(str, enum.Enum):
    """Audience and disclosure policy named by a task-command payload."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    UNSUPPORTED = "unsupported"


def classify_task_command_audience(
    kind: TaskCommandKind | str,
    payload: object,
) -> TaskCommandAudience:
    """Classify a complete payload without accepting malformed cancel scopes."""

    normalized_kind = kind.value if isinstance(kind, TaskCommandKind) else kind
    if normalized_kind != TaskCommandKind.CANCEL.value:
        return TaskCommandAudience.INTERNAL
    if not isinstance(payload, dict):
        return TaskCommandAudience.UNSUPPORTED
    if "scope" not in payload:
        return TaskCommandAudience.INTERNAL
    if payload["scope"] == EXTERNAL_COMMAND_SCOPE:
        return TaskCommandAudience.EXTERNAL
    return TaskCommandAudience.UNSUPPORTED
