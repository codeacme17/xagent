"""Deployment hooks for classifying ``source="external"`` conversation logs.

The OSS runtime never stamps ``Task.source = "external"`` itself. The SaaS
session transport (session widget, header-WS, v1 external API) does, and the
same stored value covers both REST/SDK tasks created through a client
application and widget-session tasks. Telling them apart needs deployment-side
state (widget session linkage) that this repository does not model, so the
Conversation Logs page asks the deployment layer through the two hooks below.

Both hooks are optional. With none registered every ``external`` row is shown
under "REST API" and carries no public context, which keeps the rows visible
instead of dropping them from every channel filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from ..models.task import Task

EXTERNAL_TASK_SOURCE = "external"

# Hook signature: (db: Session) -> Sequence[tuple[<SQL predicate>, ui_source]]
# Each pair is a SQLAlchemy boolean expression over ``Task`` columns and the UI
# source ("widget", "rest_api", "shared_link", "webhook") it selects. The
# Conversation Logs query evaluates the pairs in order inside its
# classification CASE, restricted to rows whose stored source is "external";
# rows no predicate matches fall back to "rest_api". The hook is SQL-shaped so
# per-source counts, channel filters and pagination stay in the database
# instead of loading every task to classify it in Python. A predicate may
# reference ``Task`` columns directly; anything in another table must be
# reached through a self-contained subquery -- ``exists().where(Other.task_id
# == Task.id)`` or ``Task.id.in_(subquery)`` -- because the consuming queries
# join different tables and a bare cross-table comparison would multiply rows.
# Entries that are not a ``(SQL expression, "widget" | "rest_api" |
# "shared_link")`` pair (tuple or list) are skipped one at a time with a
# warning, and a hook that raises is treated as unregistered. Application layers inject it via
# set_external_task_source_hook().
ExternalTaskSourceHook = Callable[[Session], Sequence[tuple[Any, str]]]
_external_task_source_hook: ExternalTaskSourceHook | None = None

# Hook signature: (db: Session, task: Task, ui_source: str) -> dict | None
# Returns the ``public_context`` block for one external task's detail view,
# for example the widget session and end-user the task belongs to. ``None``
# means "no deployment context"; external rows never fall back to the
# agent_config-derived context of the legacy widget/share transports because
# the session transport does not populate those keys. Application layers
# inject it via set_external_task_context_hook().
ExternalTaskContextHook = Callable[[Session, "Task", str], dict[str, Any] | None]
_external_task_context_hook: ExternalTaskContextHook | None = None


def set_external_task_source_hook(hook: ExternalTaskSourceHook | None) -> None:
    global _external_task_source_hook
    _external_task_source_hook = hook


def external_task_source_branches(db: Session) -> list[tuple[Any, str]]:
    """Return the deployment's ``(predicate, ui_source)`` pairs, in order."""
    if _external_task_source_hook is None:
        return []
    return list(_external_task_source_hook(db))


def set_external_task_context_hook(hook: ExternalTaskContextHook | None) -> None:
    global _external_task_context_hook
    _external_task_context_hook = hook


def external_task_public_context(
    db: Session, task: Task, ui_source: str
) -> dict[str, Any] | None:
    """Return the deployment-provided public context for one external task."""
    if _external_task_context_hook is None:
        return None
    return _external_task_context_hook(db, task, ui_source)


__all__ = [
    "EXTERNAL_TASK_SOURCE",
    "ExternalTaskContextHook",
    "ExternalTaskSourceHook",
    "external_task_public_context",
    "external_task_source_branches",
    "set_external_task_context_hook",
    "set_external_task_source_hook",
]
