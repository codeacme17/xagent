from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from ..auth_dependencies import get_current_user
from ..models.agent import Agent
from ..models.chat_message import TaskChatMessage
from ..models.database import get_db
from ..models.task import Task
from ..models.trigger import AgentTrigger, TriggerRun
from ..models.user import User
from ..utils.db_timezone import format_datetime_for_api

router = APIRouter(prefix="/api/conversation-logs", tags=["conversation-logs"])

SOURCE_REST_API = "rest_api"
SOURCE_WEBHOOK = "webhook"
SOURCE_WIDGET = "widget"
SOURCE_SHARED_LINK = "shared_link"

SOURCE_LABELS = {
    SOURCE_REST_API: "REST API",
    SOURCE_WEBHOOK: "Webhook",
    SOURCE_WIDGET: "Widget",
    SOURCE_SHARED_LINK: "Shareable Link",
}
SOURCE_ORDER = [
    SOURCE_WIDGET,
    SOURCE_REST_API,
    SOURCE_SHARED_LINK,
    SOURCE_WEBHOOK,
]
EXTERNAL_TASK_SOURCES = {"sdk", "trigger", "widget", "shared_link"}


def _status_value(task: Task) -> str:
    status = getattr(task, "status", None)
    return str(status.value if hasattr(status, "value") else status or "unknown")


def _agent_config(task: Task) -> dict[str, Any]:
    config = getattr(task, "agent_config", None)
    return config if isinstance(config, dict) else {}


def _trigger_run_for_task(db: Session, task_id: int) -> TriggerRun | None:
    return (
        db.query(TriggerRun)
        .filter(TriggerRun.task_id == task_id)
        .order_by(TriggerRun.id.desc())
        .first()
    )


def _trigger_type_for_task(db: Session, task: Task) -> str | None:
    config_type = _agent_config(task).get("trigger_type")
    if config_type:
        return str(config_type)
    run = _trigger_run_for_task(db, int(task.id))
    if run and run.trigger:
        return str(run.trigger.type)
    return None


def _ui_source_for_task(db: Session, task: Task) -> str | None:
    source = str(getattr(task, "source", "") or "")
    if source == "sdk":
        return SOURCE_REST_API
    if source == "widget":
        return SOURCE_WIDGET
    if source == "shared_link":
        return SOURCE_SHARED_LINK
    if source == "trigger" and _trigger_type_for_task(db, task) == "webhook":
        return SOURCE_WEBHOOK
    return None


def _matches_search(task: Task, search: str | None) -> bool:
    if not search:
        return True
    needle = search.casefold()
    haystack = [
        task.title,
        task.description,
        task.input,
        task.output,
    ]
    return any(needle in str(value).casefold() for value in haystack if value)


def _message_sort_key(message: TaskChatMessage) -> tuple[bool, Any, int]:
    return (
        message.created_at is not None,
        message.created_at,
        int(message.id),
    )


def _base_task_query(db: Session, user: User):
    query = (
        db.query(Task)
        .options(selectinload(Task.agent), selectinload(Task.chat_messages))
        .filter(
            Task.is_visible.is_(False),
            Task.source.in_(sorted(EXTERNAL_TASK_SOURCES)),
        )
    )
    if not bool(user.is_admin):
        query = query.filter(Task.user_id == int(user.id))
    return query


def _load_candidate_tasks(
    db: Session,
    user: User,
    *,
    agent_id: int | None,
    search: str | None,
) -> list[tuple[Task, str]]:
    query = _base_task_query(db, user)
    if agent_id is not None:
        query = query.filter(Task.agent_id == agent_id)
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                Task.title.ilike(like),
                Task.description.ilike(like),
                Task.input.ilike(like),
                Task.output.ilike(like),
            )
        )

    tasks = query.order_by(Task.updated_at.desc(), Task.id.desc()).all()
    external_tasks: list[tuple[Task, str]] = []
    for task in tasks:
        ui_source = _ui_source_for_task(db, task)
        if ui_source is None:
            continue
        if not _matches_search(task, search):
            continue
        external_tasks.append((task, ui_source))
    return external_tasks


def _last_activity_at(task: Task) -> Any:
    messages = list(getattr(task, "chat_messages", []) or [])
    if messages:
        latest_message = max(messages, key=_message_sort_key)
        return latest_message.created_at or task.updated_at or task.created_at
    return task.updated_at or task.created_at


def _serialize_log_summary(task: Task, ui_source: str) -> dict[str, Any]:
    agent = task.agent if isinstance(task.agent, Agent) else None
    return {
        "task_id": int(task.id),
        "title": task.title,
        "description": task.description,
        "status": _status_value(task),
        "source": ui_source,
        "source_label": SOURCE_LABELS[ui_source],
        "stored_source": task.source,
        "agent_id": int(task.agent_id) if task.agent_id is not None else None,
        "agent_name": agent.name if agent else None,
        "agent_logo_url": agent.logo_url if agent else None,
        "created_at": format_datetime_for_api(task.created_at),
        "updated_at": format_datetime_for_api(task.updated_at),
        "last_activity_at": format_datetime_for_api(_last_activity_at(task)),
        "input_tokens": task.input_tokens or 0,
        "output_tokens": task.output_tokens or 0,
        "total_tokens": task.total_tokens or 0,
        "llm_calls": task.llm_calls or 0,
        "message_count": len(getattr(task, "chat_messages", []) or []),
    }


def _serialize_transcript(messages: list[TaskChatMessage]) -> list[dict[str, Any]]:
    return [
        {
            "id": int(message.id),
            "role": message.role,
            "content": message.content,
            "message_type": message.message_type,
            "interactions": message.interactions,
            "turn_id": message.turn_id,
            "attachments": message.attachments or [],
            "created_at": format_datetime_for_api(message.created_at),
        }
        for message in sorted(messages, key=_message_sort_key)
    ]


def _serialize_trigger_metadata(
    db: Session,
    task: Task,
) -> dict[str, Any] | None:
    run = _trigger_run_for_task(db, int(task.id))
    trigger = run.trigger if run else None
    config = _agent_config(task)
    trigger_type = str(
        getattr(trigger, "type", None) or config.get("trigger_type") or ""
    )
    if trigger_type != "webhook":
        return None

    return {
        "trigger_id": int(trigger.id)
        if isinstance(trigger, AgentTrigger)
        else config.get("trigger_id"),
        "trigger_run_id": int(run.id) if run else config.get("trigger_run_id"),
        "trigger_type": "webhook",
        "source_event_id": run.source_event_id if run else None,
        "status": str(run.status) if run else None,
        "test": bool(config.get("trigger_test", False)),
    }


def _serialize_public_context(task: Task, ui_source: str) -> dict[str, Any] | None:
    config = _agent_config(task)
    if ui_source == SOURCE_WIDGET:
        return {
            "guest_id": config.get("guest_id"),
            "auth_mode": config.get("auth_mode") or "widget",
            "channel_name": task.channel_name,
            "widget_agent_id": config.get("widget_agent_id"),
        }
    if ui_source == SOURCE_SHARED_LINK:
        return {
            "auth_mode": config.get("auth_mode") or "share",
            "channel_name": task.channel_name,
            "share_agent_id": config.get("share_agent_id") or task.agent_id,
        }
    return None


def _source_counts(items: list[tuple[Task, str]]) -> dict[str, int]:
    counts = {source: 0 for source in SOURCE_ORDER}
    for _task, source in items:
        counts[source] += 1
    return {"all": len(items), **counts}


def _agent_options(items: list[tuple[Task, str]]) -> list[dict[str, Any]]:
    options: dict[int, dict[str, Any]] = {}
    for task, _source in items:
        if task.agent_id is None:
            continue
        agent_id = int(task.agent_id)
        if agent_id in options:
            continue
        agent = task.agent if isinstance(task.agent, Agent) else None
        options[agent_id] = {
            "agent_id": agent_id,
            "agent_name": agent.name if agent else f"Agent {agent_id}",
            "agent_logo_url": agent.logo_url if agent else None,
        }
    return sorted(options.values(), key=lambda item: item["agent_name"].casefold())


@router.get("")
async def list_conversation_logs(
    source: str = Query("all"),
    agent_id: int | None = Query(None),
    search: str | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    normalized_source = source.strip().lower()
    allowed_sources = {"all", *SOURCE_ORDER}
    if normalized_source not in allowed_sources:
        raise HTTPException(status_code=400, detail="Unsupported conversation source")

    candidate_items = _load_candidate_tasks(
        db,
        user,
        agent_id=agent_id,
        search=search.strip() if search else None,
    )
    filtered_items = (
        candidate_items
        if normalized_source == "all"
        else [item for item in candidate_items if item[1] == normalized_source]
    )
    total = len(filtered_items)
    start = (page - 1) * per_page
    paged_items = filtered_items[start : start + per_page]

    return {
        "logs": [
            _serialize_log_summary(task, ui_source)
            for task, ui_source in paged_items
        ],
        "source_counts": _source_counts(candidate_items),
        "agents": _agent_options(candidate_items),
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": max(1, math.ceil(total / per_page)),
        },
    }


@router.get("/{task_id}")
async def get_conversation_log_detail(
    task_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    query = _base_task_query(db, user).filter(Task.id == task_id)
    task = query.first()
    if task is None:
        raise HTTPException(status_code=404, detail="Conversation log not found")

    ui_source = _ui_source_for_task(db, task)
    if ui_source is None:
        raise HTTPException(status_code=404, detail="Conversation log not found")

    messages = list(task.chat_messages or [])
    return {
        "log": _serialize_log_summary(task, ui_source),
        "transcript": _serialize_transcript(messages),
        "metadata": {
            "task": {
                "task_id": int(task.id),
                "input": task.input,
                "output": task.output,
                "error_message": task.error_message,
                "description": task.description,
            },
            "trigger": _serialize_trigger_metadata(db, task),
            "public_context": _serialize_public_context(task, ui_source),
        },
        "read_only": True,
    }
