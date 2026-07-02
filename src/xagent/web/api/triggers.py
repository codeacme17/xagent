from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.trigger import AgentTrigger, TriggerAuditOutcome, TriggerRun
from ..models.user import User
from ..services.trigger_providers import (
    CallbackRequestContext,
    process_trigger_callback,
    record_trigger_audit,
)
from ..services.trigger_rate_limit import (
    check_callback_rate_limit,
    check_trigger_crud_rate_limit,
    remote_ip_from_request,
)
from ..services.triggers import (
    TriggerNotFoundError,
    TriggerSecretError,
    TriggerServiceError,
    create_agent_trigger,
    decrypt_trigger_run_payload,
    delete_agent_trigger,
    fire_trigger,
    get_owned_agent,
    get_owned_trigger,
    update_agent_trigger,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["triggers"])


class TriggerCreateRequest(BaseModel):
    type: Literal["webhook", "scheduled", "gmail"]
    name: str | None = Field(default=None, max_length=200)
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)
    prompt_template: str | None = None
    secret: str | None = None


class TriggerUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    enabled: bool | None = None
    config: dict[str, Any] | None = None
    prompt_template: str | None = None
    secret: str | None = None
    rotate_secret: bool = False


class TriggerTestRequest(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    source_event_id: str | None = None


class TriggerResponse(BaseModel):
    id: int
    user_id: int
    agent_id: int
    type: str
    name: str
    enabled: bool
    config: dict[str, Any]
    prompt_template: str | None
    webhook_token: str | None
    webhook_secret: str | None = None
    callback_id: str | None = None
    provisioning_status: str | None = None
    provisioning_error: str | None = None
    next_run_at: str | None
    last_run_at: str | None
    last_error: str | None
    created_at: str | None
    updated_at: str | None


class TriggerRunResponse(BaseModel):
    id: int
    trigger_id: int
    task_id: int | None
    background_job_id: str | None
    status: str
    source_event_id: str | None
    payload_snapshot: dict[str, Any] | None
    payload_stored: bool = False
    payload: Any | None = None
    idempotency_key: str
    error_message: str | None
    started_at: str | None
    finished_at: str | None
    created_at: str | None
    updated_at: str | None


class TriggerFireResponse(BaseModel):
    trigger_run: TriggerRunResponse
    duplicate: bool = False


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _serialize_trigger(
    trigger: AgentTrigger, *, webhook_secret: str | None = None
) -> TriggerResponse:
    return TriggerResponse(
        id=int(trigger.id),
        user_id=int(trigger.user_id),
        agent_id=int(trigger.agent_id),
        type=str(trigger.type),
        name=str(trigger.name),
        enabled=bool(trigger.enabled),
        config=dict(trigger.config or {}),
        prompt_template=trigger.prompt_template,
        webhook_token=trigger.webhook_token,
        webhook_secret=webhook_secret,
        callback_id=trigger.callback_id,
        provisioning_status=trigger.provisioning_status,
        provisioning_error=trigger.provisioning_error,
        next_run_at=_dt(cast(datetime | None, trigger.next_run_at)),
        last_run_at=_dt(cast(datetime | None, trigger.last_run_at)),
        last_error=trigger.last_error,
        created_at=_dt(cast(datetime | None, trigger.created_at)),
        updated_at=_dt(cast(datetime | None, trigger.updated_at)),
    )


def _serialize_run(
    run: TriggerRun, *, payload: Any | None = None
) -> TriggerRunResponse:
    snapshot = run.payload_snapshot if isinstance(run.payload_snapshot, dict) else None
    payload_stored = bool(snapshot and "encrypted_payload" in snapshot)
    if snapshot and payload_stored:
        # Never ship ciphertext to clients; decrypted content is only
        # available through the audited include_payload read.
        snapshot = {k: v for k, v in snapshot.items() if k != "encrypted_payload"}
    return TriggerRunResponse(
        id=int(run.id),
        trigger_id=int(run.trigger_id),
        task_id=int(run.task_id) if run.task_id is not None else None,
        background_job_id=run.background_job_id,
        status=str(run.status),
        source_event_id=run.source_event_id,
        payload_snapshot=snapshot,
        payload_stored=payload_stored,
        payload=payload,
        idempotency_key=str(run.idempotency_key),
        error_message=run.error_message,
        started_at=_dt(getattr(run, "started_at", None)),
        finished_at=_dt(getattr(run, "finished_at", None)),
        created_at=_dt(getattr(run, "created_at", None)),
        updated_at=_dt(getattr(run, "updated_at", None)),
    )


def _agent_or_404(db: Session, *, user_id: int, agent_id: int) -> None:
    if get_owned_agent(db, user_id=user_id, agent_id=agent_id) is None:
        raise HTTPException(status_code=404, detail="Agent not found")


def _trigger_or_404(
    db: Session,
    *,
    user_id: int,
    agent_id: int,
    trigger_id: int,
) -> AgentTrigger:
    trigger = get_owned_trigger(
        db, user_id=user_id, agent_id=agent_id, trigger_id=trigger_id
    )
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    return trigger


def _enforce_crud_rate_limit(user_id: int) -> None:
    if not check_trigger_crud_rate_limit(user_id):
        raise HTTPException(
            status_code=429, detail="Too many trigger management requests"
        )


def _handle_service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, TriggerNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, TriggerSecretError):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, TriggerServiceError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.exception("Unhandled trigger API error")
    return HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/api/agents/{agent_id}/triggers",
    response_model=list[TriggerResponse],
)
async def list_triggers(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[TriggerResponse]:
    user_id = int(current_user.id)
    _agent_or_404(db, user_id=user_id, agent_id=agent_id)
    rows = (
        db.query(AgentTrigger)
        .filter(AgentTrigger.user_id == user_id, AgentTrigger.agent_id == agent_id)
        .order_by(AgentTrigger.created_at.desc(), AgentTrigger.id.desc())
        .all()
    )
    return [_serialize_trigger(row) for row in rows]


@router.post(
    "/api/agents/{agent_id}/triggers",
    response_model=TriggerResponse,
)
async def create_trigger(
    agent_id: int,
    request: TriggerCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TriggerResponse:
    _enforce_crud_rate_limit(int(current_user.id))
    try:
        # Runs in a worker thread: Gmail trigger provisioning can block on
        # cloud calls up to the registration timeout.
        trigger, secret = await asyncio.to_thread(
            create_agent_trigger,
            db,
            user_id=int(current_user.id),
            agent_id=agent_id,
            trigger_type=request.type,
            name=request.name,
            enabled=request.enabled,
            config=request.config,
            prompt_template=request.prompt_template,
            secret=request.secret,
        )
        return _serialize_trigger(trigger, webhook_secret=secret)
    except Exception as exc:
        raise _handle_service_error(exc)


@router.patch(
    "/api/agents/{agent_id}/triggers/{trigger_id}",
    response_model=TriggerResponse,
)
async def update_trigger(
    agent_id: int,
    trigger_id: int,
    request: TriggerUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TriggerResponse:
    _enforce_crud_rate_limit(int(current_user.id))
    try:
        trigger, secret = await asyncio.to_thread(
            update_agent_trigger,
            db,
            user_id=int(current_user.id),
            agent_id=agent_id,
            trigger_id=trigger_id,
            updates=request.model_dump(exclude_unset=True),
        )
        return _serialize_trigger(trigger, webhook_secret=secret)
    except Exception as exc:
        raise _handle_service_error(exc)


@router.delete("/api/agents/{agent_id}/triggers/{trigger_id}")
async def delete_trigger(
    agent_id: int,
    trigger_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    _enforce_crud_rate_limit(int(current_user.id))
    try:
        await asyncio.to_thread(
            delete_agent_trigger,
            db,
            user_id=int(current_user.id),
            agent_id=agent_id,
            trigger_id=trigger_id,
        )
        return {"message": "Trigger deleted"}
    except Exception as exc:
        raise _handle_service_error(exc)


@router.get(
    "/api/agents/{agent_id}/triggers/{trigger_id}/runs",
    response_model=list[TriggerRunResponse],
)
async def list_trigger_runs(
    agent_id: int,
    trigger_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[TriggerRunResponse]:
    trigger = _trigger_or_404(
        db,
        user_id=int(current_user.id),
        agent_id=agent_id,
        trigger_id=trigger_id,
    )
    rows = (
        db.query(TriggerRun)
        .filter(TriggerRun.trigger_id == int(trigger.id))
        .order_by(TriggerRun.created_at.desc(), TriggerRun.id.desc())
        .limit(100)
        .all()
    )
    return [_serialize_run(row) for row in rows]


@router.get(
    "/api/agents/{agent_id}/triggers/{trigger_id}/runs/{run_id}",
    response_model=TriggerRunResponse,
)
async def get_trigger_run(
    agent_id: int,
    trigger_id: int,
    run_id: int,
    request: Request,
    include_payload: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TriggerRunResponse:
    trigger = _trigger_or_404(
        db,
        user_id=int(current_user.id),
        agent_id=agent_id,
        trigger_id=trigger_id,
    )
    run = (
        db.query(TriggerRun)
        .filter(TriggerRun.id == run_id, TriggerRun.trigger_id == int(trigger.id))
        .first()
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Trigger run not found")

    payload: Any | None = None
    if include_payload:
        try:
            payload = decrypt_trigger_run_payload(run)
        except Exception as exc:
            raise _handle_service_error(exc)
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.PAYLOAD_READ,
            provider=str(trigger.provider) if trigger.provider else None,
            callback_id=str(trigger.callback_id) if trigger.callback_id else None,
            trigger_id=int(trigger.id),
            detail={
                "trigger_run_id": int(run.id),
                "user_id": int(current_user.id),
            },
            remote_ip=request.client.host if request.client else None,
        )
    return _serialize_run(run, payload=payload)


@router.post(
    "/api/agents/{agent_id}/triggers/{trigger_id}/test",
    response_model=TriggerFireResponse,
)
async def test_trigger(
    agent_id: int,
    trigger_id: int,
    request: TriggerTestRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TriggerFireResponse:
    trigger = _trigger_or_404(
        db,
        user_id=int(current_user.id),
        agent_id=agent_id,
        trigger_id=trigger_id,
    )
    try:
        run, created = await fire_trigger(
            db,
            trigger=trigger,
            event_payload=request.payload,
            source_event_id=request.source_event_id,
            test=True,
            event_type="test",
        )
        return TriggerFireResponse(
            trigger_run=_serialize_run(run), duplicate=not created
        )
    except Exception as exc:
        raise _handle_service_error(exc)


_SECRET_QUERY_PARAMS = frozenset(
    {"token", "secret", "signature", "key", "apikey", "api_key", "auth"}
)


class TriggerCallbackResponse(BaseModel):
    outcome: str | None
    detail: str | None = None
    trigger_run_ids: list[int] = Field(default_factory=list)
    duplicates: int = 0


@router.post(
    "/api/triggers/callback/{provider}/{callback_id}",
    response_model=TriggerCallbackResponse,
)
async def receive_trigger_callback(
    provider: str,
    callback_id: str,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> Response | TriggerCallbackResponse:
    """Public unified trigger callback endpoint.

    The callback id is only a locator; authentication happens inside the
    provider pipeline via provider-specific proof (e.g. HMAC signature
    headers). The raw body is preserved for signature verification.
    """
    remote_ip = remote_ip_from_request(request)
    # Rate limiting runs before any parsing or audit write, so hostile or
    # garbage traffic cannot amplify database writes.
    if not check_callback_rate_limit(callback_id, remote_ip):
        raise HTTPException(status_code=429, detail="Too many callback requests")

    # Secrets are accepted only via headers. A request that smuggles its
    # proof through the query string is rejected outright.
    for query_key in request.query_params:
        if query_key.lower() in _SECRET_QUERY_PARAMS:
            raise HTTPException(
                status_code=400,
                detail="Callback credentials must be sent via headers",
            )

    raw_body = await request.body()
    context = CallbackRequestContext(
        provider=provider,
        callback_id=callback_id,
        method=request.method,
        url_path=request.url.path,
        headers=dict(request.headers),
        query_params=dict(request.query_params),
        remote_ip=remote_ip,
    )
    result = await process_trigger_callback(db, context=context, raw_body=raw_body)
    if result.challenge is not None:
        return Response(
            content=result.challenge.body,
            media_type=result.challenge.media_type,
            status_code=result.challenge.status_code,
        )
    response.status_code = result.status_code
    return TriggerCallbackResponse(
        outcome=result.outcome.value if result.outcome else None,
        detail=result.detail,
        trigger_run_ids=[int(run.id) for run in result.runs],
        duplicates=result.duplicates,
    )
