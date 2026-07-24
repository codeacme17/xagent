"""Scheduled maintenance tasks (Celery Beat entrypoints)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ...config import get_taskless_upload_ttl_seconds
from ..models.database import get_session_local, init_db
from ..services.orphan_upload_gc import cleanup_orphaned_taskless_uploads
from .celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="xagent.web.jobs.maintenance_tasks.sweep_orphaned_uploads")
def sweep_orphaned_uploads() -> dict[str, Any]:
    """Celery Beat entrypoint: GC task-less public uploads never bound to a task.

    Thin wrapper around :func:`cleanup_orphaned_taskless_uploads` (which holds
    the reap logic and is unit-tested directly); this only owns the session
    lifecycle and TTL wiring, mirroring ``scan_due_triggers``.
    """
    logger.info("Orphan task-less upload GC tick")
    try:
        SessionLocal = get_session_local()
    except RuntimeError:
        init_db()
        SessionLocal = get_session_local()

    db = SessionLocal()
    try:
        deleted = cleanup_orphaned_taskless_uploads(
            db, older_than_seconds=get_taskless_upload_ttl_seconds()
        )
        return {
            "status": "ok",
            "deleted": deleted,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        db.close()
