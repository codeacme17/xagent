"""Garbage collection of orphaned task-less public uploads (#973).

A task-less public-share upload (workforce first-turn attachment) is created
BEFORE its run/task exists, then bound to the task at run start. If the guest
never completes task creation, the row + on-disk file are never bound and
never cleaned up. This reaps those orphans.

The predicate is deliberately narrow. ``task_id IS NULL`` is a system-wide
normal intermediate state (plain ``/api/files/upload`` allows an optional
task id, and turn handling binds unbound rows across every channel), so a
coarse "NULL + aged" sweep would delete logged-in users' un-sent draft
attachments. The ``upload_source`` marker (stamped only on the task-less
public-share path) scopes GC to exactly those uploads.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from ..models.uploaded_file import UploadedFile
from .uploaded_file_store import UploadedFileStore

logger = logging.getLogger(__name__)

# Provenance marker stamped on task-less public-share uploads. Orphan GC keys
# off it so the sweep only ever touches uploads created before any task
# binding on the public share path — never any other path's unbound draft.
TASKLESS_SHARE_UPLOAD_SOURCE = "taskless_share_upload"


def cleanup_orphaned_taskless_uploads(
    db: Session,
    *,
    older_than_seconds: int,
    now: datetime | None = None,
) -> int:
    """Delete task-less public-share uploads that were never bound to a task.

    Reaps rows that (a) carry the task-less-share marker, (b) still have no
    ``task_id``, and (c) are older than the TTL. The still-unbound state is
    re-confirmed per row at delete time so a row bound between the query and
    the delete is spared. On-disk file, durable object, and preview cache are
    removed via :class:`UploadedFileStore` (same semantics as a normal
    delete). Per-row failures are logged and skipped so one bad row does not
    abort the sweep. Returns the number of rows deleted.
    """
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - timedelta(seconds=older_than_seconds)
    rows = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.upload_source == TASKLESS_SHARE_UPLOAD_SOURCE,
            UploadedFile.task_id.is_(None),
            UploadedFile.created_at < cutoff,
        )
        .all()
    )
    store = UploadedFileStore(db)
    deleted = 0
    for row in rows:
        try:
            # Force a fresh read so a bind committed by a concurrent run-start
            # (between the query above and now) is visible — a stale identity-
            # map snapshot would otherwise let GC delete a now-bound
            # attachment. refresh raises if the row was deleted meanwhile, in
            # which case there is nothing to GC.
            db.refresh(row)
            if row.task_id is not None:
                continue
            store.delete(row)
            db.commit()
            deleted += 1
        except Exception:
            db.rollback()
            logger.warning(
                "Failed to GC orphaned task-less upload id=%s",
                getattr(row, "id", "?"),
                exc_info=True,
            )
    return deleted
