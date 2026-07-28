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

Reaping is race-free against run-start binding via the repo's established
claim protocol (see ``compensate_registered_uploads_sync``): binding is a
conditional ``UPDATE ... SET task_id WHERE task_id IS NULL AND
storage_status != 'compensating'``, and GC claims with a conditional
``UPDATE ... SET storage_status = 'compensating' WHERE task_id IS NULL``.
The database serializes the two row-level updates, so a row is either bound
or reaped — never both.
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

# Bounded sweep shape: rows are reaped in deterministic batches so a large
# backlog can neither materialize wholesale into worker memory nor run
# unbounded into the next scheduled sweep. Whatever a capped sweep leaves
# behind still matches the predicate and is picked up by the next tick.
GC_BATCH_SIZE = 500
GC_MAX_BATCHES = 20


def _claim_orphan(db: Session, row_id: int) -> bool:
    """Atomically claim one still-unbound row for deletion.

    The conditional UPDATE is the counterpart of run-start binding
    (``bind_turn_files_no_commit``: ``WHERE task_id IS NULL AND
    storage_status != 'compensating'``): if binding committed first the
    ``task_id IS NULL`` predicate no longer matches and the claim returns
    False (row spared); if the claim commits first, binding skips the row
    because it is now ``compensating``. Committed immediately so the claim is
    visible to concurrent binders before any storage I/O starts.
    """
    claimed = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.id == row_id,
            UploadedFile.task_id.is_(None),
        )
        .update(
            {UploadedFile.storage_status: "compensating"},
            synchronize_session=False,
        )
    )
    if claimed != 1:
        db.rollback()
        return False
    db.commit()
    return True


def cleanup_orphaned_taskless_uploads(
    db: Session,
    *,
    older_than_seconds: int,
    now: datetime | None = None,
    batch_size: int = GC_BATCH_SIZE,
    max_batches: int = GC_MAX_BATCHES,
) -> int:
    """Delete task-less public-share uploads that were never bound to a task.

    Reaps rows that (a) carry the task-less-share marker, (b) still have no
    ``task_id``, and (c) are older than the TTL. Each row is first claimed
    with :func:`_claim_orphan` so a concurrent run-start bind is spared, then
    on-disk file, durable object, and preview cache are removed via
    :class:`UploadedFileStore` (same semantics as a normal delete).

    The sweep processes at most ``max_batches`` batches of ``batch_size``
    rows, oldest first. Per-row failures are logged and skipped so one bad
    row does not abort the sweep; a failed row stays ``compensating`` (no
    longer bindable) and still matches the predicate, so a later sweep
    retries it. Returns the number of rows deleted.
    """
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - timedelta(seconds=older_than_seconds)
    store = UploadedFileStore(db)
    deleted = 0
    for _ in range(max_batches):
        rows = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.upload_source == TASKLESS_SHARE_UPLOAD_SOURCE,
                UploadedFile.task_id.is_(None),
                UploadedFile.created_at < cutoff,
            )
            .order_by(UploadedFile.created_at, UploadedFile.id)
            .limit(batch_size)
            .all()
        )
        if not rows:
            break
        batch_deleted = 0
        for row in rows:
            try:
                if not _claim_orphan(db, int(row.id)):
                    continue
                store.delete(row)
                db.commit()
                deleted += 1
                batch_deleted += 1
            except Exception:
                db.rollback()
                logger.warning(
                    "Failed to GC orphaned task-less upload id=%s",
                    getattr(row, "id", "?"),
                    exc_info=True,
                )
        if len(rows) < batch_size:
            break
        if batch_deleted == 0:
            # A full batch produced no deletions (all spared or failing):
            # refetching would return the same stuck set, so yield to the
            # next scheduled sweep instead of spinning.
            break
    return deleted
