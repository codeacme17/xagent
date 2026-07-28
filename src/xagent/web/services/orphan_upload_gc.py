"""Garbage collection of orphaned task-less public uploads (#973).

A task-less public-share upload (workforce first-turn attachment) is created
BEFORE its run/task exists, then bound to the task at run start. If the guest
never completes task creation, the row + stored bytes are never bound and
never cleaned up. This reaps those orphans.

The predicate is deliberately narrow. ``task_id IS NULL`` is a system-wide
normal intermediate state (plain ``/api/files/upload`` allows an optional
task id, and turn handling binds unbound rows across every channel), so a
coarse "NULL + aged" sweep would delete logged-in users' un-sent draft
attachments. The ``upload_source`` marker (stamped only on the task-less
public-share path) scopes GC to exactly those uploads.

Deletion rides the existing uploaded-file compensation protocol rather than
a bespoke one, so every crash window is already owned by shipped machinery:

1. **Local file first**, before any claim. At that point the row is still
   ``available`` and durable-backed, so a consumer that wins the bind can
   re-materialize the bytes from the durable object (``ensure_local``) — a
   crash here leaves a fully consistent row that the next sweep retries.
2. **Exact claim** — the same CAS as ``compensate_registered_uploads_sync``:
   ``SET storage_status='compensating', updated_at=<token> WHERE id/user/
   file_id/storage_key match AND storage_status='available' AND task_id IS
   NULL``. Requiring the exact prior status makes overlapping sweeps
   mutually exclusive (the loser matches zero rows), and the ``task_id IS
   NULL`` predicate serializes against binders. The persisted ``updated_at``
   is the generation token fencing the later settlement.
3. **Durable delete + settle** via the compensation helpers
   (:func:`delete_uploaded_file_compensation_object` /
   :func:`settle_uploaded_file_compensation_no_commit`). A crash or deferred
   presence after the claim leaves an aged ``compensating`` row that the
   stale-compensation recovery loop (``uploaded_file_recovery``) takes over
   and finishes — the local file is already gone by step 1, so that generic
   path (which knows no ``storage_path``) never leaks anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from ..models.uploaded_file import UploadedFile
from .uploaded_file_store import (
    delete_registered_preview_caches,
    delete_uploaded_file_compensation_object,
    settle_uploaded_file_compensation_no_commit,
)

logger = logging.getLogger(__name__)

# Provenance marker stamped on task-less public-share uploads. Orphan GC keys
# off it so the sweep only ever touches uploads created before any task
# binding on the public share path — never any other path's unbound draft.
TASKLESS_SHARE_UPLOAD_SOURCE = "taskless_share_upload"

# Bounded sweep shape: rows are reaped in deterministic keyset-paged batches
# so a large backlog can neither materialize wholesale into worker memory nor
# run unbounded into the next scheduled sweep. Whatever a capped sweep leaves
# behind still matches the predicate and is picked up by the next tick.
GC_BATCH_SIZE = 500
GC_MAX_BATCHES = 20

_OrphanCursor = tuple[datetime, int]


@dataclass(frozen=True)
class _OrphanUploadCandidate:
    """Detached exact identity of one reap-eligible row.

    Captured before any mutation so the storage claim and settlement fence on
    the same version the scan saw, and so the local ``storage_path`` survives
    the metadata deletion.
    """

    row_id: int
    user_id: int
    file_id: str
    storage_key: str
    storage_path: str
    created_at: datetime

    @property
    def cursor(self) -> _OrphanCursor:
        return self.created_at, self.row_id


def _orphan_candidates(
    db: Session,
    *,
    cutoff: datetime,
    limit: int,
    after: _OrphanCursor | None,
) -> tuple[_OrphanUploadCandidate, ...]:
    """One keyset page of reap-eligible rows, oldest first.

    Scoped to ``storage_status == 'available'`` with a durable key: that is
    the only state the registration path ever leaves a marked row in, it is
    the state the claim CAS requires, and it keeps rows another owner already
    claimed (``compensating`` — in-flight GC, request compensation, or stale-
    claim recovery) out of the scan entirely.
    """
    query = db.query(UploadedFile).filter(
        UploadedFile.upload_source == TASKLESS_SHARE_UPLOAD_SOURCE,
        UploadedFile.task_id.is_(None),
        UploadedFile.created_at < cutoff,
        UploadedFile.storage_status == "available",
        UploadedFile.storage_key.isnot(None),
        UploadedFile.storage_key != "",
    )
    if after is not None:
        after_created_at, after_row_id = after
        query = query.filter(
            or_(
                UploadedFile.created_at > after_created_at,
                and_(
                    UploadedFile.created_at == after_created_at,
                    UploadedFile.id > after_row_id,
                ),
            )
        )
    records = (
        query.order_by(UploadedFile.created_at.asc(), UploadedFile.id.asc())
        .limit(limit)
        .all()
    )
    return tuple(
        _OrphanUploadCandidate(
            row_id=int(record.id),
            user_id=int(record.user_id),
            file_id=str(record.file_id),
            storage_key=str(record.storage_key),
            storage_path=str(record.storage_path),
            created_at=cast(datetime, record.created_at),
        )
        for record in records
    )


def _delete_local_file(storage_path: str) -> None:
    """Remove the staged local copy (mirrors ``UploadedFileStore._delete_local``)."""
    local_path = Path(storage_path)
    if local_path.exists() and local_path.is_file():
        local_path.unlink()


def _claim_orphan(db: Session, candidate: _OrphanUploadCandidate) -> datetime | None:
    """CAS-claim one still-unbound row; return its persisted generation token.

    Identical shape to the ``compensate_registered_uploads_sync`` claim: the
    exact expected status makes concurrent claimers (an overlapping sweep, a
    request compensation) mutually exclusive, and ``task_id IS NULL`` makes
    the claim lose to any bind that committed first. Binders in turn use
    conditional updates excluding ``compensating`` rows, so whichever side
    commits first wins outright. Committed immediately so the claim is
    visible before any storage I/O starts.
    """
    claimed_at = datetime.now(timezone.utc)
    claimed = (
        db.query(UploadedFile)
        .filter(
            UploadedFile.id == candidate.row_id,
            UploadedFile.user_id == candidate.user_id,
            UploadedFile.file_id == candidate.file_id,
            UploadedFile.storage_key == candidate.storage_key,
            UploadedFile.storage_status == "available",
            UploadedFile.task_id.is_(None),
        )
        .update(
            {
                UploadedFile.storage_status: "compensating",
                UploadedFile.updated_at: claimed_at,
            },
            synchronize_session=False,
        )
    )
    if claimed != 1:
        db.rollback()
        return None
    # Read the token back as persisted: the database may round the datetime,
    # and the settlement fences on exact equality.
    token = cast(
        "datetime | None",
        db.query(UploadedFile.updated_at)
        .filter(
            UploadedFile.id == candidate.row_id,
            UploadedFile.storage_status == "compensating",
        )
        .scalar(),
    )
    if token is None:
        db.rollback()
        return None
    db.commit()
    return token


def _reap_orphan(db: Session, candidate: _OrphanUploadCandidate) -> bool:
    """Reap one candidate; True only when its metadata row was deleted."""
    # Local bytes first, while the row is still available: a consumer that
    # binds after this re-materializes from the durable object, and the
    # generic stale-compensation recovery (which owns every post-claim crash
    # window but knows no storage_path) then never has a local file to leak.
    _delete_local_file(candidate.storage_path)

    token = _claim_orphan(db, candidate)
    if token is None:
        return False  # bound, or claimed by another owner — spared

    presence = delete_uploaded_file_compensation_object(
        user_id=candidate.user_id,
        storage_key=candidate.storage_key,
    )
    if presence != "absent":
        # Durable state unresolved: leave the claimed row to the stale-
        # compensation recovery loop, which retries the delete under a
        # takeover token. Deleting metadata now could strand a live object.
        logger.warning(
            "Deferred orphan upload GC for file %s (durable presence: %s)",
            candidate.file_id,
            presence,
        )
        return False

    settlement = settle_uploaded_file_compensation_no_commit(
        db,
        row_id=candidate.row_id,
        user_id=candidate.user_id,
        file_id=candidate.file_id,
        task_id=None,
        storage_key=candidate.storage_key,
        expected_updated_at=token,
        presence=presence,
    )
    if settlement is None:
        db.rollback()
        return False
    db.commit()
    delete_registered_preview_caches(candidate.file_id)
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
    ``task_id``, and (c) are older than the TTL, in oldest-first keyset pages
    of ``batch_size`` (at most ``max_batches`` per sweep). The cursor always
    advances past spared or failing rows, so a bad oldest page can never
    starve newer orphans; whatever one sweep defers is retried by the next.
    Per-row failures are logged and skipped so one bad row does not abort the
    sweep. Returns the number of metadata rows deleted.
    """
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - timedelta(seconds=older_than_seconds)
    deleted = 0
    cursor: _OrphanCursor | None = None
    for _ in range(max_batches):
        candidates = _orphan_candidates(
            db, cutoff=cutoff, limit=batch_size, after=cursor
        )
        if not candidates:
            break
        cursor = candidates[-1].cursor
        for candidate in candidates:
            try:
                if _reap_orphan(db, candidate):
                    deleted += 1
            except Exception:
                db.rollback()
                logger.warning(
                    "Failed to GC orphaned task-less upload id=%s",
                    candidate.row_id,
                    exc_info=True,
                )
        if len(candidates) < batch_size:
            break
    return deleted
