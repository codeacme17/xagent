"""Orphan GC of task-less public uploads (#973, PR3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

import xagent.web.services.orphan_upload_gc as orphan_upload_gc
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.file_turn import bind_turn_files_no_commit
from xagent.web.services.orphan_upload_gc import (
    TASKLESS_SHARE_UPLOAD_SOURCE,
    _claim_orphan,
    cleanup_orphaned_taskless_uploads,
)
from xagent.web.services.uploaded_file_store import UploadedFileStore

DAY = 24 * 60 * 60


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'orphan_gc.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture()
def owner(db_session) -> User:
    user = User(username="gc-owner", password_hash="h", is_admin=False)
    db_session.add(user)
    db_session.commit()
    return user


def _mk_upload(
    db_session,
    owner: User,
    tmp_path: Path,
    *,
    name: str,
    marker: str | None,
    task_id: int | None,
    age_days: float,
) -> tuple[UploadedFile, Path]:
    path = tmp_path / name
    path.write_bytes(b"payload")
    now = datetime.now(timezone.utc)
    row = UploadedFile(
        file_id=str(uuid4()),
        user_id=int(owner.id),
        task_id=task_id,
        filename=name,
        storage_path=str(path),
        storage_status="legacy",
        file_size=path.stat().st_size,
        upload_source=marker,
        created_at=now - timedelta(days=age_days),
    )
    db_session.add(row)
    db_session.commit()
    db_session.refresh(row)
    return row, path


def _make_task(db_session, owner: User) -> int:
    task = Task(
        user_id=int(owner.id),
        title="t",
        description="t",
        status=TaskStatus.PENDING,
    )
    db_session.add(task)
    db_session.commit()
    return int(task.id)


def test_reaps_aged_marked_unbound_upload(db_session, owner, tmp_path) -> None:
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="orphan.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    row_id = int(row.id)

    deleted = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert deleted == 1
    assert db_session.query(UploadedFile).filter_by(id=row_id).first() is None
    assert not path.exists()  # on-disk file removed too


def test_spares_marked_but_recent_upload(db_session, owner, tmp_path) -> None:
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="fresh.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=0,
    )
    row_id = int(row.id)

    deleted = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert deleted == 0
    assert db_session.query(UploadedFile).filter_by(id=row_id).first() is not None
    assert path.exists()


def test_spares_unmarked_unbound_upload(db_session, owner, tmp_path) -> None:
    """A logged-in user's aged, un-sent draft (no marker) must never be reaped
    by the task_id-IS-NULL sweep."""
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="draft.txt",
        marker=None,
        task_id=None,
        age_days=10,
    )
    row_id = int(row.id)

    deleted = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert deleted == 0
    assert db_session.query(UploadedFile).filter_by(id=row_id).first() is not None
    assert path.exists()


def test_spares_marked_but_bound_upload(db_session, owner, tmp_path) -> None:
    """Once a marked upload is bound to a task (run started), it is no longer an
    orphan and must be kept."""
    task_id = _make_task(db_session, owner)
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="bound.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=task_id,
        age_days=5,
    )
    row_id = int(row.id)

    deleted = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert deleted == 0
    assert db_session.query(UploadedFile).filter_by(id=row_id).first() is not None
    assert path.exists()


def test_claim_fails_on_bound_row_and_marks_unbound_row(
    db_session, owner, tmp_path
) -> None:
    """The claim is a conditional UPDATE: False once ``task_id`` is set, and a
    successful claim leaves the row ``compensating`` (unbindable)."""
    task_id = _make_task(db_session, owner)
    bound, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="claim-bound.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=task_id,
        age_days=5,
    )
    unbound, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="claim-unbound.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )

    assert _claim_orphan(db_session, int(bound.id)) is False
    assert _claim_orphan(db_session, int(unbound.id)) is True
    db_session.refresh(unbound)
    assert unbound.storage_status == "compensating"


def test_claimed_row_is_excluded_from_run_start_binding(
    db_session, owner, tmp_path
) -> None:
    """The other half of the interlock: once GC has claimed a row, the
    run-start conditional bind must skip it rather than resurrect it."""
    task_id = _make_task(db_session, owner)
    row, _ = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="claimed.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    assert _claim_orphan(db_session, int(row.id)) is True

    missing = bind_turn_files_no_commit(
        file_ids=[str(row.file_id)],
        task_id=task_id,
        owner_user_id=int(owner.id),
        db=db_session,
    )

    assert missing == [str(row.file_id)]  # bind refused the claimed row
    db_session.rollback()
    db_session.refresh(row)
    assert row.task_id is None


def test_row_bound_between_fetch_and_claim_is_spared(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    """A run-start bind that commits after the batch query but before the
    claim must win: the claim's ``task_id IS NULL`` predicate no longer
    matches, so the row (and its bytes) survive."""
    task_id = _make_task(db_session, owner)
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="raced.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    row_id = int(row.id)

    real_claim = orphan_upload_gc._claim_orphan

    def bind_then_claim(db, claimed_row_id: int) -> bool:
        # Emulate the concurrent run-start committing its bind first.
        db.query(UploadedFile).filter(UploadedFile.id == claimed_row_id).update(
            {UploadedFile.task_id: task_id}, synchronize_session=False
        )
        db.commit()
        return real_claim(db, claimed_row_id)

    monkeypatch.setattr(orphan_upload_gc, "_claim_orphan", bind_then_claim)

    deleted = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)

    assert deleted == 0
    survivor = db_session.query(UploadedFile).filter_by(id=row_id).one()
    assert survivor.task_id == task_id
    assert path.exists()


def test_sweep_is_batched_and_bounded(db_session, owner, tmp_path) -> None:
    """The sweep never materializes more than ``batch_size`` rows at once and
    stops after ``max_batches``; the remainder is left for the next tick."""
    for i in range(5):
        _mk_upload(
            db_session,
            owner,
            tmp_path,
            name=f"backlog-{i}.txt",
            marker=TASKLESS_SHARE_UPLOAD_SOURCE,
            task_id=None,
            age_days=5,
        )

    deleted = cleanup_orphaned_taskless_uploads(
        db_session, older_than_seconds=2 * DAY, batch_size=2, max_batches=2
    )
    assert deleted == 4
    assert db_session.query(UploadedFile).count() == 1

    # The next sweep drains the remainder.
    deleted = cleanup_orphaned_taskless_uploads(
        db_session, older_than_seconds=2 * DAY, batch_size=2, max_batches=2
    )
    assert deleted == 1
    assert db_session.query(UploadedFile).count() == 0


def test_failed_delete_stays_claimed_and_is_retried(
    db_session, owner, tmp_path, monkeypatch
) -> None:
    """A row whose storage deletion fails is skipped (sweep continues), stays
    ``compensating`` (unbindable), and is reaped by a later sweep."""
    row, path = _mk_upload(
        db_session,
        owner,
        tmp_path,
        name="flaky.txt",
        marker=TASKLESS_SHARE_UPLOAD_SOURCE,
        task_id=None,
        age_days=5,
    )
    row_id = int(row.id)

    def boom(self, file_record, **kwargs):
        raise RuntimeError("storage backend down")

    monkeypatch.setattr(UploadedFileStore, "delete", boom)
    deleted = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)
    assert deleted == 0
    survivor = db_session.query(UploadedFile).filter_by(id=row_id).one()
    assert survivor.storage_status == "compensating"

    monkeypatch.undo()
    deleted = cleanup_orphaned_taskless_uploads(db_session, older_than_seconds=2 * DAY)
    assert deleted == 1
    assert db_session.query(UploadedFile).filter_by(id=row_id).first() is None
    assert not path.exists()
