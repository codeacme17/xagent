"""Orphan GC of task-less public uploads (#973, PR3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.orphan_upload_gc import (
    TASKLESS_SHARE_UPLOAD_SOURCE,
    cleanup_orphaned_taskless_uploads,
)

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
