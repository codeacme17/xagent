"""Resource discovery yields without losing the production cleanup claim."""

# ruff: noqa: F401, F811

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest
import sqlalchemy as sa
from filelock import FileLock

from tests.web.services.test_complete_upload_cleanup import (
    PAYLOAD,
    claim,
    collect,
    completed,
    lifecycle,
    recover,
)
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.file_storage.storage import materialized_key_directory
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services import uploaded_file_cleanup_discovery as discovery


def test_large_source_directory_yields_and_recovery_finishes(lifecycle, monkeypatch):
    sessions, source, _, _, key, _ = lifecycle
    for index in range(600):
        (source.parent / f"unrelated-{index}").touch()
    scanned = 0
    original = os.scandir

    class CountedScan:
        def __init__(self, entries):
            self.entries = entries

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal scanned
            entry = next(self.entries)
            scanned += 1
            return entry

        def close(self):
            self.entries.close()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(os, "scandir", lambda *a, **kw: CountedScan(original(*a, **kw)))
    assert collect(lifecycle).deleted == 0
    assert scanned <= 256
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.storage_status == "compensating"
        assert row.cleanup_manifest["done"] == []
    assert source.exists()
    assert get_unscoped_file_storage().exists(key)
    for _ in range(10):
        scanned = 0
        result = recover(lifecycle)
        assert scanned <= 256
        if result.deleted:
            break
    else:
        raise AssertionError("Bounded recovery did not finish")
    completed(lifecycle)
    assert len(list(source.parent.glob("unrelated-*"))) == 600


@pytest.fixture(autouse=True)
def process_streams():
    discovery.close_discovery_streams()
    yield
    discovery.close_discovery_streams()


def finish(lifecycle, limit=40):
    for _ in range(limit):
        result = recover(lifecycle)
        assert result.failed == 0
        if result.deleted:
            completed(lifecycle)
            return
    raise AssertionError("Recovery did not converge")


def test_claim_does_not_enumerate_directories(lifecycle, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Discovery ran while reference locks were held")

    monkeypatch.setattr(os, "scandir", forbidden)
    candidate, token = claim(lifecycle)
    with lifecycle[0]() as db:
        row = db.get(UploadedFile, candidate.row_id)
        assert row.cleanup_manifest["version"] == 2
        assert row.updated_at == token
        assert not row.cleanup_manifest["discovery"]["local"]["complete"]


def test_many_materialization_generations_and_partial_copies_are_discovered(lifecycle):
    _, source, materialized, _, key, _ = lifecycle
    directory = materialized_key_directory(materialized.parents[2], key)
    copies = []
    for index in range(300):
        cached = (
            directory / hashlib.sha256(str(index).encode()).hexdigest() / source.name
        )
        cached.parent.mkdir(exist_ok=True)
        cached.write_bytes(PAYLOAD)
        temporary = cached.with_name(f".{cached.name}.interrupted.tmp")
        temporary.write_bytes(b"partial")
        copies.extend((cached, temporary))
    assert collect(lifecycle).deleted == 0
    finish(lifecycle)
    assert not any(path.exists() for path in copies)


def test_large_preview_namespaces_are_discovered_and_verified_before_receipt(lifecycle):
    sessions, _, _, previews, _, file_id = lifecycle
    for index in range(400):
        (previews[0].parent / f"unrelated-{index}").touch()
    owned = []
    for index in range(280):
        path = previews[0].parent / f"{file_id}.preview.pdf.{index}.tmp"
        path.write_bytes(b"partial")
        owned.append(path)
    assert collect(lifecycle).deleted == 0
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert "previews" not in row.cleanup_manifest["done"]
    finish(lifecycle)
    assert not any(path.exists() for path in owned)
    assert len(list(previews[0].parent.glob("unrelated-*"))) == 400


def test_converter_tree_is_drained_without_unbounded_rmtree(lifecycle, monkeypatch):
    sessions, source, _, previews, _, file_id = lifecycle
    converter = previews[0].parent / f".{file_id}.preview-abandoned"
    converter.mkdir()
    for index in range(400):
        (converter / f"partial-{index}.pdf").touch()
    nested = converter / "nested"
    nested.mkdir()
    (nested / "external-link").symlink_to(source)
    unrelated = previews[0].parent / "unrelated-tree"
    unrelated.mkdir()
    (unrelated / "keep").touch()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unbounded recursive disposal")

    monkeypatch.setattr(shutil, "rmtree", forbidden)
    assert collect(lifecycle).deleted == 0
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert "previews" not in row.cleanup_manifest["done"]
        resource = next(
            r for r in row.cleanup_manifest["previews"] if r["path"] == str(converter)
        )
        assert (converter.parent / resource["quarantine"]).is_dir()
    finish(lifecycle)
    assert not converter.exists()
    assert (unrelated / "keep").exists()


def test_failed_progress_commit_cannot_skip_owned_resources(lifecycle):
    sessions, source, _, _, _, file_id = lifecycle
    prefix = hashlib.sha256(file_id.encode()).hexdigest()[:24]
    owned = []
    for index in range(350):
        path = source.parent / f".{prefix}.{source.name}.{index}.tmp"
        path.touch()
        owned.append(path)
    engine = sessions.kw["bind"]
    failed = False

    def fail_progress(_connection, _cursor, statement, *_args):
        nonlocal failed
        changed = statement.partition(" WHERE ")[0]
        if (
            changed.startswith("UPDATE uploaded_files")
            and "cleanup_manifest=" in changed
            and "storage_status=" not in changed
            and not failed
        ):
            failed = True
            raise sa.exc.OperationalError(
                statement, (), RuntimeError("progress commit interrupted")
            )

    sa.event.listen(engine, "before_cursor_execute", fail_progress)
    try:
        assert collect(lifecycle).deleted == 0
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail_progress)
    assert failed
    finish(lifecycle)
    assert not any(path.exists() for path in owned)


def test_directory_mutation_restarts_discovery_and_preserves_new_obligation(lifecycle):
    _, source, _, _, _, file_id = lifecycle
    for index in range(600):
        (source.parent / f"unrelated-{index}").touch()
    assert collect(lifecycle).deleted == 0
    prefix = hashlib.sha256(file_id.encode()).hexdigest()[:24]
    late = source.parent / f".{prefix}.{source.name}.late.tmp"
    late.touch()
    assert recover(lifecycle).deleted == 0
    finish(lifecycle)
    assert not late.exists()


def test_fresh_process_resumes_retained_partial_discovery(lifecycle):
    sessions, source, _, _, _, file_id = lifecycle
    prefix = hashlib.sha256(file_id.encode()).hexdigest()[:24]
    owned = []
    for index in range(600):
        path = source.parent / f".{prefix}.{source.name}.{index}.tmp"
        path.touch()
        owned.append(path)
    assert collect(lifecycle).deleted == 0
    engine = sessions.kw["bind"]
    if engine.dialect.name == "sqlite":
        url = str(engine.url)
    else:
        with engine.connect() as conn:
            name = conn.exec_driver_sql("SELECT current_database()").scalar_one()
        url = (
            sa.engine.make_url(os.environ["XAGENT_TEST_POSTGRES_URL"])
            .set(database=name)
            .render_as_string(hide_password=False)
        )
    environment = dict(os.environ, DATABASE_URL=url)
    script = """
from datetime import UTC, datetime, timedelta
from xagent.web.models.database import configure_db, get_session_local
from xagent.web.services.uploaded_file_recovery import recover_stale_uploaded_file_compensations_batch_isolated
configure_db()
for _ in range(40):
    result = recover_stale_uploaded_file_compensations_batch_isolated(
        session_factory=get_session_local(), cutoff=datetime.now(UTC) + timedelta(days=1), batch_size=10)
    assert result.failed == 0
    if result.deleted:
        break
else:
    raise AssertionError('Restart failed to converge')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    completed(lifecycle)
    assert not any(path.exists() for path in owned)


@pytest.mark.parametrize("restart", [False, True])
def test_directory_replacement_retains_original_ownership_evidence(lifecycle, restart):
    sessions, source, _, _, key, _ = lifecycle
    for index in range(600):
        (source.parent / f"unrelated-{index}").touch()
    assert collect(lifecycle).deleted == 0
    original_directory = source.parent.with_name("original-owner-directory")
    source.parent.rename(original_directory)
    source.parent.mkdir()
    source.write_bytes(b"replacement")
    if restart:
        discovery.close_discovery_streams()
    result = recover(lifecycle)
    assert result.deleted == 0 and result.failed == 1
    assert source.read_bytes() == b"replacement"
    assert (original_directory / source.name).read_bytes() == PAYLOAD
    assert get_unscoped_file_storage().exists(key)
    with sessions() as db:
        assert db.query(UploadedFile).one().cleanup_manifest["done"] == []


def test_overlapping_workers_resume_only_after_inflight_discovery_commits(
    lifecycle, monkeypatch
):
    from xagent.web.services.uploaded_file_cleanup import run_uploaded_file_cleanup
    from xagent.web.services.uploaded_file_store import (
        delete_uploaded_file_compensation_object,
    )

    sessions, source, _, _, key, file_id = lifecycle
    for index in range(600):
        (source.parent / f"unrelated-{index}").touch()
    candidate, token = claim(lifecycle)
    entered, proceed, attempted = Event(), Event(), Event()
    original = os.scandir
    engine = sessions.kw["bind"]
    checked_out = 0

    def checkout(*_args):
        nonlocal checked_out
        checked_out += 1

    def returned(*_args):
        nonlocal checked_out
        checked_out -= 1

    sa.event.listen(engine, "checkout", checkout)
    sa.event.listen(engine, "checkin", returned)  # codespell:ignore checkin

    def blocked(*args, **kwargs):
        if not entered.is_set():
            assert checked_out == 0
            lock = (
                Path(os.environ["LANCEDB_DIR"])
                / ".file-references"
                / (hashlib.sha256(file_id.encode()).hexdigest() + ".lock")
            )
            with FileLock(lock, timeout=0):
                entered.set()
                assert proceed.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(os, "scandir", blocked)

    def first():
        return run_uploaded_file_cleanup(
            session_factory=sessions,
            row_id=candidate.row_id,
            user_id=candidate.user_id,
            file_id=file_id,
            task_id=None,
            storage_key=key,
            expected_updated_at=token,
            compensation_delete=delete_uploaded_file_compensation_object,
        )

    def second():
        attempted.set()
        return recover(lifecycle)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_worker = pool.submit(first)
        try:
            if not entered.wait(10):
                first_worker.result(timeout=10)
                raise AssertionError("First worker did not enter discovery")
            second_worker = pool.submit(second)
            assert attempted.wait(10)
            assert not second_worker.done()
        finally:
            proceed.set()
        assert first_worker.result(timeout=10) == "yielded"
        assert second_worker.result(timeout=10).deferred_budget == 1
    assert first() == "stale"
    finish(lifecycle)
    sa.event.remove(engine, "checkout", checkout)
    sa.event.remove(engine, "checkin", returned)  # codespell:ignore checkin


@pytest.mark.asyncio
async def test_cancelled_collector_drains_one_bounded_activation_and_retains_claim(
    lifecycle, monkeypatch
):
    from xagent.web.services.db_runtime import run_db_io_cancellation_safe
    from xagent.web.services.orphan_upload_gc import (
        sweep_orphaned_taskless_uploads_isolated,
    )

    sessions, source, _, _, _, _ = lifecycle
    for index in range(600):
        (source.parent / f"unrelated-{index}").touch()
    entered, proceed = Event(), Event()
    original = os.scandir

    def blocked(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert proceed.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(os, "scandir", blocked)
    task = asyncio.create_task(
        run_db_io_cancellation_safe(
            lambda: sweep_orphaned_taskless_uploads_isolated(
                session_factory=sessions, older_than_seconds=1
            )
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    with sessions() as db:
        row = db.query(UploadedFile).one()
        assert row.storage_status == "compensating"
        assert row.cleanup_manifest["done"] == []
    finish(lifecycle)


@pytest.mark.parametrize(
    "done", [[], ["durable"], ["durable", "local"], ["durable", "local", "previews"]]
)
def test_version_one_manifest_adoption_preserves_existing_phase_receipts(
    lifecycle, done
):
    sessions, source, materialized, previews, key, _ = lifecycle
    claim(lifecycle)
    if "durable" in done:
        get_unscoped_file_storage().delete(key)
    if "local" in done:
        source.unlink()
        materialized.unlink()
    if "previews" in done:
        for preview in previews:
            preview.unlink()
    with sessions.begin() as db:
        row = db.query(UploadedFile).one()
        old = dict(row.cleanup_manifest)
        old["version"] = 1
        old["done"] = done
        del old["discovery"]
        del old["preview_check"]
        row.cleanup_manifest = old
        row.updated_at = datetime.now(UTC)
    assert recover(lifecycle).deleted == 1
    completed(lifecycle)


def test_stream_capacity_does_not_lose_waiting_claims(lifecycle):
    sessions, source, _, _, _, _ = lifecycle
    for index in range(600):
        (source.parent / f"unrelated-{index}").touch()
    extra_sources = []
    storage = get_unscoped_file_storage()
    with sessions.begin() as db:
        for index in range(8):
            file_id = str(uuid4())
            extra = source.with_name(f"extra-{index}.txt")
            extra.write_bytes(PAYLOAD)
            key = f"users/1/uploads/{file_id}/{extra.name}"
            durable = storage.put_file(extra, key)
            db.add(
                UploadedFile(
                    file_id=file_id,
                    user_id=1,
                    filename=extra.name,
                    storage_path=str(extra),
                    storage_key=key,
                    storage_backend=durable.backend,
                    storage_uri=durable.uri,
                    storage_status="available",
                    checksum=hashlib.sha256(PAYLOAD).hexdigest(),
                    upload_source="taskless_share_upload",
                    created_at=datetime.now(UTC) - timedelta(days=10),
                )
            )
            extra_sources.append(extra)
    assert collect(lifecycle).scanned == 9
    with sessions() as db:
        assert (
            db.query(UploadedFile)
            .filter(UploadedFile.storage_status == "compensating")
            .count()
            == 9
        )
    for _ in range(50):
        recover(lifecycle)
        with sessions() as db:
            if db.query(UploadedFile).count() == 0:
                break
    else:
        raise AssertionError("Waiting claims starved behind active streams")
    completed(lifecycle)
    assert not any(path.exists() for path in extra_sources)
