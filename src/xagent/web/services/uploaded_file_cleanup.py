"""Complete a committed upload claim without losing unfinished obligations."""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Any, Callable, Literal

from sqlalchemy.orm import Session

from ...core.tools.core.RAG_tools.storage.file_reference import file_cleanup_lock
from ..models.uploaded_file import UploadedFile
from .uploaded_file_cleanup_discovery import (
    CleanupBudgetExhausted,
    CleanupWorkBudget,
    advance_discovery,
    close_discovery_streams,
    dispose_resource_page,
    prepare_discovery,
    validate_local_evidence,
)
from .uploaded_file_cleanup_resources import (
    CleanupResourceUncertain,
    build_cleanup_manifest,
    validate_configuration,
)

logger = logging.getLogger(__name__)
CleanupOutcome = Literal["deleted", "stale", "exists", "unknown", "pending", "yielded"]
CLEANUP_PHASES = ("durable", "local", "previews")


def cleanup_complete(manifest: Any) -> bool:
    return bool(
        isinstance(manifest, dict)
        and manifest.get("version") in {1, 2}
        and all(phase in manifest.get("done", []) for phase in CLEANUP_PHASES)
        and (
            manifest["version"] == 1
            or all(
                manifest.get("discovery", {}).get(phase, {}).get("complete")
                for phase in ("local", "previews", "preview_check")
            )
        )
    )


def run_uploaded_file_cleanup(
    *,
    session_factory: Callable[[], Session],
    row_id: int,
    user_id: int,
    file_id: str,
    task_id: int | None,
    storage_key: str,
    expected_updated_at: datetime | None,
    compensation_delete: Callable[..., str],
    take_over: bool = False,
) -> CleanupOutcome:
    """Fence execution, takeover, phase receipts and settlement by exact token.

    The execution lock prevents a takeover from settling while an older worker
    still has destructive I/O in flight. It is distinct from the reference lock;
    no SQL connection or reference lock spans any storage operation.
    """
    from .uploaded_file_store import (
        settle_uploaded_file_compensation_no_commit,
        snapshot_uploaded_file_version,
        take_over_uploaded_file_compensation_no_commit,
    )

    with file_cleanup_lock(file_id):
        token = expected_updated_at
        with session_factory() as db:
            if take_over:
                token = take_over_uploaded_file_compensation_no_commit(
                    db,
                    row_id=row_id,
                    user_id=user_id,
                    file_id=file_id,
                    task_id=task_id,
                    storage_key=storage_key,
                    expected_updated_at=token,
                )
                if token is None:
                    return "stale"
                db.commit()
            record = _claim_query(
                db, row_id, user_id, file_id, task_id, storage_key, token
            ).first()
            if record is None or token is None:
                return "stale"
            manifest = copy.deepcopy(record.cleanup_manifest)
            snapshot = snapshot_uploaded_file_version(record)
        # Upgrade-era compensations still retain their source metadata. Adopt
        # only what that metadata and current resource evidence can prove.
        if manifest is None:
            manifest = build_cleanup_manifest(snapshot)
            if not _save_manifest(
                session_factory,
                row_id,
                user_id,
                file_id,
                task_id,
                storage_key,
                token,
                manifest,
            ):
                return "stale"
        try:
            if (
                manifest.get("version") not in {1, 2}
                or manifest.get("file_id") != file_id
                or manifest.get("user_id") != user_id
                or manifest.get("storage_key") != storage_key
            ):
                raise CleanupResourceUncertain(
                    "Cleanup manifest identity does not match its claim"
                )
            validate_configuration(manifest)
            prepare_discovery(manifest, snapshot.filename)
            budget = CleanupWorkBudget()

            def save_progress() -> bool:
                saved = _save_manifest(
                    session_factory,
                    row_id,
                    user_id,
                    file_id,
                    task_id,
                    storage_key,
                    token,
                    manifest,
                )
                if not saved:
                    close_discovery_streams(manifest["generation"])
                return saved

            validate_local_evidence(manifest)
            if not manifest["discovery"]["local"]["complete"]:
                advance_discovery(manifest, "local", budget)
                if not save_progress():
                    return "stale"
                if not manifest["discovery"]["local"]["complete"]:
                    return "yielded"
            validate_local_evidence(manifest)
            for phase in CLEANUP_PHASES:
                if phase in manifest["done"]:
                    continue
                if phase == "durable":
                    presence = compensation_delete(
                        user_id=user_id, storage_key=storage_key
                    )
                    if presence != "absent":
                        return "exists" if presence == "exists" else "unknown"
                else:
                    if (
                        phase == "previews"
                        and not manifest["discovery"][phase]["complete"]
                    ):
                        advance_discovery(manifest, phase, budget)
                        if not save_progress():
                            return "stale"
                        if not manifest["discovery"][phase]["complete"]:
                            return "yielded"
                    dispose_resource_page(manifest, phase, budget)
                    if phase == "previews":
                        advance_discovery(manifest, "preview_check", budget)
                        if not save_progress():
                            return "stale"
                        if not manifest["discovery"]["preview_check"]["complete"]:
                            return "yielded"
                        if manifest["preview_check"]:
                            raise CleanupResourceUncertain(
                                "Owned previews remain after disposal"
                            )
                manifest["done"].append(phase)
                if not _save_manifest(
                    session_factory,
                    row_id,
                    user_id,
                    file_id,
                    task_id,
                    storage_key,
                    token,
                    manifest,
                ):
                    return "stale"
        except CleanupBudgetExhausted:
            if not save_progress():
                return "stale"
            return "yielded"
        except (CleanupResourceUncertain, OSError):
            close_discovery_streams(manifest["generation"])
            logger.warning(
                "Upload cleanup needs reconciliation for %s", file_id, exc_info=True
            )
            return "pending"
        except BaseException:
            close_discovery_streams(manifest["generation"])
            raise
        with session_factory() as db:
            result = settle_uploaded_file_compensation_no_commit(
                db,
                row_id=row_id,
                user_id=user_id,
                file_id=file_id,
                task_id=task_id,
                storage_key=storage_key,
                expected_updated_at=token,
                presence="absent",
            )
            if result is None:
                return "stale"
            db.commit()
            return "deleted"


def _claim_query(
    db: Session,
    row_id: int,
    user_id: int,
    file_id: str,
    task_id: int | None,
    storage_key: str,
    token: datetime | None,
) -> Any:
    return db.query(UploadedFile).filter(
        UploadedFile.id == row_id,
        UploadedFile.user_id == user_id,
        UploadedFile.file_id == file_id,
        UploadedFile.task_id == task_id,
        UploadedFile.storage_key == storage_key,
        UploadedFile.storage_status == "compensating",
        UploadedFile.updated_at == token,
    )


def _save_manifest(
    sessions: Callable[[], Session],
    row_id: int,
    user_id: int,
    file_id: str,
    task_id: int | None,
    storage_key: str,
    token: datetime,
    manifest: dict[str, Any],
) -> bool:
    with sessions() as db:
        changed = _claim_query(
            db, row_id, user_id, file_id, task_id, storage_key, token
        ).update(
            {UploadedFile.cleanup_manifest: manifest, UploadedFile.updated_at: token},
            synchronize_session=False,
        )
        if changed != 1:
            return False
        db.commit()
        return True
