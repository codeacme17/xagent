from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import IngestionResult
from xagent.core.tools.core.RAG_tools.kb import KBApiOperationResult
from xagent.web.api import kb as kb_module
from xagent.web.models.user import User


class _Facade:
    def __init__(self) -> None:
        self.rollback_complete_inputs: list[tuple[KBApiOperationResult[Any], bool]] = []
        self.single_cleanup_inputs: list[
            tuple[KBApiOperationResult[Any], int | None]
        ] = []
        self.batch_cleanup_inputs: list[
            tuple[list[KBApiOperationResult[Any]], int | None]
        ] = []

    def with_rollback_complete(
        self,
        api_result: KBApiOperationResult[Any],
        rollback_complete: bool,
    ) -> KBApiOperationResult[Any]:
        self.rollback_complete_inputs.append((api_result, rollback_complete))
        return KBApiOperationResult(
            result=api_result.result,
            operation_outcome=api_result.operation_outcome,
            rollback_complete=rollback_complete,
        )

    def failed_ingest_cleanup_decision(
        self,
        api_result: KBApiOperationResult[Any],
        *,
        successful_documents: int | None = None,
    ) -> Any:
        self.single_cleanup_inputs.append((api_result, successful_documents))
        return type(
            "Decision",
            (),
            {
                "successful_documents": 3,
                "side_effects_may_remain": api_result.rollback_complete is False,
            },
        )()

    def failed_batch_ingest_cleanup_decision(
        self,
        api_results: list[KBApiOperationResult[Any]],
        *,
        successful_documents: int | None = None,
    ) -> Any:
        self.batch_cleanup_inputs.append((api_results, successful_documents))
        return type(
            "Decision",
            (),
            {
                "successful_documents": 5,
                "side_effects_may_remain": True,
            },
        )()


@pytest.mark.asyncio
async def test_api_failed_ingest_config_cleanup_uses_api_outcome_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = _Facade()
    restore_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(kb_module, "_get_api_compatibility_facade", lambda: facade)

    async def fake_restore(**kwargs: Any) -> None:
        restore_calls.append(kwargs)

    monkeypatch.setattr(
        kb_module,
        "_restore_or_cleanup_collection_config_after_failed_ingest",
        fake_restore,
    )
    api_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="failed")
    )
    user = User()
    user.id = 7

    updated = (
        await kb_module._restore_or_cleanup_collection_config_after_failed_api_ingest(
            api_result=api_result,
            snapshot="snapshot",
            collection_existed_before=True,
            collection_name="demo",
            user=user,
            context="ingest",
            successful_documents=1,
            rollback_complete=False,
        )
    )

    assert updated.rollback_complete is False
    assert facade.rollback_complete_inputs == [(api_result, False)]
    assert facade.single_cleanup_inputs == [(updated, 1)]
    assert restore_calls == [
        {
            "snapshot": "snapshot",
            "collection_existed_before": True,
            "collection_name": "demo",
            "user": user,
            "context": "ingest",
            "successful_documents": 3,
            "side_effects_may_remain": True,
        }
    ]


@pytest.mark.asyncio
async def test_api_failed_batch_ingest_config_cleanup_uses_api_outcome_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = _Facade()
    restore_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(kb_module, "_get_api_compatibility_facade", lambda: facade)

    async def fake_restore(**kwargs: Any) -> None:
        restore_calls.append(kwargs)

    monkeypatch.setattr(
        kb_module,
        "_restore_or_cleanup_collection_config_after_failed_ingest",
        fake_restore,
    )
    api_results = [
        KBApiOperationResult(result=IngestionResult(status="success", message="ok")),
        KBApiOperationResult(result=IngestionResult(status="error", message="failed")),
    ]
    user = User()
    user.id = 9

    await kb_module._restore_or_cleanup_collection_config_after_failed_batch_api_ingest(
        api_results=api_results,
        snapshot="snapshot",
        collection_existed_before=False,
        collection_name="demo",
        user=user,
        context="ingest_cloud",
        successful_documents=1,
    )

    assert facade.batch_cleanup_inputs == [(api_results, 1)]
    assert restore_calls == [
        {
            "snapshot": "snapshot",
            "collection_existed_before": False,
            "collection_name": "demo",
            "user": user,
            "context": "ingest_cloud",
            "successful_documents": 5,
            "side_effects_may_remain": True,
        }
    ]


def test_background_failed_ingest_config_cleanup_reuses_api_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.jobs import kb_tasks

    api_result = KBApiOperationResult(
        result=IngestionResult(status="error", message="failed")
    )
    user = User()
    user.id = 11
    db = object()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(kb_tasks, "_get_job_user", lambda *args, **kwargs: user)

    async def fake_api_helper(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        kb_module,
        "_restore_or_cleanup_collection_config_after_failed_api_ingest",
        fake_api_helper,
    )

    kb_tasks._restore_or_cleanup_failed_job_collection_config_after_api_ingest(
        db,  # type: ignore[arg-type]
        {
            "collection": "job-kb",
            "collection_existed_before": False,
        },
        api_result=api_result,
        snapshot="snapshot",
        context="background document ingest",
        successful_documents=2,
        rollback_complete=True,
    )

    assert calls == [
        {
            "api_result": api_result,
            "snapshot": "snapshot",
            "collection_existed_before": False,
            "collection_name": "job-kb",
            "user": user,
            "context": "background document ingest",
            "successful_documents": 2,
            "rollback_complete": True,
        }
    ]


def test_background_web_cleanup_keeps_early_exception_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.jobs import kb_tasks

    fallback = MagicMock()
    api_helper = MagicMock()
    db = object()
    payload = {"collection": "job-kb"}
    monkeypatch.setattr(
        kb_tasks,
        "_restore_or_cleanup_failed_job_collection_config",
        fallback,
    )
    monkeypatch.setattr(
        kb_tasks,
        "_restore_or_cleanup_failed_job_collection_config_after_api_ingest",
        api_helper,
    )

    kb_tasks._cleanup_failed_web_collection_metadata_if_new(
        db,  # type: ignore[arg-type]
        payload,
        snapshot="snapshot",
    )

    fallback.assert_called_once_with(
        db,
        payload,
        snapshot="snapshot",
        context="background web ingest",
        successful_documents=0,
    )
    api_helper.assert_not_called()
