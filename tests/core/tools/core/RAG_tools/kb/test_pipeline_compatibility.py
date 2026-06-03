"""Tests for KB pipeline compatibility facade."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    CrawlResult,
    IngestionResult,
    IngestionStepResult,
    WebCrawlConfig,
    WebIngestionResult,
)
from xagent.core.tools.core.RAG_tools.kb import (
    KBOperationCompatibilityFacade,
    KBPipelineCompatibilityFacade,
    RollbackStatus,
    SideEffectPlane,
)


class _FakeMetadataStore:
    def __init__(self, collection: Optional[CollectionInfo]) -> None:
        self.collection = collection
        self.saved: list[CollectionInfo] = []

    async def get_collection(self, collection: str) -> CollectionInfo:
        if self.collection is None or self.collection.name != collection:
            raise ValueError(f"Collection {collection!r} not found")
        return self.collection

    async def save_collection(self, collection: CollectionInfo) -> None:
        self.saved.append(collection)
        self.collection = collection


class _FakeStorageShim:
    def __init__(self, metadata_store: _FakeMetadataStore) -> None:
        self.metadata_store = metadata_store

    def get_metadata_store(self) -> _FakeMetadataStore:
        return self.metadata_store


def test_ensure_collection_backend_binding_sets_lancedb_when_missing() -> None:
    metadata_store = _FakeMetadataStore(CollectionInfo(name="demo"))
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(metadata_store)
    )

    updated = facade.ensure_collection_backend_binding("demo")

    assert updated is not None
    assert updated.extra_metadata["kb_storage"] == {"backend": "lancedb"}
    assert metadata_store.saved == [updated]


@pytest.mark.asyncio
async def test_ensure_collection_backend_binding_async_sets_lancedb_when_missing() -> (
    None
):
    metadata_store = _FakeMetadataStore(CollectionInfo(name="demo"))
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(metadata_store)
    )

    updated = await facade.ensure_collection_backend_binding_async("demo")

    assert updated is not None
    assert updated.extra_metadata["kb_storage"] == {"backend": "lancedb"}
    assert metadata_store.saved == [updated]


def test_ensure_collection_backend_binding_preserves_existing_binding() -> None:
    existing_binding = {"backend": "postgresql", "dsn": "kept"}
    metadata_store = _FakeMetadataStore(
        CollectionInfo(
            name="demo",
            extra_metadata={"kb_storage": existing_binding, "other": "value"},
        )
    )
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(metadata_store)
    )

    existing = facade.ensure_collection_backend_binding("demo")

    assert existing is metadata_store.collection
    assert existing is not None
    assert existing.extra_metadata["kb_storage"] == existing_binding
    assert existing.extra_metadata["other"] == "value"
    assert metadata_store.saved == []


def test_ensure_collection_backend_binding_ignores_missing_collection() -> None:
    metadata_store = _FakeMetadataStore(None)
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(metadata_store)
    )

    assert facade.ensure_collection_backend_binding("missing") is None
    assert metadata_store.saved == []


def _ingestion_step(name: str, **metadata: object) -> IngestionStepResult:
    return IngestionStepResult(name=name, metadata=dict(metadata))


def _successful_ingestion_result(doc_id: str = "doc-ok") -> IngestionResult:
    return IngestionResult(
        status="success",
        doc_id=doc_id,
        parse_hash="parse-ok",
        chunk_count=1,
        embedding_count=1,
        vector_count=1,
        completed_steps=[
            _ingestion_step("initialize_collection", embedding_model_id="model-a"),
            _ingestion_step("register_document", doc_id=doc_id, created=True),
            _ingestion_step("parse_document", parse_hash="parse-ok", written=True),
            _ingestion_step("chunk_document", chunk_count=1, created=True),
            _ingestion_step("write_vectors_to_db", vector_count=1),
        ],
        message="ok",
    )


def test_step_metadata_preserves_present_step_with_empty_metadata() -> None:
    empty_metadata_step = IngestionStepResult(name="initialize_collection")
    none_metadata_step = IngestionStepResult.model_construct(
        name="register_document", metadata=None
    )

    assert (
        KBPipelineCompatibilityFacade._step_metadata(
            [empty_metadata_step], "initialize_collection"
        )
        == {}
    )
    assert (
        KBPipelineCompatibilityFacade._step_metadata(
            [none_metadata_step], "register_document"
        )
        == {}
    )
    assert KBPipelineCompatibilityFacade._step_metadata([], "missing") is None


def test_process_document_binds_collection_after_first_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion

    metadata_store = _FakeMetadataStore(CollectionInfo(name="demo"))
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(metadata_store)
    )

    def fake_process_document_impl(**_: object) -> IngestionResult:
        return _successful_ingestion_result()

    monkeypatch.setattr(
        document_ingestion,
        "_process_document_impl",
        fake_process_document_impl,
    )

    result = facade.process_document("demo", "/tmp/doc.md")

    assert result.status == "success"
    assert metadata_store.collection is not None
    assert metadata_store.collection.extra_metadata["kb_storage"] == {
        "backend": "lancedb"
    }
    assert metadata_store.saved == [metadata_store.collection]


@pytest.mark.parametrize(
    ("failed_step", "completed_steps", "chunk_count", "vector_count", "planes"),
    [
        (
            "register_document",
            [
                _ingestion_step("initialize_collection", embedding_model_id="model-a"),
                _ingestion_step("register_document", doc_id="doc-1", created=True),
            ],
            0,
            0,
            {
                SideEffectPlane.COLLECTION,
                SideEffectPlane.DOCUMENT,
                SideEffectPlane.STATUS,
            },
        ),
        (
            "parse_document",
            [
                _ingestion_step("initialize_collection", embedding_model_id="model-a"),
                _ingestion_step("register_document", doc_id="doc-1", created=True),
                _ingestion_step("parse_document", parse_hash="parse-1", written=True),
            ],
            0,
            0,
            {
                SideEffectPlane.COLLECTION,
                SideEffectPlane.DOCUMENT,
                SideEffectPlane.STATUS,
                SideEffectPlane.PARSE,
            },
        ),
        (
            "chunk_document",
            [
                _ingestion_step("initialize_collection", embedding_model_id="model-a"),
                _ingestion_step("register_document", doc_id="doc-1", created=True),
                _ingestion_step("parse_document", parse_hash="parse-1", written=True),
                _ingestion_step("chunk_document", chunk_count=2, created=True),
            ],
            2,
            0,
            {
                SideEffectPlane.COLLECTION,
                SideEffectPlane.DOCUMENT,
                SideEffectPlane.STATUS,
                SideEffectPlane.PARSE,
                SideEffectPlane.CHUNK,
            },
        ),
        (
            "write_vectors_to_db",
            [
                _ingestion_step("initialize_collection", embedding_model_id="model-a"),
                _ingestion_step("register_document", doc_id="doc-1", created=True),
                _ingestion_step("parse_document", parse_hash="parse-1", written=True),
                _ingestion_step("chunk_document", chunk_count=2, created=True),
                _ingestion_step("write_vectors_to_db", vector_count=2),
            ],
            2,
            2,
            {
                SideEffectPlane.COLLECTION,
                SideEffectPlane.DOCUMENT,
                SideEffectPlane.STATUS,
                SideEffectPlane.PARSE,
                SideEffectPlane.CHUNK,
                SideEffectPlane.EMBEDDING,
            },
        ),
    ],
)
def test_process_document_records_failed_ingest_operation_outcome(
    monkeypatch: pytest.MonkeyPatch,
    failed_step: str,
    completed_steps: list[IngestionStepResult],
    chunk_count: int,
    vector_count: int,
    planes: set[SideEffectPlane],
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion

    operation_facade = KBOperationCompatibilityFacade()
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeMetadataStore(CollectionInfo(name="demo"))),
        operation_compatibility=operation_facade,
    )

    def fake_process_document_impl(**_: object) -> IngestionResult:
        return IngestionResult(
            status="partial",
            doc_id="doc-1",
            parse_hash="parse-1" if failed_step != "register_document" else None,
            chunk_count=chunk_count,
            embedding_count=vector_count,
            vector_count=vector_count,
            completed_steps=completed_steps,
            failed_step=failed_step,
            message=f"failed at {failed_step}",
        )

    monkeypatch.setattr(
        document_ingestion,
        "_process_document_impl",
        fake_process_document_impl,
    )

    result = facade.process_document("demo", "/tmp/doc.md")

    assert result.failed_step == failed_step
    outcome = operation_facade.last_outcome
    assert outcome is not None
    assert outcome.status == "partial"
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert {step.plane for step in outcome.compensation_steps} == planes
    assert [step.name for step in outcome.compensation_plan] == [
        step.name for step in reversed(outcome.compensation_steps)
    ]


def test_run_document_ingestion_preserves_legacy_non_result_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion

    expected_result = object()
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeMetadataStore(CollectionInfo(name="demo"))),
        operation_compatibility=KBOperationCompatibilityFacade(),
    )

    def fake_run_document_ingestion_impl(**_: object) -> object:
        return expected_result

    def fail_if_called(*_: object, **__: object) -> None:
        raise AssertionError("structured ingestion hooks should not run")

    monkeypatch.setattr(
        document_ingestion,
        "_run_document_ingestion_impl",
        fake_run_document_ingestion_impl,
    )
    monkeypatch.setattr(
        facade, "_record_document_ingestion_side_effects", fail_if_called
    )
    monkeypatch.setattr(facade, "ensure_collection_backend_binding", fail_if_called)
    monkeypatch.setattr(facade, "_finish_document_ingestion_outcome", fail_if_called)

    result = facade.run_document_ingestion("demo", "/tmp/doc.md")

    assert result is expected_result


@pytest.mark.asyncio
async def test_web_ingestion_records_page_child_outcomes_and_preserve_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines import (
        document_ingestion,
        web_ingestion,
    )

    operation_facade = KBOperationCompatibilityFacade()
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeMetadataStore(CollectionInfo(name="demo"))),
        operation_compatibility=operation_facade,
    )

    def fake_run_document_ingestion_impl(
        collection: str,
        source_path: str,
        **_: object,
    ) -> IngestionResult:
        if source_path.endswith("ok.md"):
            return _successful_ingestion_result("doc-ok")
        return IngestionResult(
            status="partial",
            doc_id="doc-bad",
            parse_hash=None,
            completed_steps=[
                _ingestion_step("initialize_collection", embedding_model_id="model-a"),
                _ingestion_step("register_document", doc_id="doc-bad", created=True),
            ],
            failed_step="parse_document",
            message="parse failed",
        )

    async def fake_run_web_ingestion_impl(
        collection: str,
        crawl_config: WebCrawlConfig,
        **_: object,
    ) -> WebIngestionResult:
        facade.run_document_ingestion(collection, "/tmp/ok.md")
        facade.run_document_ingestion(collection, "/tmp/bad.md")
        return WebIngestionResult(
            status="partial",
            collection=collection,
            total_urls_found=2,
            pages_crawled=2,
            pages_failed=1,
            documents_created=1,
            chunks_created=1,
            embeddings_created=1,
            crawled_urls=[crawl_config.start_url, "https://example.com/bad"],
            failed_urls={"https://example.com/bad": "parse failed"},
            message="partial",
            warnings=["parse failed"],
            elapsed_time_ms=1,
        )

    monkeypatch.setattr(
        document_ingestion,
        "_run_document_ingestion_impl",
        fake_run_document_ingestion_impl,
    )
    monkeypatch.setattr(
        web_ingestion,
        "_run_web_ingestion_impl",
        fake_run_web_ingestion_impl,
    )

    result = await facade.run_web_ingestion(
        "demo",
        WebCrawlConfig(start_url="https://example.com", max_pages=2),
    )

    assert result.status == "partial"
    outcome = operation_facade.last_outcome
    assert outcome is not None
    assert outcome.operation_type == "web_ingestion"
    assert outcome.status == "partial"
    assert outcome.rollback_status is RollbackStatus.SKIPPED_BY_POLICY
    assert outcome.side_effects_may_remain is True
    assert [child.operation_type for child in outcome.child_outcomes] == [
        "web_page_ingestion",
        "web_page_ingestion",
    ]
    assert [child.status for child in outcome.child_outcomes] == [
        "success",
        "partial",
    ]


@pytest.mark.asyncio
async def test_web_ingestion_zero_success_reports_incomplete_rollback_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines import (
        document_ingestion,
        web_ingestion,
    )

    operation_facade = KBOperationCompatibilityFacade()
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeMetadataStore(CollectionInfo(name="demo"))),
        operation_compatibility=operation_facade,
    )

    def fake_run_document_ingestion_impl(**_: object) -> IngestionResult:
        return IngestionResult(
            status="partial",
            doc_id="doc-failed",
            parse_hash=None,
            completed_steps=[
                _ingestion_step("initialize_collection", embedding_model_id="model-a"),
                _ingestion_step("register_document", doc_id="doc-failed", created=True),
            ],
            failed_step="parse_document",
            message="parse failed",
        )

    async def fake_run_web_ingestion_impl(
        collection: str,
        crawl_config: WebCrawlConfig,
        **_: object,
    ) -> WebIngestionResult:
        facade.run_document_ingestion(collection, "/tmp/failed.md")
        return WebIngestionResult(
            status="error",
            collection=collection,
            total_urls_found=1,
            pages_crawled=1,
            pages_failed=1,
            documents_created=0,
            chunks_created=0,
            embeddings_created=0,
            crawled_urls=[crawl_config.start_url],
            failed_urls={crawl_config.start_url: "parse failed"},
            message="failed",
            warnings=["parse failed"],
            elapsed_time_ms=1,
        )

    monkeypatch.setattr(
        document_ingestion,
        "_run_document_ingestion_impl",
        fake_run_document_ingestion_impl,
    )
    monkeypatch.setattr(
        web_ingestion,
        "_run_web_ingestion_impl",
        fake_run_web_ingestion_impl,
    )

    result = await facade.run_web_ingestion(
        "demo",
        WebCrawlConfig(start_url="https://example.com", max_pages=1),
    )

    assert result.status == "error"
    outcome = operation_facade.last_outcome
    assert outcome is not None
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert len(outcome.child_outcomes) == 1
    assert outcome.child_outcomes[0].rollback_status is RollbackStatus.INCOMPLETE


class _FailingSaveMetadataStore(_FakeMetadataStore):
    async def save_collection(self, collection: CollectionInfo) -> None:
        raise RuntimeError("save failed")


def test_process_document_binding_failure_marks_operation_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion

    operation_facade = KBOperationCompatibilityFacade()
    facade = KBPipelineCompatibilityFacade(
        storage_shim=_FakeStorageShim(
            _FailingSaveMetadataStore(CollectionInfo(name="demo"))
        ),
        operation_compatibility=operation_facade,
    )

    def fake_process_document_impl(**_: object) -> IngestionResult:
        return _successful_ingestion_result("doc-binding")

    monkeypatch.setattr(
        document_ingestion,
        "_process_document_impl",
        fake_process_document_impl,
    )

    with pytest.raises(RuntimeError, match="save failed"):
        facade.process_document("demo", "/tmp/doc.md")

    outcome = operation_facade.last_outcome
    assert outcome is not None
    assert outcome.status == "error"
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert {step.plane for step in outcome.compensation_steps} >= {
        SideEffectPlane.DOCUMENT,
        SideEffectPlane.PARSE,
        SideEffectPlane.CHUNK,
        SideEffectPlane.EMBEDDING,
    }


class _SinglePageCrawler:
    def __init__(
        self, config: WebCrawlConfig, progress_callback: object = None
    ) -> None:
        self.total_urls_found = 1
        self.failed_urls: dict[str, str] = {}

    async def crawl(self) -> list[CrawlResult]:
        return [
            CrawlResult(
                url="https://example.com/page",
                title="Example Page",
                content_markdown="body",
                status="success",
                depth=0,
                timestamp=datetime.now(timezone.utc),
                content_length=4,
            )
        ]


@pytest.mark.asyncio
async def test_web_ingestion_file_handler_failure_records_page_child_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines import web_ingestion

    operation_facade = KBOperationCompatibilityFacade()
    facade = KBPipelineCompatibilityFacade(operation_compatibility=operation_facade)
    monkeypatch.setattr(web_ingestion, "WebCrawler", _SinglePageCrawler)

    def failing_file_handler(
        temp_file: Path, title: str, collection: str, url: str
    ) -> dict[str, str]:
        raise RuntimeError("file handler failed")

    result = await facade.run_web_ingestion(
        "demo",
        WebCrawlConfig(start_url="https://example.com", max_pages=1),
        file_handler=failing_file_handler,
    )

    assert result.status == "error"
    outcome = operation_facade.last_outcome
    assert outcome is not None
    assert outcome.status == "error"
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert outcome.side_effects_may_remain is True
    assert len(outcome.child_outcomes) == 1
    child = outcome.child_outcomes[0]
    assert child.operation_type == "web_page_ingestion"
    assert child.status == "error"
    assert child.rollback_status is RollbackStatus.INCOMPLETE
    assert {step.plane for step in child.compensation_steps} == {SideEffectPlane.FILE}


@pytest.mark.asyncio
async def test_web_ingestion_empty_file_handler_result_is_explicit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines import web_ingestion

    operation_facade = KBOperationCompatibilityFacade()
    facade = KBPipelineCompatibilityFacade(operation_compatibility=operation_facade)
    monkeypatch.setattr(web_ingestion, "WebCrawler", _SinglePageCrawler)

    def empty_file_handler(
        temp_file: Path, title: str, collection: str, url: str
    ) -> dict[str, str]:
        return {}

    result = await facade.run_web_ingestion(
        "demo",
        WebCrawlConfig(start_url="https://example.com", max_pages=1),
        file_handler=empty_file_handler,
    )

    assert result.status == "error"
    assert "File handler returned no file information" in next(
        iter(result.failed_urls.values())
    )
    outcome = operation_facade.last_outcome
    assert outcome is not None
    assert outcome.rollback_status is RollbackStatus.INCOMPLETE
    assert len(outcome.child_outcomes) == 1
    child = outcome.child_outcomes[0]
    assert child.status == "error"
    assert {step.plane for step in child.compensation_steps} == {SideEffectPlane.FILE}


@pytest.mark.asyncio
async def test_web_ingestion_none_file_handler_path_uses_temp_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines import web_ingestion

    operation_facade = KBOperationCompatibilityFacade()
    facade = KBPipelineCompatibilityFacade(operation_compatibility=operation_facade)
    monkeypatch.setattr(web_ingestion, "WebCrawler", _SinglePageCrawler)
    captured: dict[str, object] = {}

    def file_handler(
        temp_file: Path, title: str, collection: str, url: str
    ) -> dict[str, object]:
        return {"file_path": None, "file_id": "file-1"}

    def fake_run_document_ingestion(**kwargs: object) -> IngestionResult:
        captured.update(kwargs)
        return _successful_ingestion_result(doc_id="doc-ok")

    monkeypatch.setattr(
        web_ingestion,
        "run_document_ingestion",
        fake_run_document_ingestion,
    )

    result = await facade.run_web_ingestion(
        "demo",
        WebCrawlConfig(start_url="https://example.com", max_pages=1),
        file_handler=file_handler,
    )

    assert result.status == "success"
    assert captured["file_id"] == "file-1"
    assert "xagent_web_ingest" in str(captured["source_path"])
    assert str(captured["source_path"]).endswith(".md")


@pytest.mark.asyncio
async def test_web_ingestion_file_and_document_side_effects_share_page_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.core.tools.core.RAG_tools.pipelines import (
        document_ingestion,
        web_ingestion,
    )

    operation_facade = KBOperationCompatibilityFacade()
    facade = KBPipelineCompatibilityFacade(operation_compatibility=operation_facade)
    monkeypatch.setattr(web_ingestion, "WebCrawler", _SinglePageCrawler)
    monkeypatch.setattr(
        web_ingestion, "run_document_ingestion", facade.run_document_ingestion
    )

    def fake_run_document_ingestion_impl(**_: object) -> IngestionResult:
        return IngestionResult(
            status="partial",
            doc_id="doc-failed",
            parse_hash=None,
            completed_steps=[
                _ingestion_step("initialize_collection", embedding_model_id="model-a"),
                _ingestion_step("register_document", doc_id="doc-failed", created=True),
            ],
            failed_step="parse_document",
            message="parse failed",
        )

    def file_handler(
        temp_file: Path, title: str, collection: str, url: str
    ) -> dict[str, str]:
        return {"file_path": str(temp_file), "file_id": "file-1"}

    monkeypatch.setattr(
        document_ingestion,
        "_run_document_ingestion_impl",
        fake_run_document_ingestion_impl,
    )

    result = await facade.run_web_ingestion(
        "demo",
        WebCrawlConfig(start_url="https://example.com", max_pages=1),
        file_handler=file_handler,
    )

    assert result.status == "error"
    outcome = operation_facade.last_outcome
    assert outcome is not None
    assert len(outcome.child_outcomes) == 1
    child = outcome.child_outcomes[0]
    assert child.status == "partial"
    assert child.rollback_status is RollbackStatus.INCOMPLETE
    assert {step.plane for step in child.compensation_steps} >= {
        SideEffectPlane.FILE,
        SideEffectPlane.DOCUMENT,
        SideEffectPlane.STATUS,
    }
