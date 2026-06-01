"""Tests for the KB parse display compatibility facade."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    DatabaseOperationError,
    DocumentNotFoundError,
)


class _FakeRow:
    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    def to_dict(self) -> dict[str, Any]:
        return dict(self._record)


class _FakeDataFrame:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def iterrows(self):
        for index, record in enumerate(self._records):
            yield index, _FakeRow(record)


class _FakeBatch:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def to_pandas(self) -> _FakeDataFrame:
        return _FakeDataFrame(self._records)


class _FakeVectorStore:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.count_calls: list[dict[str, Any]] = []
        self.iter_calls: list[dict[str, Any]] = []

    def _matches(
        self,
        record: dict[str, Any],
        filters: dict[str, Any],
        user_id: Optional[int],
        is_admin: bool,
    ) -> bool:
        for field, value in filters.items():
            if record.get(field) != value:
                return False
        if not is_admin and record.get("user_id") != user_id:
            return False
        return True

    def _matching_records(
        self,
        filters: dict[str, Any],
        user_id: Optional[int],
        is_admin: bool,
    ) -> list[dict[str, Any]]:
        return [
            record
            for record in self._records
            if self._matches(record, filters, user_id, is_admin)
        ]

    def count_rows_or_zero(
        self,
        table_name: str,
        *,
        filters: dict[str, Any],
        user_id: Optional[int],
        is_admin: bool,
    ) -> int:
        self.count_calls.append(
            {
                "table_name": table_name,
                "filters": filters,
                "user_id": user_id,
                "is_admin": is_admin,
            }
        )
        assert table_name == "parses"
        return len(self._matching_records(filters, user_id, is_admin))

    def iter_batches(
        self,
        *,
        table_name: str,
        filters: dict[str, Any],
        user_id: Optional[int],
        is_admin: bool,
    ):
        self.iter_calls.append(
            {
                "table_name": table_name,
                "filters": filters,
                "user_id": user_id,
                "is_admin": is_admin,
            }
        )
        assert table_name == "parses"
        yield _FakeBatch(self._matching_records(filters, user_id, is_admin))


class _FakeStorageShim:
    def __init__(self, vector_store: _FakeVectorStore) -> None:
        self.vector_store = vector_store

    def get_vector_index_store(self) -> _FakeVectorStore:
        return self.vector_store


def _parse_record(
    *,
    parse_hash: str,
    created_at: datetime,
    text: str,
    layout_type: str = "text",
    user_id: int = 1,
) -> dict[str, Any]:
    return {
        "collection": "docs",
        "doc_id": "doc-1",
        "parse_hash": parse_hash,
        "created_at": created_at,
        "parsed_content": json.dumps(
            [{"text": text, "metadata": {"layout_type": layout_type}}]
        ),
        "user_id": user_id,
    }


def _signature_shape(callable_obj: Any) -> list[tuple[str, Any, Any]]:
    return [
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(callable_obj).parameters.items()
    ]


def test_kb_parse_display_facade_public_surface_imports() -> None:
    """Given the KB package, the parse display facade is publicly importable."""
    import xagent.core.tools.core.RAG_tools.kb as kb
    from xagent.core.tools.core.RAG_tools.kb import (
        KBParseDisplayCompatibilityFacade,
        get_kb_coordinator,
        reset_kb_coordinator_for_tests,
    )

    assert hasattr(kb, "KBParseDisplayCompatibilityFacade")
    reset_kb_coordinator_for_tests()
    coordinator = get_kb_coordinator()
    assert isinstance(
        coordinator.parse_display_compatibility,
        KBParseDisplayCompatibilityFacade,
    )
    assert coordinator.parse_display is coordinator.parse_display_compatibility


def test_parse_display_facade_methods_match_public_helper_signatures() -> None:
    """Given legacy helpers, facade methods preserve their call signatures."""
    from xagent.core.tools.core.RAG_tools.kb import KBParseDisplayCompatibilityFacade
    from xagent.core.tools.core.RAG_tools.parse import parse_display

    facade = KBParseDisplayCompatibilityFacade(
        storage_shim=_FakeStorageShim(_FakeVectorStore([]))
    )

    assert _signature_shape(facade.reconstruct_parse_result_from_db) == (
        _signature_shape(parse_display.reconstruct_parse_result_from_db)
    )
    assert _signature_shape(facade.paginate_parse_results) == _signature_shape(
        parse_display.paginate_parse_results
    )


def test_public_parse_display_helpers_remain_sync_and_route_through_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a public parse display helper call, it routes through the facade."""
    from xagent.core.tools.core.RAG_tools.parse import parse_display

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class _FakeFacade:
        def reconstruct_parse_result_from_db(self, *args: Any, **kwargs: Any):
            calls.append(("reconstruct", args, kwargs))
            return ([{"type": "text", "text": "ok", "metadata": {}}], "hash-1")

        def paginate_parse_results(self, *args: Any, **kwargs: Any):
            calls.append(("paginate", args, kwargs))
            return (["page"], {"page": kwargs["page"]})

    monkeypatch.setattr(
        parse_display,
        "_get_parse_display_compatibility_facade",
        lambda: _FakeFacade(),
    )

    assert not inspect.iscoroutinefunction(
        parse_display.reconstruct_parse_result_from_db
    )
    assert not inspect.iscoroutinefunction(parse_display.paginate_parse_results)
    assert parse_display.reconstruct_parse_result_from_db(
        "docs",
        "doc-1",
        parse_hash="hash-1",
        user_id=7,
        is_admin=True,
    ) == ([{"type": "text", "text": "ok", "metadata": {}}], "hash-1")
    assert parse_display.paginate_parse_results([], page=2, page_size=3) == (
        ["page"],
        {"page": 2},
    )
    assert calls == [
        (
            "reconstruct",
            ("docs", "doc-1"),
            {"parse_hash": "hash-1", "user_id": 7, "is_admin": True},
        ),
        ("paginate", ([],), {"page": 2, "page_size": 3}),
    ]


def test_parse_display_facade_preserves_sync_tuple_shapes_and_latest_selection() -> (
    None
):
    """Given direct sync calls, latest fallback and explicit hash behavior stay stable."""
    from xagent.core.tools.core.RAG_tools.kb import KBParseDisplayCompatibilityFacade

    created_at = datetime(2026, 1, 1, 12, 0, 0)
    vector_store = _FakeVectorStore(
        [
            _parse_record(parse_hash="old", created_at=created_at, text="old body"),
            _parse_record(
                parse_hash="new",
                created_at=created_at + timedelta(seconds=1),
                text="new body",
            ),
        ]
    )
    facade = KBParseDisplayCompatibilityFacade(
        storage_shim=_FakeStorageShim(vector_store)
    )

    elements, actual_hash = facade.reconstruct_parse_result_from_db(
        "docs",
        "doc-1",
        user_id=1,
        is_admin=False,
    )
    assert actual_hash == "new"
    assert elements == [
        {"type": "text", "text": "new body", "metadata": {"layout_type": "text"}}
    ]

    explicit_elements, explicit_hash = facade.reconstruct_parse_result_from_db(
        "docs",
        "doc-1",
        parse_hash="old",
        user_id=1,
        is_admin=False,
    )
    assert explicit_hash == "old"
    assert explicit_elements[0]["text"] == "old body"

    page_elements, pagination = facade.paginate_parse_results(
        elements + explicit_elements,
        page=1,
        page_size=1,
    )
    assert len(page_elements) == 1
    assert pagination == {
        "page": 1,
        "page_size": 1,
        "total_elements": 2,
        "total_pages": 2,
        "has_next": True,
        "has_previous": False,
    }
    assert vector_store.count_calls[0]["filters"] == {
        "collection": "docs",
        "doc_id": "doc-1",
    }
    assert vector_store.count_calls[1]["filters"] == {
        "collection": "docs",
        "doc_id": "doc-1",
        "parse_hash": "old",
    }


def test_parse_display_lookup_after_rolled_back_ingest_keeps_not_found_behavior() -> (
    None
):
    """Rolled-back ingest leaves no parse row, so lookup stays legacy not-found."""
    from xagent.core.tools.core.RAG_tools.kb import KBParseDisplayCompatibilityFacade

    vector_store = _FakeVectorStore([])
    facade = KBParseDisplayCompatibilityFacade(
        storage_shim=_FakeStorageShim(vector_store)
    )

    with pytest.raises(
        DocumentNotFoundError,
        match="No parse results found for document: doc_id=doc-rolled-back",
    ):
        facade.reconstruct_parse_result_from_db(
            "docs",
            "doc-rolled-back",
            user_id=1,
            is_admin=False,
        )

    assert vector_store.count_calls[0]["filters"] == {
        "collection": "docs",
        "doc_id": "doc-rolled-back",
    }
    assert vector_store.iter_calls == []


def test_parse_display_facade_rebinds_coordinator_storage_for_legacy_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Facade delegation rebinds storage factory access to its coordinator shim."""
    from xagent.core.tools.core.RAG_tools.kb import KBParseDisplayCompatibilityFacade
    from xagent.core.tools.core.RAG_tools.parse import parse_display
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        bind_storage_shim_for_current_context,
        get_vector_index_store,
    )

    outer_store = _FakeVectorStore([])
    inner_store = _FakeVectorStore([])
    outer_shim = _FakeStorageShim(outer_store)
    inner_shim = _FakeStorageShim(inner_store)

    class _FakeCoordinator:
        @property
        def storage_shim(self) -> _FakeStorageShim:
            return inner_shim

        def get_context_sync(self, _request: Any) -> Any:
            return SimpleNamespace(vector_index_store=inner_store)

    def fake_impl(
        collection: str,
        doc_id: str,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
        **_kwargs: Any,
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        assert collection == "docs"
        assert doc_id == "doc-inner"
        assert parse_hash is None
        assert user_id == 1
        assert is_admin is False
        assert get_vector_index_store() is inner_store
        return ([{"type": "text", "text": "inner", "metadata": {}}], "inner")

    monkeypatch.setattr(
        parse_display,
        "_reconstruct_parse_result_from_db_impl",
        fake_impl,
    )

    facade = KBParseDisplayCompatibilityFacade(coordinator=_FakeCoordinator())
    with bind_storage_shim_for_current_context(outer_shim):
        elements, parse_hash = facade.reconstruct_parse_result_from_db(
            "docs", "doc-inner", user_id=1, is_admin=False
        )
        assert get_vector_index_store() is outer_store

    assert parse_hash == "inner"
    assert elements[0]["text"] == "inner"


def test_parse_display_facade_preserves_json_corruption_mapping() -> None:
    """Given corrupt parse JSON, facade preserves the legacy DatabaseOperationError."""
    from xagent.core.tools.core.RAG_tools.kb import KBParseDisplayCompatibilityFacade

    vector_store = _FakeVectorStore(
        [
            {
                "collection": "docs",
                "doc_id": "doc-1",
                "parse_hash": "bad",
                "created_at": datetime(2026, 1, 1, 12, 0, 0),
                "parsed_content": "{not-json",
                "user_id": 1,
            }
        ]
    )
    facade = KBParseDisplayCompatibilityFacade(
        storage_shim=_FakeStorageShim(vector_store)
    )

    with pytest.raises(DatabaseOperationError, match="Failed to read parse result"):
        facade.reconstruct_parse_result_from_db(
            "docs",
            "doc-1",
            user_id=1,
            is_admin=False,
        )
