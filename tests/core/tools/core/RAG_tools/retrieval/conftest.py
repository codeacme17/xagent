"""Shared fixtures for driving the public ``search_*`` functions through the
coordinator-opened collection handle (#511).

After the search facade was re-routed (Task 6) the public
``retrieval.search_dense/search_sparse/search_hybrid`` functions no longer touch
the ``search_engine`` free functions. They go:

    public search_*()  ->  facade.search_*()  ->  facade._open_collection_handle()
                       ->  coordinator.open_collection_sync()  ->  LanceDBCollectionHandle
                       ->  handle.search_*()  (runs against context.vector_index_store)

These fixtures build a real ``LanceDBCollectionHandle`` whose context carries a
MOCK ``vector_index_store`` (the canonical store-mock seam used by
``tests/core/tools/core/RAG_tools/kb/test_collection_handle_search.py``), wrap it
in a facade, and return that facade from ``_get_legacy_step_compatibility_facade``
on the target retrieval module. The real handle logic (index-status mapping,
score conversion, filter building, FTS fallback, fusion) runs unchanged against
the mock store, so the existing behavioral assertions stay valid.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)
from xagent.core.tools.core.RAG_tools.kb.legacy_step_compatibility import (
    KBLegacyStepCompatibilityFacade,
)


def _make_handle(
    *, collection: str = "test_col", supports_search: bool = True
) -> tuple[LanceDBCollectionHandle, MagicMock, MagicMock]:
    handle = LanceDBCollectionHandle.__new__(LanceDBCollectionHandle)
    ctx = MagicMock()
    ctx.collection = collection
    object.__setattr__(handle, "context", ctx)

    store = MagicMock()
    ctx.vector_index_store = store

    caps = MagicMock()
    caps.supports_search = supports_search
    ctx.capabilities = caps

    return handle, store, caps


def _facade_for_handle(
    handle: LanceDBCollectionHandle,
) -> KBLegacyStepCompatibilityFacade:
    facade = KBLegacyStepCompatibilityFacade()
    facade._open_collection_handle = MagicMock(return_value=handle)  # type: ignore[method-assign]

    @contextmanager
    def _noop_storage_context() -> Iterator[None]:
        yield

    facade._storage_context = _noop_storage_context  # type: ignore[method-assign]
    return facade


@pytest.fixture
def make_handle():
    """Factory: build a real handle with a mock context/store/capabilities.

    Returns a callable ``make_handle(*, collection=..., supports_search=...)``
    yielding ``(handle, store, capabilities)``. Configure ``store`` exactly as
    the legacy tests configured their ``mock_vector_store``.
    """
    return _make_handle


@pytest.fixture
def routed_facade():
    """Context manager: route a retrieval module's public ``search_*`` to a handle.

    Usage::

        with routed_facade(search_dense_module, handle):
            response = search_dense(...)

    Patches ``module._get_legacy_step_compatibility_facade`` to a facade whose
    ``_open_collection_handle`` returns ``handle``, so the public function runs
    the real handle logic against the handle's mock store.
    """

    @contextmanager
    def _routed(
        module, handle: LanceDBCollectionHandle
    ) -> Iterator[KBLegacyStepCompatibilityFacade]:
        facade = _facade_for_handle(handle)
        with patch.object(
            module, "_get_legacy_step_compatibility_facade", return_value=facade
        ):
            yield facade

    return _routed
