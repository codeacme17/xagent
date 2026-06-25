"""Routing tests: verify facade search methods open a READ handle and delegate to it."""

from unittest.mock import MagicMock, patch

from xagent.core.tools.core.RAG_tools.kb.legacy_step_compatibility import (
    KBLegacyStepCompatibilityFacade,
)


def _make_facade() -> KBLegacyStepCompatibilityFacade:
    return KBLegacyStepCompatibilityFacade()


# ---------------------------------------------------------------------------
# search_dense
# ---------------------------------------------------------------------------


def test_search_dense_opens_handle_and_delegates():
    facade = _make_facade()
    handle = MagicMock()
    with patch.object(facade, "_open_collection_handle", return_value=handle) as opener:
        facade.search_dense("col1", "model-x", [0.1], top_k=4, user_id=5, is_admin=True)
    opener.assert_called_once()
    handle.search_dense.assert_called_once()
    kwargs = handle.search_dense.call_args.kwargs
    assert kwargs["top_k"] == 4


def test_search_dense_passes_read_access_mode():
    from xagent.core.tools.core.RAG_tools.kb.models import KBAccessMode

    facade = _make_facade()
    handle = MagicMock()
    with patch.object(facade, "_open_collection_handle", return_value=handle) as opener:
        facade.search_dense("col1", "model-x", [0.1, 0.2], top_k=7)
    _, opener_kwargs = opener.call_args
    assert opener_kwargs.get("access_mode") == KBAccessMode.READ


# ---------------------------------------------------------------------------
# search_sparse
# ---------------------------------------------------------------------------


def test_search_sparse_opens_handle_and_delegates():
    facade = _make_facade()
    handle = MagicMock()
    with patch.object(facade, "_open_collection_handle", return_value=handle) as opener:
        facade.search_sparse("col2", "model-y", "query text", top_k=5)
    opener.assert_called_once()
    handle.search_sparse.assert_called_once()
    kwargs = handle.search_sparse.call_args.kwargs
    assert kwargs["top_k"] == 5


def test_search_sparse_passes_read_access_mode():
    from xagent.core.tools.core.RAG_tools.kb.models import KBAccessMode

    facade = _make_facade()
    handle = MagicMock()
    with patch.object(facade, "_open_collection_handle", return_value=handle) as opener:
        facade.search_sparse("col2", "model-y", "hello", top_k=3)
    _, opener_kwargs = opener.call_args
    assert opener_kwargs.get("access_mode") == KBAccessMode.READ


# ---------------------------------------------------------------------------
# search_hybrid
# ---------------------------------------------------------------------------


def test_search_hybrid_opens_handle_and_delegates():
    facade = _make_facade()
    handle = MagicMock()
    with patch.object(facade, "_open_collection_handle", return_value=handle) as opener:
        facade.search_hybrid("col3", "model-z", "query", [0.3, 0.4], top_k=6)
    opener.assert_called_once()
    handle.search_hybrid.assert_called_once()
    kwargs = handle.search_hybrid.call_args.kwargs
    assert kwargs["top_k"] == 6


def test_search_hybrid_passes_read_access_mode():
    from xagent.core.tools.core.RAG_tools.kb.models import KBAccessMode

    facade = _make_facade()
    handle = MagicMock()
    with patch.object(facade, "_open_collection_handle", return_value=handle) as opener:
        facade.search_hybrid("col3", "model-z", "q", [0.1], top_k=2)
    _, opener_kwargs = opener.call_args
    assert opener_kwargs.get("access_mode") == KBAccessMode.READ


# ---------------------------------------------------------------------------
# search_dense_async
# ---------------------------------------------------------------------------


def test_search_dense_async_delegates_to_handle():
    import asyncio

    facade = _make_facade()
    handle = MagicMock()
    handle.search_dense_async = MagicMock(return_value=_async_return(MagicMock()))

    mock_coord = MagicMock()
    mock_coord.open_collection = MagicMock(return_value=_async_return(handle))
    with patch.object(facade, "_active_coordinator", return_value=mock_coord):
        asyncio.run(facade.search_dense_async("col1", "model-x", [0.1], top_k=8))
    handle.search_dense_async.assert_called_once()
    kwargs = handle.search_dense_async.call_args.kwargs
    assert kwargs["top_k"] == 8


def test_search_sparse_async_delegates_to_handle():
    import asyncio

    facade = _make_facade()
    handle = MagicMock()
    handle.search_sparse_async = MagicMock(return_value=_async_return(MagicMock()))

    mock_coord = MagicMock()
    mock_coord.open_collection = MagicMock(return_value=_async_return(handle))
    with patch.object(facade, "_active_coordinator", return_value=mock_coord):
        asyncio.run(facade.search_sparse_async("col2", "model-y", "hello", top_k=9))
    handle.search_sparse_async.assert_called_once()
    kwargs = handle.search_sparse_async.call_args.kwargs
    assert kwargs["top_k"] == 9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _async_return(value):
    return value
