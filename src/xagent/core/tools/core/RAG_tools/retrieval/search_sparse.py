from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, cast

import pandas as pd
import pyarrow as pa  # type: ignore
from pyarrow import Table as PyArrowTable

from ..core.schemas import (
    SearchFallbackAction,
    SearchResult,
    SearchWarning,
    SparseSearchResponse,
)
from ..LanceDB.schema_manager import _safe_close_table
from ..storage.contracts import FilterExpression
from ..storage.factory import (
    get_vector_index_store,
)
from ..utils.filter_utils import parse_legacy_filters, validate_filter_depth
from ..utils.metadata_utils import deserialize_metadata

if TYPE_CHECKING:
    from ..kb import KBLegacyStepCompatibilityFacade

logger = logging.getLogger(__name__)


def _get_legacy_step_compatibility_facade() -> "KBLegacyStepCompatibilityFacade":
    """Return the coordinator-owned legacy step compatibility facade."""
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().legacy_step_compatibility


def search_sparse(
    collection: str,
    model_tag: str,
    query_text: str,
    *,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> SparseSearchResponse:
    """Performs sparse (Full-Text Search) retrieval on the specified collection."""
    return _get_legacy_step_compatibility_facade().search_sparse(
        collection=collection,
        model_tag=model_tag,
        query_text=query_text,
        top_k=top_k,
        filters=filters,
        readonly=readonly,
        nprobes=nprobes,
        refine_factor=refine_factor,
        user_id=user_id,
        is_admin=is_admin,
    )


def _search_sparse_impl(
    collection: str,
    model_tag: str,
    query_text: str,
    *,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> SparseSearchResponse:
    """Performs sparse (Full-Text Search) retrieval on the specified collection."""

    _fts_enabled = False
    current_warnings: List[SearchWarning] = []

    if readonly:
        current_warnings.append(
            SearchWarning(
                code="READONLY_MODE",
                message=f"Readonly mode enabled for sparse search on {model_tag}. No FTS index operations will be performed.",
                fallback_action=SearchFallbackAction.REBUILD_INDEX,
                affected_models=[model_tag],
            )
        )

    table = None
    try:
        vector_store = get_vector_index_store()

        # Open embeddings table with legacy fallback (handled by abstraction layer)
        # open_embeddings_table will handle adding the "embeddings_" prefix
        table, actual_table_name = vector_store.open_embeddings_table(model_tag)

        # Use storage abstraction for index management
        index_result_obj = vector_store.create_index(model_tag, readonly)

        # Use FTS enabled status from index result
        _fts_enabled = index_result_obj.fts_enabled

        if not _fts_enabled:
            current_warnings.append(
                SearchWarning(
                    code="FTS_INDEX_MISSING",
                    message=f"FTS index not found on 'text' column for {model_tag}. Sparse search performance may be degraded.",
                    fallback_action=SearchFallbackAction.REBUILD_INDEX,
                    affected_models=[model_tag],
                )
            )

        search_query = table.search(query_text, query_type="fts").limit(top_k)

        # Convert legacy dict format to FilterExpression if needed
        filter_expr: Optional[FilterExpression] = None
        if collection or filters:
            # Build filter conditions
            conditions: List[FilterExpression] = []

            # Add collection filter
            if collection:
                from ..storage.contracts import FilterCondition, FilterOperator

                conditions.append(
                    FilterCondition(
                        field="collection", operator=FilterOperator.EQ, value=collection
                    )
                )

            # Add custom filters
            if filters:
                if isinstance(filters, dict):
                    # Legacy format: use parser
                    parsed_filters = parse_legacy_filters(filters)
                    # parsed_filters can be FilterCondition or tuple (AND combination)
                    if parsed_filters is not None:
                        if isinstance(parsed_filters, tuple):
                            # Type narrowing: tuple of FilterConditions
                            conditions.extend(parsed_filters)
                        else:
                            # Type narrowing: single FilterCondition
                            conditions.append(parsed_filters)
                elif isinstance(filters, (tuple, list)):
                    # Already FilterExpression
                    conditions.extend(
                        filters if isinstance(filters, tuple) else list(filters)
                    )
                else:
                    # Single FilterCondition
                    conditions.append(filters)

            # Combine conditions with AND
            if len(conditions) == 1:
                filter_expr = conditions[0]
            elif len(conditions) > 1:
                filter_expr = tuple(conditions)

        # Validate filter expression depth to prevent DoS
        if filter_expr is not None:
            validate_filter_depth(filter_expr)

        # Use abstract filter builder to get backend-specific syntax
        if filter_expr:
            backend_filter = vector_store.build_filter_expression(
                filters=filter_expr,
                user_id=user_id,
                is_admin=is_admin,
            )
            if backend_filter:
                search_query = search_query.where(backend_filter)

        # LanceDB's search().to_pandas() returns Any due to missing type stubs
        raw_results_df = pd.DataFrame(search_query.to_pandas())

        if not raw_results_df.empty:
            search_results: List[SearchResult] = []
            for _, row in raw_results_df.iterrows():
                # LanceDB FTS returns TF-IDF score (higher is better),
                # normalize to similarity score (0-1) similar to dense search
                # Using score/(1+score) formula to convert TF-IDF to normalized similarity
                raw_score_value = row.get("_score")
                raw_score = float(raw_score_value) if pd.notna(raw_score_value) else 0.0
                # Normalize TF-IDF score to [0, 1) range using x/(1+x) formula
                score = raw_score / (1.0 + raw_score)
                # Deserialize metadata from JSON string to dictionary
                metadata = deserialize_metadata(row.get("metadata"))
                search_results.append(
                    SearchResult(
                        doc_id=row["doc_id"],
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        score=score,
                        parse_hash=row["parse_hash"],
                        model_tag=model_tag,
                        created_at=row["created_at"],
                        metadata=metadata,
                    )
                )

            return _build_sparse_response(
                results=search_results,
                warnings=current_warnings,
                fts_enabled=_fts_enabled,
                query_text=query_text,
            )

        logger.warning(
            "FTS lookup returned no rows for query '%s'; falling back to substring match",
            query_text,
        )
        fallback_results = _substring_fallback(
            table=table,
            collection=collection,
            query_text=query_text,
            model_tag=model_tag,
            top_k=top_k,
            filters=filters,
            current_warnings=current_warnings,
        )

        return _build_sparse_response(
            results=fallback_results,
            warnings=current_warnings,
            fts_enabled=_fts_enabled,
            query_text=query_text,
        )

    except Exception as e:
        logger.error(
            "Sparse search failed for %s with query '%s': %s", model_tag, query_text, e
        )
        error_warnings = current_warnings + [
            SearchWarning(
                code="FTS_SEARCH_FAILED",
                message=f"An unexpected error occurred during sparse search: {str(e)}",
                fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                affected_models=[model_tag],
            )
        ]
        return _build_sparse_response(
            results=[],
            warnings=error_warnings,
            fts_enabled=_fts_enabled,
            query_text=query_text,
            status="failed",
        )
    finally:
        _safe_close_table(table)


def _substring_fallback(
    *,
    table: Any,
    collection: str,
    query_text: str,
    model_tag: str,
    top_k: int,
    filters: Optional[Dict[str, Any]],
    current_warnings: List[SearchWarning],
    batch_size: int = 2048,
) -> List[SearchResult]:
    """Perform a memory-friendly substring scan across the table when FTS misses."""

    desired_columns: Set[str] = {
        "collection",
        "doc_id",
        "chunk_id",
        "text",
        "parse_hash",
        "created_at",
        "metadata",
    }
    if filters:
        desired_columns.update(filters.keys())

    results: List[SearchResult] = []

    try:
        if hasattr(table, "to_batches"):
            batch_iter: Iterable[Any] = table.to_batches(
                columns=list(desired_columns), batch_size=batch_size
            )
        else:
            if pa is None:  # pragma: no cover - Safety guard when pyarrow missing
                raise ImportError(
                    "pyarrow is required for substring fallback when LanceDB table does not expose to_batches()."
                )
            arrow_table: PyArrowTable = table.to_arrow()  # type: ignore
            arrow_table = arrow_table.select(list(desired_columns))
            batch_iter = arrow_table.to_batches(max_chunksize=batch_size)
    except Exception as exc:  # noqa: BLE001
        logger.error("Substring fallback failed to read batches: %s", exc)
        return results

    for batch in batch_iter:
        batch_df = batch.to_pandas()

        mask = batch_df["collection"] == collection
        if filters:
            for key, value in filters.items():
                if key not in batch_df.columns:
                    continue
                if isinstance(value, (list, tuple, set)):
                    mask &= batch_df[key].isin(list(value))
                else:
                    mask &= batch_df[key] == value

        if not mask.any():
            continue

        text_mask = (
            batch_df["text"].astype(str).str.contains(query_text, na=False, regex=False)
        )
        mask &= text_mask

        if not mask.any():
            continue

        for _, row in batch_df.loc[mask].iterrows():
            # Deserialize metadata from JSON string to dictionary
            metadata = deserialize_metadata(row.get("metadata"))
            results.append(
                SearchResult(
                    doc_id=row["doc_id"],
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    score=1.0,
                    parse_hash=row["parse_hash"],
                    model_tag=model_tag,
                    created_at=row["created_at"],
                    metadata=metadata,
                )
            )
            if len(results) >= top_k:
                break

        if len(results) >= top_k:
            break

    if results:
        current_warnings.append(
            SearchWarning(
                code="FTS_FALLBACK",
                message=(
                    "Full-text index returned no matches; used substring search fallback. "
                    "Check FTS tokenizer configuration or update LanceDB to ensure proper tokenisation for query language."
                ),
                fallback_action=SearchFallbackAction.BRUTE_FORCE,
                affected_models=[model_tag],
            )
        )

    return results


def _build_sparse_response(
    *,
    results: List[SearchResult],
    warnings: List[SearchWarning],
    fts_enabled: bool,
    query_text: str,
    status: str = "success",
) -> SparseSearchResponse:
    """Helper to assemble `SparseSearchResponse`. Allows fallback reuse."""

    return SparseSearchResponse(
        results=results,
        total_count=len(results),
        status=status,
        warnings=warnings,
        fts_enabled=fts_enabled,
        query_text=query_text,
    )


# --- Async variant (Phase 1A Option C) ---


async def search_sparse_async(
    collection: str,
    model_tag: str,
    query_text: str,
    *,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> SparseSearchResponse:
    """Perform sparse retrieval using async vector store abstraction."""
    return await _get_legacy_step_compatibility_facade().search_sparse_async(
        collection=collection,
        model_tag=model_tag,
        query_text=query_text,
        top_k=top_k,
        filters=filters,
        readonly=readonly,
        nprobes=nprobes,
        refine_factor=refine_factor,
        user_id=user_id,
        is_admin=is_admin,
    )


async def _search_sparse_async_impl(
    collection: str,
    model_tag: str,
    query_text: str,
    *,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> SparseSearchResponse:
    """
    Perform sparse (Full-Text Search) retrieval using async vector store abstraction.

    This is the async variant of search_sparse. It uses VectorIndexStore.search_fts_async()
    instead of raw LanceDB connection for the main search path.

    Note: FTS index creation uses VectorIndexStore.create_index() for full decoupling.
    """
    vector_store = get_vector_index_store()

    _fts_enabled = False
    current_warnings: List[SearchWarning] = []

    if readonly:
        current_warnings.append(
            SearchWarning(
                code="READONLY_MODE",
                message=f"Readonly mode enabled for sparse search on {model_tag}. No FTS index operations will be performed.",
                fallback_action=SearchFallbackAction.REBUILD_INDEX,
                affected_models=[model_tag],
            )
        )

    try:
        # Check and create FTS index if needed (using storage abstraction layer)
        if not readonly:
            index_result_obj = vector_store.create_index(model_tag, readonly=False)
            _fts_enabled = index_result_obj.fts_enabled

        if not _fts_enabled:
            current_warnings.append(
                SearchWarning(
                    code="FTS_INDEX_MISSING",
                    message=f"FTS index may not be enabled on 'text' column for {model_tag}. Sparse search performance may be degraded.",
                    fallback_action=SearchFallbackAction.REBUILD_INDEX,
                    affected_models=[model_tag],
                )
            )

        # Convert API-facing dict filters into abstract FilterExpression
        filter_expr: Optional[FilterExpression] = None
        if collection or filters:
            conditions: List[FilterExpression] = []

            if collection:
                from ..storage.contracts import FilterCondition, FilterOperator

                conditions.append(
                    FilterCondition(
                        field="collection", operator=FilterOperator.EQ, value=collection
                    )
                )

            if filters:
                if isinstance(filters, dict):
                    parsed_filters = parse_legacy_filters(filters)
                    if parsed_filters is not None:
                        if isinstance(parsed_filters, tuple):
                            conditions.extend(parsed_filters)
                        else:
                            conditions.append(parsed_filters)
                elif isinstance(filters, (tuple, list)):
                    conditions.extend(
                        filters if isinstance(filters, tuple) else list(filters)
                    )
                else:
                    conditions.append(filters)

            if len(conditions) == 1:
                filter_expr = conditions[0]
            elif len(conditions) > 1:
                filter_expr = tuple(conditions)

        # Validate filter expression depth to prevent DoS
        if filter_expr is not None:
            validate_filter_depth(filter_expr)

        # Execute async FTS search using abstraction layer (by model_tag)
        raw_results = await vector_store.search_fts_by_model_async(
            model_tag=model_tag,
            query_text=query_text,
            top_k=top_k,
            filters=filter_expr,
            text_column_name="text",
        )

        if not raw_results:
            logger.warning(
                "FTS lookup returned no results for query '%s'; falling back to substring match",
                query_text,
            )
            # Use async iter_batches for fallback
            fallback_results = await _substring_fallback_async(
                model_tag=model_tag,
                collection=collection,
                query_text=query_text,
                top_k=top_k,
                filters=filters,
                current_warnings=current_warnings,
                user_id=user_id,
                is_admin=is_admin,
            )

            return _build_sparse_response(
                results=fallback_results,
                warnings=current_warnings,
                fts_enabled=_fts_enabled,
                query_text=query_text,
            )

        # Convert raw results to SearchResult objects
        search_results: List[SearchResult] = []
        for row in raw_results:
            # LanceDB FTS returns TF-IDF score (higher is better)
            raw_score_value = row.get("_score")
            raw_score = float(raw_score_value) if raw_score_value is not None else 0.0
            # Normalize TF-IDF score to [0, 1) range
            score = raw_score / (1.0 + raw_score)

            # Deserialize metadata
            metadata = deserialize_metadata(row.get("metadata"))

            search_results.append(
                SearchResult(
                    doc_id=row["doc_id"],
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    score=score,
                    parse_hash=row.get("parse_hash"),
                    model_tag=model_tag,
                    created_at=row.get("created_at"),
                    metadata=metadata,
                )
            )

        return _build_sparse_response(
            results=search_results,
            warnings=current_warnings,
            fts_enabled=_fts_enabled,
            query_text=query_text,
        )

    except Exception as e:
        logger.error(
            "Async sparse search failed for %s with query '%s': %s",
            model_tag,
            query_text,
            e,
        )
        error_warnings = current_warnings + [
            SearchWarning(
                code="FTS_SEARCH_FAILED",
                message=f"An unexpected error occurred during sparse search: {str(e)}",
                fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                affected_models=[model_tag],
            )
        ]
        return _build_sparse_response(
            results=[],
            warnings=error_warnings,
            fts_enabled=_fts_enabled,
            query_text=query_text,
            status="failed",
        )


async def _substring_fallback_async(
    *,
    model_tag: str,
    collection: str,
    query_text: str,
    top_k: int,
    filters: Optional[Dict[str, Any]],
    current_warnings: List[SearchWarning],
    user_id: Optional[int] = None,
    is_admin: bool = False,
    batch_size: int = 2048,
) -> List[SearchResult]:
    """Perform async substring scan using iter_batches_async when FTS misses."""

    vector_store = get_vector_index_store()
    results: List[SearchResult] = []

    # Build query filters
    query_filters: Dict[str, Any] = {"collection": collection}
    if filters:
        query_filters.update(filters)

    _table = None
    try:
        # Open embeddings table with legacy fallback
        _table, table_name = vector_store.open_embeddings_table(model_tag)

        # Use async batch iteration for memory-efficient scanning
        # Specify only required columns to minimize memory usage
        async for batch in cast(
            AsyncIterator[Any],
            vector_store.iter_batches_async(
                table_name=table_name,
                columns=[
                    "doc_id",
                    "chunk_id",
                    "text",
                    "parse_hash",
                    "created_at",
                    "metadata",
                ],
                batch_size=batch_size,
                filters=query_filters,
                user_id=user_id,
                is_admin=is_admin,
            ),
        ):
            batch_df = batch.to_pandas()

            # Apply substring filter
            text_mask = (
                batch_df["text"]
                .astype(str)
                .str.contains(query_text, na=False, regex=False)
            )
            matching_rows = batch_df[text_mask]

            # Early exit: stop processing if we already have enough results
            if len(results) >= top_k:
                break

            for _, row in matching_rows.iterrows():
                metadata = deserialize_metadata(row.get("metadata"))
                results.append(
                    SearchResult(
                        doc_id=row["doc_id"],
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        score=1.0,
                        parse_hash=row["parse_hash"],
                        model_tag=model_tag,
                        created_at=row["created_at"],
                        metadata=metadata,
                    )
                )

                # Early exit: stop as soon as we have enough results
                if len(results) >= top_k:
                    break

        if results:
            current_warnings.append(
                SearchWarning(
                    code="FTS_FALLBACK",
                    message=(
                        "Full-text index returned no matches; used async substring search fallback. "
                        "Check FTS tokenizer configuration or update LanceDB to ensure proper tokenisation for query language."
                    ),
                    fallback_action=SearchFallbackAction.BRUTE_FORCE,
                    affected_models=[model_tag],
                )
            )

    except Exception as exc:
        logger.error("Async substring fallback failed: %s", exc)
    finally:
        _safe_close_table(_table)

    return results
