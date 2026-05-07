"""
Core search engine implementation for dense vector retrieval.

This module provides the low-level search functionality that interacts
with the vector store abstraction layer for performing ANN searches.

Phase 1A Option C: Provides both sync and async search functions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..core.schemas import SearchResult
from ..storage.contracts import FilterExpression
from ..storage.factory import get_vector_index_store
from ..utils.filter_utils import parse_legacy_filters, validate_filter_depth
from ..utils.metadata_utils import deserialize_metadata

logger = logging.getLogger(__name__)


def search_dense_engine(
    collection: str,
    model_tag: str,
    query_vector: List[float],
    *,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Tuple[List[SearchResult], str, Optional[str]]:
    """
    Execute dense vector search against LanceDB embeddings table.

    Args:
        collection: Collection name for data isolation
        model_tag: Model tag to determine which embeddings table to search
        query_vector: Query vector for similarity search
        top_k: Number of top results to return
        filters: Optional filters to apply to the search
        readonly: If True, don't trigger index creation
        nprobes: Number of partitions to probe for ANN search (LanceDB specific).
        refine_factor: Refine factor for re-ranking results in memory (LanceDB specific).
        user_id: Optional user ID for multi-tenancy filtering.
        is_admin: Whether the user has admin privileges.

    Returns:
        Tuple of (search_results, index_status, index_advice)
    """
    try:
        vector_store = get_vector_index_store()

        # Check and create index if needed (using storage abstraction)
        index_result_obj = vector_store.create_index(model_tag, readonly)
        index_status = index_result_obj.status
        index_advice = index_result_obj.advice

        # Convert API-facing dict filters into abstract FilterExpression
        filter_expr: Optional[FilterExpression] = None
        if collection or filters:
            conditions: List[FilterExpression] = []

            if collection:
                from ..storage.contracts import FilterCondition, FilterOperator

                conditions.append(
                    FilterCondition(
                        field="collection",
                        operator=FilterOperator.EQ,
                        value=collection,
                    )
                )

            if filters:
                parsed = (
                    parse_legacy_filters(filters) if isinstance(filters, dict) else None
                )
                if parsed is not None:
                    if isinstance(parsed, tuple):
                        # Type narrowing: tuple of FilterConditions
                        # Cast to list for extend since tuple is also Iterable
                        conditions.extend(parsed)
                    else:
                        # Type narrowing: single FilterCondition
                        conditions.append(parsed)

            if len(conditions) == 1:
                filter_expr = conditions[0]
            elif len(conditions) > 1:
                filter_expr = tuple(conditions)

        # Validate filter expression depth to prevent DoS
        if filter_expr is not None:
            validate_filter_depth(filter_expr)

        # Execute vector search using abstraction layer (by model_tag)
        raw_results = vector_store.search_vectors_by_model(
            model_tag=model_tag,
            query_vector=query_vector,
            top_k=top_k,
            filters=filter_expr,
            vector_column_name="vector",
            user_id=user_id,
            is_admin=is_admin,
        )

        # OPTIMIZATION: Use list comprehension instead of iterrows()
        # Convert raw results to SearchResult objects
        search_results = []
        for row in raw_results:
            # LanceDB returns Squared Euclidean Distance (L_2^{2} distance),
            # lower is better, convert to similarity score (higher is better)
            # Using 1/(1+distance) formula to convert distance to similarity
            # Arrow/to_list() returns None instead of NaN, so direct None check is sufficient
            distance_value = row.get("_distance")
            distance = float(distance_value) if distance_value is not None else 0.0
            score = 1.0 / (1.0 + distance)

            # Deserialize metadata from JSON string to dictionary
            metadata = deserialize_metadata(row.get("metadata"))

            search_result = SearchResult(
                doc_id=row["doc_id"],
                chunk_id=row["chunk_id"],
                text=row["text"],
                score=score,
                parse_hash=row["parse_hash"],
                model_tag=model_tag,
                created_at=row["created_at"],
                metadata=metadata,
            )
            search_results.append(search_result)

        return search_results, index_status, index_advice

    except Exception as e:
        logger.error("Failed to execute dense search: %s", str(e))
        raise


# --- Async variant (Phase 1A Option C) ---


async def search_dense_engine_async(
    collection: str,
    model_tag: str,
    query_vector: List[float],
    *,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> Tuple[List[SearchResult], str, Optional[str]]:
    """
    Execute dense vector search using async vector store abstraction.

    This is the async variant of search_dense_engine. It uses the
    VectorIndexStore.search_vectors_async() method instead of raw
    LanceDB connection.

    Args:
        collection: Collection name for data isolation
        model_tag: Model tag to determine which embeddings table to search
        query_vector: Query vector for similarity search
        top_k: Number of top results to return
        filters: Optional filters to apply to the search
        readonly: If True, don't trigger index creation
        nprobes: Number of partitions to probe (passed to underlying store if supported)
        refine_factor: Refine factor for re-ranking (passed to underlying store if supported)
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges

    Returns:
        Tuple of (search_results, index_status, index_advice)
    """
    try:
        vector_store = get_vector_index_store()

        # Check and create index if needed (using storage abstraction)
        index_result_obj = vector_store.create_index(model_tag, readonly)
        index_status = index_result_obj.status
        index_advice = index_result_obj.advice

        # Convert API-facing dict filters into abstract FilterExpression
        filter_expr: Optional[FilterExpression] = None
        if collection or filters:
            conditions: List[FilterExpression] = []

            if collection:
                from ..storage.contracts import FilterCondition, FilterOperator

                conditions.append(
                    FilterCondition(
                        field="collection",
                        operator=FilterOperator.EQ,
                        value=collection,
                    )
                )

            if filters:
                parsed = (
                    parse_legacy_filters(filters) if isinstance(filters, dict) else None
                )
                if parsed is not None:
                    if isinstance(parsed, tuple):
                        conditions.extend(parsed)
                    else:
                        conditions.append(parsed)

            if len(conditions) == 1:
                filter_expr = conditions[0]
            elif len(conditions) > 1:
                filter_expr = tuple(conditions)

        # Validate filter expression depth to prevent DoS
        if filter_expr is not None:
            validate_filter_depth(filter_expr)

        # Execute async vector search using abstraction layer (by model_tag)
        raw_results = await vector_store.search_vectors_by_model_async(
            model_tag=model_tag,
            query_vector=query_vector,
            top_k=top_k,
            filters=filter_expr,
            vector_column_name="vector",
            user_id=user_id,
            is_admin=is_admin,
        )

        # Convert raw results to SearchResult objects
        search_results = []
        for row in raw_results:
            # LanceDB returns Squared Euclidean Distance (L_2^{2} distance)
            distance_value = row.get("_distance")
            distance = float(distance_value) if distance_value is not None else 0.0
            score = 1.0 / (1.0 + distance)

            # Deserialize metadata from JSON string to dictionary
            metadata = deserialize_metadata(row.get("metadata"))

            search_result = SearchResult(
                doc_id=row["doc_id"],
                chunk_id=row["chunk_id"],
                text=row["text"],
                score=score,
                parse_hash=row.get("parse_hash"),
                model_tag=model_tag,
                created_at=row.get("created_at"),
                metadata=metadata,
            )
            search_results.append(search_result)

        return search_results, index_status, index_advice

    except Exception as e:
        logger.error("Failed to execute async dense search: %s", str(e))
        raise
