"""
Dense vector search implementation for RAG retrieval.

This module provides the main entry point for dense vector search operations,
handling input validation and orchestrating the search execution.

Phase 1A Option C: Provides both sync and async search functions.
"""

import logging
from typing import Any, Dict, List, Optional

from ..core.exceptions import DocumentValidationError
from ..core.schemas import (
    DenseSearchResponse,
    IndexStatus,
    SearchFallbackAction,
    SearchWarning,
)
from ..vector_storage.vector_manager import validate_query_vector
from .search_engine import search_dense_engine

logger = logging.getLogger(__name__)


def search_dense(
    collection: str,
    model_tag: str,
    query_vector: List[float],
    *,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> DenseSearchResponse:
    """
    Execute dense vector search for RAG retrieval.

    This function performs input validation and orchestrates the dense vector
    search against the specified embeddings table, returning structured results.

    Args:
        collection: Collection name for data isolation
        model_tag: Model tag identifying which embeddings table to search
        query_vector: Query vector for similarity search
        top_k: Number of top results to return (default: 10)
        filters: Optional filters to apply to the search
        readonly: If True, don't trigger index operations
        nprobes: Number of partitions to probe for ANN search (LanceDB specific).
        refine_factor: Refine factor for re-ranking results in memory (LanceDB specific).
        user_id: Optional user ID for multi-tenancy filtering.
        is_admin: Whether the user has admin privileges (bypasses user filtering).

    Returns:
        DenseSearchResponse with search results and metadata. Returns a failed
        response with error warnings if an exception occurs.

    Raises:
        DocumentValidationError: If input validation fails
        VectorValidationError: If vector validation fails
    """
    # Input validation
    if not collection or not isinstance(collection, str):
        raise DocumentValidationError("Collection must be a non-empty string")

    if not model_tag or not isinstance(model_tag, str):
        raise DocumentValidationError("model_tag must be a non-empty string")

    if top_k <= 0 or top_k > 1000:
        raise DocumentValidationError("top_k must be between 1 and 1000")

    # Validate query vector (basic validation without DB connection)
    # Note: Dimension validation is handled by the storage abstraction layer during search
    validate_query_vector(query_vector)

    try:
        # Execute search using search engine
        search_results, index_status, index_advice = search_dense_engine(
            collection=collection,
            model_tag=model_tag,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            readonly=readonly,
            nprobes=nprobes,
            refine_factor=refine_factor,
            user_id=user_id,
            is_admin=is_admin,
        )

        # Map index status to enum
        index_status_enum = IndexStatus.INDEX_READY
        if index_status == "index_building":
            index_status_enum = IndexStatus.INDEX_BUILDING
        elif index_status == "no_index":
            index_status_enum = IndexStatus.NO_INDEX
        elif index_status == "index_corrupted":
            index_status_enum = IndexStatus.INDEX_CORRUPTED
        elif index_status == "readonly":
            index_status_enum = IndexStatus.READONLY
        elif index_status == "below_threshold":
            index_status_enum = IndexStatus.BELOW_THRESHOLD

        # Build response
        response = DenseSearchResponse(
            results=search_results,
            total_count=len(search_results),
            status="success",
            warnings=[],
            index_status=index_status_enum,
            index_advice=index_advice,
            # TODO: Generate idempotency_key based on search parameters hash
            # (collection, model_tag, query_vector, filters, top_k, nprobes, refine_factor)
            # for request deduplication, caching, and observability tracking.
            # Implementation planned for PR21 (caching strategy).
            idempotency_key=None,
            fallback_info=None,
            nprobes=nprobes,
            refine_factor=refine_factor,
        )

        logger.info(
            "Dense search completed: collection=%s, model_tag=%s, top_k=%s, returned=%s, index_status=%s",
            collection,
            model_tag,
            top_k,
            len(search_results),
            index_status,
        )

        return response

    except Exception as e:
        logger.error(
            "Dense search failed for %s in collection '%s': %s",
            model_tag,
            collection,
            e,
        )
        # Return structured error response instead of raising exception
        # This matches the behavior of search_sparse for API consistency
        return DenseSearchResponse(
            results=[],
            total_count=0,
            status="failed",
            warnings=[
                SearchWarning(
                    code="DENSE_SEARCH_FAILED",
                    message=f"An unexpected error occurred during dense search: {str(e)}",
                    fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                    affected_models=[model_tag],
                )
            ],
            index_status=IndexStatus.NO_INDEX,
            index_advice=None,
            idempotency_key=None,
            fallback_info=None,
            nprobes=nprobes,
            refine_factor=refine_factor,
        )


# --- Async variant (Phase 1A Option C) ---


async def search_dense_async(
    collection: str,
    model_tag: str,
    query_vector: List[float],
    *,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    readonly: bool = False,
    nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> DenseSearchResponse:
    """
    Execute dense vector search using async vector store abstraction.

    This is the async variant of search_dense. It performs the same input
    validation but uses search_dense_engine_async() internally.

    Args:
        collection: Collection name for data isolation
        model_tag: Model tag identifying which embeddings table to search
        query_vector: Query vector for similarity search
        top_k: Number of top results to return (default: 10)
        filters: Optional filters to apply to the search
        readonly: If True, don't trigger index operations
        nprobes: Number of partitions to probe for ANN search (LanceDB specific).
        refine_factor: Refine factor for re-ranking results in memory (LanceDB specific).
        user_id: Optional user ID for multi-tenancy filtering.
        is_admin: Whether the user has admin privileges (bypasses user filtering).

    Returns:
        DenseSearchResponse with search results and metadata. Returns a failed
        response with error warnings if an exception occurs.

    Raises:
        DocumentValidationError: If input validation fails
        VectorValidationError: If vector validation fails
    """
    # Input validation (same as sync version)
    if not collection or not isinstance(collection, str):
        raise DocumentValidationError("Collection must be a non-empty string")

    if not model_tag or not isinstance(model_tag, str):
        raise DocumentValidationError("model_tag must be a non-empty string")

    if top_k <= 0 or top_k > 1000:
        raise DocumentValidationError("top_k must be between 1 and 1000")

    # Validate query vector (basic validation without DB connection)
    # Note: Dimension validation is handled by the storage abstraction layer during search
    validate_query_vector(query_vector)

    # Import async search engine
    from .search_engine import search_dense_engine_async

    try:
        # Execute async search
        search_results, index_status, index_advice = await search_dense_engine_async(
            collection=collection,
            model_tag=model_tag,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            readonly=readonly,
            nprobes=nprobes,
            refine_factor=refine_factor,
            user_id=user_id,
            is_admin=is_admin,
        )

        # Map index status to enum
        index_status_enum = IndexStatus.INDEX_READY
        if index_status == "index_building":
            index_status_enum = IndexStatus.INDEX_BUILDING
        elif index_status == "no_index":
            index_status_enum = IndexStatus.NO_INDEX
        elif index_status == "index_corrupted":
            index_status_enum = IndexStatus.INDEX_CORRUPTED
        elif index_status == "readonly":
            index_status_enum = IndexStatus.READONLY
        elif index_status == "below_threshold":
            index_status_enum = IndexStatus.BELOW_THRESHOLD

        # Build response
        response = DenseSearchResponse(
            results=search_results,
            total_count=len(search_results),
            status="success",
            warnings=[],
            index_status=index_status_enum,
            index_advice=index_advice,
            idempotency_key=None,
            fallback_info=None,
            nprobes=nprobes,
            refine_factor=refine_factor,
        )

        logger.info(
            "Async dense search completed: collection=%s, model_tag=%s, top_k=%s, returned=%s, index_status=%s",
            collection,
            model_tag,
            top_k,
            len(search_results),
            index_status,
        )

        return response

    except Exception as e:
        logger.error(
            "Async dense search failed for %s in collection '%s': %s",
            model_tag,
            collection,
            e,
        )
        # Return structured error response instead of raising exception
        # This matches the behavior of search_sparse for API consistency
        return DenseSearchResponse(
            results=[],
            total_count=0,
            status="failed",
            warnings=[
                SearchWarning(
                    code="DENSE_SEARCH_FAILED",
                    message=f"An unexpected error occurred during dense search: {str(e)}",
                    fallback_action=SearchFallbackAction.PARTIAL_RESULTS,
                    affected_models=[model_tag],
                )
            ],
            index_status=IndexStatus.NO_INDEX,
            index_advice=None,
            idempotency_key=None,
            fallback_info=None,
            nprobes=nprobes,
            refine_factor=refine_factor,
        )
