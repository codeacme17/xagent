"""List candidates functionality for version management.

This module provides functionality for listing candidate versions
across different processing stages (parse, chunk, embed).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..core.exceptions import DatabaseOperationError, VersionManagementError
from ..core.schemas import StepType
from ..LanceDB.schema_manager import _safe_close_table
from ..storage.factory import get_vector_store_raw_connection
from ..utils.lancedb_query_utils import list_table_names, query_to_list
from ..utils.string_utils import build_lancedb_filter_expression

if TYPE_CHECKING:
    from ..kb import KBVersionCompatibilityFacade


def _get_version_compatibility_facade() -> "KBVersionCompatibilityFacade":
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().version_compatibility


def _resolve_step_type(step_type_input: Union[StepType, str]) -> StepType:
    """
    Resolves the step type, converting string inputs to StepType enum members.

    Args:
        step_type_input: The input step type, which can be a StepType enum or a string.

    Returns:
        The resolved StepType enum member.

    Raises:
        VersionManagementError: If the input string does not correspond to a valid
                                  StepType member, or if the input type is unsupported.
    """
    if isinstance(step_type_input, StepType):
        return step_type_input
    elif isinstance(step_type_input, str):
        try:
            return StepType(step_type_input)
        except ValueError:
            raise VersionManagementError(
                f"Invalid step_type string: '{step_type_input}'. Expected one of: "
                + ", ".join(["'" + s.value + "'" for s in StepType])
            )
    else:
        raise VersionManagementError(
            f"Unsupported step_type type: {type(step_type_input)}. Expected StepType or str."
        )


def _query_table(
    connection: Any,
    table_name: str,
    filters: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run a where query on a table with given filters and return List[Dict].

    This small helper unifies filter building and querying across candidate getters.
    It safely escapes filter values to prevent injection attacks.
    Uses unified query_to_list() with three-tier fallback for maximum compatibility.

    Args:
        connection: LanceDB connection
        table_name: Name of the table to query
        filters: Dictionary of filters for the table

    Returns:
        List of dictionaries representing query results. Empty list if table doesn't exist or no results.
    """
    if table_name not in list_table_names(connection):
        return []
    table = None
    try:
        table = connection.open_table(table_name)

        filter_expr = build_lancedb_filter_expression(filters)
        return query_to_list(table.search().where(filter_expr))
    finally:
        _safe_close_table(table)


def _generate_semantic_id(
    step_type: StepType, technical_id: str, params: Optional[Dict[str, Any]] = None
) -> str:
    """Generate a semantic ID from technical ID and parameters.

    Args:
        step_type: Processing stage type (parse, chunk, embed)
        technical_id: Technical identifier (hash)
        params: Optional parameters for context

    Returns:
        Semantic identifier
    """
    # For now, use a simple format based on step_type and hash prefix
    # In the future, this could be more sophisticated based on actual parameters
    hash_prefix = technical_id[:8] if technical_id else "unknown"

    if step_type == StepType.PARSE:
        method = params.get("parse_method", "unknown") if params else "unknown"
        return f"parse_{method}_{hash_prefix}"
    elif step_type == StepType.CHUNK:
        strategy = params.get("chunk_strategy", "unknown") if params else "unknown"
        size = params.get("chunk_size", "unknown") if params else "unknown"
        return f"chunk_{strategy}_{size}_{hash_prefix}"
    elif step_type == StepType.EMBED:
        model = params.get("model", "unknown") if params else "unknown"
        return f"embed_{model}_{hash_prefix}"
    else:
        return f"{step_type.value}_{hash_prefix}"


def _get_parse_candidates(
    connection: Any,
    filters: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Get parse candidates from the database.

    Args:
        connection: LanceDB connection
        filters: Dictionary of filters for the parse table

    Returns:
        List of parse candidate dictionaries
    """
    result = _query_table(connection, "parses", filters)
    if not result:
        return []
    parse_candidates: List[Dict[str, Any]] = []
    for row in result:
        # NOTE: This legacy path shares the same latent bug as #705 (the parses
        # table has no top-level parse_method column, so this resolves to
        # "unknown"). It is intentionally left unfixed because it is unreachable:
        # the only callers are the KBVersionCompatibilityFacade fallbacks
        # (coordinator is None), and the facade is always constructed with a
        # coordinator. The live path is LanceDBVectorIndexStore._vis_get_parse_candidates.
        params = {
            "parse_method": row.get("parse_method", "unknown"),
            "parser": row.get("parser", "unknown"),
        }
        semantic_id = _generate_semantic_id(StepType.PARSE, row["parse_hash"], params)
        stats = {
            "paragraphs_count": 0,
            "elapsed_ms": 0,
            "parse_method": row.get("parse_method", "unknown"),
            "parser": row.get("parser", "unknown"),
        }
        parse_candidates.append(
            {
                "semantic_id": semantic_id,
                "technical_id": row["parse_hash"],
                "params_brief": params,
                "stats": stats,
                "state": "candidate",
                "created_at": row.get(
                    "created_at", datetime.now(timezone.utc).replace(tzinfo=None)
                ),
                "operator": "unknown",
            }
        )
    return parse_candidates


def _get_chunk_candidates(
    connection: Any,
    filters: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Get chunk candidates from the database.

    Args:
        connection: LanceDB connection
        filters: Dictionary of filters for the chunk table

    Returns:
        List of chunk candidate dictionaries
    """
    result = _query_table(connection, "chunks", filters)
    if not result:
        return []
    chunk_configs: Dict[str, Dict[str, Any]] = {}
    for row in result:
        parse_hash = row["parse_hash"]
        if parse_hash not in chunk_configs:
            chunk_configs[parse_hash] = {
                "chunk_count": 0,
                "avg_length": 0,
                "created_at": row.get(
                    "created_at", datetime.now(timezone.utc).replace(tzinfo=None)
                ),
            }
        chunk_configs[parse_hash]["chunk_count"] += 1
        text_len = len(row.get("text", ""))
        cfg = chunk_configs[parse_hash]
        cfg["avg_length"] = (
            cfg["avg_length"] * (cfg["chunk_count"] - 1) + text_len
        ) / cfg["chunk_count"]

    chunk_candidates: List[Dict[str, Any]] = []
    for parse_hash, cfg in chunk_configs.items():
        semantic_id = _generate_semantic_id(
            StepType.CHUNK, parse_hash, {"chunk_count": cfg["chunk_count"]}
        )
        stats = {
            "chunks_count": cfg["chunk_count"],
            "avg_length": int(cfg["avg_length"]),
            "parse_hash": parse_hash,
        }
        chunk_candidates.append(
            {
                "semantic_id": semantic_id,
                "technical_id": parse_hash,
                "params_brief": {"chunk_count": cfg["chunk_count"]},
                "stats": stats,
                "state": "candidate",
                "created_at": cfg["created_at"],
                "operator": "unknown",
            }
        )
    return chunk_candidates


def _get_embed_candidates(
    connection: Any,
    filters: Dict[str, Any],
    model_tag: str,
) -> List[Dict[str, Any]]:
    """Get embedding candidates from the database.

    Args:
        connection: LanceDB connection
        filters: Dictionary of filters for the embeddings table
        model_tag: Model tag for the embeddings table

    Returns:
        List of embedding candidate dictionaries
    """
    table_names = list_table_names(connection)
    embed_tables = [name for name in table_names if name.startswith("embeddings_")]
    if not embed_tables:
        return []
    embed_candidates: List[Dict[str, Any]] = []
    for table_name in embed_tables:
        table_model_tag = table_name.replace("embeddings_", "")
        if model_tag != table_model_tag:
            continue
        result = _query_table(connection, table_name, filters)
        if not result:
            continue
        embed_configs: Dict[tuple[str, str], Dict[str, Any]] = {}
        for row in result:
            model = row.get("model", "unknown")
            parse_hash = row.get("parse_hash", "unknown")
            key = (model, parse_hash)
            if key not in embed_configs:
                embed_configs[key] = {
                    "vector_count": 0,
                    "vector_dim": 0,
                    "created_at": row.get(
                        "created_at", datetime.now(timezone.utc).replace(tzinfo=None)
                    ),
                }
            embed_configs[key]["vector_count"] += 1
            vector = row.get("vector", [])
            if vector and embed_configs[key]["vector_dim"] == 0:
                embed_configs[key]["vector_dim"] = len(vector)

        for (model, parse_hash), cfg in embed_configs.items():
            semantic_id = _generate_semantic_id(
                StepType.EMBED,
                parse_hash,
                {"model": model, "model_tag": model_tag},
            )
            stats = {
                "upsert_count": cfg["vector_count"],
                "vector_dim": cfg["vector_dim"],
                "model": model,
                "model_tag": model_tag,
                "parse_hash": parse_hash,
            }
            embed_candidates.append(
                {
                    "semantic_id": semantic_id,
                    "technical_id": parse_hash,
                    "params_brief": {
                        "model": model,
                        "model_tag": model_tag,
                    },
                    "stats": stats,
                    "state": "candidate",
                    "created_at": cfg["created_at"],
                    "operator": "unknown",
                }
            )
    return embed_candidates


def _get_candidates(
    connection: Any,
    step_type: StepType,
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Unified candidate getter by step_type.

    Internally applies minimal branching and reuses common helpers.
    """
    try:
        filters = {"collection": collection, "doc_id": doc_id}

        if step_type == StepType.PARSE:
            return _get_parse_candidates(connection, filters)

        if step_type == StepType.CHUNK:
            return _get_chunk_candidates(connection, filters)

        if step_type == StepType.EMBED:
            if not model_tag:
                raise VersionManagementError("model_tag is required for embed step")
            return _get_embed_candidates(connection, filters, model_tag)

        # Handle invalid step_type
        step_type_str = step_type.value
        raise VersionManagementError(f"Unknown step_type: {step_type_str}")
    except Exception as e:
        raise DatabaseOperationError(f"Failed to get candidates: {e}") from e


def list_candidates(
    collection: str,
    doc_id: str,
    step_type: Union[StepType, str],
    model_tag: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50,
    order_by: str = "created_at desc",
) -> Dict[str, Any]:
    return _get_version_compatibility_facade().list_candidates(
        collection=collection,
        doc_id=doc_id,
        step_type=step_type,
        model_tag=model_tag,
        state=state,
        limit=limit,
        order_by=order_by,
    )


def _list_candidates_impl(
    collection: str,
    doc_id: str,
    step_type: Union[StepType, str],
    model_tag: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50,
    order_by: str = "created_at desc",
) -> Dict[str, Any]:
    """List candidate versions for a specific document and processing stage.

    Args:
        collection: Collection name
        doc_id: Document ID
        step_type: Processing stage type (StepType enum: PARSE, CHUNK, EMBED) or its string representation.
        model_tag: Model tag for embed stage (optional)
        state: State filter (experimental, candidate, main) (optional)
        limit: Maximum number of candidates to return (default: 50)
        order_by: Sort order (default: "created_at desc")

    Returns:
        Dictionary containing candidates list and metadata

    Raises:
        DatabaseOperationError: If database connection or operation fails
        VersionManagementError: If there's a version management logic error
    """
    resolved_step_type = _resolve_step_type(step_type)
    try:
        # Get LanceDB connection from environment (uses default path if LANCEDB_DIR not set)
        connection = get_vector_store_raw_connection()

        # Get candidates based on step_type
        candidates = _get_candidates(
            connection=connection,
            step_type=resolved_step_type,
            collection=collection,
            doc_id=doc_id,
            model_tag=model_tag,
        )

        # Apply state filter if specified
        if state is not None:
            candidates = [c for c in candidates if c["state"] == state]

        # Record total count before limit
        total_count = len(candidates)

        # Sort by order_by (must happen before limit)
        if order_by == "created_at desc":
            candidates.sort(key=lambda x: x["created_at"], reverse=True)
        elif order_by == "created_at asc":
            candidates.sort(key=lambda x: x["created_at"], reverse=False)

        # Apply limit after sorting
        if limit > 0:
            candidates = candidates[:limit]

        return {
            "candidates": candidates,
            "total_count": total_count,
            "returned_count": len(candidates),
            "step_type": resolved_step_type.value,
            "model_tag": model_tag,
            "filters": {"state": state, "limit": limit, "order_by": order_by},
        }

    except (DatabaseOperationError, VersionManagementError):
        # Re-raise known exceptions without wrapping
        raise
    except Exception as e:
        # Wrap unknown exceptions in VersionManagementError
        raise VersionManagementError(f"Failed to list candidates: {e}") from e
