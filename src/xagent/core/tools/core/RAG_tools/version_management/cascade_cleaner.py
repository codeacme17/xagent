"""Cascade cleanup functions for version management.

Provide cascade cleanup utilities when promoting main versions,
ensuring data consistency across processing stages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from typing_extensions import Literal

from ..core.exceptions import CascadeCleanupError
from ..kb.cleanup_filters import (
    append_user_filter_for_table,
    append_user_filter_without_schema,
    build_embedding_cleanup_filters_from_base,
    select_embedding_tables,
)
from ..kb.cleanup_filters import table_has_column as _table_has_column
from ..LanceDB.schema_manager import (
    _safe_close_table,
    ensure_chunks_table,
    ensure_documents_table,
    ensure_ingestion_runs_table,
    ensure_main_pointers_table,
    ensure_parses_table,
)
from ..storage.factory import get_vector_store_raw_connection
from ..utils.lancedb_query_utils import _safe_count_rows, list_table_names
from ..utils.string_utils import (
    build_lancedb_filter_expression,
    build_user_id_filter_for_table,
    escape_lancedb_string,
)
from ..utils.user_permissions import UserPermissions
from ..utils.user_scope import resolve_user_scope

logger = logging.getLogger(__name__)

# NOTE (#494): The cascade-cleanup predicate pipeline now lives authoritatively in
# the vector_index_store (LanceDBVectorIndexStore.cleanup_cascade_by_scope). The
# COLLECTION-DELETE path below (_cascade_delete_impl / cascade_delete /
# cascade_delete_documents / _build_collection_filter) intentionally keeps its own
# copy for now; that duplication is tracked by #494. The version-cleanup
# _cleanup_cascade_impl is now a thin shell that routes through the
# version-compatibility facade (-> coordinator -> handle -> store).

if TYPE_CHECKING:
    from ..kb import KBVersionCompatibilityFacade


def _get_version_compatibility_facade() -> "KBVersionCompatibilityFacade":
    from ..kb import get_kb_coordinator

    return get_kb_coordinator().version_compatibility


FilterPredicateMap = Dict[str, list[str]]


def _replace_predicate(
    predicates: FilterPredicateMap, table_name: str, filter_expr: str
) -> None:
    _replace_predicates(predicates, table_name, [filter_expr])


def _replace_predicates(
    predicates: FilterPredicateMap, table_name: str, filter_exprs: list[str]
) -> None:
    predicates[table_name] = list(filter_exprs)


def _append_predicates(
    predicates: FilterPredicateMap, table_name: str, filter_exprs: list[str]
) -> None:
    predicates.setdefault(table_name, []).extend(filter_exprs)


def _count_rows_by_filters(table: Any, filter_exprs: list[str]) -> int:
    return sum(_safe_count_rows(table, filter_expr) for filter_expr in filter_exprs)


def _delete_rows_by_filters(table: Any, filter_exprs: list[str]) -> int:
    deleted_count = 0
    for filter_expr in filter_exprs:
        count = _safe_count_rows(table, filter_expr)
        if count > 0:
            table.delete(filter_expr)
        deleted_count += count
    return deleted_count


def _build_collection_filter(
    *,
    conn: Any,
    table_name: str,
    collection: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Build a safe filter for collection-scoped deletion.

    Adds user_id filtering only when the target table contains a user_id column.
    """
    base: Dict[str, str] = {"collection": collection}
    table = None
    try:
        table = conn.open_table(table_name)
        if not is_admin and user_id is not None:
            if _table_has_column(table, "user_id"):
                base_expr = build_lancedb_filter_expression(
                    base, user_id=user_id, is_admin=is_admin, skip_user_filter=True
                )
                user_expr = build_user_id_filter_for_table(table, int(user_id))
                return f"{base_expr} AND {user_expr}"
            # Legacy schemas without user_id must remain compatible.
            return build_lancedb_filter_expression(base, skip_user_filter=True)
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)
    except Exception:
        # If table introspection fails, keep tenant-safe fallback.
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)
    finally:
        _safe_close_table(table)


def _build_document_filter(
    *,
    conn: Any,
    table_name: str,
    collection: str,
    doc_id: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Build a safe filter for document-scoped deletion."""
    base: Dict[str, str] = {"collection": collection, "doc_id": doc_id}
    table = None
    try:
        table = conn.open_table(table_name)
        if not is_admin and user_id is not None:
            if _table_has_column(table, "user_id"):
                base_expr = build_lancedb_filter_expression(
                    base, user_id=user_id, is_admin=is_admin, skip_user_filter=True
                )
                user_expr = build_user_id_filter_for_table(table, int(user_id))
                return f"{base_expr} AND {user_expr}"
            # Legacy schemas without user_id must remain compatible.
            return build_lancedb_filter_expression(base, skip_user_filter=True)
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)
    except Exception:
        # If table introspection fails, keep tenant-safe fallback.
        return build_lancedb_filter_expression(base, user_id=user_id, is_admin=is_admin)
    finally:
        _safe_close_table(table)


def _doc_ids_filter(doc_ids: list[str]) -> str:
    if len(doc_ids) == 1:
        return f"doc_id == '{escape_lancedb_string(doc_ids[0])}'"
    values = ", ".join(f"'{escape_lancedb_string(doc_id)}'" for doc_id in doc_ids)
    return f"doc_id IN ({values})"


def _build_documents_filter(
    *,
    conn: Any,
    table_name: str,
    collection: str,
    doc_ids: list[str],
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Build a safe filter for deleting multiple document-scoped rows."""
    base_expr = build_lancedb_filter_expression(
        {"collection": collection}, skip_user_filter=True
    )
    doc_expr = _doc_ids_filter(doc_ids)
    scoped_expr = f"{base_expr} AND {doc_expr}"

    if is_admin:
        return scoped_expr
    if user_id is None:
        return f"{scoped_expr} AND ({UserPermissions.get_no_access_filter()})"

    table = None
    try:
        table = conn.open_table(table_name)
        if _table_has_column(table, "user_id"):
            return (
                f"{scoped_expr} AND "
                f"{build_user_id_filter_for_table(table, int(user_id))}"
            )
        # Legacy schemas without user_id stay document-scoped by doc_id.
        return scoped_expr
    except Exception:
        # If introspection fails, fail closed with an explicit user_id predicate.
        return f"{scoped_expr} AND user_id == {int(user_id)}"
    finally:
        _safe_close_table(table)


def _append_user_filter_if_needed(
    *,
    conn: Any,
    table_name: str,
    base_expr: str,
    user_id: Optional[int],
    is_admin: bool,
) -> str:
    """Append user_id filter when non-admin and table contains user_id."""
    table = None
    try:
        table = conn.open_table(table_name)
        return append_user_filter_for_table(
            table=table,
            filter_expr=base_expr,
            user_id=user_id,
            is_admin=is_admin,
        )
    except Exception:
        return append_user_filter_without_schema(
            filter_expr=base_expr,
            user_id=user_id,
            is_admin=is_admin,
        )
    finally:
        _safe_close_table(table)


def _replace_embedding_predicates(
    *,
    predicates: FilterPredicateMap,
    conn: Any,
    base_expr: str,
    user_id: Optional[int],
    is_admin: bool,
    model_tag: Optional[str] = None,
) -> None:
    """Expand an embeddings cleanup predicate per target embeddings table."""
    table_filters = build_embedding_cleanup_filters_from_base(
        conn,
        base_filter=base_expr,
        user_id=user_id,
        is_admin=is_admin,
        model_tag=model_tag,
    )
    for table_name, filter_exprs in table_filters.items():
        if filter_exprs:
            _replace_predicates(predicates, table_name, filter_exprs)


def _get_table_names(conn: Any) -> list[str]:
    """Get table names from LanceDB connection with mypy-safe access."""
    try:
        return list_table_names(conn)
    except Exception:
        return []


def _plan_by_predicates(
    conn: Any, table_to_filter: FilterPredicateMap, model_tag: Optional[str] = None
) -> Dict[str, int]:
    """Count rows that match each table predicate without deleting.

    Args:
        conn: LanceDB connection
        table_to_filter: Mapping of table name -> filter expressions
        model_tag: Optional model tag to filter embeddings tables. If specified,
                   only the embeddings table matching this model will be counted.

    Returns:
        Mapping of table name -> matched row count
    """
    counts: Dict[str, int] = {}
    table_names = _get_table_names(conn)

    # If predicates explicitly include embeddings tables, plan them first.
    for t in table_names:
        if t.startswith("embeddings_") and t in table_to_filter:
            table = None
            try:
                table = conn.open_table(t)
                counts[t] = _count_rows_by_filters(table, table_to_filter[t])
            finally:
                _safe_close_table(table)

    for table_name, filter_exprs in table_to_filter.items():
        # Special fan-out handling for embeddings preview like deleter
        if table_name == "__embeddings__":
            total = 0
            target_tables = select_embedding_tables(conn, model_tag=model_tag)
            for t in target_tables:
                table = None
                try:
                    table = conn.open_table(t)
                    total += _count_rows_by_filters(table, filter_exprs)
                finally:
                    _safe_close_table(table)
            counts[table_name] = total
            continue
        if table_name.startswith("embeddings_"):
            continue

        if table_name not in table_names:
            counts[table_name] = 0
            continue
        table = None
        try:
            table = conn.open_table(table_name)
            count = _count_rows_by_filters(table, filter_exprs)
            counts[table_name] = count
        finally:
            _safe_close_table(table)
    return counts


def _delete_by_predicates(
    conn: Any, table_to_filter: FilterPredicateMap, model_tag: Optional[str] = None
) -> Dict[str, int]:
    """Delete rows by table predicates in a fixed, safe order.

    Order: embeddings_* -> chunks -> parses -> main_pointers -> documents
    Unknown tables are executed after the known order, in given insertion order.

    Args:
        conn: LanceDB connection
        table_to_filter: Dictionary mapping table names to filter expressions
        model_tag: Optional model tag to filter embeddings tables. If specified,
                   only the embeddings table matching this model will be processed.
    """
    deleted: Dict[str, int] = {}
    table_names = _get_table_names(conn)

    # If predicates explicitly include embeddings tables, delete them first.
    for t in table_names:
        if not t.startswith("embeddings_") or t not in table_to_filter:
            continue
        filter_exprs = table_to_filter[t]
        table = None
        try:
            table = conn.open_table(t)
            cnt = _delete_rows_by_filters(table, filter_exprs)
            if cnt > 0:
                logger.info("Cascade cleanup: deleted %s rows from %s", cnt, t)
            deleted[t] = cnt
        finally:
            _safe_close_table(table)

    order = [
        # embeddings handled specially below (fan-out across many tables)
        "__embeddings__",
        "chunks",
        "parses",
        "main_pointers",
        "ingestion_runs",
        "documents",
    ]

    # First handle embeddings fan-out
    if "__embeddings__" in table_to_filter:
        filter_exprs = table_to_filter["__embeddings__"]
        total = 0

        target_tables = select_embedding_tables(conn, model_tag=model_tag)

        for t in target_tables:
            table = None
            try:
                table = conn.open_table(t)
                total += _delete_rows_by_filters(table, filter_exprs)
            finally:
                _safe_close_table(table)
        deleted["embeddings"] = total
        if total > 0:
            logger.info(
                "Cascade cleanup: deleted %s rows from embeddings tables", total
            )

    # Then handle known tables
    for name in order[1:]:
        if name in table_to_filter and name in table_names:
            table = None
            try:
                table = conn.open_table(name)
                cnt = _delete_rows_by_filters(table, table_to_filter[name])
                if cnt > 0:
                    logger.info("Cascade cleanup: deleted %s rows from %s", cnt, name)
                deleted[name] = cnt
            finally:
                _safe_close_table(table)

    # Finally, handle any remaining custom tables once
    for name, filter_exprs in table_to_filter.items():
        if name in (
            "__embeddings__",
            "chunks",
            "parses",
            "main_pointers",
            "ingestion_runs",
            "documents",
        ) or name.startswith("embeddings_"):
            continue
        if name not in table_names:
            deleted[name] = 0
            continue
        table = None
        try:
            table = conn.open_table(name)
            cnt = _delete_rows_by_filters(table, filter_exprs)
            if cnt > 0:
                logger.info("Cascade cleanup: deleted %s rows from %s", cnt, name)
            deleted[name] = cnt
        finally:
            _safe_close_table(table)

    return deleted


def _should_execute_delete(*, preview_only: bool, confirm: bool) -> bool:
    """Return whether a destructive cleanup should execute."""
    return bool(confirm and not preview_only)


def _execute_or_plan_by_predicates(
    conn: Any,
    predicates: FilterPredicateMap,
    *,
    preview_only: bool,
    confirm: bool,
    model_tag: Optional[str] = None,
) -> Dict[str, int]:
    """Plan unless deletion is explicitly confirmed outside preview mode."""
    if not _should_execute_delete(preview_only=preview_only, confirm=confirm):
        return _plan_by_predicates(conn, predicates, model_tag=model_tag)
    return _delete_by_predicates(conn, predicates, model_tag=model_tag)


def cascade_delete(
    *,
    target: Literal["collection", "document"],
    collection: str,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    model_tag: Optional[str] = None,
    preview_only: bool = True,
    confirm: bool = False,
    conn: Any | None = None,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cascade_delete(
        target=target,
        collection=collection,
        doc_id=doc_id,
        user_id=user_id,
        is_admin=is_admin,
        model_tag=model_tag,
        preview_only=preview_only,
        confirm=confirm,
        conn=conn,
    )


def _cascade_delete_impl(
    *,
    target: Literal["collection", "document"],
    collection: str,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    model_tag: Optional[str] = None,
    preview_only: bool = True,
    confirm: bool = False,
    conn: Any | None = None,
) -> Dict[str, int]:
    """Unified cascade delete for collection or document targets.

    This is intended for user-facing destructive operations (e.g. KB delete)
    and is separate from version promotion cleanup scopes.

    Args:
        target: "collection" or "document".
        collection: Collection name.
        doc_id: Required when target == "document".
        user_id: Optional user ID for multi-tenancy filtering.
        is_admin: Whether the caller is an admin (None to fallback to context).
        model_tag: Optional embeddings model tag limiter.
        preview_only: If True, only plan counts.
        confirm: If True, execute deletions.

    Returns:
        Mapping of table name -> deleted (or planned) row count.
    """
    user_scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
    user_id = user_scope.user_id
    is_admin = user_scope.is_admin

    if target == "document" and not doc_id:
        raise CascadeCleanupError("doc_id is required for document cascade delete")

    if conn is None:
        conn = get_vector_store_raw_connection()
    ensure_documents_table(conn)
    ensure_parses_table(conn)
    ensure_chunks_table(conn)
    ensure_main_pointers_table(conn)
    ensure_ingestion_runs_table(conn)

    table_names = _get_table_names(conn)
    predicates: FilterPredicateMap = {}

    core_tables = ["documents", "parses", "chunks", "main_pointers", "ingestion_runs"]
    for table_name in core_tables:
        if table_name not in table_names:
            continue
        if target == "collection":
            filter_expr = _build_collection_filter(
                conn=conn,
                table_name=table_name,
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )
        else:
            filter_expr = _build_document_filter(
                conn=conn,
                table_name=table_name,
                collection=collection,
                doc_id=str(doc_id),
                user_id=user_id,
                is_admin=is_admin,
            )
        _replace_predicate(predicates, table_name, filter_expr)

    for table_name in select_embedding_tables(conn, model_tag=model_tag):
        if target == "collection":
            filter_expr = _build_collection_filter(
                conn=conn,
                table_name=table_name,
                collection=collection,
                user_id=user_id,
                is_admin=is_admin,
            )
        else:
            filter_expr = _build_document_filter(
                conn=conn,
                table_name=table_name,
                collection=collection,
                doc_id=str(doc_id),
                user_id=user_id,
                is_admin=is_admin,
            )
        _replace_predicate(predicates, table_name, filter_expr)

    return _execute_or_plan_by_predicates(
        conn,
        predicates,
        preview_only=preview_only,
        confirm=confirm,
        model_tag=None,
    )


def cascade_delete_documents(
    *,
    collection: str,
    doc_ids: list[str],
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    preview_only: bool = True,
    confirm: bool = False,
    conn: Any | None = None,
) -> Dict[str, int]:
    """Cascade delete several documents using one predicate set.

    This keeps non-admin deletes document-scoped even for legacy tables that do
    not expose user_id, avoiding collection-wide mutation while reducing the
    number of full cascade passes.
    """
    normalized_doc_ids = sorted({str(doc_id) for doc_id in doc_ids if str(doc_id)})
    if not normalized_doc_ids:
        return {}

    user_scope = resolve_user_scope(user_id=user_id, is_admin=is_admin)
    user_id = user_scope.user_id
    is_admin = user_scope.is_admin
    if not is_admin and user_id is None:
        return {}

    if conn is None:
        conn = get_vector_store_raw_connection()
    ensure_documents_table(conn)
    ensure_parses_table(conn)
    ensure_chunks_table(conn)
    ensure_main_pointers_table(conn)
    ensure_ingestion_runs_table(conn)

    table_names = _get_table_names(conn)
    predicates: FilterPredicateMap = {}

    core_tables = ["documents", "parses", "chunks", "main_pointers", "ingestion_runs"]
    for table_name in core_tables:
        if table_name not in table_names:
            continue
        filter_expr = _build_documents_filter(
            conn=conn,
            table_name=table_name,
            collection=collection,
            doc_ids=normalized_doc_ids,
            user_id=user_id,
            is_admin=is_admin,
        )
        _replace_predicate(predicates, table_name, filter_expr)

    for table_name in select_embedding_tables(conn):
        filter_expr = _build_documents_filter(
            conn=conn,
            table_name=table_name,
            collection=collection,
            doc_ids=normalized_doc_ids,
            user_id=user_id,
            is_admin=is_admin,
        )
        _replace_predicate(predicates, table_name, filter_expr)

    return _execute_or_plan_by_predicates(
        conn,
        predicates,
        preview_only=preview_only,
        confirm=confirm,
        model_tag=None,
    )


def cleanup_cascade(
    collection: str,
    doc_id: str,
    scope: str,
    new_parse_hash: Optional[str] = None,
    old_parse_hash: Optional[str] = None,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cleanup_cascade(
        collection=collection,
        doc_id=doc_id,
        scope=scope,
        new_parse_hash=new_parse_hash,
        old_parse_hash=old_parse_hash,
        model_tag=model_tag,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _cleanup_cascade_impl(
    collection: str,
    doc_id: str,
    scope: str,
    new_parse_hash: Optional[str] = None,
    old_parse_hash: Optional[str] = None,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: Optional[bool] = None,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Unified cascade cleanup by scope with preview/confirm semantics.

    Args:
        collection: Collection name
        doc_id: Document ID
        scope: "document" | "parse" | "chunk" | "embeddings" | "pointers"
        new_parse_hash: New main parse hash for parse/chunk scopes
        old_parse_hash: Optional old main parse hash (auto-filled from pointers if None)
        model_tag: Optional embed model tag limiter
        user_id: Optional user ID for tenant scoping
        is_admin: Whether caller is admin (None to fallback to context, defaults to True
                   for system-level version promotion operations)
        preview_only: If True, only plan counts
        confirm: If True, execute deletions

    Returns:
        Deleted (or planned) counts per table scope
    """
    return _get_version_compatibility_facade().cleanup_cascade(
        collection=collection,
        doc_id=doc_id,
        scope=scope,
        new_parse_hash=new_parse_hash,
        old_parse_hash=old_parse_hash,
        model_tag=model_tag,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _collapse_embedding_table_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """Collapse concrete embeddings table counts into the legacy summary key."""
    collapsed = dict(counts)
    embeddings_total = int(collapsed.pop("__embeddings__", 0)) + int(
        collapsed.pop("embeddings", 0)
    )
    for table_name in list(collapsed):
        if str(table_name).startswith("embeddings_"):
            embeddings_total += int(collapsed.pop(table_name, 0))
    collapsed["embeddings"] = embeddings_total
    return collapsed


def cleanup_document_cascade(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cleanup_document_cascade(
        collection=collection,
        doc_id=doc_id,
        model_tag=model_tag,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _cleanup_document_cascade_impl(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Cascade delete all data for a document across all stages.

    Order: embeddings_* -> chunks -> parses -> main_pointers -> documents

    Args:
        collection: Collection name
        doc_id: Document ID
        model_tag: Optional model tag to limit embeddings deletion

    Returns:
        Deleted counts per scope
    """
    try:
        # Delegate to unified entry
        return _cleanup_cascade_impl(
            collection=collection,
            doc_id=doc_id,
            scope="document",
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup document cascade: {e}")


def cleanup_parse_cascade(
    collection: str,
    doc_id: str,
    old_parse_hash: Optional[str] = None,
    new_parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cleanup_parse_cascade(
        collection=collection,
        doc_id=doc_id,
        old_parse_hash=old_parse_hash,
        new_parse_hash=new_parse_hash,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _cleanup_parse_cascade_impl(
    collection: str,
    doc_id: str,
    old_parse_hash: Optional[str] = None,
    new_parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Clean up cascade when promoting a new parse version.

    This method:
    1. Deletes old parse's chunks and embeddings
    2. Deletes other parse candidates and their downstream data

    Args:
        collection: Collection name
        doc_id: Document ID
        old_parse_hash: Old main parse hash (optional)
        new_parse_hash: New main parse hash (optional)

    Returns:
        Dictionary with deletion counts

    Raises:
        CascadeCleanupError: If cleanup fails
    """
    try:
        return _cleanup_cascade_impl(
            collection=collection,
            doc_id=doc_id,
            scope="parse",
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup parse cascade: {e}")


def cleanup_chunk_cascade(
    collection: str,
    doc_id: str,
    old_parse_hash: Optional[str] = None,
    new_parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cleanup_chunk_cascade(
        collection=collection,
        doc_id=doc_id,
        old_parse_hash=old_parse_hash,
        new_parse_hash=new_parse_hash,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _cleanup_chunk_cascade_impl(
    collection: str,
    doc_id: str,
    old_parse_hash: Optional[str] = None,
    new_parse_hash: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Clean up cascade when promoting a new chunk version.

    This method:
    1. Deletes old chunk's embeddings
    2. Deletes other chunk candidates

    Args:
        collection: Collection name
        doc_id: Document ID
        old_parse_hash: Old main parse hash (optional)
        new_parse_hash: New main parse hash (optional)

    Returns:
        Dictionary with deletion counts

    Raises:
        CascadeCleanupError: If cleanup fails
    """
    try:
        return _cleanup_cascade_impl(
            collection=collection,
            doc_id=doc_id,
            scope="chunk",
            new_parse_hash=new_parse_hash,
            old_parse_hash=old_parse_hash,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup chunk cascade: {e}")


def cleanup_embed_cascade(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    old_technical_id: Optional[str] = None,
    new_technical_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    return _get_version_compatibility_facade().cleanup_embed_cascade(
        collection=collection,
        doc_id=doc_id,
        model_tag=model_tag,
        old_technical_id=old_technical_id,
        new_technical_id=new_technical_id,
        user_id=user_id,
        is_admin=is_admin,
        preview_only=preview_only,
        confirm=confirm,
    )


def _cleanup_embed_cascade_impl(
    collection: str,
    doc_id: str,
    model_tag: Optional[str] = None,
    old_technical_id: Optional[str] = None,
    new_technical_id: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = True,
    preview_only: bool = True,
    confirm: bool = False,
) -> Dict[str, int]:
    """Clean up cascade when promoting a new embeddings version.

    This method:
    1. Deletes other embeddings candidates (optionally filtered by model_tag)

    Args:
        collection: Collection name
        doc_id: Document ID
        model_tag: Model tag filter (optional)
        old_technical_id: Old main technical ID (optional)
        new_technical_id: New main technical ID (optional)

    Returns:
        Dictionary with deletion counts

    Raises:
        CascadeCleanupError: If cleanup fails
    """
    try:
        # Delegate to unified entry; old/new technical ids are not used in current schema
        return _cleanup_cascade_impl(
            collection=collection,
            doc_id=doc_id,
            scope="embeddings",
            model_tag=model_tag,
            user_id=user_id,
            is_admin=is_admin,
            preview_only=preview_only,
            confirm=confirm,
        )

    except Exception as e:
        raise CascadeCleanupError(f"Failed to cleanup embed cascade: {e}")
