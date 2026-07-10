"""Derived filter columns for the LanceDB memory store (#822).

Promotes the two fields that scope searches filter on — the owner ``user_id``
and the execution-scope memory dimensions — out of the JSON ``metadata`` string
into real, top-level columns. LanceDB ``where`` cannot reach fields inside the
metadata JSON string, so these columns are the prerequisite for pushing the
filters into a ``where`` prefilter (slice 002) instead of the Python
post-filtering that collapses recall under shared collections (see #822).

The ``metadata`` JSON stays authoritative: these columns are **derived
projections**, computed from a note's metadata on write and back-filled from it
on migration. Nothing reconstructs a note from them.

``scope_dims`` is a ``list<string>`` column holding one ``"key=value"`` element
per dimension (e.g. ``["agent=x", "tenant=acme"]``). Membership is tested with
DataFusion's ``array_contains`` (slice 002), which is exact per-element string
equality — so values may contain any character (``=``, ``/``, ``_`` …) with no
escaping, and there is no substring-collision surface (``tenant=a`` never matches
``tenant=ab``).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from ..execution_scope import MEMORY_DIMENSION_METADATA_PREFIX

# Real column names the memory dimensions are promoted into.
USER_ID_COLUMN = "user_id"
SCOPE_DIMS_COLUMN = "scope_dims"


def scope_dim_element(dim_key: str, value: Any) -> str:
    """The ``"key=value"`` list element stored/matched for one dimension."""
    return f"{dim_key}={value}"


def encode_scope_dims(metadata: Mapping[str, Any]) -> list[str]:
    """The scope-dimension list for a note: one ``"key=value"`` per stamp.

    Reads the ``execution_scope_<key>`` entries from ``metadata``, sorted by key
    (order-independent). Returns ``[]`` when the note carries no dimensions.
    """
    return [
        scope_dim_element(key[len(MEMORY_DIMENSION_METADATA_PREFIX) :], metadata[key])
        for key in sorted(metadata)
        if key.startswith(MEMORY_DIMENSION_METADATA_PREFIX)
    ]


def coerce_user_id(value: Any) -> Optional[int]:
    """Best-effort integer owner id for the column; ``None`` when absent/bad."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sql_string_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def user_id_where_term(value: Any) -> Optional[str]:
    """Equality predicate on the ``user_id`` column, or ``None`` if unparsable."""
    user_id = coerce_user_id(value)
    if user_id is None:
        return None
    return f"{USER_ID_COLUMN} = {user_id}"


def scope_dim_where_term(dim_key: str, value: Any) -> str:
    """An ``array_contains`` predicate matching notes carrying this dimension.

    Exact per-element string equality, so a note carrying a superset of the
    queried dimensions still matches while a different value or a prefix does
    not — no escaping or collision reasoning required.
    """
    element = _sql_string_literal(scope_dim_element(dim_key, value))
    return f"array_contains({SCOPE_DIMS_COLUMN}, {element})"


def build_scope_where(
    filters: Optional[Mapping[str, Any]],
) -> tuple[Optional[str], dict[str, Any]]:
    """Split ``filters`` into a pushable ``where`` clause and residual filters.

    Extracts ``user_id`` and the ``execution_scope_*`` dimensions from the nested
    ``filters["metadata"]`` into column predicates (``user_id`` equality +
    ``array_contains`` per dimension), AND-ed together and applied as a prefilter
    so the ANN returns ``k`` already-scoped neighbours. Everything the clause
    cannot express — category, arbitrary metadata keys, an unparsable
    ``user_id`` — is returned as residual filters for the Python post-filter.

    Returns ``(where_sql or None, residual_filters)``.
    """
    if not filters:
        return None, {}
    residual: dict[str, Any] = dict(filters)
    clauses: list[str] = []
    metadata = filters.get("metadata")
    if isinstance(metadata, Mapping):
        residual_metadata = dict(metadata)
        if "user_id" in residual_metadata:
            term = user_id_where_term(residual_metadata["user_id"])
            if term is not None:
                clauses.append(term)
                residual_metadata.pop("user_id")
        for key in list(residual_metadata):
            if key.startswith(MEMORY_DIMENSION_METADATA_PREFIX):
                dim = key[len(MEMORY_DIMENSION_METADATA_PREFIX) :]
                clauses.append(scope_dim_where_term(dim, residual_metadata[key]))
                residual_metadata.pop(key)
        if residual_metadata:
            residual["metadata"] = residual_metadata
        else:
            residual.pop("metadata", None)
    where_sql = " AND ".join(clauses) if clauses else None
    return where_sql, residual


def derive_scope_columns(
    metadata_json: Optional[str],
) -> tuple[Optional[int], list[str]]:
    """Derive ``(user_id, scope_dims)`` from a stored metadata JSON string.

    Used to back-fill the columns for existing rows during migration. Malformed
    or non-object metadata yields ``(None, [])`` rather than raising, so one bad
    row cannot abort a migration.
    """
    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    if not isinstance(metadata, dict):
        return None, []
    return coerce_user_id(metadata.get("user_id")), encode_scope_dims(metadata)
