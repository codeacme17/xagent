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
