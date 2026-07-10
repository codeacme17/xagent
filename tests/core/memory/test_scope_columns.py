"""Unit tests for the derived scope-column encoding (#822, slice 001/002)."""

from __future__ import annotations

from xagent.core.execution_scope import MEMORY_DIMENSION_METADATA_PREFIX
from xagent.core.memory.scope_columns import (
    SCOPE_DIMS_COLUMN,
    SCOPE_EXCLUSIVE_FILTER_KEY,
    USER_ID_COLUMN,
    build_scope_where,
    coerce_user_id,
    derive_scope_columns,
    encode_scope_dims,
    scope_dim_where_term,
    user_id_where_term,
)

P = MEMORY_DIMENSION_METADATA_PREFIX


def test_encode_empty_when_no_dimensions():
    assert encode_scope_dims({}) == []
    assert encode_scope_dims({"user_id": 7, "content": "x"}) == []


def test_encode_single_and_multiple_dimensions():
    assert encode_scope_dims({f"{P}tenant": "acme"}) == ["tenant=acme"]
    # Sorted by key (order-independent).
    assert encode_scope_dims({f"{P}tenant": "acme", f"{P}agent": "x"}) == [
        "agent=x",
        "tenant=acme",
    ]


def test_encoding_is_order_independent():
    a = encode_scope_dims({f"{P}tenant": "acme", f"{P}agent": "x"})
    b = encode_scope_dims({f"{P}agent": "x", f"{P}tenant": "acme"})
    assert a == b


def test_encode_preserves_raw_values_without_escaping():
    # array_contains is exact per-element equality, so no delimiter escaping.
    assert encode_scope_dims({f"{P}path": "/a=b/c"}) == ["path=/a=b/c"]
    assert encode_scope_dims({f"{P}t": "a_b"}) == ["t=a_b"]


def test_coerce_user_id():
    assert coerce_user_id(7) == 7
    assert coerce_user_id("7") == 7
    assert coerce_user_id(None) is None
    assert coerce_user_id("not-an-int") is None


def test_derive_scope_columns_from_json():
    uid, dims = derive_scope_columns(
        '{"user_id": 7, "' + P + 'tenant": "acme", "content": "x"}'
    )
    assert uid == 7
    assert dims == ["tenant=acme"]


def test_derive_scope_columns_tolerates_bad_input():
    assert derive_scope_columns(None) == (None, [])
    assert derive_scope_columns("") == (None, [])
    assert derive_scope_columns("not json") == (None, [])
    assert derive_scope_columns("[1, 2, 3]") == (None, [])


# --- where-clause construction (slice 002) ---------------------------------


def test_user_id_where_term():
    assert user_id_where_term(7) == f"{USER_ID_COLUMN} = 7"
    assert user_id_where_term("7") == f"{USER_ID_COLUMN} = 7"
    assert user_id_where_term(None) is None
    assert user_id_where_term("bad") is None


def test_scope_dim_where_term_is_exact_array_contains():
    assert (
        scope_dim_where_term("tenant", "acme")
        == f"array_contains({SCOPE_DIMS_COLUMN}, 'tenant=acme')"
    )
    # Values with SQL-special quotes are doubled; no LIKE wildcards to escape.
    assert scope_dim_where_term("t", "a'b") == (
        f"array_contains({SCOPE_DIMS_COLUMN}, 't=a''b')"
    )
    # Underscore/equals are stored/matched literally (no wildcard semantics).
    assert scope_dim_where_term("t", "a_b") == (
        f"array_contains({SCOPE_DIMS_COLUMN}, 't=a_b')"
    )


def test_build_scope_where_pushes_user_and_dims():
    where_sql, residual = build_scope_where(
        {"metadata": {"user_id": 1, f"{P}tenant": "acme"}}
    )
    assert where_sql is not None
    assert f"{USER_ID_COLUMN} = 1" in where_sql
    assert f"array_contains({SCOPE_DIMS_COLUMN}, 'tenant=acme')" in where_sql
    assert " AND " in where_sql
    assert residual == {}


def test_build_scope_where_keeps_residual_filters():
    where_sql, residual = build_scope_where(
        {
            "category": "work",
            "metadata": {"user_id": 1, f"{P}tenant": "acme", "custom": "v"},
        }
    )
    assert where_sql is not None
    assert residual == {"category": "work", "metadata": {"custom": "v"}}


def test_build_scope_where_unscoped_and_empty():
    assert build_scope_where(None) == (None, {})
    assert build_scope_where({}) == (None, {})
    where_sql, residual = build_scope_where({"metadata": {"user_id": 5}})
    assert where_sql == f"{USER_ID_COLUMN} = 5"
    assert residual == {}


def test_build_scope_where_unparsable_user_id_falls_back_to_python():
    where_sql, residual = build_scope_where({"metadata": {"user_id": "abc"}})
    assert where_sql is None
    assert residual == {"metadata": {"user_id": "abc"}}


def test_build_scope_where_exclusive_directive():
    where_sql, residual = build_scope_where(
        {"metadata": {"user_id": 1}, SCOPE_EXCLUSIVE_FILTER_KEY: True}
    )
    assert where_sql == (
        f"{USER_ID_COLUMN} = 1 AND array_length({SCOPE_DIMS_COLUMN}) = 0"
    )
    # The reserved directive is consumed, never left for equality matching.
    assert residual == {}
