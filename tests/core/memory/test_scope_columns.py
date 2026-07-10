"""Unit tests for the derived scope-column encoding (#822, slice 001)."""

from __future__ import annotations

from xagent.core.execution_scope import MEMORY_DIMENSION_METADATA_PREFIX
from xagent.core.memory.scope_columns import (
    coerce_user_id,
    derive_scope_columns,
    encode_scope_dims,
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
